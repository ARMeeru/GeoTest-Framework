# Performance testing framework for load, stress, and benchmark testing
# Provides utilities for concurrent API testing and performance analysis

import asyncio
import time
import json
import statistics
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading

from .api_client import RestCountriesClient
from .monitoring import get_metrics_collector

logger = logging.getLogger(__name__)


@dataclass
class PerformanceResult:
    # Single performance test result
    test_name: str
    endpoint: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    response_size: Optional[int] = None
    error_message: Optional[str] = None
    status_code: Optional[int] = None


@dataclass
class LoadTestResult:
    # Results from a load test execution
    test_name: str
    total_requests: int
    concurrent_users: int
    duration: float
    success_count: int
    error_count: int
    success_rate: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    errors_per_second: float
    total_data_transferred: int
    timestamp: str


@dataclass
class StressTestResult:
    # Results from a stress test execution
    test_name: str
    max_users_tested: int
    breaking_point: Optional[int]  # Users where system starts failing
    ramp_up_duration: float
    total_duration: float
    peak_rps: float
    peak_success_rate: float
    degradation_point: Optional[int]  # Users where performance degrades
    recovery_time: Optional[float]
    load_phases: List[Dict[str, Any]]
    timestamp: str


@dataclass
class BenchmarkResult:
    # Results from a benchmark test
    test_name: str
    endpoint: str
    baseline_response_time: float
    current_response_time: float
    performance_change_percent: float
    regression_detected: bool
    regression_threshold: float
    iterations: int
    timestamp: str


class PerformanceTestRunner:
    # Main performance testing orchestrator
    
    def __init__(self, base_url: str = None, max_workers: int = 50):
        self.base_url = base_url or "https://restcountries.com/v3.1"
        self.max_workers = max_workers
        self.metrics_collector = get_metrics_collector()
        self._results: List[PerformanceResult] = []
        self._lock = threading.Lock()
        
    def create_client(self) -> RestCountriesClient:
        # Create a new API client instance for testing
        return RestCountriesClient()
    
    def single_request_test(self, endpoint: str, test_name: str = None, 
                          client: RestCountriesClient = None) -> PerformanceResult:
        # Execute a single API request and measure performance
        if not client:
            client = self.create_client()
        
        test_name = test_name or f"single_{endpoint.replace('/', '_')}"
        
        start_time = time.time()
        success = False
        response_size = None
        error_message = None
        status_code = None
        
        try:
            # Make the API request based on endpoint
            if endpoint == "all":
                response = client.get_all_countries(fields="name,cca2,region")
            elif endpoint.startswith("name/"):
                country_name = endpoint.split("/", 1)[1]
                response = client.get_country_by_name(country_name)
            elif endpoint.startswith("alpha/"):
                code = endpoint.split("/", 1)[1]
                response = client.get_country_by_code(code)
            elif endpoint.startswith("currency/"):
                currency = endpoint.split("/", 1)[1]
                response = client.get_countries_by_currency(currency)
            elif endpoint.startswith("region/"):
                region = endpoint.split("/", 1)[1]
                response = client.get_countries_by_region(region)
            else:
                # Default to all countries
                response = client.get_all_countries(fields="name,cca2,region")
            
            # Calculate response size
            response_size = len(str(response)) if response else 0
            success = True
            status_code = 200  # Assume success if no exception
            
        except Exception as e:
            error_message = str(e)
            logger.debug(f"Request failed for {endpoint}: {error_message}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        result = PerformanceResult(
            test_name=test_name,
            endpoint=endpoint,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            success=success,
            response_size=response_size,
            error_message=error_message,
            status_code=status_code
        )
        
        # Record metrics
        self.metrics_collector.record_custom_metric(
            f"performance.{test_name}.response_time", 
            duration,
            tags={"endpoint": endpoint, "success": str(success)}
        )
        
        with self._lock:
            self._results.append(result)
        
        return result
    
    def load_test(self, endpoint: str, concurrent_users: int, 
                  requests_per_user: int = 1, test_name: str = None) -> LoadTestResult:
        # Execute load test with specified concurrent users
        test_name = test_name or f"load_{endpoint.replace('/', '_')}_{concurrent_users}users"
        total_requests = concurrent_users * requests_per_user
        
        logger.info(f"Starting load test: {test_name} with {concurrent_users} users, {total_requests} total requests")
        
        start_time = time.time()
        results = []
        
        def user_simulation(user_id: int) -> List[PerformanceResult]:
            # Simulate a single user making requests
            user_results = []
            client = self.create_client()
            
            for req_num in range(requests_per_user):
                result = self.single_request_test(
                    endpoint, 
                    f"{test_name}_user{user_id}_req{req_num}",
                    client
                )
                user_results.append(result)
                # Small delay between requests from same user
                time.sleep(0.1)
            
            client.close()
            return user_results
        
        # Execute concurrent users
        with ThreadPoolExecutor(max_workers=min(concurrent_users, self.max_workers)) as executor:
            future_to_user = {
                executor.submit(user_simulation, user_id): user_id 
                for user_id in range(concurrent_users)
            }
            
            for future in as_completed(future_to_user):
                user_id = future_to_user[future]
                try:
                    user_results = future.result()
                    results.extend(user_results)
                except Exception as e:
                    logger.error(f"User {user_id} simulation failed: {e}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Analyze results
        success_results = [r for r in results if r.success]
        error_results = [r for r in results if not r.success]
        
        response_times = [r.duration for r in success_results]
        
        # Calculate statistics
        success_count = len(success_results)
        error_count = len(error_results)
        success_rate = (success_count / total_requests * 100) if total_requests > 0 else 0
        
        avg_response_time = statistics.mean(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        median_response_time = statistics.median(response_times) if response_times else 0
        
        # Percentiles
        if response_times:
            sorted_times = sorted(response_times)
            p95_index = int(0.95 * len(sorted_times))
            p99_index = int(0.99 * len(sorted_times))
            p95_response_time = sorted_times[p95_index] if p95_index < len(sorted_times) else max_response_time
            p99_response_time = sorted_times[p99_index] if p99_index < len(sorted_times) else max_response_time
        else:
            p95_response_time = p99_response_time = 0
        
        # Throughput calculations
        requests_per_second = total_requests / total_duration if total_duration > 0 else 0
        errors_per_second = error_count / total_duration if total_duration > 0 else 0
        
        # Data transfer calculation
        total_data_transferred = sum(r.response_size or 0 for r in success_results)
        
        load_result = LoadTestResult(
            test_name=test_name,
            total_requests=total_requests,
            concurrent_users=concurrent_users,
            duration=total_duration,
            success_count=success_count,
            error_count=error_count,
            success_rate=success_rate,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            median_response_time=median_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            errors_per_second=errors_per_second,
            total_data_transferred=total_data_transferred,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        logger.info(f"Load test completed: {success_count}/{total_requests} successful, "
                   f"{success_rate:.1f}% success rate, {requests_per_second:.1f} RPS")
        
        return load_result
    
    def stress_test(self, endpoint: str, min_users: int = 1, max_users: int = 100, 
                   step_size: int = 10, step_duration: int = 30, 
                   test_name: str = None) -> StressTestResult:
        # Execute stress test with ramping load
        test_name = test_name or f"stress_{endpoint.replace('/', '_')}"
        
        logger.info(f"Starting stress test: {test_name} from {min_users} to {max_users} users")
        
        start_time = time.time()
        load_phases = []
        breaking_point = None
        degradation_point = None
        peak_rps = 0
        peak_success_rate = 0
        
        # Ramp up load in steps
        current_users = min_users
        while current_users <= max_users:
            phase_start = time.time()
            
            # Run load test for current user count
            load_result = self.load_test(
                endpoint, 
                current_users, 
                requests_per_user=1,
                test_name=f"{test_name}_phase_{current_users}users"
            )
            
            phase_duration = time.time() - phase_start
            
            # Record phase results
            phase_data = {
                'users': current_users,
                'duration': phase_duration,
                'success_rate': load_result.success_rate,
                'rps': load_result.requests_per_second,
                'avg_response_time': load_result.avg_response_time,
                'p95_response_time': load_result.p95_response_time
            }
            load_phases.append(phase_data)
            
            # Track peak performance
            if load_result.requests_per_second > peak_rps:
                peak_rps = load_result.requests_per_second
                peak_success_rate = load_result.success_rate
            
            # Detect degradation (success rate drops below 95% or response time > 5s)
            if (load_result.success_rate < 95 or load_result.avg_response_time > 5.0) and not degradation_point:
                degradation_point = current_users
                logger.warning(f"Performance degradation detected at {current_users} users")
            
            # Detect breaking point (success rate drops below 50% or response time > 10s)
            if (load_result.success_rate < 50 or load_result.avg_response_time > 10.0) and not breaking_point:
                breaking_point = current_users
                logger.warning(f"System breaking point detected at {current_users} users")
                break
            
            current_users += step_size
            
            # Brief pause between phases
            if current_users <= max_users:
                time.sleep(5)
        
        total_duration = time.time() - start_time
        ramp_up_duration = sum(phase['duration'] for phase in load_phases)
        
        # Test recovery time (if system broke)
        recovery_time = None
        if breaking_point:
            logger.info("Testing system recovery...")
            recovery_start = time.time()
            
            # Test with minimal load to see recovery
            recovery_result = self.load_test(endpoint, 1, test_name=f"{test_name}_recovery")
            
            if recovery_result.success_rate > 90:
                recovery_time = time.time() - recovery_start
                logger.info(f"System recovered in {recovery_time:.2f} seconds")
            else:
                logger.warning("System did not recover during test period")
        
        stress_result = StressTestResult(
            test_name=test_name,
            max_users_tested=current_users - step_size,
            breaking_point=breaking_point,
            ramp_up_duration=ramp_up_duration,
            total_duration=total_duration,
            peak_rps=peak_rps,
            peak_success_rate=peak_success_rate,
            degradation_point=degradation_point,
            recovery_time=recovery_time,
            load_phases=load_phases,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        logger.info(f"Stress test completed: Peak {peak_rps:.1f} RPS at {peak_success_rate:.1f}% success rate")
        
        return stress_result
    
    def benchmark_test(self, endpoint: str, baseline_file: Path = None, 
                      iterations: int = 10, test_name: str = None) -> BenchmarkResult:
        # Execute benchmark test against baseline
        test_name = test_name or f"benchmark_{endpoint.replace('/', '_')}"
        
        logger.info(f"Starting benchmark test: {test_name} with {iterations} iterations")
        
        # Execute multiple iterations for stable measurement
        response_times = []
        for i in range(iterations):
            result = self.single_request_test(endpoint, f"{test_name}_iter_{i}")
            if result.success:
                response_times.append(result.duration)
        
        current_response_time = statistics.median(response_times) if response_times else 0
        
        # Load baseline if available
        baseline_response_time = None
        regression_threshold = 0.2  # 20% degradation threshold
        
        if baseline_file and baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                    baseline_response_time = baseline_data.get(endpoint, {}).get('response_time')
                    regression_threshold = baseline_data.get('regression_threshold', 0.2)
            except Exception as e:
                logger.warning(f"Failed to load baseline: {e}")
        
        # Calculate performance change
        if baseline_response_time:
            performance_change = ((current_response_time - baseline_response_time) / baseline_response_time) * 100
            regression_detected = performance_change > (regression_threshold * 100)
        else:
            baseline_response_time = current_response_time  # Set current as baseline
            performance_change = 0.0
            regression_detected = False
        
        benchmark_result = BenchmarkResult(
            test_name=test_name,
            endpoint=endpoint,
            baseline_response_time=baseline_response_time,
            current_response_time=current_response_time,
            performance_change_percent=performance_change,
            regression_detected=regression_detected,
            regression_threshold=regression_threshold,
            iterations=len(response_times),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        logger.info(f"Benchmark completed: {current_response_time:.3f}s "
                   f"({performance_change:+.1f}% vs baseline)")
        
        return benchmark_result
    
    def save_baseline(self, results: List[BenchmarkResult], baseline_file: Path):
        # Save benchmark results as new baseline
        baseline_data = {
            'created': datetime.now(timezone.utc).isoformat(),
            'regression_threshold': 0.2,
            'endpoints': {}
        }
        
        for result in results:
            baseline_data['endpoints'][result.endpoint] = {
                'response_time': result.current_response_time,
                'test_name': result.test_name,
                'iterations': result.iterations
            }
        
        baseline_file.parent.mkdir(exist_ok=True)
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        logger.info(f"Saved baseline with {len(results)} endpoints to {baseline_file}")
    
    def export_results(self, output_dir: Path, 
                      load_results: List[LoadTestResult] = None,
                      stress_results: List[StressTestResult] = None,
                      benchmark_results: List[BenchmarkResult] = None) -> Dict[str, str]:
        # Export all performance test results
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        files_created = {}
        
        # Export load test results
        if load_results:
            load_file = output_dir / "load_test_results.json"
            with open(load_file, 'w') as f:
                json.dump([asdict(r) for r in load_results], f, indent=2)
            files_created['load_tests'] = str(load_file)
        
        # Export stress test results
        if stress_results:
            stress_file = output_dir / "stress_test_results.json"
            with open(stress_file, 'w') as f:
                json.dump([asdict(r) for r in stress_results], f, indent=2)
            files_created['stress_tests'] = str(stress_file)
        
        # Export benchmark results
        if benchmark_results:
            benchmark_file = output_dir / "benchmark_results.json"
            with open(benchmark_file, 'w') as f:
                json.dump([asdict(r) for r in benchmark_results], f, indent=2)
            files_created['benchmarks'] = str(benchmark_file)
        
        # Export individual request results
        individual_file = output_dir / "individual_requests.json"
        with open(individual_file, 'w') as f:
            json.dump([asdict(r) for r in self._results], f, indent=2, default=str)
        files_created['individual_requests'] = str(individual_file)
        
        # Create summary report
        summary = {
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'total_individual_requests': len(self._results),
            'load_tests_count': len(load_results) if load_results else 0,
            'stress_tests_count': len(stress_results) if stress_results else 0,
            'benchmark_tests_count': len(benchmark_results) if benchmark_results else 0
        }
        
        summary_file = output_dir / "performance_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        files_created['summary'] = str(summary_file)
        
        logger.info(f"Exported performance results to {len(files_created)} files in {output_dir}")
        return files_created