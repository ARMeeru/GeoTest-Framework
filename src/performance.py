# Unified Performance Testing and Analysis Framework
# Consolidates performance_testing.py, performance_analyzer.py, and load_generator.py
# Provides comprehensive performance testing, analysis, and reporting capabilities

import asyncio
import time
import json
import statistics
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .api_client import RestCountriesClient
from .monitoring import get_metrics_collector

logger = logging.getLogger(__name__)


# ==== DATA MODELS ====

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
    timestamp: str


@dataclass
class StressTestResult:
    # Results from stress test execution
    test_name: str
    min_users: int
    max_users: int
    ramp_duration: float
    hold_duration: float
    breaking_point: Optional[int]
    max_sustained_users: int
    total_requests: int
    total_errors: int
    avg_response_time: float
    error_rate_at_breaking_point: float
    recovery_time: Optional[float]
    timestamp: str


@dataclass
class BenchmarkResult:
    # Benchmark test result for comparison
    test_name: str
    endpoint: str
    iterations: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    std_deviation: float
    success_rate: float
    requests_per_second: float
    timestamp: str
    comparison_baseline: Optional[float] = None
    performance_change: Optional[float] = None


@dataclass
class UserBehavior:
    # User behavior pattern for load testing
    name: str
    think_time_range: Tuple[float, float]  # Min/max think time between requests
    request_pattern: List[str]  # List of endpoints to request
    session_duration: float  # How long user stays active
    error_tolerance: float  # Percentage of errors before user gives up


@dataclass
class PerformanceInsight:
    # Individual performance insight or recommendation
    category: str  # 'performance', 'reliability', 'scalability', 'efficiency'
    severity: str  # 'info', 'warning', 'critical'
    title: str
    description: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    recommendation: Optional[str] = None


@dataclass
class PerformanceTrend:
    # Performance trend analysis over time
    metric_name: str
    time_period: str
    trend_direction: str  # 'improving', 'degrading', 'stable'
    change_percentage: float
    statistical_significance: float
    data_points: int


@dataclass
class PerformanceReport:
    # Comprehensive performance analysis report
    test_type: str
    timestamp: str
    summary: Dict[str, Any]
    metrics: Dict[str, Any]
    insights: List[PerformanceInsight]
    trends: List[PerformanceTrend]
    charts: List[str]
    recommendations: List[str]


# ==== LOAD PATTERNS ====

class LoadPattern:
    # Different load testing patterns
    
    @staticmethod
    def constant_load(users: int, duration: float) -> Callable:
        """Constant number of users for specified duration"""
        def pattern():
            return [(users, duration)]
        return pattern
    
    @staticmethod
    def ramp_up(start_users: int, end_users: int, duration: float) -> Callable:
        """Gradually increase users from start to end over duration"""
        def pattern():
            steps = 10
            step_duration = duration / steps
            user_increment = (end_users - start_users) / steps
            return [(int(start_users + i * user_increment), step_duration) 
                   for i in range(steps + 1)]
        return pattern
    
    @staticmethod
    def spike_test(base_users: int, spike_users: int, spike_duration: float) -> Callable:
        """Sudden spike in users for short duration"""
        def pattern():
            return [
                (base_users, 60),  # Baseline
                (spike_users, spike_duration),  # Spike
                (base_users, 60)  # Return to baseline
            ]
        return pattern


# ==== UNIFIED PERFORMANCE ENGINE ====

class UnifiedPerformanceEngine:
    """
    Consolidated performance testing and analysis engine
    Replaces performance_testing.py, performance_analyzer.py, and load_generator.py
    """
    
    def __init__(self, base_url: str = None, max_workers: int = 50, output_dir: Path = None):
        self.base_url = base_url or "https://restcountries.com/v3.1"
        self.max_workers = max_workers
        self.output_dir = output_dir or Path("performance_analysis")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Performance thresholds from configuration
        self.thresholds = {
            'response_time_ms': {
                'excellent': 200,
                'good': 500,
                'acceptable': 1000,
                'poor': 2000
            },
            'success_rate': {
                'excellent': 99.9,
                'good': 99.5,
                'acceptable': 99.0,
                'poor': 95.0
            },
            'error_rate': {
                'excellent': 0.1,
                'good': 0.5,
                'acceptable': 1.0,
                'poor': 5.0
            }
        }
        
        # User behaviors from configuration
        self.user_behaviors = {
            'light_user': UserBehavior(
                name='light_user',
                think_time_range=(2.0, 5.0),
                request_pattern=['/all', '/name/{country}'],
                session_duration=120.0,
                error_tolerance=10.0
            ),
            'api_explorer': UserBehavior(
                name='api_explorer',
                think_time_range=(1.0, 3.0),
                request_pattern=['/all', '/name/{country}', '/alpha/{code}', '/region/{region}'],
                session_duration=300.0,
                error_tolerance=5.0
            ),
            'power_user': UserBehavior(
                name='power_user',
                think_time_range=(0.5, 1.5),
                request_pattern=['/all', '/name/{country}', '/alpha/{code}', '/currency/{currency}', '/lang/{language}'],
                session_duration=600.0,
                error_tolerance=2.0
            )
        }
        
        self._results = []
        self.metrics_collector = get_metrics_collector()
    
    def create_client(self) -> RestCountriesClient:
        """Create a new API client instance"""
        return RestCountriesClient()
    
    # ==== CORE TESTING METHODS ====
    
    def single_request_test(self, endpoint: str, test_name: str = None, 
                           params: Dict = None) -> PerformanceResult:
        """Execute a single request performance test"""
        test_name = test_name or f"single_request_{endpoint.replace('/', '_')}"
        client = self.create_client()
        
        start_time = time.time()
        try:
            # Use appropriate API client method based on endpoint
            if endpoint == "all" or endpoint == "/all":
                data = client.get_all_countries(fields="name,cca2,region")
                success = isinstance(data, list) and len(data) > 0
                status_code = 200
                response_size = len(str(data))
            elif endpoint.startswith("name/") or endpoint.startswith("/name/"):
                country_name = endpoint.split("/name/")[-1] if "/name/" in endpoint else endpoint.split("name/")[-1]
                data = client.get_country_by_name(country_name)
                success = data is not None
                status_code = 200 if success else 404
                response_size = len(str(data)) if data else 0
            elif endpoint.startswith("alpha/") or endpoint.startswith("/alpha/"):
                alpha_code = endpoint.split("/alpha/")[-1] if "/alpha/" in endpoint else endpoint.split("alpha/")[-1]
                data = client.get_country_by_code(alpha_code)
                success = isinstance(data, list) and len(data) > 0
                status_code = 200 if success else 404
                response_size = len(str(data)) if data else 0
            elif endpoint.startswith("region/") or endpoint.startswith("/region/"):
                region = endpoint.split("/region/")[-1] if "/region/" in endpoint else endpoint.split("region/")[-1]
                data = client.get_countries_by_region(region)
                success = isinstance(data, list) and len(data) > 0
                status_code = 200 if success else 404
                response_size = len(str(data)) if data else 0
            else:
                # Fallback to direct session request for unknown endpoints
                response = client.session.get(f"{self.base_url}/{endpoint}", params=params or {})
                success = response.status_code == 200
                status_code = response.status_code
                response_size = len(response.content) if hasattr(response, 'content') else 0
            
            end_time = time.time()
            
            result = PerformanceResult(
                test_name=test_name,
                endpoint=endpoint,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                success=success,
                response_size=response_size,
                status_code=status_code,
                error_message=None if success else f"HTTP {status_code}"
            )
            
        except Exception as e:
            end_time = time.time()
            result = PerformanceResult(
                test_name=test_name,
                endpoint=endpoint,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                success=False,
                error_message=str(e),
                status_code=None
            )
        
        self._results.append(result)
        return result
    
    def load_test(self, endpoint: str, concurrent_users: int, 
                  total_requests: int, test_name: str = None) -> LoadTestResult:
        """Execute a load test with concurrent users"""
        test_name = test_name or f"load_test_{endpoint.replace('/', '_')}"
        logger.info(f"Starting load test: {test_name} with {concurrent_users} users, {total_requests} total requests")
        
        start_time = time.time()
        results = []
        requests_per_user = total_requests // concurrent_users
        
        def user_simulation(user_id: int) -> List[PerformanceResult]:
            client = self.create_client()
            user_results = []
            
            for i in range(requests_per_user):
                request_start = time.time()
                try:
                    # Use appropriate API client method based on endpoint
                    if endpoint == "/all":
                        data = client.get_all_countries(fields="name,cca2,region")
                    elif endpoint.startswith("/name/"):
                        country_name = endpoint.split("/name/")[1]
                        data = client.get_country_by_name(country_name)
                    elif endpoint.startswith("/alpha/"):
                        alpha_code = endpoint.split("/alpha/")[1] 
                        data = client.get_country_by_code(alpha_code)
                    elif endpoint.startswith("/region/"):
                        region = endpoint.split("/region/")[1]
                        data = client.get_countries_by_region(region)
                    else:
                        # Fallback to raw request for unknown endpoints
                        response = client.session.get(f"{self.base_url}{endpoint}")
                        data = response.json() if response.status_code == 200 else None
                    
                    request_end = time.time()
                    
                    # Determine success based on data returned
                    success = data is not None and (isinstance(data, list) and len(data) > 0 or isinstance(data, dict))
                    response_size = len(str(data)) if data else 0
                    
                    user_results.append(PerformanceResult(
                        test_name=f"{test_name}_user_{user_id}_req_{i}",
                        endpoint=endpoint,
                        start_time=request_start,
                        end_time=request_end,
                        duration=request_end - request_start,
                        success=success,
                        response_size=response_size,
                        status_code=200 if success else 500
                    ))
                    
                except Exception as e:
                    request_end = time.time()
                    user_results.append(PerformanceResult(
                        test_name=f"{test_name}_user_{user_id}_req_{i}",
                        endpoint=endpoint,
                        start_time=request_start,
                        end_time=request_end,
                        duration=request_end - request_start,
                        success=False,
                        error_message=str(e)
                    ))
            
            return user_results
        
        # Execute concurrent load test
        with ThreadPoolExecutor(max_workers=min(concurrent_users, self.max_workers)) as executor:
            futures = [executor.submit(user_simulation, user_id) for user_id in range(concurrent_users)]
            
            for future in as_completed(futures):
                try:
                    user_results = future.result()
                    results.extend(user_results)
                except Exception as e:
                    logger.error(f"User simulation failed: {e}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Calculate metrics
        success_count = sum(1 for r in results if r.success)
        error_count = len(results) - success_count
        success_rate = (success_count / len(results)) * 100 if results else 0
        
        response_times = [r.duration for r in results if r.success]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        median_response_time = statistics.median(response_times) if response_times else 0
        
        # Calculate percentiles
        if response_times:
            sorted_times = sorted(response_times)
            p95_response_time = sorted_times[int(0.95 * len(sorted_times))]
            p99_response_time = sorted_times[int(0.99 * len(sorted_times))]
        else:
            p95_response_time = p99_response_time = 0
        
        requests_per_second = len(results) / total_duration if total_duration > 0 else 0
        errors_per_second = error_count / total_duration if total_duration > 0 else 0
        
        load_result = LoadTestResult(
            test_name=test_name,
            total_requests=len(results),
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
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        logger.info(f"Load test completed: {success_count}/{len(results)} successful, {success_rate:.1f}% success rate, {requests_per_second:.1f} RPS")
        return load_result
    
    def stress_test(self, endpoint: str, min_users: int = 1, max_users: int = 100, 
                    ramp_duration: float = 300, hold_duration: float = 300, 
                    test_name: str = None) -> StressTestResult:
        """Execute a stress test to find breaking point"""
        test_name = test_name or f"stress_test_{endpoint.replace('/', '_')}"
        logger.info(f"Starting stress test: {test_name} from {min_users} to {max_users} users")
        
        start_time = time.time()
        breaking_point = None
        max_sustained_users = min_users
        total_requests = 0
        total_errors = 0
        response_times = []
        
        # Ramp up users gradually
        user_increment = (max_users - min_users) / 10
        current_users = min_users
        
        while current_users <= max_users:
            # Run load test at current user level
            load_result = self.load_test(endpoint, int(current_users), int(current_users * 5), 
                                       f"{test_name}_users_{int(current_users)}")
            
            total_requests += load_result.total_requests
            total_errors += load_result.error_count
            response_times.extend([load_result.avg_response_time])
            
            # Check if this is the breaking point (error rate > 5% or avg response time > 5s)
            if load_result.success_rate < 95.0 or load_result.avg_response_time > 5.0:
                breaking_point = int(current_users)
                break
            else:
                max_sustained_users = int(current_users)
            
            current_users += user_increment
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        avg_response_time = statistics.mean(response_times) if response_times else 0
        error_rate_at_breaking_point = (total_errors / total_requests * 100) if total_requests > 0 else 0
        
        stress_result = StressTestResult(
            test_name=test_name,
            min_users=min_users,
            max_users=max_users,
            ramp_duration=ramp_duration,
            hold_duration=hold_duration,
            breaking_point=breaking_point,
            max_sustained_users=max_sustained_users,
            total_requests=total_requests,
            total_errors=total_errors,
            avg_response_time=avg_response_time,
            error_rate_at_breaking_point=error_rate_at_breaking_point,
            recovery_time=None,  # Would need additional testing
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        logger.info(f"Stress test completed: Breaking point at {breaking_point} users, max sustained {max_sustained_users} users")
        return stress_result
    
    def benchmark_test(self, endpoint: str, iterations: int = 20, baseline_file: Path = None, 
                       test_name: str = None) -> BenchmarkResult:
        """Execute benchmark test and compare with baseline"""
        test_name = test_name or f"benchmark_{endpoint.replace('/', '_')}"
        logger.info(f"Starting benchmark test: {test_name} with {iterations} iterations")
        
        results = []
        for i in range(iterations):
            result = self.single_request_test(endpoint, f"{test_name}_iteration_{i}")
            if result.success:
                results.append(result.duration)
        
        if not results:
            raise ValueError("No successful requests in benchmark test")
        
        avg_response_time = statistics.mean(results)
        min_response_time = min(results)
        max_response_time = max(results)
        std_deviation = statistics.stdev(results) if len(results) > 1 else 0
        success_rate = (len(results) / iterations) * 100
        requests_per_second = 1 / avg_response_time if avg_response_time > 0 else 0
        
        # Compare with baseline if available
        comparison_baseline = None
        performance_change = None
        
        if baseline_file and baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                    if baseline_data and len(baseline_data) > 0:
                        comparison_baseline = baseline_data[-1]['avg_response_time']
                        performance_change = ((avg_response_time - comparison_baseline) / comparison_baseline) * 100
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.warning(f"Could not load baseline data from {baseline_file}: {e}")
        
        benchmark_result = BenchmarkResult(
            test_name=test_name,
            endpoint=endpoint,
            iterations=iterations,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            std_deviation=std_deviation,
            success_rate=success_rate,
            requests_per_second=requests_per_second,
            comparison_baseline=comparison_baseline,
            performance_change=performance_change,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        logger.info(f"Benchmark completed: {avg_response_time:.3f}s avg, {success_rate:.1f}% success rate")
        return benchmark_result
    
    # ==== ANALYSIS METHODS ====
    
    def analyze_load_test(self, result: LoadTestResult) -> PerformanceReport:
        """Analyze load test results and generate insights"""
        insights = []
        recommendations = []
        
        # Response time analysis
        if result.avg_response_time <= self.thresholds['response_time_ms']['excellent'] / 1000:
            insights.append(PerformanceInsight(
                category='performance',
                severity='info',
                title='Excellent Response Time',
                description=f'Average response time of {result.avg_response_time:.3f}s is excellent',
                metric_value=result.avg_response_time,
                threshold=self.thresholds['response_time_ms']['excellent'] / 1000
            ))
        elif result.avg_response_time > self.thresholds['response_time_ms']['poor'] / 1000:
            insights.append(PerformanceInsight(
                category='performance',
                severity='critical',
                title='Poor Response Time',
                description=f'Average response time of {result.avg_response_time:.3f}s exceeds acceptable threshold',
                metric_value=result.avg_response_time,
                threshold=self.thresholds['response_time_ms']['poor'] / 1000,
                recommendation='Consider optimizing API endpoints or scaling infrastructure'
            ))
        
        # Success rate analysis
        if result.success_rate >= self.thresholds['success_rate']['excellent']:
            insights.append(PerformanceInsight(
                category='reliability',
                severity='info',
                title='Excellent Reliability',
                description=f'Success rate of {result.success_rate:.1f}% is excellent',
                metric_value=result.success_rate,
                threshold=self.thresholds['success_rate']['excellent']
            ))
        elif result.success_rate < self.thresholds['success_rate']['poor']:
            insights.append(PerformanceInsight(
                category='reliability',
                severity='critical',
                title='Poor Reliability',
                description=f'Success rate of {result.success_rate:.1f}% is below acceptable threshold',
                metric_value=result.success_rate,
                threshold=self.thresholds['success_rate']['poor'],
                recommendation='Investigate error causes and implement retry mechanisms'
            ))
        
        # Generate charts
        charts = self._generate_load_test_charts(result)
        
        return PerformanceReport(
            test_type='load_test',
            timestamp=result.timestamp,
            summary={
                'test_name': result.test_name,
                'concurrent_users': result.concurrent_users,
                'total_requests': result.total_requests,
                'success_rate': result.success_rate,
                'avg_response_time': result.avg_response_time,
                'requests_per_second': result.requests_per_second
            },
            metrics=asdict(result),
            insights=insights,
            trends=[],  # Would require historical data
            charts=charts,
            recommendations=recommendations
        )
    
    def _generate_load_test_charts(self, result: LoadTestResult) -> List[str]:
        """Generate charts for load test results"""
        charts = []
        
        # Response time distribution chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Min', 'Avg', 'P95', 'P99', 'Max'],
            y=[result.min_response_time, result.avg_response_time, 
               result.p95_response_time, result.p99_response_time, result.max_response_time],
            name='Response Times'
        ))
        fig.update_layout(title=f'Response Time Distribution - {result.test_name}')
        
        chart_file = self.output_dir / f"{result.test_name}_response_times.html"
        fig.write_html(str(chart_file))
        charts.append(str(chart_file))
        
        # Success/Error summary chart
        fig = go.Figure(data=[
            go.Pie(labels=['Success', 'Errors'], values=[result.success_count, result.error_count])
        ])
        fig.update_layout(title=f'Success Rate - {result.test_name}')
        
        chart_file = self.output_dir / f"{result.test_name}_summary.html"
        fig.write_html(str(chart_file))
        charts.append(str(chart_file))
        
        return charts
    
    def export_results(self, output_dir: Path = None, format: str = 'json') -> str:
        """Export all performance results"""
        output_dir = output_dir or self.output_dir
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            output_file = output_dir / f"performance_results_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump([asdict(r) for r in self._results], f, indent=2, default=str)
        
        return str(output_file)


# ==== BACKWARDS COMPATIBILITY ====

# Create aliases for existing imports
PerformanceTester = UnifiedPerformanceEngine  # Alias for performance_testing.py
PerformanceAnalyzer = UnifiedPerformanceEngine  # Alias for performance_analyzer.py  
LoadGenerator = UnifiedPerformanceEngine  # Alias for load_generator.py


def get_performance_engine() -> UnifiedPerformanceEngine:
    """Get the global performance engine instance"""
    return UnifiedPerformanceEngine()