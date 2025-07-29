# Benchmark testing suite for Phase 5 performance testing
# Tests baseline performance and regression detection

import pytest
import json
import time
import statistics
import logging
from pathlib import Path
from typing import Dict, Any, List

from src.performance import UnifiedPerformanceEngine, BenchmarkResult
from src.monitoring import get_metrics_collector

logger = logging.getLogger(__name__)


class TestBenchmarkPerformance:
    # Benchmark testing suite for baseline validation and regression detection
    
    @pytest.fixture(scope="class")
    def performance_runner(self):
        # Fixture for performance test runner
        runner = UnifiedPerformanceEngine(max_workers=50)
        yield runner
        runner._results.clear()
    
    @pytest.fixture(scope="class")
    def performance_analyzer(self):
        # Fixture for performance analyzer  
        output_dir = Path("performance_analysis") / "benchmarks"
        analyzer = UnifiedPerformanceEngine(output_dir)
        yield analyzer
    
    @pytest.fixture(scope="class")
    def benchmark_config(self):
        # Load benchmark test configuration
        config_file = Path("config/performance.json")
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Fallback configuration
            return {
                "benchmark_scenarios": {
                    "api_endpoints": {
                        "endpoints": [
                            {
                                "name": "all_countries",
                                "endpoint": "all",
                                "iterations": 10,
                                "baseline_response_time": 1.5,
                                "regression_threshold": 0.2
                            }
                        ]
                    }
                }
            }
    
    @pytest.fixture(scope="class")
    def baseline_file(self):
        # Baseline file for benchmark comparisons
        return Path("performance_baseline") / "api_benchmarks.json"
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_api_endpoints_benchmark(self, performance_runner: UnifiedPerformanceEngine,
                                   performance_analyzer: UnifiedPerformanceEngine,
                                   benchmark_config: Dict[str, Any],
                                   baseline_file: Path):
        # Benchmark all major API endpoints
        scenario = benchmark_config["benchmark_scenarios"]["api_endpoints"]
        
        logger.info("Starting API endpoints benchmark test")
        
        benchmark_results = []
        
        for endpoint_config in scenario["endpoints"]:
            endpoint = endpoint_config["endpoint"]
            iterations = endpoint_config.get("iterations", 10)
            test_name = endpoint_config["name"]
            
            logger.info(f"Benchmarking {test_name} endpoint ({endpoint})")
            
            # Execute benchmark test
            result = performance_runner.benchmark_test(
                endpoint=endpoint,
                baseline_file=baseline_file,
                iterations=iterations,
                test_name=f"benchmark_{test_name}"
            )
            
            benchmark_results.append(result)
            
            # Validate individual endpoint benchmark
            assert result.iterations > 0, f"No iterations completed for {test_name}"
            assert result.avg_response_time > 0, f"Invalid response time for {test_name}"
            
            # Check for significant regressions
            if result.performance_change is not None and result.performance_change > 50:
                logger.warning(f"Performance regression detected for {test_name}: "
                             f"{result.performance_change:+.1f}%")
                
                # Critical regressions should fail the test
                if result.performance_change > 100:  # 100% slower
                    pytest.fail(f"Critical performance regression in {test_name}: "
                               f"{result.performance_change:+.1f}%")
            else:
                change_text = f"({result.performance_change:+.1f}% vs baseline)" if result.performance_change is not None else "(no baseline)"
                logger.info(f"{test_name} performance: {result.avg_response_time:.3f}s {change_text}")
        
        # Basic validation of benchmark results
        logger.info(f"Completed benchmark testing for {len(benchmark_results)} endpoints")
        
        # Overall benchmark validation
        regressions = len([r for r in benchmark_results if r.performance_change is not None and r.performance_change > 50])
        total_endpoints = len(benchmark_results)
        
        logger.info(f"Benchmark completed: {regressions}/{total_endpoints} endpoints with regressions")
        
        # No more than 30% of endpoints should have regressions
        regression_rate = (regressions / total_endpoints) if total_endpoints > 0 else 0
        assert regression_rate <= 0.3, f"Too many endpoint regressions: {regression_rate:.1%}"
        
        # Export benchmark results to JSON
        results_export = performance_runner.export_results(format='json')
        logger.info(f"Benchmark results exported to: {results_export}")
        
        # Save current results as new baseline if no critical issues
        critical_regressions = len([r for r in benchmark_results if r.performance_change is not None and r.performance_change > 100])
        if critical_regressions == 0:
            # Create simple baseline data for future comparisons
            baseline_data = [{"endpoint": r.endpoint, "avg_response_time": r.avg_response_time, "timestamp": r.timestamp} for r in benchmark_results]
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            logger.info(f"Updated baseline with {len(benchmark_results)} endpoint measurements")
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_single_endpoint_detailed_benchmark(self, performance_runner: UnifiedPerformanceEngine,
                                              performance_analyzer: UnifiedPerformanceEngine):
        # Detailed benchmark of a single endpoint with statistical analysis
        
        logger.info("Starting detailed single endpoint benchmark")
        
        # High iteration count for statistical significance
        iterations = 30
        endpoint = "all"
        
        # Execute detailed benchmark
        result = performance_runner.benchmark_test(
            endpoint=endpoint,
            baseline_file=None,  # No baseline comparison for detailed analysis
            iterations=iterations,
            test_name="detailed_benchmark_all_countries"
        )
        
        # Get raw performance data
        raw_results = [r for r in performance_runner._results 
                      if r.test_name.startswith("detailed_benchmark") and r.success]
        
        assert len(raw_results) >= iterations * 0.8, f"Not enough successful iterations: {len(raw_results)}"
        
        # Statistical analysis
        response_times = [r.duration for r in raw_results]
        
        mean_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
        min_time = min(response_times)
        max_time = max(response_times)
        
        # Calculate percentiles
        sorted_times = sorted(response_times)
        p90_time = sorted_times[int(0.9 * len(sorted_times))]
        p95_time = sorted_times[int(0.95 * len(sorted_times))]
        p99_time = sorted_times[int(0.99 * len(sorted_times))]
        
        # Coefficient of variation (stability measure)
        cv = (std_dev / mean_time) if mean_time > 0 else 0
        
        logger.info(f"Detailed benchmark statistics:")
        logger.info(f"  Mean: {mean_time:.3f}s, Median: {median_time:.3f}s")
        logger.info(f"  Min: {min_time:.3f}s, Max: {max_time:.3f}s")
        logger.info(f"  P90: {p90_time:.3f}s, P95: {p95_time:.3f}s, P99: {p99_time:.3f}s")
        logger.info(f"  Std Dev: {std_dev:.3f}s, CV: {cv:.3f}")
        
        # Performance assertions
        assert mean_time <= 3.0, f"Mean response time too high: {mean_time:.3f}s"
        assert p95_time <= 5.0, f"P95 response time too high: {p95_time:.3f}s"
        assert p99_time <= 8.0, f"P99 response time too high: {p99_time:.3f}s"
        
        # Stability assertions
        assert cv <= 0.5, f"Response time too variable (CV: {cv:.3f})"
        assert max_time / min_time <= 10, f"Too much variation between fastest and slowest requests"
        
        # Validate that median is close to mean (normal distribution)
        mean_median_ratio = abs(mean_time - median_time) / mean_time if mean_time > 0 else 0
        assert mean_median_ratio <= 0.2, f"Response time distribution appears skewed: {mean_median_ratio:.3f}"
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_endpoint_comparison_benchmark(self, performance_runner: UnifiedPerformanceEngine,
                                         performance_analyzer: UnifiedPerformanceEngine,
                                         benchmark_config: Dict[str, Any]):
        # Compare performance across different endpoint types
        
        scenario = benchmark_config["benchmark_scenarios"].get("performance_comparison", {
            "test_sets": [
                {
                    "name": "single_country_lookups",
                    "endpoints": ["alpha/us", "alpha/de", "alpha/fr"],
                    "iterations": 15
                }
            ]
        })
        
        logger.info("Starting endpoint comparison benchmark")
        
        comparison_results = {}
        
        for test_set in scenario["test_sets"]:
            set_name = test_set["name"]
            endpoints = test_set["endpoints"]
            iterations = test_set.get("iterations", 10)
            
            logger.info(f"Benchmarking {set_name} endpoint set")
            
            set_results = []
            
            for endpoint in endpoints:
                result = performance_runner.benchmark_test(
                    endpoint=endpoint,
                    baseline_file=None,
                    iterations=iterations,
                    test_name=f"comparison_{set_name}_{endpoint.replace('/', '_')}"
                )
                set_results.append((endpoint, result))
            
            comparison_results[set_name] = set_results
            
            # Analyze performance within the set
            response_times = [result.current_response_time for _, result in set_results]
            
            if len(response_times) > 1:
                fastest_time = min(response_times)
                slowest_time = max(response_times)
                performance_spread = slowest_time / fastest_time if fastest_time > 0 else 1
                
                logger.info(f"{set_name} performance spread: {performance_spread:.2f}x")
                logger.info(f"  Fastest: {fastest_time:.3f}s, Slowest: {slowest_time:.3f}s")
                
                # Endpoints in the same category shouldn't vary too much
                assert performance_spread <= 5.0, f"Too much performance variation in {set_name}: {performance_spread:.2f}x"
        
        # Cross-set comparison if multiple sets
        if len(comparison_results) > 1:
            set_averages = {}
            for set_name, set_results in comparison_results.items():
                avg_time = statistics.mean([result.current_response_time for _, result in set_results])
                set_averages[set_name] = avg_time
            
            fastest_set = min(set_averages.items(), key=lambda x: x[1])
            slowest_set = max(set_averages.items(), key=lambda x: x[1])
            
            logger.info(f"Fastest endpoint set: {fastest_set[0]} ({fastest_set[1]:.3f}s avg)")
            logger.info(f"Slowest endpoint set: {slowest_set[0]} ({slowest_set[1]:.3f}s avg)")
        
        # Validate that all benchmarks completed successfully
        total_benchmarks = sum(len(set_results) for set_results in comparison_results.values())
        assert total_benchmarks > 0, "No benchmarks completed"
        
        successful_benchmarks = sum(
            1 for set_results in comparison_results.values()
            for endpoint, result in set_results
            if result.iterations > 0
        )
        
        success_rate = successful_benchmarks / total_benchmarks
        assert success_rate >= 0.9, f"Too many benchmark failures: {success_rate:.1%} success rate"
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_baseline_establishment(self, performance_runner: UnifiedPerformanceEngine,
                                  baseline_file: Path):
        # Establish performance baselines for future regression testing
        
        logger.info("Establishing performance baselines")
        
        # Define key endpoints for baseline establishment
        baseline_endpoints = [
            ("all", "all_countries", 20),
            ("name/germany", "country_by_name", 25), 
            ("alpha/us", "country_by_alpha2", 25),
            ("alpha/deu", "country_by_alpha3", 25),
            ("currency/usd", "countries_by_currency", 20),
            ("region/europe", "countries_by_region", 20),
            ("subregion/northern europe", "countries_by_subregion", 20)
        ]
        
        baseline_results = []
        
        for endpoint, name, iterations in baseline_endpoints:
            logger.info(f"Establishing baseline for {name}")
            
            # Execute multiple iterations for stable baseline
            result = performance_runner.benchmark_test(
                endpoint=endpoint,
                baseline_file=None,  # No existing baseline for establishment
                iterations=iterations,
                test_name=f"baseline_{name}"
            )
            
            baseline_results.append(result)
            
            # Validate baseline quality
            assert result.iterations >= iterations * 0.8, f"Not enough iterations for {name} baseline"
            assert result.current_response_time > 0, f"Invalid response time for {name} baseline"
            
            # Get raw results for statistical validation
            raw_results = [r for r in performance_runner._results 
                          if r.test_name.startswith(f"baseline_{name}") and r.success]
            
            if len(raw_results) >= 5:  # Need minimum data for statistics
                response_times = [r.duration for r in raw_results]
                std_dev = statistics.stdev(response_times)
                mean_time = statistics.mean(response_times)
                cv = std_dev / mean_time if mean_time > 0 else 0
                
                # Baseline should be stable (low coefficient of variation)
                assert cv <= 0.4, f"Baseline too unstable for {name}: CV={cv:.3f}"
                
                logger.info(f"{name} baseline: {mean_time:.3f}s ± {std_dev:.3f}s (CV: {cv:.3f})")
        
        # Save baselines
        performance_runner.save_baseline(baseline_results, baseline_file)
        
        # Validate baseline file creation
        assert baseline_file.exists(), "Baseline file was not created"
        
        # Load and validate baseline file content
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        assert 'endpoints' in baseline_data, "Baseline file missing endpoints data"
        assert len(baseline_data['endpoints']) == len(baseline_results), "Baseline file incomplete"
        
        logger.info(f"Baseline established with {len(baseline_results)} endpoints")
        logger.info(f"Baseline saved to: {baseline_file}")
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_regression_detection(self, performance_runner: UnifiedPerformanceEngine,
                                 performance_analyzer: UnifiedPerformanceEngine,
                                 baseline_file: Path):
        # Test regression detection against existing baselines
        
        logger.info("Testing regression detection capabilities")
        
        # Ensure baseline exists
        if not baseline_file.exists():
            pytest.skip("No baseline file available for regression testing")
        
        # Load baseline for validation
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        baseline_endpoints = list(baseline_data.get('endpoints', {}).keys())
        
        if not baseline_endpoints:
            pytest.skip("No baseline endpoints available for regression testing")
        
        # Test a subset of baseline endpoints for regression
        test_endpoints = baseline_endpoints[:5]  # Test first 5 endpoints
        
        regression_results = []
        detected_regressions = []
        
        for endpoint in test_endpoints:
            logger.info(f"Testing regression for {endpoint}")
            
            result = performance_runner.benchmark_test(
                endpoint=endpoint,
                baseline_file=baseline_file,
                iterations=15,
                test_name=f"regression_test_{endpoint.replace('/', '_')}"
            )
            
            regression_results.append(result)
            
            if result.regression_detected:
                detected_regressions.append((endpoint, result))
                logger.warning(f"Regression detected for {endpoint}: "
                             f"{result.performance_change_percent:+.1f}%")
            else:
                logger.info(f"No regression for {endpoint}: "
                           f"{result.performance_change_percent:+.1f}%")
            
            # Validate benchmark execution
            assert result.iterations > 0, f"No iterations completed for {endpoint}"
            assert result.baseline_response_time > 0, f"Invalid baseline for {endpoint}"
        
        # Analyze regression detection results
        report = performance_analyzer.analyze_benchmark_results(regression_results)
        
        # Validate regression detection functionality
        regression_count = len(detected_regressions)
        total_tests = len(regression_results)
        
        logger.info(f"Regression detection test: {regression_count}/{total_tests} regressions detected")
        
        # Most endpoints should not have regressions in a stable system
        regression_rate = regression_count / total_tests if total_tests > 0 else 0
        
        if regression_rate > 0.5:
            logger.warning(f"High regression rate detected: {regression_rate:.1%}")
            # This might indicate system issues, but we'll log rather than fail
            
        # Critical regressions (>100% slower) should be investigated
        critical_regressions = [
            (endpoint, result) for endpoint, result in detected_regressions
            if result.performance_change_percent > 100
        ]
        
        if critical_regressions:
            logger.error(f"Critical regressions detected: {len(critical_regressions)}")
            for endpoint, result in critical_regressions:
                logger.error(f"  {endpoint}: {result.performance_change_percent:+.1f}%")
        
        # Export regression analysis
        performance_analyzer.export_report(report, format='html')
        
        # Fail test only for excessive critical regressions
        assert len(critical_regressions) <= 1, f"Too many critical regressions: {len(critical_regressions)}"
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    @pytest.mark.monitoring_integration  
    def test_benchmark_monitoring_integration(self, performance_runner: UnifiedPerformanceEngine):
        # Test monitoring integration during benchmark testing
        
        logger.info("Testing benchmark monitoring integration")
        
        # Get initial metrics
        metrics_collector = get_metrics_collector()
        initial_metrics_count = len(metrics_collector.test_metrics)
        
        # Execute benchmark with monitoring
        result = performance_runner.benchmark_test(
            endpoint="all",
            baseline_file=None,
            iterations=10,
            test_name="benchmark_monitoring_integration"
        )
        
        # Validate monitoring captured benchmark data
        final_metrics_count = len(metrics_collector.test_metrics)
        new_metrics = final_metrics_count - initial_metrics_count
        
        assert new_metrics >= result.iterations, \
               f"Expected at least {result.iterations} new metrics, got {new_metrics}"
        
        # Check custom metrics for benchmark
        custom_metrics = metrics_collector.custom_metrics
        benchmark_metrics = [
            key for key in custom_metrics.keys()
            if 'benchmark_monitoring_integration' in key
        ]
        
        assert len(benchmark_metrics) > 0, "No custom benchmark metrics recorded"
        
        # Validate benchmark results
        assert result.iterations > 0, "Benchmark didn't complete iterations"
        assert result.current_response_time > 0, "Invalid benchmark response time"
        
        # Get monitoring summary
        test_summary = metrics_collector.get_test_summary(time_window_minutes=5)
        assert test_summary['total_tests'] >= new_metrics, "Test summary doesn't reflect benchmark"
        
        logger.info(f"Benchmark monitoring verified: {new_metrics} metrics recorded")
        logger.info(f"Benchmark result: {result.current_response_time:.3f}s over {result.iterations} iterations")
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_benchmark_stability_validation(self, performance_runner: UnifiedPerformanceEngine):
        # Test benchmark stability and consistency over multiple runs
        
        logger.info("Testing benchmark stability and consistency")
        
        endpoint = "alpha/us"
        runs = 5
        iterations_per_run = 8
        
        run_results = []
        
        # Execute multiple benchmark runs
        for run_num in range(runs):
            logger.info(f"Benchmark stability run {run_num + 1}/{runs}")
            
            result = performance_runner.benchmark_test(
                endpoint=endpoint,
                baseline_file=None,
                iterations=iterations_per_run,
                test_name=f"stability_run_{run_num + 1}"
            )
            
            run_results.append(result.current_response_time)
            
            assert result.iterations > 0, f"Run {run_num + 1} failed to complete"
            assert result.current_response_time > 0, f"Invalid response time in run {run_num + 1}"
        
        # Analyze stability across runs
        mean_response_time = statistics.mean(run_results)
        std_dev = statistics.stdev(run_results) if len(run_results) > 1 else 0
        min_time = min(run_results)
        max_time = max(run_results)
        
        # Coefficient of variation for stability
        cv = std_dev / mean_response_time if mean_response_time > 0 else 0
        variation_range = max_time / min_time if min_time > 0 else 1
        
        logger.info(f"Benchmark stability analysis:")
        logger.info(f"  Mean: {mean_response_time:.3f}s, Std Dev: {std_dev:.3f}s")
        logger.info(f"  Min: {min_time:.3f}s, Max: {max_time:.3f}s")
        logger.info(f"  CV: {cv:.3f}, Range: {variation_range:.2f}x")
        
        # Stability assertions
        assert cv <= 0.3, f"Benchmark results too variable across runs: CV={cv:.3f}"
        assert variation_range <= 3.0, f"Too much variation between runs: {variation_range:.2f}x"
        
        # All runs should complete successfully
        assert len(run_results) == runs, f"Not all benchmark runs completed: {len(run_results)}/{runs}"
        
        # Performance should be consistent
        outlier_threshold = mean_response_time + (2 * std_dev)
        outliers = [time for time in run_results if time > outlier_threshold]
        
        assert len(outliers) <= 1, f"Too many outlier runs: {len(outliers)}"
        
        logger.info(f"Benchmark stability validated: {len(run_results)} consistent runs")