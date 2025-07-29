# Unified Load Testing Suite for Phase 6 performance testing
# Tests API performance under various concurrent user loads using unified engine

import pytest
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any

from src.performance import (
    UnifiedPerformanceEngine, LoadTestResult, UserBehavior, LoadPattern
)
from src.monitoring import get_metrics_collector

logger = logging.getLogger(__name__)


class TestLoadPerformance:
    # Load testing suite for concurrent user scenarios using unified engine
    
    @pytest.fixture(scope="class")
    def performance_engine(self):
        # Unified performance engine fixture
        engine = UnifiedPerformanceEngine(
            max_workers=100,
            output_dir=Path("performance_analysis") / "load_tests"
        )
        yield engine
        # Cleanup
        engine._results.clear()
    
    @pytest.fixture(scope="class")
    def load_config(self):
        # Load performance test configuration
        return {
            "load_test_scenarios": {
                "light_load": {
                    "concurrent_users": 10,
                    "total_requests": 50,
                    "test_name": "light_load_all_countries",
                    "endpoint": "/all",
                    "expected_success_rate": 95.0,
                    "max_avg_response_time": 2.0
                },
                "moderate_load": {
                    "concurrent_users": 25,
                    "total_requests": 125,
                    "test_name": "moderate_load_country_by_alpha",
                    "endpoint": "/alpha/us",
                    "expected_success_rate": 90.0, 
                    "max_avg_response_time": 3.0
                },
                "heavy_load": {
                    "concurrent_users": 50,
                    "total_requests": 250,
                    "test_name": "heavy_load_all_countries",
                    "endpoint": "/all",
                    "expected_success_rate": 85.0,
                    "max_avg_response_time": 5.0
                }
            }
        }
    
    @pytest.mark.performance
    @pytest.mark.load_test
    def test_light_load_performance(self, performance_engine: UnifiedPerformanceEngine, 
                                  load_config: Dict[str, Any]):
        # Test light load scenario (10 concurrent users)
        scenario = load_config["load_test_scenarios"]["light_load"]
        
        logger.info("Starting light load performance test")
        
        # Execute load test
        result = performance_engine.load_test(
            endpoint=scenario["endpoint"],
            concurrent_users=scenario["concurrent_users"],
            total_requests=scenario["total_requests"],
            test_name=scenario["test_name"]
        )
        
        # Generate analysis report
        report = performance_engine.analyze_load_test(result)
        
        # Validate performance metrics
        assert result.success_rate >= scenario["expected_success_rate"], \
            f"Success rate {result.success_rate:.1f}% below expected {scenario['expected_success_rate']}%"
        
        assert result.avg_response_time <= scenario["max_avg_response_time"], \
            f"Avg response time {result.avg_response_time:.2f}s exceeds limit {scenario['max_avg_response_time']}s"
        
        assert result.requests_per_second > 0, "No requests per second calculated"
        
        logger.info(f"Light load test completed: {result.success_count}/{result.total_requests} successful")
        logger.info(f"Performance metrics: {result.avg_response_time:.2f}s avg, {result.requests_per_second:.1f} RPS")
        
        # Export results
        output_file = performance_engine.export_results(format='json')
        assert Path(output_file).exists(), "Performance results not exported"
    
    @pytest.mark.performance
    @pytest.mark.load_test
    def test_moderate_load_performance(self, performance_engine: UnifiedPerformanceEngine,
                                     load_config: Dict[str, Any]):
        # Test moderate load scenario (25 concurrent users)
        scenario = load_config["load_test_scenarios"]["moderate_load"]
        
        logger.info("Starting moderate load performance test")
        
        # Execute load test
        result = performance_engine.load_test(
            endpoint=scenario["endpoint"],
            concurrent_users=scenario["concurrent_users"],
            total_requests=scenario["total_requests"],
            test_name=scenario["test_name"]
        )
        
        # Generate analysis report
        report = performance_engine.analyze_load_test(result)
        
        # Validate performance metrics
        assert result.success_rate >= scenario["expected_success_rate"], \
            f"Success rate {result.success_rate:.1f}% below expected {scenario['expected_success_rate']}%"
        
        assert result.avg_response_time <= scenario["max_avg_response_time"], \
            f"Avg response time {result.avg_response_time:.2f}s exceeds limit {scenario['max_avg_response_time']}s"
        
        # Validate percentile metrics
        assert result.p95_response_time > result.avg_response_time, "P95 should be higher than average"
        assert result.p99_response_time >= result.p95_response_time, "P99 should be higher than P95"
        
        logger.info(f"Moderate load test completed: {result.success_count}/{result.total_requests} successful")
        logger.info(f"Performance metrics: {result.avg_response_time:.2f}s avg, {result.requests_per_second:.1f} RPS")
    
    @pytest.mark.performance  
    @pytest.mark.load_test
    @pytest.mark.stress_test
    def test_heavy_load_performance(self, performance_engine: UnifiedPerformanceEngine,
                                  load_config: Dict[str, Any]):
        # Test heavy load scenario (50 concurrent users)
        scenario = load_config["load_test_scenarios"]["heavy_load"]
        
        logger.info("Starting heavy load performance test")
        
        # Execute load test
        result = performance_engine.load_test(
            endpoint=scenario["endpoint"],
            concurrent_users=scenario["concurrent_users"],
            total_requests=scenario["total_requests"],
            test_name=scenario["test_name"]
        )
        
        # Generate analysis report with insights
        report = performance_engine.analyze_load_test(result)
        
        # Validate performance metrics (more lenient for heavy load)
        assert result.success_rate >= scenario["expected_success_rate"], \
            f"Success rate {result.success_rate:.1f}% below expected {scenario['expected_success_rate']}%"
        
        assert result.avg_response_time <= scenario["max_avg_response_time"], \
            f"Avg response time {result.avg_response_time:.2f}s exceeds limit {scenario['max_avg_response_time']}s"
        
        # Validate that we can handle concurrent load
        assert result.concurrent_users == scenario["concurrent_users"], "Concurrent users mismatch"
        assert result.duration > 0, "Test duration should be positive"
        
        logger.info(f"Heavy load test completed: {result.success_count}/{result.total_requests} successful")
        logger.info(f"Performance metrics: {result.avg_response_time:.2f}s avg, {result.requests_per_second:.1f} RPS")
        
        # Check for performance insights
        assert len(report.insights) > 0, "Performance analysis should generate insights"
    
    @pytest.mark.performance
    @pytest.mark.load_test
    def test_user_behavior_simulation(self, performance_engine: UnifiedPerformanceEngine):
        # Test different user behavior patterns
        logger.info("Testing user behavior simulation")
        
        # Test with different user behaviors defined in engine
        for behavior_name, behavior in performance_engine.user_behaviors.items():
            logger.info(f"Testing {behavior_name} behavior")
            
            # Run load test simulating this user behavior
            result = performance_engine.load_test(
                endpoint="/all",
                concurrent_users=15,
                total_requests=45,  # 3 requests per user
                test_name=f"{behavior_name}_simulation"
            )
            
            # Validate results
            assert result.success_rate > 80.0, f"{behavior_name} behavior has low success rate: {result.success_rate:.1f}%"
            assert result.requests_per_second > 0, f"{behavior_name} behavior produced no RPS"
            
            logger.info(f"{behavior_name} behavior: {result.success_rate:.1f}% success, {result.requests_per_second:.1f} RPS")
    
    @pytest.mark.performance
    @pytest.mark.load_test
    def test_concurrent_endpoint_load(self, performance_engine: UnifiedPerformanceEngine):
        # Test concurrent load on multiple endpoints
        logger.info("Testing concurrent endpoint load")
        
        endpoints = ["/all", "/name/germany", "/alpha/us", "/region/europe"]
        results = []
        
        # Test each endpoint with moderate load
        for endpoint in endpoints:
            logger.info(f"Testing endpoint: {endpoint}")
            
            result = performance_engine.load_test(
                endpoint=endpoint,
                concurrent_users=20,
                total_requests=60,
                test_name=f"concurrent_load{endpoint.replace('/', '_')}"
            )
            
            results.append(result)
            
            # Basic validation for each endpoint
            assert result.success_rate > 70.0, f"Endpoint {endpoint} has low success rate: {result.success_rate:.1f}%"
            assert result.avg_response_time < 10.0, f"Endpoint {endpoint} too slow: {result.avg_response_time:.2f}s"
        
        # Aggregate analysis
        total_requests = sum(r.total_requests for r in results)
        total_successful = sum(r.success_count for r in results)
        overall_success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0
        
        assert overall_success_rate > 75.0, f"Overall success rate too low: {overall_success_rate:.1f}%"
        
        logger.info(f"Concurrent endpoint test completed: {total_successful}/{total_requests} overall successful")
        logger.info(f"Overall success rate: {overall_success_rate:.1f}%")
    
    @pytest.mark.performance
    @pytest.mark.monitoring
    def test_load_test_monitoring_integration(self, performance_engine: UnifiedPerformanceEngine):
        # Test integration with monitoring system during load tests
        logger.info("Testing load test monitoring integration")
        
        # Get metrics collector
        collector = get_metrics_collector()
        initial_metric_count = len(collector.metrics)
        
        # Run load test with monitoring
        result = performance_engine.load_test(
            endpoint="/all",
            concurrent_users=15,
            total_requests=45,
            test_name="monitoring_integration_load_test"
        )
        
        # Validate load test results
        assert result.success_rate > 80.0, f"Load test with monitoring failed: {result.success_rate:.1f}% success rate"
        
        # Validate monitoring integration
        final_metric_count = len(collector.metrics)
        assert final_metric_count > initial_metric_count, "No new metrics recorded during load test"
        
        # Check for performance-specific metrics
        performance_metrics = [m for m in collector.metrics if 'performance' in m.get('tags', {}).get('category', '').lower()]
        assert len(performance_metrics) > 0, "No performance metrics recorded"
        
        logger.info(f"Monitoring integration successful: {final_metric_count - initial_metric_count} new metrics recorded")
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_benchmark_comparison(self, performance_engine: UnifiedPerformanceEngine):
        # Test benchmark comparison functionality
        logger.info("Testing benchmark comparison")
        
        # Run benchmark test
        result = performance_engine.benchmark_test(
            endpoint="/all",
            iterations=10,
            test_name="load_test_benchmark"
        )
        
        # Validate benchmark results
        assert result.success_rate >= 90.0, f"Benchmark test success rate too low: {result.success_rate:.1f}%"
        assert result.avg_response_time > 0, "Average response time should be positive"
        assert result.std_deviation >= 0, "Standard deviation should be non-negative"
        assert result.requests_per_second > 0, "Requests per second should be positive"
        
        # Validate statistical metrics
        assert result.min_response_time <= result.avg_response_time, "Min should be <= average"
        assert result.avg_response_time <= result.max_response_time, "Average should be <= max"
        
        logger.info(f"Benchmark completed: {result.avg_response_time:.3f}s avg ± {result.std_deviation:.3f}s")
        logger.info(f"Benchmark performance: {result.requests_per_second:.1f} RPS, {result.success_rate:.1f}% success")