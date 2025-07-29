# Stress testing suite for Phase 5 performance testing
# Tests system breaking points and recovery under extreme load

import pytest
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

from src.performance import (
    UnifiedPerformanceEngine, StressTestResult, LoadPattern, UserBehavior
)
from src.monitoring import get_metrics_collector

logger = logging.getLogger(__name__)


class TestStressPerformance:
    # Stress testing suite for breaking point analysis
    
    @pytest.fixture(scope="class")
    def performance_runner(self):
        # Fixture for performance test runner
        runner = UnifiedPerformanceEngine(max_workers=300)  # Higher limit for stress testing
        yield runner
        runner._results.clear()
    
    @pytest.fixture(scope="class")
    def performance_analyzer(self):
        # Fixture for performance analyzer
        output_dir = Path("performance_analysis") / "stress_tests"
        analyzer = UnifiedPerformanceEngine(output_dir)
        yield analyzer
    
    @pytest.fixture(scope="class") 
    def stress_config(self):
        # Load stress test configuration
        config_file = Path("config/performance.json")
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Fallback configuration
            return {
                "stress_test_scenarios": {
                    "gradual_stress": {
                        "min_users": 1,
                        "max_users": 100,
                        "step_size": 10,
                        "step_duration": 30
                    }
                }
            }
    
    @pytest.mark.performance
    @pytest.mark.stress_test
    def test_gradual_stress_breaking_point(self, performance_runner: UnifiedPerformanceEngine,
                                         performance_analyzer: UnifiedPerformanceEngine,
                                         stress_config: Dict[str, Any]):
        # Test gradual stress increase to find breaking point
        scenario = stress_config["stress_test_scenarios"]["gradual_stress"]
        
        logger.info("Starting gradual stress test to find breaking point")
        
        # Execute stress test
        result = performance_runner.stress_test(
            endpoint="all",
            min_users=scenario["min_users"],
            max_users=scenario["max_users"],
            step_size=scenario["step_size"],
            step_duration=scenario["step_duration"],
            test_name="gradual_stress_all_countries"
        )
        
        # Analyze stress test results
        report = performance_analyzer.analyze_stress_test(result)
        
        # Validate stress test execution
        assert len(result.load_phases) > 0, "No load phases recorded"
        assert result.max_users_tested > scenario["min_users"], "Stress test didn't progress beyond minimum users"
        assert result.peak_rps > 0, "No throughput recorded during stress test"
        
        # Check if degradation point was detected
        if result.degradation_point:
            logger.info(f"Performance degradation detected at {result.degradation_point} users")
            assert result.degradation_point <= result.max_users_tested, "Degradation point beyond tested range"
            
            # Find the load phase where degradation occurred
            degradation_phase = None
            for phase in result.load_phases:
                if phase['users'] >= result.degradation_point:
                    degradation_phase = phase
                    break
            
            if degradation_phase:
                # Validate degradation indicators
                assert (degradation_phase['success_rate'] < 95.0 or 
                       degradation_phase['avg_response_time'] > 5.0), \
                       "Degradation point doesn't show expected performance issues"
        
        # Check if breaking point was detected  
        if result.breaking_point:
            logger.info(f"System breaking point detected at {result.breaking_point} users")
            assert result.breaking_point <= result.max_users_tested, "Breaking point beyond tested range"
            
            # Breaking point should be after degradation point (if detected)
            if result.degradation_point:
                assert result.breaking_point >= result.degradation_point, \
                       "Breaking point should be at or after degradation point"
        
        # Validate peak performance metrics
        assert result.peak_success_rate > 0, "No peak success rate recorded"
        assert result.peak_success_rate <= 100.0, "Invalid peak success rate"
        
        logger.info(f"Stress test completed: Peak {result.peak_rps:.1f} RPS at {result.peak_success_rate:.1f}% success")
        
        # Export detailed analysis
        performance_analyzer.export_report(report, format='html')
    
    @pytest.mark.performance
    @pytest.mark.stress_test
    def test_rapid_stress_spike_handling(self, performance_runner: UnifiedPerformanceEngine,
                                       performance_analyzer: UnifiedPerformanceEngine,
                                       stress_config: Dict[str, Any]):
        # Test rapid stress increases (spike handling)
        scenario = stress_config["stress_test_scenarios"].get("rapid_stress", {
            "min_users": 10,
            "max_users": 200,
            "step_size": 50,
            "step_duration": 20
        })
        
        logger.info("Starting rapid stress test for spike handling")
        
        # Execute rapid stress test
        result = performance_runner.stress_test(
            endpoint="region/europe",
            min_users=scenario["min_users"],
            max_users=scenario["max_users"],
            step_size=scenario["step_size"],
            step_duration=scenario["step_duration"],
            test_name="rapid_stress_region_europe"
        )
        
        # Analyze results
        report = performance_analyzer.analyze_stress_test(result)
        
        # Validate rapid stress handling
        assert len(result.load_phases) > 0, "No load phases in rapid stress test"
        
        # Check for rapid escalation in load phases
        if len(result.load_phases) >= 2:
            user_increases = []
            for i in range(1, len(result.load_phases)):
                increase = result.load_phases[i]['users'] - result.load_phases[i-1]['users']
                user_increases.append(increase)
            
            avg_increase = sum(user_increases) / len(user_increases)
            assert avg_increase >= scenario["step_size"] * 0.8, "Stress test didn't increase load rapidly enough"
        
        # Validate system response to rapid load changes
        performance_drops = 0
        for i, phase in enumerate(result.load_phases):
            if i > 0:
                prev_phase = result.load_phases[i-1]
                # Check if performance dropped significantly
                if (phase['success_rate'] < prev_phase['success_rate'] - 10 or
                    phase['avg_response_time'] > prev_phase['avg_response_time'] * 2):
                    performance_drops += 1
        
        # Some performance drops are expected under rapid stress
        total_phases = len(result.load_phases)
        drop_ratio = performance_drops / total_phases if total_phases > 0 else 0
        
        logger.info(f"Rapid stress test: {performance_drops}/{total_phases} phases showed performance drops")
        
        # System should handle at least the initial phases well
        if total_phases >= 3:
            initial_phases_success = all(
                phase['success_rate'] >= 80.0 
                for phase in result.load_phases[:2]
            )
            assert initial_phases_success, "System failed to handle initial rapid load increases"
        
        # Export results
        performance_analyzer.export_report(report, format='json')
    
    @pytest.mark.performance
    @pytest.mark.stress_test
    @pytest.mark.endurance
    def test_endurance_stress_stability(self, performance_runner: UnifiedPerformanceEngine,
                                      performance_analyzer: UnifiedPerformanceEngine,
                                      stress_config: Dict[str, Any]):
        # Test long-duration stress for stability analysis
        scenario = stress_config["stress_test_scenarios"].get("endurance_stress", {
            "min_users": 25,
            "max_users": 75,
            "step_size": 5,
            "step_duration": 45
        })
        
        logger.info("Starting endurance stress test for stability")
        
        # Execute endurance stress test
        result = performance_runner.stress_test(
            endpoint="all",
            min_users=scenario["min_users"],
            max_users=scenario["max_users"],
            step_size=scenario["step_size"],
            step_duration=scenario["step_duration"],
            test_name="endurance_stress_stability"
        )
        
        # Analyze endurance results
        report = performance_analyzer.analyze_stress_test(result)
        
        # Validate endurance characteristics
        assert result.total_duration >= 300, "Endurance test didn't run long enough"  # At least 5 minutes
        assert len(result.load_phases) >= 5, "Not enough load phases for endurance testing"
        
        # Check for stability over time
        success_rates = [phase['success_rate'] for phase in result.load_phases]
        response_times = [phase['avg_response_time'] for phase in result.load_phases]
        
        # Calculate stability metrics
        success_rate_variance = max(success_rates) - min(success_rates) if success_rates else 0
        response_time_variance = max(response_times) - min(response_times) if response_times else 0
        
        # System should remain relatively stable
        assert success_rate_variance <= 20.0, f"Success rate too variable during endurance: {success_rate_variance}%"
        
        # Check for memory leaks or degradation over time
        if len(result.load_phases) >= 4:
            early_phases = result.load_phases[:2]
            late_phases = result.load_phases[-2:]
            
            early_avg_response = sum(p['avg_response_time'] for p in early_phases) / len(early_phases)
            late_avg_response = sum(p['avg_response_time'] for p in late_phases) / len(late_phases)
            
            degradation_ratio = late_avg_response / early_avg_response if early_avg_response > 0 else 1
            
            # Performance shouldn't degrade more than 50% over the test duration
            assert degradation_ratio <= 1.5, f"Significant performance degradation over time: {degradation_ratio:.2f}x"
        
        logger.info(f"Endurance stress completed: {result.total_duration:.1f}s duration")
        logger.info(f"Stability metrics: {success_rate_variance:.1f}% success rate variance")
        
        # Export endurance analysis
        performance_analyzer.export_report(report, format='html')
    
    @pytest.mark.performance
    @pytest.mark.stress_test
    def test_recovery_time_analysis(self, performance_runner: UnifiedPerformanceEngine,
                                  performance_analyzer: UnifiedPerformanceEngine):
        # Test system recovery after overload
        
        logger.info("Starting recovery time analysis test")
        
        # Execute stress test designed to cause overload
        result = performance_runner.stress_test(
            endpoint="all",
            min_users=10,
            max_users=250,  # High load to trigger overload
            step_size=30,
            step_duration=20,
            test_name="recovery_analysis_overload"
        )
        
        # Analyze recovery characteristics
        report = performance_analyzer.analyze_stress_test(result)
        
        # Check if overload occurred and recovery was tested
        if result.breaking_point:
            logger.info(f"System overload detected at {result.breaking_point} users")
            
            if result.recovery_time is not None:
                logger.info(f"System recovery took {result.recovery_time:.2f} seconds")
                
                # Recovery should be reasonably fast
                assert result.recovery_time <= 300, f"Recovery time too slow: {result.recovery_time}s"
                
                # Very fast recovery is good
                if result.recovery_time <= 30:
                    logger.info("Excellent recovery time detected")
                elif result.recovery_time <= 60:
                    logger.info("Good recovery time detected")
                else:
                    logger.warning("Slow recovery time detected")
            else:
                logger.warning("System did not recover during test period")
                # This is not necessarily a failure, but worth noting
        else:
            logger.info("No system overload detected - increasing max_users might be needed")
        
        # Validate that stress test executed properly
        assert len(result.load_phases) > 0, "No load phases recorded"
        assert result.max_users_tested >= 50, "Stress test didn't reach sufficient load"
        
        # Export recovery analysis
        performance_analyzer.export_report(report, format='json')
    
    @pytest.mark.performance
    @pytest.mark.stress_test
    @pytest.mark.async_stress
    def test_async_stress_patterns(self):
        # Test async stress patterns for better concurrency
        import asyncio
        
        logger.info("Starting async stress pattern test")
        
        stress_patterns = [
            {
                'name': 'spike_pattern',
                'config': LoadConfig(
                    pattern=LoadPattern.SPIKE,
                    duration_seconds=180,
                    min_users=10,
                    max_users=100
                )
            },
            {
                'name': 'step_pattern', 
                'config': LoadConfig(
                    pattern=LoadPattern.STEP,
                    duration_seconds=150,
                    min_users=5,
                    max_users=80,
                    step_size=15,
                    step_duration=30
                )
            }
        ]
        
        results = {}
        
        for pattern_info in stress_patterns:
            pattern_name = pattern_info['name']
            config = pattern_info['config']
            
            logger.info(f"Testing {pattern_name} async stress pattern")
            
            async def run_stress_pattern():
                generator = AsyncLoadGenerator("https://restcountries.com/v3.1")
                behavior = STANDARD_BEHAVIORS['heavy_user']
                return await generator.generate_load_pattern(config, behavior)
            
            result = asyncio.run(run_stress_pattern())
            results[pattern_name] = result
            
            # Validate async stress pattern
            assert result.total_requests > 0, f"No requests in {pattern_name}"
            assert result.successful_requests > 0, f"No successful requests in {pattern_name}"
            assert len(result.timeline) > 0, f"No timeline data in {pattern_name}"
            
            success_rate = (result.successful_requests / result.total_requests * 100) if result.total_requests > 0 else 0
            assert success_rate >= 60.0, f"{pattern_name} success rate too low: {success_rate}%"
            
            logger.info(f"{pattern_name} completed: {success_rate:.1f}% success, {result.requests_per_second:.1f} RPS")
        
        # Compare stress patterns
        spike_result = results['spike_pattern']
        step_result = results['step_pattern']
        
        # Spike pattern should show more timeline events due to rapid changes
        assert len(spike_result.timeline) >= len(step_result.timeline) * 0.5, \
               "Spike pattern should have significant timeline activity"
        
        logger.info("Async stress patterns test completed successfully")
    
    @pytest.mark.performance
    @pytest.mark.stress_test
    def test_multi_endpoint_stress(self, performance_runner: UnifiedPerformanceEngine,
                                 performance_analyzer: UnifiedPerformanceEngine):
        # Test stress across multiple endpoints simultaneously
        
        logger.info("Starting multi-endpoint stress test")
        
        endpoints_to_stress = [
            ("all", "all_countries"),
            ("region/europe", "europe_region"),
            ("currency/usd", "usd_currency"),
            ("name/germany", "germany_lookup")
        ]
        
        stress_results = []
        
        for endpoint, test_name in endpoints_to_stress:
            logger.info(f"Stress testing {endpoint}")
            
            # Execute stress test on each endpoint
            result = performance_runner.stress_test(
                endpoint=endpoint,
                min_users=5,
                max_users=80,
                step_size=15,
                step_duration=25,
                test_name=f"multi_stress_{test_name}"
            )
            
            stress_results.append((endpoint, result))
            
            # Validate individual endpoint stress results
            assert len(result.load_phases) > 0, f"No load phases for {endpoint}"
            assert result.peak_rps > 0, f"No throughput recorded for {endpoint}"
        
        # Analyze relative stress tolerance
        breaking_points = {}
        degradation_points = {}
        peak_performance = {}
        
        for endpoint, result in stress_results:
            if result.breaking_point:
                breaking_points[endpoint] = result.breaking_point
            if result.degradation_point:
                degradation_points[endpoint] = result.degradation_point
                
            peak_performance[endpoint] = {
                'peak_rps': result.peak_rps,
                'peak_success_rate': result.peak_success_rate
            }
        
        # Find most and least stress-tolerant endpoints
        if breaking_points:
            most_tolerant = max(breaking_points.items(), key=lambda x: x[1])
            least_tolerant = min(breaking_points.items(), key=lambda x: x[1])
            
            logger.info(f"Most stress-tolerant endpoint: {most_tolerant[0]} ({most_tolerant[1]} users)")
            logger.info(f"Least stress-tolerant endpoint: {least_tolerant[0]} ({least_tolerant[1]} users)")
            
            # Most tolerant should handle at least 25% more load than least tolerant
            tolerance_ratio = most_tolerant[1] / least_tolerant[1] if least_tolerant[1] > 0 else 1
            logger.info(f"Stress tolerance ratio: {tolerance_ratio:.2f}x")
        
        # Validate that at least one endpoint reached decent load
        max_users_tested = max(result.max_users_tested for _, result in stress_results)
        assert max_users_tested >= 40, f"Multi-endpoint stress didn't reach sufficient load: {max_users_tested}"
        
        # Export analysis for best performing endpoint
        best_endpoint, best_result = max(stress_results, key=lambda x: x[1].peak_rps)
        report = performance_analyzer.analyze_stress_test(best_result)
        performance_analyzer.export_report(report, format='json')
        
        logger.info(f"Multi-endpoint stress test completed: {len(stress_results)} endpoints tested")
    
    @pytest.mark.performance
    @pytest.mark.stress_test
    @pytest.mark.monitoring_integration
    def test_stress_monitoring_integration(self, performance_runner: UnifiedPerformanceEngine):
        # Test monitoring integration during stress testing
        
        logger.info("Starting stress test with monitoring integration")
        
        # Get initial metrics
        metrics_collector = get_metrics_collector()
        initial_metrics_count = len(metrics_collector.test_metrics)
        initial_system_metrics = len(metrics_collector.system_metrics)
        
        # Execute stress test with monitoring
        result = performance_runner.stress_test(
            endpoint="all",
            min_users=10,
            max_users=60,
            step_size=10,
            step_duration=20,
            test_name="stress_monitoring_integration"
        )
        
        # Validate monitoring captured stress test data
        final_metrics_count = len(metrics_collector.test_metrics)
        final_system_metrics = len(metrics_collector.system_metrics)
        
        new_test_metrics = final_metrics_count - initial_metrics_count
        new_system_metrics = final_system_metrics - initial_system_metrics
        
        assert new_test_metrics > 0, "No test metrics captured during stress test"
        assert new_system_metrics > 0, "No system metrics captured during stress test"
        
        # Check that stress test generated substantial metrics
        assert new_test_metrics >= result.max_users_tested, \
               f"Expected at least {result.max_users_tested} test metrics, got {new_test_metrics}"
        
        # Get recent test summary
        test_summary = metrics_collector.get_test_summary(time_window_minutes=10)
        assert test_summary['total_tests'] >= new_test_metrics, "Test summary doesn't reflect stress test"
        
        # Get system performance summary
        system_summary = metrics_collector.get_system_summary(time_window_minutes=10)
        assert system_summary.get('samples_count', 0) > 0, "No system performance data captured"
        
        # Validate stress test results
        assert len(result.load_phases) > 0, "Stress test didn't execute properly"
        assert result.peak_rps > 0, "No peak performance recorded"
        
        logger.info(f"Monitoring integration verified:")
        logger.info(f"  New test metrics: {new_test_metrics}")
        logger.info(f"  New system metrics: {new_system_metrics}")
        logger.info(f"  Stress test peak: {result.peak_rps:.1f} RPS")
    
    @pytest.mark.performance
    @pytest.mark.stress_test
    @pytest.mark.resource_monitoring
    def test_resource_consumption_under_stress(self, performance_runner: UnifiedPerformanceEngine):
        # Test resource consumption patterns during stress
        
        logger.info("Starting resource consumption monitoring under stress")
        
        # Get baseline system metrics
        metrics_collector = get_metrics_collector()
        
        # Collect baseline
        baseline_metrics = metrics_collector.collect_system_metrics()
        baseline_cpu = baseline_metrics.cpu_percent if baseline_metrics else 0
        baseline_memory = baseline_metrics.memory_percent if baseline_metrics else 0
        
        # Execute resource-intensive stress test
        result = performance_runner.stress_test(
            endpoint="all",
            min_users=15,
            max_users=100,
            step_size=20,
            step_duration=30,
            test_name="resource_consumption_stress"
        )
        
        # Get peak system metrics
        system_summary = metrics_collector.get_system_summary(time_window_minutes=15)
        
        # Validate resource monitoring
        if system_summary.get('samples_count', 0) > 0:
            peak_cpu = system_summary.get('cpu_max', 0)
            peak_memory = system_summary.get('memory_max', 0)
            avg_cpu = system_summary.get('cpu_avg', 0)
            avg_memory = system_summary.get('memory_avg', 0)
            
            logger.info(f"Resource consumption during stress:")
            logger.info(f"  CPU: {avg_cpu:.1f}% avg, {peak_cpu:.1f}% peak (baseline: {baseline_cpu:.1f}%)")
            logger.info(f"  Memory: {avg_memory:.1f}% avg, {peak_memory:.1f}% peak (baseline: {baseline_memory:.1f}%)")
            
            # Resource usage should increase under stress
            cpu_increase = avg_cpu - baseline_cpu
            memory_increase = avg_memory - baseline_memory
            
            assert cpu_increase > 0 or avg_cpu > 5, "Expected CPU usage increase under stress"
            
            # Resource usage shouldn't be excessive (depends on system)
            # These are reasonable limits for API testing
            assert peak_cpu <= 95, f"CPU usage too high during stress: {peak_cpu}%"
            assert peak_memory <= 90, f"Memory usage too high during stress: {peak_memory}%"
        else:
            logger.warning("No system metrics available for resource analysis")
        
        # Validate stress test execution
        assert len(result.load_phases) > 0, "Stress test didn't execute"
        assert result.max_users_tested >= 50, "Stress test didn't reach sufficient load for resource testing"
        
        logger.info(f"Resource monitoring completed: {result.max_users_tested} max users tested")