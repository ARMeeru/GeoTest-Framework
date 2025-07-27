# Advanced pytest plugins for monitoring and reporting integration
# Integrates metrics collection and alerting with pytest execution

import pytest
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from src.monitoring import get_metrics_collector, SystemMonitor
from src.alerting import get_alert_manager, AlertType, AlertSeverity
from src.dashboard_generator import DashboardGenerator


class MonitoringPlugin:
    # Pytest plugin that integrates with monitoring system
    
    def __init__(self):
        self.metrics_collector = get_metrics_collector()
        self.alert_manager = get_alert_manager()
        self.system_monitor = None
        self.session_start_time = None
        self.test_start_times = {}
        self.failed_test_count = 0
        self.total_test_count = 0
        
    def pytest_configure(self, config):
        # Configure plugin at start of test session
        # Start system monitoring
        self.system_monitor = SystemMonitor(self.metrics_collector, interval_seconds=5)
        self.system_monitor.start()
        
        # Set up reports directory
        reports_dir = Path.cwd() / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        print("GeoTest Framework monitoring started")
    
    def pytest_sessionstart(self, session):
        # Called at start of test session
        self.session_start_time = time.time()
        self.metrics_collector.record_custom_metric(
            "session.started", 
            1, 
            tags={"session_id": str(int(self.session_start_time))}
        )
    
    def pytest_runtest_setup(self, item):
        # Called before each test
        test_id = item.nodeid
        test_name = item.name
        
        # Extract API endpoint from test if available
        api_endpoint = getattr(item.function, '__api_endpoint__', None)
        if not api_endpoint and hasattr(item, 'funcargs'):
            # Try to infer from test parameters
            if 'api_client' in item.funcargs:
                api_endpoint = "inferred_from_client"
        
        # Start tracking test
        self.metrics_collector.start_test(test_id, test_name, api_endpoint)
        self.test_start_times[test_id] = time.time()
        self.total_test_count += 1
    
    def pytest_runtest_call(self, item):
        # Called during test execution
        test_id = item.nodeid
        
        # Record test execution metric
        self.metrics_collector.record_custom_metric(
            "test.executing",
            1,
            tags={"test_id": test_id, "test_name": item.name}
        )
    
    def pytest_runtest_teardown(self, item, nextitem):
        # Called after each test
        test_id = item.nodeid
        
        # Determine test outcome
        if hasattr(item, 'rep_call'):
            if item.rep_call.passed:
                status = "passed"
            elif item.rep_call.failed:
                status = "failed"
                self.failed_test_count += 1
            elif item.rep_call.skipped:
                status = "skipped"
            else:
                status = "error"
                self.failed_test_count += 1
        else:
            status = "unknown"
        
        # Get error message if failed
        error_message = None
        if hasattr(item, 'rep_call') and item.rep_call.failed:
            error_message = str(item.rep_call.longrepr)[:500]  # First 500 chars
        
        # End test tracking
        self.metrics_collector.end_test(
            test_id=test_id,
            status=status,
            error_message=error_message
        )
        
        # Check for alerting conditions
        self._check_test_alerts(item, status, error_message)
    
    def pytest_sessionfinish(self, session, exitstatus):
        # Called at end of test session
        session_duration = time.time() - self.session_start_time if self.session_start_time else 0
        
        # Record session metrics
        self.metrics_collector.record_custom_metric(
            "session.completed",
            session_duration,
            tags={
                "exit_status": str(exitstatus),
                "total_tests": str(self.total_test_count),
                "failed_tests": str(self.failed_test_count)
            }
        )
        
        # Check session-level alerts
        self._check_session_alerts(exitstatus)
        
        # Generate dashboard
        self._generate_session_dashboard()
        
        # Export metrics
        self._export_session_metrics()
        
        # Stop system monitoring
        if self.system_monitor:
            self.system_monitor.stop()
        
        print(f"Session completed: {self.total_test_count} tests, {self.failed_test_count} failed")
        print(f"Duration: {session_duration:.2f}s")
    
    def _check_test_alerts(self, item, status: str, error_message: str = None):
        # Check if test results should trigger alerts
        test_name = item.name
        
        # Alert on test failure
        if status == "failed":
            self.alert_manager.create_alert(
                alert_type=AlertType.TEST_FAILURE,
                severity=AlertSeverity.MEDIUM,
                title=f"Test Failed: {test_name}",
                message=f"Test '{test_name}' failed.\n\nError: {error_message or 'No error message'}",
                source=f"test:{item.nodeid}",
                tags={"test_name": test_name, "test_file": str(item.fspath)}
            )
        
        # Alert on critical test failure (if marked as critical)
        if status == "failed" and hasattr(item, 'pytestmark'):
            for mark in item.pytestmark:
                if mark.name == 'critical':
                    self.alert_manager.create_alert(
                        alert_type=AlertType.TEST_FAILURE,
                        severity=AlertSeverity.CRITICAL,
                        title=f"Critical Test Failed: {test_name}",
                        message=f"Critical test '{test_name}' failed!\n\nError: {error_message or 'No error message'}",
                        source=f"test:{item.nodeid}",
                        tags={"test_name": test_name, "critical": "true"}
                    )
                    break
    
    def _check_session_alerts(self, exit_status: int):
        # Check if session results should trigger alerts
        failure_rate = (self.failed_test_count / self.total_test_count * 100) if self.total_test_count > 0 else 0
        
        # Alert on high failure rate
        if failure_rate >= 50:
            self.alert_manager.create_alert(
                alert_type=AlertType.TEST_FAILURE,
                severity=AlertSeverity.HIGH,
                title="High Test Failure Rate",
                message=f"Test session completed with {failure_rate:.1f}% failure rate ({self.failed_test_count}/{self.total_test_count} tests failed).",
                source="session",
                tags={"failure_rate": str(failure_rate), "total_tests": str(self.total_test_count)}
            )
        elif failure_rate >= 25:
            self.alert_manager.create_alert(
                alert_type=AlertType.TEST_FAILURE,
                severity=AlertSeverity.MEDIUM,
                title="Elevated Test Failure Rate",
                message=f"Test session completed with {failure_rate:.1f}% failure rate ({self.failed_test_count}/{self.total_test_count} tests failed).",
                source="session",
                tags={"failure_rate": str(failure_rate), "total_tests": str(self.total_test_count)}
            )
        
        # Alert on session failure
        if exit_status != 0:
            self.alert_manager.create_alert(
                alert_type=AlertType.TEST_FAILURE,
                severity=AlertSeverity.HIGH,
                title="Test Session Failed",
                message=f"Test session exited with status {exit_status}. {self.failed_test_count} out of {self.total_test_count} tests failed.",
                source="session",
                tags={"exit_status": str(exit_status)}
            )
    
    def _generate_session_dashboard(self):
        # Generate dashboard after session completion
        try:
            dashboard_generator = DashboardGenerator(self.metrics_collector, self.alert_manager)
            reports_dir = Path.cwd() / "reports"
            dashboard_file = reports_dir / "dashboard.html"
            
            dashboard_generator.generate_dashboard(dashboard_file, time_window_hours=24)
            print(f"Dashboard generated: {dashboard_file}")
            
        except Exception as e:
            print(f"Failed to generate dashboard: {e}")
    
    def _export_session_metrics(self):
        # Export metrics after session completion
        try:
            reports_dir = Path.cwd() / "reports"
            files_created = self.metrics_collector.export_metrics(reports_dir)
            print(f"Metrics exported: {len(files_created)} files created")
            
        except Exception as e:
            print(f"Failed to export metrics: {e}")
    
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        # Capture test reports for outcome analysis
        outcome = yield
        rep = outcome.get_result()
        
        # Store report on item for later access
        setattr(item, f"rep_{rep.when}", rep)
        
        # Record API metrics if this is an API test
        if call.when == "call" and hasattr(item.instance, '__class__'):
            self._record_api_metrics(item, rep)
    
    def _record_api_metrics(self, item, rep):
        # Record API-specific metrics if available
        try:
            # Try to extract API metrics from test instance
            if hasattr(item.instance, 'last_response_time'):
                response_time = item.instance.last_response_time
                self.metrics_collector.record_api_metric(
                    endpoint=getattr(item.instance, 'last_endpoint', 'unknown'),
                    metric_name="response_time",
                    value=response_time,
                    tags={"test_name": item.name}
                )
            
            if hasattr(item.instance, 'last_status_code'):
                status_code = item.instance.last_status_code
                self.metrics_collector.record_api_metric(
                    endpoint=getattr(item.instance, 'last_endpoint', 'unknown'),
                    metric_name="status_code",
                    value=status_code,
                    tags={"test_name": item.name}
                )
                
        except Exception:
            pass  # Ignore errors in metric collection


class PerformancePlugin:
    # Pytest plugin for performance monitoring and alerting
    
    def __init__(self):
        self.metrics_collector = get_metrics_collector()
        self.alert_manager = get_alert_manager()
        self.performance_thresholds = {
            'max_response_time': 5.0,  # seconds
            'max_test_duration': 30.0,  # seconds
            'max_memory_usage': 80.0,   # percentage
            'max_cpu_usage': 90.0       # percentage
        }
    
    def pytest_runtest_teardown(self, item, nextitem):
        # Check performance metrics after each test
        test_id = item.nodeid
        
        # Get test metrics
        with self.metrics_collector._lock:
            test_metrics = [m for m in self.metrics_collector.test_metrics if m.test_id == test_id]
        
        if not test_metrics:
            return
        
        test_metric = test_metrics[-1]  # Most recent
        
        # Check response time threshold
        if test_metric.response_time and test_metric.response_time > self.performance_thresholds['max_response_time']:
            self.alert_manager.create_alert(
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                severity=AlertSeverity.MEDIUM,
                title=f"Slow API Response: {item.name}",
                message=f"Test '{item.name}' had slow API response time: {test_metric.response_time:.3f}s (threshold: {self.performance_thresholds['max_response_time']}s)",
                source=f"test:{test_id}",
                tags={"response_time": str(test_metric.response_time)}
            )
        
        # Check test duration threshold
        if test_metric.duration and test_metric.duration > self.performance_thresholds['max_test_duration']:
            self.alert_manager.create_alert(
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                severity=AlertSeverity.MEDIUM,
                title=f"Slow Test Execution: {item.name}",
                message=f"Test '{item.name}' took too long to execute: {test_metric.duration:.3f}s (threshold: {self.performance_thresholds['max_test_duration']}s)",
                source=f"test:{test_id}",
                tags={"duration": str(test_metric.duration)}
            )
        
        # Check system resource usage
        if test_metric.memory_usage and test_metric.memory_usage > self.performance_thresholds['max_memory_usage']:
            self.alert_manager.create_alert(
                alert_type=AlertType.SYSTEM_RESOURCE,
                severity=AlertSeverity.MEDIUM,
                title=f"High Memory Usage During Test: {item.name}",
                message=f"Memory usage was high during test '{item.name}': {test_metric.memory_usage:.1f}% (threshold: {self.performance_thresholds['max_memory_usage']}%)",
                source=f"test:{test_id}",
                tags={"memory_usage": str(test_metric.memory_usage)}
            )
        
        if test_metric.cpu_usage and test_metric.cpu_usage > self.performance_thresholds['max_cpu_usage']:
            self.alert_manager.create_alert(
                alert_type=AlertType.SYSTEM_RESOURCE,
                severity=AlertSeverity.MEDIUM,
                title=f"High CPU Usage During Test: {item.name}",
                message=f"CPU usage was high during test '{item.name}': {test_metric.cpu_usage:.1f}% (threshold: {self.performance_thresholds['max_cpu_usage']}%)",
                source=f"test:{test_id}",
                tags={"cpu_usage": str(test_metric.cpu_usage)}
            )


# Plugin registration functions
def pytest_configure(config):
    # Register plugins with pytest
    if not hasattr(config, '_monitoring_plugin'):
        config._monitoring_plugin = MonitoringPlugin()
        config.pluginmanager.register(config._monitoring_plugin, 'monitoring')
    
    if not hasattr(config, '_performance_plugin'):
        config._performance_plugin = PerformancePlugin()
        config.pluginmanager.register(config._performance_plugin, 'performance')


def pytest_unconfigure(config):
    # Unregister plugins
    if hasattr(config, '_monitoring_plugin'):
        config.pluginmanager.unregister(config._monitoring_plugin)
    
    if hasattr(config, '_performance_plugin'):
        config.pluginmanager.unregister(config._performance_plugin)


# Pytest fixtures for accessing monitoring features
@pytest.fixture
def metrics_collector():
    # Fixture to access metrics collector in tests
    return get_metrics_collector()


@pytest.fixture
def alert_manager():
    # Fixture to access alert manager in tests
    return get_alert_manager()


@pytest.fixture
def performance_tracker(metrics_collector):
    # Fixture for tracking performance in tests
    def track_performance(operation_name: str, operation_func, **tags):
        # Track performance of an operation
        start_time = time.time()
        try:
            result = operation_func()
            duration = time.time() - start_time
            metrics_collector.record_custom_metric(
                f"operation.{operation_name}.duration",
                duration,
                tags=tags
            )
            metrics_collector.record_custom_metric(
                f"operation.{operation_name}.success",
                1,
                tags=tags
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            metrics_collector.record_custom_metric(
                f"operation.{operation_name}.duration",
                duration,
                tags=tags
            )
            metrics_collector.record_custom_metric(
                f"operation.{operation_name}.failure",
                1,
                tags={**tags, "error": str(e)[:100]}
            )
            raise
    
    return track_performance


# Custom markers for enhanced monitoring
def pytest_configure(config):
    # Register custom markers
    config.addinivalue_line(
        "markers", "critical: mark test as critical (will trigger alerts on failure)"
    )
    config.addinivalue_line(
        "markers", "performance: mark test for performance monitoring"
    )
    config.addinivalue_line(
        "markers", "api_endpoint(name): specify API endpoint being tested"
    )


# Custom pytest marks
def api_endpoint(endpoint_name: str):
    # Mark a test with the API endpoint it's testing
    def decorator(func):
        func.__api_endpoint__ = endpoint_name
        return func
    return decorator


def critical(func):
    # Mark a test as critical (failures will trigger high-severity alerts)
    return pytest.mark.critical(func)


def performance_test(func):
    # Mark a test for enhanced performance monitoring
    return pytest.mark.performance(func)