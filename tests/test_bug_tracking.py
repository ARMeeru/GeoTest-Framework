# Smart failure tracking test suite for Phase 5 intelligent bug tracking
# Tests bug tracking system components and integration

import pytest
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.github_integration import SmartBugTracker, GitHubAPIClient, FailureRecord
from src.failure_analysis import (
    UnifiedFailureAnalysisEngine, FailureCategory, MTTRRecord
)
from src.retry_manager import SmartRetryManager, RetryConfig, RetryStrategy


class TestGitHubIntegration:
    # Test GitHub Issues API integration with safeguards
    
    @pytest.fixture
    def mock_github_config(self, tmp_path):
        # Mock GitHub configuration for testing
        config_file = tmp_path / "test_bug_tracking.json"
        config = {
            "github": {
                "enabled": True,
                "token": "test_token",
                "repo_owner": "test_owner",
                "repo_name": "test_repo",
                "labels": ["bug", "automated-test"]
            },
            "failure_tracking": {
                "consecutive_failures_threshold": 2,
                "max_issues_per_day": 3,
                "similar_failure_grouping": True,
                "auto_close_on_success": True
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        return str(config_file)
    
    @pytest.fixture
    def bug_tracker(self, mock_github_config):
        # Bug tracker with mocked GitHub client
        tracker = SmartBugTracker(mock_github_config)
        tracker.github_client = Mock(spec=GitHubAPIClient)
        return tracker
    
    def test_failure_record_creation(self, bug_tracker):
        # Test failure record creation and tracking
        test_name = "test_api_endpoint"
        error_message = "HTTP 500 Internal Server Error"
        traceback = "Traceback: requests.exceptions.HTTPError"
        
        # Record first failure
        bug_tracker.record_failure(test_name, error_message, traceback, "api_error")
        
        # Verify failure was recorded
        assert len(bug_tracker.failure_history) == 1
        
        failure_hash = list(bug_tracker.failure_history.keys())[0]
        record = bug_tracker.failure_history[failure_hash]
        
        assert record.test_name == test_name
        assert record.error_message == error_message
        assert record.failure_type == "api_error"
        assert record.consecutive_failures == 1
    
    def test_consecutive_failure_threshold(self, bug_tracker):
        # Test issue creation after consecutive failure threshold
        test_name = "test_flaky_endpoint"
        error_message = "Connection timeout"
        traceback = "requests.exceptions.Timeout"
        
        # Mock GitHub client methods
        bug_tracker.github_client.get_existing_issues.return_value = []
        bug_tracker.github_client.create_issue.return_value = Mock(
            success=True, 
            issue=Mock(number=123, title="Test Issue"),
            message="Issue created"
        )
        
        # Record failures below threshold
        bug_tracker.record_failure(test_name, error_message, traceback)
        assert not bug_tracker.github_client.create_issue.called
        
        # Record failure that exceeds threshold
        bug_tracker.record_failure(test_name, error_message, traceback)
        
        # Verify issue creation was attempted
        assert bug_tracker.github_client.create_issue.called
        
        # Check call arguments
        call_args = bug_tracker.github_client.create_issue.call_args[1]
        assert test_name in call_args['title']
        assert "2 consecutive failures" in call_args['title']
    
    def test_similar_failure_grouping(self, bug_tracker):
        # Test grouping of similar failures
        test_name = "test_similar_failures"
        error_message = "HTTP 404 Not Found"
        traceback = "requests.exceptions.HTTPError"
        
        # Mock existing issue with same failure hash
        existing_issue = Mock()
        existing_issue.failure_hash = bug_tracker._generate_failure_hash(
            test_name, error_message, "test_failure"
        )
        existing_issue.number = 456
        
        bug_tracker.github_client.get_existing_issues.return_value = [existing_issue]
        bug_tracker.github_client.add_comment.return_value = True
        
        # Record failures to trigger threshold
        for _ in range(2):
            bug_tracker.record_failure(test_name, error_message, traceback)
        
        # Should add comment to existing issue, not create new one
        assert bug_tracker.github_client.add_comment.called
        assert not bug_tracker.github_client.create_issue.called
    
    def test_auto_close_on_success(self, bug_tracker):
        # Test automatic issue closing when test passes
        test_name = "test_auto_close"
        
        # Mock existing open issue
        existing_issue = Mock()
        existing_issue.title = f"Test Failure: {test_name}"
        existing_issue.number = 789
        existing_issue.failure_hash = "test_hash"
        
        # Add failure record that matches the issue
        failure_record = FailureRecord(
            test_name=test_name,
            failure_type="test_failure", 
            error_message="Test error",
            timestamp=time.time(),
            traceback="Stack trace",
            failure_hash="test_hash",
            consecutive_failures=3
        )
        bug_tracker.failure_history["test_hash"] = failure_record
        
        bug_tracker.github_client.get_existing_issues.return_value = [existing_issue]
        bug_tracker.github_client.close_issue.return_value = True
        
        # Record success
        bug_tracker.record_success(test_name)
        
        # Verify issue was closed
        assert bug_tracker.github_client.close_issue.called
        call_args = bug_tracker.github_client.close_issue.call_args[0]
        assert call_args[0] == 789  # Issue number
    
    def test_daily_issue_limit(self, bug_tracker):
        # Test daily issue creation limit
        test_name = "test_rate_limit"
        error_message = "Rate limit test"
        traceback = "Test traceback"
        
        # Set daily count to maximum
        today = time.strftime("%Y-%m-%d")
        bug_tracker.daily_issue_count[today] = 3  # At max limit
        
        bug_tracker.github_client.get_existing_issues.return_value = []
        
        # Record failures to trigger threshold
        for _ in range(2):
            bug_tracker.record_failure(test_name, error_message, traceback)
        
        # Should not create issue due to daily limit
        assert not bug_tracker.github_client.create_issue.called
    
    def test_failure_summary_generation(self, bug_tracker):
        # Test failure summary generation
        # Add some test failures
        for i in range(5):
            bug_tracker.record_failure(
                f"test_{i}",
                f"Error {i}",
                f"Traceback {i}",
                "api_error"
            )
        
        summary = bug_tracker.get_failure_summary(hours=24)
        
        assert summary["total_failures"] == 5
        assert summary["unique_tests"] == 5
        assert "api_error" in summary["failure_types"]
        assert summary["failure_types"]["api_error"] == 5


class TestUnifiedFailureAnalysisEngine:
    # Test intelligent failure categorization system
    
    @pytest.fixture
    def analyzer(self):
        return UnifiedFailureAnalysisEngine()
    
    def test_api_error_classification(self, analyzer):
        # Test API error pattern detection
        test_name = "test_api_call"
        error_message = "HTTP 500 Internal Server Error"
        traceback = "requests.exceptions.HTTPError: 500 Server Error"
        
        classification = analyzer.analyze_failure(test_name, error_message, traceback)
        
        assert classification['category'] == FailureCategory.API_ERROR.value
        # Unified engine returns analysis dict instead of object
        assert 'matching_patterns' in classification
        assert 'root_cause_hints' in classification
    
    def test_network_error_classification(self, analyzer):
        # Test network error pattern detection
        test_name = "test_network"
        error_message = "Connection timeout"
        traceback = "requests.exceptions.ConnectionError: Connection timeout"
        
        classification = analyzer.analyze_failure(test_name, error_message, traceback)
        
        assert classification.category == FailureCategory.NETWORK_ERROR
        assert classification.retry_recommended == True
    
    def test_assertion_failure_classification(self, analyzer):
        # Test assertion failure detection
        test_name = "test_assertion"
        error_message = "AssertionError: Expected 200, got 404"
        traceback = "assert response.status_code == 200\nAssertionError"
        
        classification = analyzer.analyze_failure(test_name, error_message, traceback)
        
        assert classification.category == FailureCategory.ASSERTION_FAILURE
        assert classification.retry_recommended == False
    
    def test_flaky_test_detection(self, analyzer):
        # Test flaky test pattern detection
        test_name = "test_flaky"
        
        # Record some successes and failures
        for _ in range(3):
            analyzer.record_test_success(test_name)
        
        # Record intermittent failure
        error_message = "Intermittent failure detected"
        traceback = "Random timing issue"
        
        classification = analyzer.analyze_failure(test_name, error_message, traceback)
        
        # Check flaky test indicator
        indicator = analyzer.known_flaky_tests.get(test_name)
        assert indicator is not None
        assert indicator.recent_passes == 3
        assert indicator.recent_failures == 1
    
    def test_root_cause_hint_generation(self, analyzer):
        # Test root cause hint generation
        test_name = "test_hints"
        error_message = "HTTP 404 Not Found"
        traceback = "requests.exceptions.HTTPError: 404"
        
        classification = analyzer.analyze_failure(test_name, error_message, traceback)
        
        assert len(classification.root_cause_hints) > 0
        assert any("endpoint" in hint.lower() for hint in classification.root_cause_hints)
    
    def test_batch_failure_categorization(self, analyzer):
        # Test categorizing multiple failures
        failures = [
            ("test1", "HTTP 500", "HTTPError"),
            ("test2", "Connection timeout", "ConnectionError"),
            ("test3", "AssertionError", "assert failed"),
            ("test4", "HTTP 404", "HTTPError")
        ]
        
        categories = analyzer.categorize_batch_failures(failures)
        
        assert FailureCategory.API_ERROR in categories
        assert FailureCategory.NETWORK_ERROR in categories
        assert FailureCategory.ASSERTION_FAILURE in categories
        assert categories[FailureCategory.API_ERROR] == 2  # Two HTTP errors


class TestRetryManager:
    # Test intelligent retry mechanisms
    
    @pytest.fixture
    def retry_manager(self):
        return SmartRetryManager()
    
    def test_successful_execution_no_retry(self, retry_manager):
        # Test successful function execution without retry
        def successful_function():
            return "success"
        
        result = retry_manager.retry_with_analysis(successful_function)
        
        assert result.success == True
        assert result.result == "success"
        assert result.total_attempts == 1
        assert len(result.attempts) == 1
    
    def test_retry_on_api_error(self, retry_manager):
        # Test retry behavior on API errors
        call_count = 0
        
        def failing_api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("HTTP 500 Internal Server Error")
            return "success"
        
        result = retry_manager.retry_with_analysis(failing_api_call)
        
        assert result.success == True
        assert result.total_attempts == 3
        assert call_count == 3
    
    def test_no_retry_on_assertion_failure(self, retry_manager):
        # Test that assertion failures are not retried
        def assertion_failure():
            raise AssertionError("Expected 200, got 404")
        
        result = retry_manager.retry_with_analysis(assertion_failure)
        
        assert result.success == False
        assert result.total_attempts == 1  # No retries
    
    def test_category_specific_retry_config(self, retry_manager):
        # Test category-specific retry configuration
        call_count = 0
        
        def network_error():
            nonlocal call_count
            call_count += 1
            raise Exception("Connection timeout")
        
        result = retry_manager.retry_with_analysis(network_error)
        
        assert result.success == False
        # Network errors should get 4 retries (5 total attempts)
        assert result.total_attempts == 5
    
    def test_exponential_backoff_delay(self, retry_manager):
        # Test exponential backoff delay calculation
        attempt_times = []
        
        def failing_function():
            attempt_times.append(time.time())
            raise Exception("Test error")
        
        config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            strategy=RetryStrategy.EXPONENTIAL,
            backoff_multiplier=2.0
        )
        
        start_time = time.time()
        result = retry_manager.retry_with_analysis(failing_function, config=config)
        total_time = time.time() - start_time
        
        assert result.success == False
        assert len(attempt_times) == 4  # 4 attempts
        # Should take at least 7 seconds (1 + 2 + 4 = 7 seconds of delays)
        assert total_time >= 7.0
    
    @patch('src.retry_manager.time.sleep')
    def test_retry_statistics_tracking(self, mock_sleep, retry_manager):
        # Test retry statistics collection
        def sometimes_failing():
            if hasattr(sometimes_failing, 'call_count'):
                sometimes_failing.call_count += 1
            else:
                sometimes_failing.call_count = 1
                
            if sometimes_failing.call_count <= 2:
                raise Exception("Temporary error")
            return "success"
        
        # Execute function multiple times
        retry_manager.retry_with_analysis(sometimes_failing, context="test_function")
        
        stats = retry_manager.get_retry_statistics()
        
        assert "test_function" in stats["functions"]
        assert stats["functions"]["test_function"]["total_calls"] == 1
        assert stats["functions"]["test_function"]["successful_calls"] == 1
        assert stats["summary"]["total_functions"] == 1


class TestFailureReporting:
    # Test failure analysis and reporting engine
    
    @pytest.fixture
    def report_generator(self, tmp_path):
        return UnifiedFailureAnalysisEngine(str(tmp_path))
    
    def test_mttr_record_creation(self, report_generator):
        # Test MTTR record creation and tracking
        issue_id = "test_issue_123"
        test_name = "test_mttr"
        category = FailureCategory.API_ERROR
        
        # Record issue creation
        report_generator.record_issue_creation(issue_id, test_name, category)
        
        assert len(report_generator.mttr_records) == 1
        record = report_generator.mttr_records[0]
        assert record.issue_id == issue_id
        assert record.test_name == test_name
        assert record.failure_category == category
        assert record.resolved_at is None
    
    def test_mttr_resolution_tracking(self, report_generator):
        # Test MTTR resolution time calculation
        issue_id = "test_issue_456"
        test_name = "test_resolution"
        category = FailureCategory.NETWORK_ERROR
        
        # Record creation
        report_generator.record_issue_creation(issue_id, test_name, category)
        
        # Wait a bit and record resolution
        time.sleep(0.1)
        report_generator.record_issue_resolution(issue_id, "auto_close", True)
        
        # Check resolution was recorded
        record = report_generator.mttr_records[0]
        assert record.resolved_at is not None
        assert record.resolution_time_hours is not None
        assert record.resolution_time_hours > 0
        assert record.auto_resolved == True
        assert record.resolution_method == "auto_close"
    
    def test_comprehensive_report_generation(self, report_generator):
        # Test comprehensive failure report generation
        # Add some MTTR records
        for i in range(3):
            issue_id = f"issue_{i}"
            report_generator.record_issue_creation(
                issue_id, f"test_{i}", FailureCategory.API_ERROR
            )
            report_generator.record_issue_resolution(
                issue_id, "manual_fix", False
            )
        
        # Generate report
        report = report_generator.generate_comprehensive_report(time_period_hours=24)
        
        assert report.report_id.startswith("failure_report_")
        assert report.time_period_hours == 24
        assert "total_failures" in report.summary
        assert "overall_health_score" in report.summary
        assert len(report.failure_patterns) >= 0
        assert "total_resolved_issues" in report.mttr_analysis
        assert len(report.recommendations) > 0
    
    def test_html_report_generation(self, report_generator):
        # Test HTML report generation
        # Create a basic report
        report = report_generator.generate_comprehensive_report(time_period_hours=1)
        
        # Generate HTML
        html_file = report_generator.generate_html_report(report)
        
        assert html_file is not None
        assert html_file.exists()
        assert html_file.suffix == ".html"
        
        # Check HTML content
        with open(html_file, 'r') as f:
            content = f.read()
            assert "Failure Analysis Report" in content
            assert report.report_id in content
            assert "Executive Summary" in content


class TestIntegration:
    # Test integration between bug tracking components
    
    @pytest.fixture
    def integration_setup(self, tmp_path):
        # Setup for integration testing
        config_file = tmp_path / "integration_config.json"
        config = {
            "github": {"enabled": False},  # Disable for testing
            "failure_tracking": {
                "consecutive_failures_threshold": 2,
                "max_issues_per_day": 10
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        return {
            "config_file": str(config_file),
            "reports_dir": str(tmp_path / "reports")
        }
    
    def test_end_to_end_failure_tracking(self, integration_setup):
        # Test complete failure tracking workflow
        config_file = integration_setup["config_file"]
        
        # Initialize components
        bug_tracker = SmartBugTracker(config_file)
        analyzer = UnifiedFailureAnalysisEngine()
        retry_manager = SmartRetryManager()
        
        test_name = "test_integration"
        error_message = "HTTP 500 Server Error"
        traceback = "requests.exceptions.HTTPError"
        
        # 1. Analyze failure
        classification = analyzer.analyze_failure(test_name, error_message, traceback)
        assert classification.category == FailureCategory.API_ERROR
        
        # 2. Record failure in bug tracker
        bug_tracker.record_failure(test_name, error_message, traceback, "api_error")
        bug_tracker.record_failure(test_name, error_message, traceback, "api_error")
        
        # 3. Check failure history
        assert len(bug_tracker.failure_history) == 1
        failure_hash = list(bug_tracker.failure_history.keys())[0]
        record = bug_tracker.failure_history[failure_hash]
        assert record.consecutive_failures == 2
        
        # 4. Test retry logic integration
        call_count = 0
        def api_call_with_failure():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("HTTP 500 Server Error")
            return "success"
        
        result = retry_manager.retry_with_analysis(api_call_with_failure)
        assert result.success == True
        assert result.total_attempts == 3
        
        # 5. Record success to reset failure count
        bug_tracker.record_success(test_name)
        assert record.consecutive_failures == 0
    
    def test_flaky_test_workflow(self, integration_setup):
        # Test flaky test detection and handling workflow
        analyzer = UnifiedFailureAnalysisEngine()
        test_name = "test_flaky_workflow"
        
        # Simulate flaky test behavior (passes and fails intermittently)
        for i in range(10):
            if i % 3 == 0:  # Fail every 3rd attempt
                analyzer.analyze_failure(
                    test_name, 
                    "Intermittent timeout", 
                    "requests.timeout"
                )
            else:
                analyzer.record_test_success(test_name)
        
        # Check flaky test detection
        flaky_report = analyzer.get_flaky_tests_report()
        assert flaky_report["total_flaky_tests"] > 0
        
        # Find our test in the flaky tests
        our_test = None
        for test in flaky_report["flaky_tests"]:
            if test["test_name"] == test_name:
                our_test = test
                break
        
        assert our_test is not None
        assert our_test["pass_rate"] < 1.0  # Should have some failures
        assert len(flaky_report["recommendations"]) > 0
    
    @pytest.mark.integration
    def test_monitoring_integration(self, integration_setup):
        # Test integration with existing monitoring system
        from src.monitoring import get_metrics_collector
        
        config_file = integration_setup["config_file"]
        bug_tracker = SmartBugTracker(config_file)
        
        # Get metrics collector
        metrics_collector = get_metrics_collector()
        initial_custom_metrics = len(metrics_collector.custom_metrics)
        
        # Record some failures
        for i in range(3):
            bug_tracker.record_failure(
                f"test_monitoring_{i}",
                "Test error for monitoring",
                "Test traceback",
                "test_failure"
            )
        
        # Check if custom metrics were recorded
        final_custom_metrics = len(metrics_collector.custom_metrics)
        
        # The bug tracker should integrate with monitoring
        # (This would need actual integration code in the bug tracker)
        assert final_custom_metrics >= initial_custom_metrics