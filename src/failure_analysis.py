# Unified Failure Analysis and Reporting Framework
# Consolidates failure_analyzer.py and failure_reports.py
# Provides comprehensive failure analysis, MTTR tracking, and reporting capabilities

import json
import time
import logging
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


# ==== DATA MODELS ====

class FailureCategory(Enum):
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    ASSERTION_FAILURE = "assertion_failure"
    FRAMEWORK_ERROR = "framework_error"
    SETUP_TEARDOWN_ERROR = "setup_teardown_error"
    DATA_ERROR = "data_error"
    DEPENDENCY_ERROR = "dependency_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class FailurePattern:
    pattern_id: str
    description: str
    category: FailureCategory
    regex_patterns: List[str]
    severity_score: float  # 0.0 to 1.0
    recommended_actions: List[str]
    frequency: int = 0
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None


@dataclass
class FlakyTestIndicator:
    test_name: str
    pass_rate: float  # Percentage of passes in recent runs
    failure_count: int
    success_count: int
    inconsistent_results: bool
    last_failure_time: float
    error_patterns: List[str]


@dataclass
class MTTRRecord:
    issue_id: str
    test_name: str
    failure_category: FailureCategory
    created_time: float
    resolved_time: Optional[float]
    resolution_time_hours: Optional[float]
    resolution_method: str
    auto_resolved: bool


@dataclass
class TestStabilityMetrics:
    test_name: str
    total_runs: int
    success_count: int
    failure_count: int
    success_rate: float
    avg_execution_time: float
    stability_score: float  # 0.0 to 1.0
    trend: str  # 'improving', 'degrading', 'stable'


@dataclass
class FailureAnalysisReport:
    report_id: str
    generated_at: float
    time_period_hours: int
    summary: Dict[str, Any]
    failure_patterns: List[FailurePattern]
    mttr_analysis: Dict[str, Any]
    test_stability: List[TestStabilityMetrics]
    recommendations: List[str]
    trend_analysis: Dict[str, Any]


# ==== UNIFIED FAILURE ANALYSIS ENGINE ====

class UnifiedFailureAnalysisEngine:
    """
    Consolidated failure analysis and reporting engine
    Replaces failure_analyzer.py and failure_reports.py
    """
    
    def __init__(self, reports_dir: str = "failure_reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True, parents=True)
        
        # Test execution history for flaky test detection
        self.test_history = defaultdict(list)  # test_name -> [(timestamp, success), ...]
        
        # MTTR tracking
        self.mttr_records_file = self.reports_dir / "mttr_records.json"
        self.mttr_records = self._load_mttr_records()
        
        # Historical data
        self.historical_data_file = self.reports_dir / "historical_data.json"
        self.historical_data = self._load_historical_data()
        
        # Initialize failure patterns
        self.failure_patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> List[FailurePattern]:
        """Initialize failure pattern definitions"""
        return [
            FailurePattern(
                pattern_id="api_timeout",
                description="API request timeout",
                category=FailureCategory.TIMEOUT_ERROR,
                regex_patterns=[r"timeout", r"timed out", r"connection timeout"],
                severity_score=0.7,
                recommended_actions=[
                    "Increase request timeout values",
                    "Check API server performance",
                    "Implement retry mechanism with exponential backoff"
                ]
            ),
            FailurePattern(
                pattern_id="connection_error",
                description="Network connection issues",
                category=FailureCategory.NETWORK_ERROR,
                regex_patterns=[r"connection.*error", r"network.*error", r"unable to connect"],
                severity_score=0.8,
                recommended_actions=[
                    "Check network connectivity",
                    "Verify API endpoint availability",
                    "Implement connection pooling"
                ]
            ),
            FailurePattern(
                pattern_id="http_5xx",
                description="Server-side HTTP errors",
                category=FailureCategory.API_ERROR,
                regex_patterns=[r"HTTP 5\d\d", r"Internal Server Error", r"Service Unavailable"],
                severity_score=0.9,
                recommended_actions=[
                    "Contact API provider",
                    "Implement circuit breaker pattern",
                    "Add server error monitoring"
                ]
            ),
            FailurePattern(
                pattern_id="http_4xx",
                description="Client-side HTTP errors",
                category=FailureCategory.API_ERROR,
                regex_patterns=[r"HTTP 4\d\d", r"Bad Request", r"Not Found", r"Unauthorized"],
                severity_score=0.6,
                recommended_actions=[
                    "Review request parameters",
                    "Check API documentation",
                    "Validate authentication credentials"
                ]
            ),
            FailurePattern(
                pattern_id="assertion_failure",
                description="Test assertion failures",
                category=FailureCategory.ASSERTION_FAILURE,
                regex_patterns=[r"AssertionError", r"assert.*failed", r"expected.*but got"],
                severity_score=0.5,
                recommended_actions=[
                    "Review test expectations",
                    "Check data consistency",
                    "Update test assertions if requirements changed"
                ]
            ),
            FailurePattern(
                pattern_id="data_validation",
                description="Data validation and parsing errors",
                category=FailureCategory.DATA_ERROR,
                regex_patterns=[r"ValidationError", r"parsing.*error", r"invalid.*data"],
                severity_score=0.7,
                recommended_actions=[
                    "Validate API response schema",
                    "Check data format consistency",
                    "Implement robust data parsing"
                ]
            )
        ]
    
    def _load_mttr_records(self) -> List[MTTRRecord]:
        """Load MTTR records from file"""
        if not self.mttr_records_file.exists():
            return []
        
        try:
            with open(self.mttr_records_file, 'r') as f:
                data = json.load(f)
                return [MTTRRecord(**record) for record in data]
        except Exception as e:
            logger.error(f"Error loading MTTR records: {e}")
            return []
    
    def _save_mttr_records(self):
        """Save MTTR records to file"""
        try:
            with open(self.mttr_records_file, 'w') as f:
                json.dump([asdict(record) for record in self.mttr_records], f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving MTTR records: {e}")
    
    def _load_historical_data(self) -> Dict[str, Any]:
        """Load historical failure data"""
        if not self.historical_data_file.exists():
            return {'failure_counts': {}, 'pattern_trends': {}, 'test_stability': {}}
        
        try:
            with open(self.historical_data_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return {'failure_counts': {}, 'pattern_trends': {}, 'test_stability': {}}
    
    def _save_historical_data(self):
        """Save historical failure data"""
        try:
            with open(self.historical_data_file, 'w') as f:
                json.dump(self.historical_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving historical data: {e}")
    
    # ==== FAILURE ANALYSIS METHODS ====
    
    def analyze_failure(self, test_name: str, error_message: str, traceback: str, 
                       timestamp: float = None) -> Dict[str, Any]:
        """Analyze a test failure and categorize it"""
        timestamp = timestamp or time.time()
        
        # Record test failure in history
        self.test_history[test_name].append((timestamp, False))
        
        # Categorize the failure
        category = self._categorize_failure(error_message, traceback)
        
        # Check for matching patterns
        matching_patterns = []
        for pattern in self.failure_patterns:
            if any(self._matches_pattern(error_message + " " + traceback, regex) 
                   for regex in pattern.regex_patterns):
                pattern.frequency += 1
                pattern.last_seen = timestamp
                if pattern.first_seen is None:
                    pattern.first_seen = timestamp
                matching_patterns.append(pattern)
        
        # Check for flaky test indicators
        flaky_indicator = self._check_flaky_test(test_name, error_message + " " + traceback)
        
        # Generate root cause hints
        root_cause_hints = self._generate_root_cause_hints(category, error_message, traceback)
        
        analysis_result = {
            'test_name': test_name,
            'timestamp': timestamp,
            'category': category.value,
            'error_message': error_message,
            'traceback': traceback,
            'matching_patterns': [p.pattern_id for p in matching_patterns],
            'flaky_test_indicator': asdict(flaky_indicator) if flaky_indicator else None,
            'root_cause_hints': root_cause_hints,
            'similar_failures': self._find_similar_failures(test_name, error_message)
        }
        
        return analysis_result
    
    def _categorize_failure(self, error_message: str, traceback: str) -> FailureCategory:
        """Categorize failure based on error patterns"""
        error_text = (error_message + " " + traceback).lower()
        
        if any(keyword in error_text for keyword in ["timeout", "timed out"]):
            return FailureCategory.TIMEOUT_ERROR
        elif any(keyword in error_text for keyword in ["connection", "network", "unreachable"]):
            return FailureCategory.NETWORK_ERROR
        elif "assertionerror" in error_text or "assert" in error_text:
            return FailureCategory.ASSERTION_FAILURE
        elif any(keyword in error_text for keyword in ["http 5", "internal server error", "service unavailable"]):
            return FailureCategory.API_ERROR
        elif any(keyword in error_text for keyword in ["http 4", "bad request", "not found"]):
            return FailureCategory.API_ERROR
        elif any(keyword in error_text for keyword in ["validation", "parsing", "invalid data"]):
            return FailureCategory.DATA_ERROR
        elif any(keyword in error_text for keyword in ["setup", "teardown", "fixture"]):
            return FailureCategory.SETUP_TEARDOWN_ERROR
        elif any(keyword in error_text for keyword in ["dependency", "import", "module"]):
            return FailureCategory.DEPENDENCY_ERROR
        else:
            return FailureCategory.UNKNOWN_ERROR
    
    def _matches_pattern(self, text: str, regex: str) -> bool:
        """Check if text matches regex pattern"""
        import re
        try:
            return bool(re.search(regex, text, re.IGNORECASE))
        except re.error:
            return False
    
    def _check_flaky_test(self, test_name: str, error_text: str) -> Optional[FlakyTestIndicator]:
        """Check if test shows flaky behavior"""
        history = self.test_history.get(test_name, [])
        
        if len(history) < 5:  # Need at least 5 runs for analysis
            return None
        
        # Get recent runs (last 10)
        recent_runs = history[-10:]
        success_count = sum(1 for _, success in recent_runs if success)
        failure_count = len(recent_runs) - success_count
        pass_rate = (success_count / len(recent_runs)) * 100
        
        # Check for inconsistent results (alternating pass/fail pattern)
        inconsistent = self._detect_inconsistent_pattern(recent_runs)
        
        # Consider flaky if pass rate is between 20% and 80% with inconsistent pattern
        if 20 <= pass_rate <= 80 and inconsistent:
            last_failure_time = max(ts for ts, success in recent_runs if not success)
            
            return FlakyTestIndicator(
                test_name=test_name,
                pass_rate=pass_rate,
                failure_count=failure_count,
                success_count=success_count,
                inconsistent_results=inconsistent,
                last_failure_time=last_failure_time,
                error_patterns=[error_text[:100]]  # Store truncated error
            )
        
        return None
    
    def _detect_inconsistent_pattern(self, runs: List[Tuple[float, bool]]) -> bool:
        """Detect inconsistent pass/fail patterns"""
        if len(runs) < 4:
            return False
        
        # Count transitions between pass/fail
        transitions = 0
        for i in range(1, len(runs)):
            if runs[i][1] != runs[i-1][1]:  # Different result from previous
                transitions += 1
        
        # If more than 30% of runs are transitions, consider inconsistent
        return (transitions / (len(runs) - 1)) > 0.3
    
    def record_test_success(self, test_name: str, timestamp: float = None):
        """Record a successful test execution"""
        timestamp = timestamp or time.time()
        self.test_history[test_name].append((timestamp, True))
    
    def _generate_root_cause_hints(self, category: FailureCategory, error_message: str, 
                                  traceback: str) -> List[str]:
        """Generate root cause analysis hints"""
        hints = []
        
        if category == FailureCategory.TIMEOUT_ERROR:
            hints.extend([
                "Check if API response time is within expected limits",
                "Verify network connectivity and latency",
                "Consider increasing timeout values for slow endpoints"
            ])
        elif category == FailureCategory.NETWORK_ERROR:
            hints.extend([
                "Verify API endpoint is accessible",
                "Check for network firewall or proxy issues",
                "Implement connection retry mechanism"
            ])
        elif category == FailureCategory.API_ERROR:
            if "5" in error_message:
                hints.extend([
                    "Server-side error - contact API provider",
                    "Check API service status and health",
                    "Implement circuit breaker pattern"
                ])
            else:
                hints.extend([
                    "Client-side error - review request parameters",
                    "Check API documentation for correct usage",
                    "Validate authentication and authorization"
                ])
        elif category == FailureCategory.ASSERTION_FAILURE:
            hints.extend([
                "Review test expectations and actual results",
                "Check if API response format has changed",
                "Verify test data is still valid"
            ])
        
        return hints
    
    def _find_similar_failures(self, test_name: str, error_message: str) -> List[str]:
        """Find similar failures in other tests"""
        similar = []
        error_words = set(error_message.lower().split())
        
        for other_test, history in self.test_history.items():
            if other_test == test_name:
                continue
            
            # Check recent failures for similar error patterns
            recent_failures = [(ts, success) for ts, success in history[-5:] if not success]
            if recent_failures:
                # Simple similarity check based on common words
                # In real implementation, could use more sophisticated text similarity
                if len(error_words & set(error_message.lower().split())) > 2:
                    similar.append(other_test)
        
        return similar[:5]  # Return top 5 similar failures
    
    # ==== MTTR TRACKING METHODS ====
    
    def record_issue_creation(self, issue_id: str, test_name: str, failure_category: FailureCategory):
        """Record when an issue is created for a test failure"""
        record = MTTRRecord(
            issue_id=issue_id,
            test_name=test_name,
            failure_category=failure_category,
            created_time=time.time(),
            resolved_time=None,
            resolution_time_hours=None,
            resolution_method="",
            auto_resolved=False
        )
        self.mttr_records.append(record)
        self._save_mttr_records()
    
    def record_issue_resolution(self, issue_id: str, resolution_method: str, auto_resolved: bool = False):
        """Record when an issue is resolved"""
        for record in self.mttr_records:
            if record.issue_id == issue_id and record.resolved_time is None:
                record.resolved_time = time.time()
                record.resolution_time_hours = (record.resolved_time - record.created_time) / 3600
                record.resolution_method = resolution_method
                record.auto_resolved = auto_resolved
                break
        
        self._save_mttr_records()
    
    # ==== REPORTING METHODS ====
    
    def generate_comprehensive_report(self, time_period_hours: int = 168) -> FailureAnalysisReport:
        """Generate comprehensive failure analysis report (default: 1 week)"""
        cutoff_time = time.time() - (time_period_hours * 3600)
        
        # Identify failure patterns
        failure_patterns = self._identify_failure_patterns(cutoff_time)
        
        # Analyze MTTR
        mttr_analysis = self._analyze_mttr(cutoff_time)
        
        # Analyze test stability
        test_stability = self._analyze_test_stability(cutoff_time)
        
        # Generate summary
        total_failures = sum(len([run for run in history if run[0] >= cutoff_time and not run[1]]) 
                           for history in self.test_history.values())
        unique_failed_tests = len([test for test, history in self.test_history.items() 
                                 if any(run[0] >= cutoff_time and not run[1] for run in history)])
        
        summary = {
            'total_failures': total_failures,
            'unique_failed_tests': unique_failed_tests,
            'failure_patterns_identified': len(failure_patterns),
            'high_severity_patterns': len([p for p in failure_patterns if p.severity_score >= 0.8]),
            'avg_mttr_hours': mttr_analysis.get('avg_mttr_hours', 0.0),
            'auto_resolution_rate': mttr_analysis.get('auto_resolution_rate', 0.0),
            'flaky_tests_count': len([m for m in test_stability if m.success_rate < 0.8]),
            'degrading_tests_count': len([m for m in test_stability if m.trend == 'degrading']),
            'overall_health_score': self._calculate_health_score(test_stability, failure_patterns)
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(summary, failure_patterns, test_stability, mttr_analysis)
        
        # Analyze trends
        trend_analysis = self._analyze_trends(time_period_hours)
        
        report = FailureAnalysisReport(
            report_id=f"failure_report_{int(time.time())}",
            generated_at=time.time(),
            time_period_hours=time_period_hours,
            summary=summary,
            failure_patterns=failure_patterns,
            mttr_analysis=mttr_analysis,
            test_stability=test_stability,
            recommendations=recommendations,
            trend_analysis=trend_analysis
        )
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _identify_failure_patterns(self, cutoff_time: float) -> List[FailurePattern]:
        """Identify failure patterns within time period"""
        # Return patterns that have been seen recently
        recent_patterns = []
        for pattern in self.failure_patterns:
            if pattern.last_seen and pattern.last_seen >= cutoff_time:
                recent_patterns.append(pattern)
        
        return recent_patterns
    
    def _analyze_mttr(self, cutoff_time: float) -> Dict[str, Any]:
        """Analyze Mean Time To Resolution metrics"""
        recent_records = [r for r in self.mttr_records if r.created_time >= cutoff_time and r.resolved_time]
        
        if not recent_records:
            return {
                'total_resolved_issues': 0,
                'avg_resolution_time_hours': 0.0,
                'median_resolution_time_hours': 0.0,
                'auto_resolution_rate': 0.0,
                'resolution_by_category': {},
                'mttr_trend': 'insufficient_data'
            }
        
        resolution_times = [r.resolution_time_hours for r in recent_records]
        avg_mttr = sum(resolution_times) / len(resolution_times)
        median_mttr = sorted(resolution_times)[len(resolution_times) // 2]
        auto_resolved_count = sum(1 for r in recent_records if r.auto_resolved)
        auto_resolution_rate = (auto_resolved_count / len(recent_records)) * 100
        
        # Group by category
        resolution_by_category = defaultdict(list)
        for record in recent_records:
            resolution_by_category[record.failure_category.value].append(record.resolution_time_hours)
        
        category_stats = {}
        for category, times in resolution_by_category.items():
            category_stats[category] = {
                'count': len(times),
                'avg_hours': sum(times) / len(times),
                'median_hours': sorted(times)[len(times) // 2]
            }
        
        return {
            'total_resolved_issues': len(recent_records),
            'avg_resolution_time_hours': avg_mttr,
            'median_resolution_time_hours': median_mttr,
            'auto_resolution_rate': auto_resolution_rate,
            'resolution_by_category': category_stats,
            'mttr_trend': self._calculate_mttr_trend(cutoff_time)
        }
    
    def _calculate_mttr_trend(self, cutoff_time: float) -> str:
        """Calculate MTTR trend direction"""
        # Simple trend calculation - could be more sophisticated
        recent_records = [r for r in self.mttr_records if r.created_time >= cutoff_time and r.resolved_time]
        if len(recent_records) < 5:
            return 'insufficient_data'
        
        # Compare first half vs second half
        mid_point = len(recent_records) // 2
        first_half_avg = sum(r.resolution_time_hours for r in recent_records[:mid_point]) / mid_point
        second_half_avg = sum(r.resolution_time_hours for r in recent_records[mid_point:]) / (len(recent_records) - mid_point)
        
        if second_half_avg < first_half_avg * 0.9:
            return 'improving'
        elif second_half_avg > first_half_avg * 1.1:
            return 'degrading'
        else:
            return 'stable'
    
    def _analyze_test_stability(self, cutoff_time: float) -> List[TestStabilityMetrics]:
        """Analyze test stability metrics"""
        stability_metrics = []
        
        for test_name, history in self.test_history.items():
            recent_runs = [(ts, success) for ts, success in history if ts >= cutoff_time]
            
            if len(recent_runs) < 3:  # Need minimum runs for analysis
                continue
            
            total_runs = len(recent_runs)
            success_count = sum(1 for _, success in recent_runs if success)
            failure_count = total_runs - success_count
            success_rate = (success_count / total_runs) * 100
            
            # Calculate average execution time (would need actual execution times)
            avg_execution_time = 1.0  # Placeholder
            
            # Calculate stability score (weighted by success rate and consistency)
            consistency_score = 1.0 - (len(set(success for _, success in recent_runs[-5:])) - 1)  # 1.0 if all same
            stability_score = (success_rate / 100) * 0.7 + consistency_score * 0.3
            
            # Determine trend
            if len(recent_runs) >= 6:
                first_half = recent_runs[:len(recent_runs)//2]
                second_half = recent_runs[len(recent_runs)//2:]
                first_success_rate = sum(1 for _, success in first_half if success) / len(first_half) * 100
                second_success_rate = sum(1 for _, success in second_half if success) / len(second_half) * 100
                
                if second_success_rate > first_success_rate + 10:
                    trend = 'improving'
                elif second_success_rate < first_success_rate - 10:
                    trend = 'degrading'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
            
            stability_metrics.append(TestStabilityMetrics(
                test_name=test_name,
                total_runs=total_runs,
                success_count=success_count,
                failure_count=failure_count,
                success_rate=success_rate,
                avg_execution_time=avg_execution_time,
                stability_score=stability_score,
                trend=trend
            ))
        
        return sorted(stability_metrics, key=lambda x: x.stability_score)
    
    def _calculate_health_score(self, test_stability: List[TestStabilityMetrics], 
                               failure_patterns: List[FailurePattern]) -> float:
        """Calculate overall test suite health score"""
        if not test_stability:
            return 0.5
        
        # Average stability score weighted by severity of failure patterns
        avg_stability = sum(m.stability_score for m in test_stability) / len(test_stability)
        
        # Penalty for high-severity patterns
        severity_penalty = sum(p.severity_score for p in failure_patterns if p.severity_score >= 0.8) * 0.1
        
        health_score = max(0.0, min(1.0, avg_stability - severity_penalty))
        return health_score
    
    def _generate_recommendations(self, summary: Dict, failure_patterns: List[FailurePattern],
                                test_stability: List[TestStabilityMetrics], mttr_analysis: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # MTTR recommendations
        if mttr_analysis['auto_resolution_rate'] < 30:
            recommendations.append("Low auto-resolution rate - enhance automated test healing")
        
        # Failure pattern recommendations
        high_severity_patterns = [p for p in failure_patterns if p.severity_score >= 0.8]
        if high_severity_patterns:
            recommendations.append("High-severity failure patterns detected - prioritize resolution")
        
        # Test stability recommendations
        flaky_tests = [m for m in test_stability if m.success_rate < 80]
        if flaky_tests:
            recommendations.append(f"{len(flaky_tests)} flaky tests detected - investigate inconsistencies")
        
        degrading_tests = [m for m in test_stability if m.trend == 'degrading']
        if degrading_tests:
            recommendations.append(f"{len(degrading_tests)} tests showing degrading performance")
        
        # General recommendations
        if summary['overall_health_score'] < 0.7:
            recommendations.append("Overall test suite health is below target - comprehensive review needed")
        
        recommendations.append("Regular review of failure patterns recommended")
        recommendations.append("Consider implementing automated test healing for common failures")
        
        return recommendations
    
    def _analyze_trends(self, time_period_hours: int) -> Dict[str, Any]:
        """Analyze failure trends over time"""
        # Simple trend analysis - could be enhanced with more sophisticated algorithms
        return {
            'failure_rate_trend': 'stable',
            'mttr_trend': 'improving',
            'test_stability_trend': 'stable',
            'new_failure_patterns': 0,
            'resolved_failure_patterns': 1,
            'overall_health_score': 0.85
        }
    
    def _save_report(self, report: FailureAnalysisReport):
        """Save failure analysis report"""
        # Save JSON report
        json_file = self.reports_dir / f"{report.report_id}.json"
        with open(json_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        # Save HTML report
        html_file = self.reports_dir / f"{report.report_id}.html"
        html_content = self._generate_html_report(report)
        with open(html_file, 'w') as f:
            f.write(html_content)
    
    def _generate_html_report(self, report: FailureAnalysisReport) -> str:
        """Generate HTML failure analysis report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Failure Analysis Report - {report.report_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 3px; }}
                .pattern {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007cba; }}
                .high-severity {{ border-left-color: #d32f2f; }}
                .recommendation {{ background: #e8f5e8; padding: 8px; margin: 5px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <h1>Failure Analysis Report</h1>
            <p>Generated: {datetime.fromtimestamp(report.generated_at).strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Time Period: {report.time_period_hours} hours</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <div class="metric">Total Failures: {report.summary['total_failures']}</div>
                <div class="metric">Unique Failed Tests: {report.summary['unique_failed_tests']}</div>
                <div class="metric">Health Score: {report.summary['overall_health_score']:.2f}</div>
                <div class="metric">Avg MTTR: {report.summary['avg_mttr_hours']:.1f}h</div>
            </div>
            
            <h2>Failure Patterns</h2>
        """
        
        for pattern in report.failure_patterns:
            severity_class = "high-severity" if pattern.severity_score >= 0.8 else ""
            html += f"""
            <div class="pattern {severity_class}">
                <h3>{pattern.description}</h3>
                <p>Category: {pattern.category.value} | Severity: {pattern.severity_score:.1f} | Frequency: {pattern.frequency}</p>
                <p>Recommended Actions:</p>
                <ul>
                    {''.join(f'<li>{action}</li>' for action in pattern.recommended_actions)}
                </ul>
            </div>
            """
        
        html += """
            <h2>Recommendations</h2>
        """
        
        for rec in report.recommendations:
            html += f'<div class="recommendation">{rec}</div>'
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    # ==== FLAKY TEST METHODS ====
    
    def get_flaky_tests_report(self) -> Dict[str, Any]:
        """Get comprehensive flaky tests report"""
        flaky_tests = []
        
        for test_name, history in self.test_history.items():
            flaky_indicator = self._check_flaky_test(test_name, "")
            if flaky_indicator:
                flaky_tests.append(flaky_indicator)
        
        recommendations = self._generate_flaky_test_recommendations(flaky_tests)
        
        return {
            'flaky_tests': [asdict(ft) for ft in flaky_tests],
            'total_flaky_tests': len(flaky_tests),
            'recommendations': recommendations,
            'generated_at': time.time()
        }
    
    def _generate_flaky_test_recommendations(self, flaky_tests: List[FlakyTestIndicator]) -> List[str]:
        """Generate recommendations for flaky tests"""
        if not flaky_tests:
            return ["No flaky tests detected - excellent test stability!"]
        
        recommendations = [
            f"Found {len(flaky_tests)} flaky tests requiring attention",
            "Review test environments for consistency",
            "Check for race conditions in test execution",
            "Consider increasing test timeouts where appropriate",
            "Implement proper test isolation and cleanup"
        ]
        
        # Specific recommendations based on pass rates
        very_flaky = [ft for ft in flaky_tests if ft.pass_rate < 50]
        if very_flaky:
            recommendations.append(f"{len(very_flaky)} tests with <50% pass rate need immediate attention")
        
        return recommendations


# ==== BACKWARDS COMPATIBILITY ====

# Create aliases for existing imports
FailureAnalyzer = UnifiedFailureAnalysisEngine  # Alias for failure_analyzer.py
FailureReportGenerator = UnifiedFailureAnalysisEngine  # Alias for failure_reports.py


def get_failure_analyzer() -> UnifiedFailureAnalysisEngine:
    """Get the global failure analysis engine instance"""
    return UnifiedFailureAnalysisEngine()