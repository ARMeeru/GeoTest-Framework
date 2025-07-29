# Failure analysis and reporting engine for Phase 5 intelligent bug tracking
# Generates comprehensive reports on failure patterns, MTTR tracking, and recommendations

import json
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from .failure_analyzer import FailureCategory, FailureSeverity, FlakyTestIndicator, get_failure_analyzer
from .github_integration import SmartBugTracker, FailureRecord

logger = logging.getLogger(__name__)


@dataclass
class MTTRRecord:
    # Mean Time To Resolution tracking record
    issue_id: str
    test_name: str
    failure_category: FailureCategory
    created_at: float
    resolved_at: Optional[float]
    resolution_time_hours: Optional[float]
    auto_resolved: bool
    resolution_method: str  # "auto_close", "manual_fix", "issue_closed"


@dataclass
class FailurePattern:
    # Identified failure pattern for reporting
    pattern_id: str
    description: str
    category: FailureCategory
    affected_tests: List[str]
    frequency: int
    first_seen: float
    last_seen: float
    severity_score: float
    recommended_actions: List[str]


@dataclass
class TestStabilityMetrics:
    # Stability metrics for individual tests
    test_name: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    pass_rate: float
    avg_failure_recovery_time: float
    flakiness_score: float
    most_common_failure: str
    stability_trend: str  # "improving", "stable", "degrading"


@dataclass
class FailureAnalysisReport:
    # Comprehensive failure analysis report
    report_id: str
    generated_at: float
    time_period_hours: int
    summary: Dict[str, Any]
    failure_patterns: List[FailurePattern]
    mttr_analysis: Dict[str, Any]
    test_stability: List[TestStabilityMetrics]
    recommendations: List[str]
    trend_analysis: Dict[str, Any]


class FailureReportGenerator:
    # Generates comprehensive failure analysis reports
    
    def __init__(self, reports_dir: str = "failure_reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        
        self.failure_analyzer = get_failure_analyzer()
        self.mttr_records = self._load_mttr_records()
        self.historical_data = self._load_historical_data()
        
    def _load_mttr_records(self) -> List[MTTRRecord]:
        # Load MTTR tracking records
        mttr_file = self.reports_dir / "mttr_records.json"
        if mttr_file.exists():
            try:
                with open(mttr_file, 'r') as f:
                    data = json.load(f)
                    return [MTTRRecord(**record) for record in data]
            except Exception as e:
                logger.warning(f"Failed to load MTTR records: {e}")
        return []
    
    def _save_mttr_records(self):
        # Save MTTR tracking records
        mttr_file = self.reports_dir / "mttr_records.json"
        try:
            data = [asdict(record) for record in self.mttr_records]
            with open(mttr_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save MTTR records: {e}")
    
    def _load_historical_data(self) -> Dict[str, Any]:
        # Load historical failure data for trend analysis
        history_file = self.reports_dir / "historical_data.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load historical data: {e}")
        return {"daily_summaries": {}, "weekly_summaries": {}}
    
    def _save_historical_data(self):
        # Save historical failure data
        history_file = self.reports_dir / "historical_data.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.historical_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save historical data: {e}")
    
    def record_issue_creation(self, issue_id: str, test_name: str, failure_category: FailureCategory):
        # Record when an issue is created for MTTR tracking
        record = MTTRRecord(
            issue_id=issue_id,
            test_name=test_name,
            failure_category=failure_category,
            created_at=time.time(),
            resolved_at=None,
            resolution_time_hours=None,
            auto_resolved=False,
            resolution_method=""
        )
        
        self.mttr_records.append(record)
        self._save_mttr_records()
        logger.info(f"Recorded issue creation for MTTR tracking: {issue_id}")
    
    def record_issue_resolution(self, issue_id: str, resolution_method: str, auto_resolved: bool = False):
        # Record when an issue is resolved for MTTR calculation
        for record in self.mttr_records:
            if record.issue_id == issue_id and record.resolved_at is None:
                record.resolved_at = time.time()
                record.resolution_time_hours = (record.resolved_at - record.created_at) / 3600
                record.auto_resolved = auto_resolved
                record.resolution_method = resolution_method
                
                self._save_mttr_records()
                logger.info(f"Recorded issue resolution: {issue_id} in {record.resolution_time_hours:.1f} hours")
                return
        
        logger.warning(f"Could not find MTTR record for issue: {issue_id}")
    
    def generate_comprehensive_report(self, time_period_hours: int = 168) -> FailureAnalysisReport:
        # Generate comprehensive failure analysis report (default: 1 week)
        
        report_id = f"failure_report_{int(time.time())}"
        cutoff_time = time.time() - (time_period_hours * 3600)
        
        logger.info(f"Generating failure analysis report for last {time_period_hours} hours")
        
        # Get failure data from bug tracker
        bug_tracker = SmartBugTracker()
        failure_summary = bug_tracker.get_failure_summary(time_period_hours)
        
        # Analyze failure patterns
        failure_patterns = self._identify_failure_patterns(cutoff_time)
        
        # Generate MTTR analysis
        mttr_analysis = self._analyze_mttr(cutoff_time)
        
        # Generate test stability metrics
        test_stability = self._analyze_test_stability(cutoff_time)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            failure_summary, failure_patterns, mttr_analysis, test_stability
        )
        
        # Generate trend analysis
        trend_analysis = self._analyze_trends(time_period_hours)
        
        # Create comprehensive summary
        summary = self._create_report_summary(
            failure_summary, failure_patterns, mttr_analysis, test_stability
        )
        
        report = FailureAnalysisReport(
            report_id=report_id,
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
        
        logger.info(f"Generated comprehensive failure report: {report_id}")
        return report
    
    def _identify_failure_patterns(self, cutoff_time: float) -> List[FailurePattern]:
        # Identify common failure patterns
        patterns = []
        
        # Get flaky tests report for pattern analysis
        flaky_report = self.failure_analyzer.get_flaky_tests_report()
        
        # Pattern 1: Highly flaky tests
        very_flaky_tests = [
            test for test in flaky_report["flaky_tests"]
            if test["pass_rate"] < 0.7
        ]
        
        if very_flaky_tests:
            patterns.append(FailurePattern(
                pattern_id="flaky_tests_high",
                description="Tests with high flakiness (pass rate < 70%)",
                category=FailureCategory.UNKNOWN,
                affected_tests=[test["test_name"] for test in very_flaky_tests],
                frequency=len(very_flaky_tests),
                first_seen=cutoff_time,
                last_seen=time.time(),
                severity_score=0.8,
                recommended_actions=[
                    "Review test environment stability",
                    "Add retry logic for flaky tests",
                    "Investigate root causes of flakiness"
                ]
            ))
        
        # Pattern 2: Network-related failures
        # This would be enhanced with actual failure data analysis
        patterns.append(FailurePattern(
            pattern_id="network_issues",
            description="Network connectivity and timeout issues",
            category=FailureCategory.NETWORK_ERROR,
            affected_tests=["network-dependent tests"],
            frequency=5,  # Placeholder
            first_seen=cutoff_time,
            last_seen=time.time(),
            severity_score=0.6,
            recommended_actions=[
                "Check network stability",
                "Implement network resilience patterns",
                "Add network monitoring"
            ]
        ))
        
        return patterns
    
    def _analyze_mttr(self, cutoff_time: float) -> Dict[str, Any]:
        # Analyze Mean Time To Resolution metrics
        
        recent_records = [
            record for record in self.mttr_records
            if record.created_at >= cutoff_time and record.resolved_at is not None
        ]
        
        if not recent_records:
            return {
                "total_resolved_issues": 0,
                "avg_resolution_time_hours": 0.0,
                "median_resolution_time_hours": 0.0,
                "auto_resolution_rate": 0.0,
                "resolution_by_category": {},
                "mttr_trend": "insufficient_data"
            }
        
        resolution_times = [record.resolution_time_hours for record in recent_records]
        auto_resolutions = len([r for r in recent_records if r.auto_resolved])
        
        # Calculate by category
        category_mttr = {}
        for category in FailureCategory:
            category_records = [r for r in recent_records if r.failure_category == category]
            if category_records:
                category_times = [r.resolution_time_hours for r in category_records]
                category_mttr[category.value] = {
                    "count": len(category_records),
                    "avg_hours": statistics.mean(category_times),
                    "median_hours": statistics.median(category_times)
                }
        
        return {
            "total_resolved_issues": len(recent_records),
            "avg_resolution_time_hours": statistics.mean(resolution_times),
            "median_resolution_time_hours": statistics.median(resolution_times),
            "auto_resolution_rate": auto_resolutions / len(recent_records),
            "resolution_by_category": category_mttr,
            "mttr_trend": self._calculate_mttr_trend(cutoff_time)
        }
    
    def _calculate_mttr_trend(self, cutoff_time: float) -> str:
        # Calculate MTTR trend over time
        # Compare current period with previous period
        
        period_duration = time.time() - cutoff_time
        previous_cutoff = cutoff_time - period_duration
        
        current_records = [
            r for r in self.mttr_records
            if cutoff_time <= r.created_at <= time.time() and r.resolved_at is not None
        ]
        
        previous_records = [
            r for r in self.mttr_records
            if previous_cutoff <= r.created_at < cutoff_time and r.resolved_at is not None
        ]
        
        if not previous_records or not current_records:
            return "insufficient_data"
        
        current_avg = statistics.mean([r.resolution_time_hours for r in current_records])
        previous_avg = statistics.mean([r.resolution_time_hours for r in previous_records])
        
        change_ratio = current_avg / previous_avg
        
        if change_ratio < 0.9:
            return "improving"
        elif change_ratio > 1.1:
            return "degrading"
        else:
            return "stable"
    
    def _analyze_test_stability(self, cutoff_time: float) -> List[TestStabilityMetrics]:
        # Analyze stability metrics for individual tests
        
        # Get flaky test data
        flaky_tests = self.failure_analyzer.known_flaky_tests
        
        stability_metrics = []
        
        for test_name, indicator in flaky_tests.items():
            if indicator.last_analysis >= cutoff_time:
                total_runs = indicator.recent_failures + indicator.recent_passes
                
                if total_runs > 0:
                    # Calculate flakiness score (0 = stable, 1 = very flaky)
                    flakiness_score = 1.0 - indicator.pass_rate
                    
                    # Determine stability trend (simplified)
                    if indicator.pass_rate > 0.9:
                        trend = "stable"
                    elif indicator.recent_passes > indicator.recent_failures:
                        trend = "improving"
                    else:
                        trend = "degrading"
                    
                    metrics = TestStabilityMetrics(
                        test_name=test_name,
                        total_runs=total_runs,
                        successful_runs=indicator.recent_passes,
                        failed_runs=indicator.recent_failures,
                        pass_rate=indicator.pass_rate,
                        avg_failure_recovery_time=0.0,  # Would need historical data
                        flakiness_score=flakiness_score,
                        most_common_failure=indicator.failure_pattern or "unknown",
                        stability_trend=trend
                    )
                    
                    stability_metrics.append(metrics)
        
        # Sort by flakiness score (most problematic first)
        stability_metrics.sort(key=lambda x: x.flakiness_score, reverse=True)
        
        return stability_metrics[:20]  # Top 20 most problematic tests
    
    def _generate_recommendations(self, failure_summary: Dict, failure_patterns: List[FailurePattern],
                                mttr_analysis: Dict, test_stability: List[TestStabilityMetrics]) -> List[str]:
        # Generate actionable recommendations
        
        recommendations = []
        
        # Based on failure summary
        if failure_summary.get("total_failures", 0) > 50:
            recommendations.append("High failure rate detected - review test environment stability")
        
        # Based on failure patterns
        high_severity_patterns = [p for p in failure_patterns if p.severity_score > 0.7]
        if high_severity_patterns:
            recommendations.append("Critical failure patterns identified - prioritize fixing high-severity issues")
        
        # Based on MTTR analysis
        if mttr_analysis.get("avg_resolution_time_hours", 0) > 24:
            recommendations.append("High MTTR detected - improve issue response time and automation")
        
        if mttr_analysis.get("auto_resolution_rate", 0) < 0.3:
            recommendations.append("Low auto-resolution rate - enhance automated test healing")
        
        # Based on test stability
        very_flaky_tests = [t for t in test_stability if t.flakiness_score > 0.5]
        if len(very_flaky_tests) > 5:
            recommendations.append("Multiple flaky tests detected - implement test stabilization program")
        
        degrading_tests = [t for t in test_stability if t.stability_trend == "degrading"]
        if degrading_tests:
            recommendations.append(f"{len(degrading_tests)} tests showing degrading stability - investigate recent changes")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Test suite appears stable - continue monitoring")
        
        recommendations.append("Regular review of failure patterns recommended")
        recommendations.append("Consider implementing automated test healing for common failures")
        
        return recommendations
    
    def _analyze_trends(self, time_period_hours: int) -> Dict[str, Any]:
        # Analyze failure trends over time
        
        # This would be enhanced with actual time-series data
        return {
            "failure_rate_trend": "stable",
            "mttr_trend": "improving",
            "test_stability_trend": "stable",
            "new_failure_patterns": 0,
            "resolved_failure_patterns": 1,
            "overall_health_score": 0.85  # 0-1 scale
        }
    
    def _create_report_summary(self, failure_summary: Dict, failure_patterns: List[FailurePattern],
                             mttr_analysis: Dict, test_stability: List[TestStabilityMetrics]) -> Dict[str, Any]:
        # Create comprehensive report summary
        
        return {
            "total_failures": failure_summary.get("total_failures", 0),
            "unique_failed_tests": failure_summary.get("unique_tests", 0),
            "failure_patterns_identified": len(failure_patterns),
            "high_severity_patterns": len([p for p in failure_patterns if p.severity_score > 0.7]),
            "avg_mttr_hours": mttr_analysis.get("avg_resolution_time_hours", 0),
            "auto_resolution_rate": mttr_analysis.get("auto_resolution_rate", 0),
            "flaky_tests_count": len([t for t in test_stability if t.flakiness_score > 0.3]),
            "degrading_tests_count": len([t for t in test_stability if t.stability_trend == "degrading"]),
            "overall_health_score": self._calculate_health_score(
                failure_summary, failure_patterns, mttr_analysis, test_stability
            )
        }
    
    def _calculate_health_score(self, failure_summary: Dict, failure_patterns: List[FailurePattern],
                              mttr_analysis: Dict, test_stability: List[TestStabilityMetrics]) -> float:
        # Calculate overall test suite health score (0-1)
        
        score = 1.0
        
        # Reduce score based on failure rate
        failure_rate = failure_summary.get("total_failures", 0) / max(100, failure_summary.get("total_failures", 100))
        score -= min(0.3, failure_rate)
        
        # Reduce score based on flaky tests
        flaky_tests = len([t for t in test_stability if t.flakiness_score > 0.3])
        if flaky_tests > 0:
            score -= min(0.2, flaky_tests * 0.02)
        
        # Reduce score based on MTTR
        mttr_hours = mttr_analysis.get("avg_resolution_time_hours", 0)
        if mttr_hours > 24:
            score -= min(0.2, (mttr_hours - 24) * 0.01)
        
        # Reduce score based on high-severity patterns
        high_severity_patterns = len([p for p in failure_patterns if p.severity_score > 0.7])
        if high_severity_patterns > 0:
            score -= min(0.3, high_severity_patterns * 0.1)
        
        return max(0.0, score)
    
    def _save_report(self, report: FailureAnalysisReport):
        # Save report to file
        report_file = self.reports_dir / f"{report.report_id}.json"
        
        try:
            # Convert dataclasses to dict for JSON serialization
            report_dict = {
                "report_id": report.report_id,
                "generated_at": report.generated_at,
                "time_period_hours": report.time_period_hours,
                "summary": report.summary,
                "failure_patterns": [asdict(p) for p in report.failure_patterns],
                "mttr_analysis": report.mttr_analysis,
                "test_stability": [asdict(t) for t in report.test_stability],
                "recommendations": report.recommendations,
                "trend_analysis": report.trend_analysis
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            logger.info(f"Saved failure report to: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def generate_html_report(self, report: FailureAnalysisReport) -> Path:
        # Generate HTML version of the report
        
        html_file = self.reports_dir / f"{report.report_id}.html"
        
        html_content = self._create_html_report(report)
        
        try:
            with open(html_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Generated HTML report: {html_file}")
            return html_file
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return None
    
    def _create_html_report(self, report: FailureAnalysisReport) -> str:
        # Create HTML content for the report
        
        generated_time = datetime.fromtimestamp(report.generated_at).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Failure Analysis Report - {report.report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: #e8f4fd; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
        .pattern {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 3px; }}
        .recommendation {{ background-color: #d4edda; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .flaky-test {{ background-color: #f8d7da; padding: 8px; margin: 5px 0; border-radius: 3px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Failure Analysis Report</h1>
        <p><strong>Report ID:</strong> {report.report_id}</p>
        <p><strong>Generated:</strong> {generated_time}</p>
        <p><strong>Time Period:</strong> {report.time_period_hours} hours</p>
    </div>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <div class="metric"><strong>Total Failures:</strong> {report.summary['total_failures']}</div>
        <div class="metric"><strong>Unique Failed Tests:</strong> {report.summary['unique_failed_tests']}</div>
        <div class="metric"><strong>Health Score:</strong> {report.summary['overall_health_score']:.2f}/1.0</div>
        <div class="metric"><strong>Avg MTTR:</strong> {report.summary['avg_mttr_hours']:.1f} hours</div>
        <div class="metric"><strong>Flaky Tests:</strong> {report.summary['flaky_tests_count']}</div>
    </div>
    
    <div class="section">
        <h2>Failure Patterns</h2>
"""
        
        for pattern in report.failure_patterns:
            html += f"""
        <div class="pattern">
            <h3>{pattern.description}</h3>
            <p><strong>Category:</strong> {pattern.category.value}</p>
            <p><strong>Affected Tests:</strong> {len(pattern.affected_tests)}</p>
            <p><strong>Severity Score:</strong> {pattern.severity_score:.2f}</p>
            <p><strong>Recommended Actions:</strong></p>
            <ul>
                {''.join(f'<li>{action}</li>' for action in pattern.recommended_actions)}
            </ul>
        </div>
"""
        
        html += """
    </div>
    
    <div class="section">
        <h2>Test Stability</h2>
        <table>
            <tr>
                <th>Test Name</th>
                <th>Pass Rate</th>
                <th>Flakiness Score</th>
                <th>Trend</th>
                <th>Total Runs</th>
            </tr>
"""
        
        for test in report.test_stability[:10]:  # Top 10 problematic tests
            html += f"""
            <tr>
                <td>{test.test_name}</td>
                <td>{test.pass_rate:.1%}</td>
                <td>{test.flakiness_score:.2f}</td>
                <td>{test.stability_trend}</td>
                <td>{test.total_runs}</td>
            </tr>
"""
        
        html += """
        </table>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
"""
        
        for rec in report.recommendations:
            html += f'<div class="recommendation">{rec}</div>'
        
        html += """
    </div>
    
    <div class="section">
        <h2>MTTR Analysis</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
"""
        
        mttr = report.mttr_analysis
        html += f"""
            <tr><td>Total Resolved Issues</td><td>{mttr.get('total_resolved_issues', 0)}</td></tr>
            <tr><td>Average Resolution Time</td><td>{mttr.get('avg_resolution_time_hours', 0):.1f} hours</td></tr>
            <tr><td>Median Resolution Time</td><td>{mttr.get('median_resolution_time_hours', 0):.1f} hours</td></tr>
            <tr><td>Auto Resolution Rate</td><td>{mttr.get('auto_resolution_rate', 0):.1%}</td></tr>
            <tr><td>MTTR Trend</td><td>{mttr.get('mttr_trend', 'unknown')}</td></tr>
"""
        
        html += """
        </table>
    </div>
</body>
</html>
"""
        
        return html
    
    def get_recent_reports(self, limit: int = 10) -> List[str]:
        # Get list of recent report files
        report_files = list(self.reports_dir.glob("failure_report_*.json"))
        report_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return [f.stem for f in report_files[:limit]]


def get_failure_report_generator() -> FailureReportGenerator:
    # Get global failure report generator instance
    global _failure_report_generator_instance
    if '_failure_report_generator_instance' not in globals():
        _failure_report_generator_instance = FailureReportGenerator()
    return _failure_report_generator_instance