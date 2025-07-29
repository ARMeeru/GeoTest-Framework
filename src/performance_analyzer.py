# Performance analysis and reporting utilities
# Provides statistical analysis, trend detection, and performance insights

import json
import statistics
import numpy as np
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .performance_testing import LoadTestResult, StressTestResult, BenchmarkResult, PerformanceResult

logger = logging.getLogger(__name__)


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
    summary_metrics: Dict[str, Any]
    insights: List[PerformanceInsight]
    trends: List[PerformanceTrend]
    recommendations: List[str]
    charts_generated: List[str]


class PerformanceAnalyzer:
    # Main performance analysis engine
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("performance_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        # Performance thresholds for analysis
        self.thresholds = {
            'response_time_warning': 2.0,      # seconds
            'response_time_critical': 5.0,     # seconds
            'success_rate_warning': 95.0,      # percentage
            'success_rate_critical': 90.0,     # percentage
            'rps_minimum': 10.0,               # requests per second
            'error_rate_warning': 1.0,         # percentage
            'error_rate_critical': 5.0,        # percentage
            'cpu_warning': 80.0,               # percentage
            'memory_warning': 85.0,            # percentage
        }
    
    def analyze_load_test(self, result: LoadTestResult) -> PerformanceReport:
        # Analyze load test results and generate insights
        insights = []
        recommendations = []
        
        # Response time analysis
        if result.avg_response_time > self.thresholds['response_time_critical']:
            insights.append(PerformanceInsight(
                category='performance',
                severity='critical',
                title='Critical Response Time',
                description=f'Average response time ({result.avg_response_time:.2f}s) exceeds critical threshold',
                metric_value=result.avg_response_time,
                threshold=self.thresholds['response_time_critical'],
                recommendation='Investigate server performance and optimize API endpoints'
            ))
            recommendations.append('Consider implementing caching or optimizing database queries')
        elif result.avg_response_time > self.thresholds['response_time_warning']:
            insights.append(PerformanceInsight(
                category='performance',
                severity='warning',
                title='Elevated Response Time',
                description=f'Average response time ({result.avg_response_time:.2f}s) above optimal levels',
                metric_value=result.avg_response_time,
                threshold=self.thresholds['response_time_warning'],
                recommendation='Monitor response times and consider performance optimization'
            ))
        
        # Success rate analysis
        if result.success_rate < self.thresholds['success_rate_critical']:
            insights.append(PerformanceInsight(
                category='reliability',
                severity='critical',
                title='Critical Success Rate',
                description=f'Success rate ({result.success_rate:.1f}%) below acceptable threshold',
                metric_value=result.success_rate,
                threshold=self.thresholds['success_rate_critical'],
                recommendation='Immediate investigation required - system may be overloaded'
            ))
            recommendations.append('Reduce load or scale infrastructure immediately')
        elif result.success_rate < self.thresholds['success_rate_warning']:
            insights.append(PerformanceInsight(
                category='reliability',
                severity='warning',
                title='Reduced Success Rate',
                description=f'Success rate ({result.success_rate:.1f}%) showing signs of stress',
                metric_value=result.success_rate,
                threshold=self.thresholds['success_rate_warning'],
                recommendation='Monitor system health and consider capacity planning'
            ))
        
        # Throughput analysis
        if result.requests_per_second < self.thresholds['rps_minimum']:
            insights.append(PerformanceInsight(
                category='performance',
                severity='warning',
                title='Low Throughput',
                description=f'Throughput ({result.requests_per_second:.1f} RPS) below expected levels',
                metric_value=result.requests_per_second,
                threshold=self.thresholds['rps_minimum'],
                recommendation='Investigate bottlenecks in request processing'
            ))
        
        # P95/P99 analysis
        p95_ratio = result.p95_response_time / result.avg_response_time if result.avg_response_time > 0 else 0
        if p95_ratio > 3.0:
            insights.append(PerformanceInsight(
                category='performance',
                severity='warning',
                title='High Response Time Variability',
                description=f'P95 response time is {p95_ratio:.1f}x higher than average',
                metric_value=p95_ratio,
                threshold=3.0,
                recommendation='Investigate outliers and ensure consistent performance'
            ))
        
        # Error rate analysis
        error_rate = (result.error_count / result.total_requests * 100) if result.total_requests > 0 else 0
        if error_rate > self.thresholds['error_rate_critical']:
            insights.append(PerformanceInsight(
                category='reliability',
                severity='critical',
                title='High Error Rate',
                description=f'Error rate ({error_rate:.1f}%) indicates system instability',
                metric_value=error_rate,
                threshold=self.thresholds['error_rate_critical'],
                recommendation='Investigate error patterns and system logs'
            ))
        
        # Generate summary metrics
        summary_metrics = {
            'test_type': 'load_test',
            'concurrent_users': result.concurrent_users,
            'total_requests': result.total_requests,
            'success_rate': result.success_rate,
            'avg_response_time': result.avg_response_time,
            'p95_response_time': result.p95_response_time,
            'p99_response_time': result.p99_response_time,
            'requests_per_second': result.requests_per_second,
            'error_rate': error_rate,
            'test_duration': result.duration
        }
        
        # Generate performance charts
        charts = self._generate_load_test_charts(result)
        
        return PerformanceReport(
            test_type='load_test',
            timestamp=datetime.now(timezone.utc).isoformat(),
            summary_metrics=summary_metrics,
            insights=insights,
            trends=[],  # Trends require historical data
            recommendations=recommendations,
            charts_generated=charts
        )
    
    def analyze_stress_test(self, result: StressTestResult) -> PerformanceReport:
        # Analyze stress test results
        insights = []
        recommendations = []
        
        # Scalability analysis
        if result.breaking_point:
            insights.append(PerformanceInsight(
                category='scalability',
                severity='critical',
                title='System Breaking Point Identified',
                description=f'System breaks at {result.breaking_point} concurrent users',
                metric_value=result.breaking_point,
                recommendation=f'Maximum safe load is approximately {int(result.breaking_point * 0.8)} users'
            ))
            recommendations.append(f'Consider scaling infrastructure before reaching {result.breaking_point} users')
        
        if result.degradation_point and result.breaking_point:
            safety_margin = result.breaking_point - result.degradation_point
            if safety_margin < result.degradation_point * 0.2:  # Less than 20% margin
                insights.append(PerformanceInsight(
                    category='scalability',
                    severity='warning',
                    title='Narrow Operating Window',
                    description=f'Only {safety_margin} user difference between degradation and failure',
                    metric_value=safety_margin,
                    recommendation='Consider proactive scaling before reaching degradation point'
                ))
        
        # Performance degradation analysis
        if result.degradation_point:
            insights.append(PerformanceInsight(
                category='performance',
                severity='warning',
                title='Performance Degradation Detected',
                description=f'Performance degrades at {result.degradation_point} users',
                metric_value=result.degradation_point,
                recommendation='Monitor performance closely when approaching this load level'
            ))
        
        # Recovery analysis
        if result.recovery_time:
            if result.recovery_time > 60:  # More than 1 minute
                insights.append(PerformanceInsight(
                    category='reliability',
                    severity='warning',
                    title='Slow System Recovery',
                    description=f'System took {result.recovery_time:.1f}s to recover after overload',
                    metric_value=result.recovery_time,
                    threshold=60.0,
                    recommendation='Investigate recovery mechanisms and consider auto-scaling'
                ))
            else:
                insights.append(PerformanceInsight(
                    category='reliability',
                    severity='info',
                    title='Good Recovery Performance',
                    description=f'System recovered quickly ({result.recovery_time:.1f}s) after overload',
                    metric_value=result.recovery_time
                ))
        
        # Peak performance analysis
        insights.append(PerformanceInsight(
            category='performance',
            severity='info',
            title='Peak Performance Metrics',
            description=f'Peak throughput: {result.peak_rps:.1f} RPS at {result.peak_success_rate:.1f}% success rate',
            metric_value=result.peak_rps
        ))
        
        summary_metrics = {
            'test_type': 'stress_test',
            'max_users_tested': result.max_users_tested,
            'breaking_point': result.breaking_point,
            'degradation_point': result.degradation_point,
            'peak_rps': result.peak_rps,
            'peak_success_rate': result.peak_success_rate,
            'recovery_time': result.recovery_time,
            'total_duration': result.total_duration
        }
        
        charts = self._generate_stress_test_charts(result)
        
        return PerformanceReport(
            test_type='stress_test',
            timestamp=datetime.now(timezone.utc).isoformat(),
            summary_metrics=summary_metrics,
            insights=insights,
            trends=[],
            recommendations=recommendations,
            charts_generated=charts
        )
    
    def analyze_benchmark_results(self, results: List[BenchmarkResult]) -> PerformanceReport:
        # Analyze benchmark test results
        insights = []
        recommendations = []
        
        for result in results:
            if result.regression_detected:
                severity = 'critical' if result.performance_change_percent > 50 else 'warning'
                insights.append(PerformanceInsight(
                    category='performance',
                    severity=severity,
                    title=f'Performance Regression: {result.endpoint}',
                    description=f'Performance degraded by {result.performance_change_percent:.1f}%',
                    metric_value=result.performance_change_percent,
                    threshold=result.regression_threshold * 100,
                    recommendation='Investigate recent changes that may have impacted performance'
                ))
                recommendations.append(f'Review {result.endpoint} endpoint optimization')
            elif result.performance_change_percent < -10:  # Significant improvement
                insights.append(PerformanceInsight(
                    category='performance',
                    severity='info',
                    title=f'Performance Improvement: {result.endpoint}',
                    description=f'Performance improved by {abs(result.performance_change_percent):.1f}%',
                    metric_value=result.performance_change_percent
                ))
        
        # Overall performance summary
        avg_change = statistics.mean([r.performance_change_percent for r in results])
        regressions = len([r for r in results if r.regression_detected])
        
        summary_metrics = {
            'test_type': 'benchmark',
            'endpoints_tested': len(results),
            'regressions_detected': regressions,
            'avg_performance_change': avg_change,
            'baseline_comparison': True
        }
        
        charts = self._generate_benchmark_charts(results)
        
        return PerformanceReport(
            test_type='benchmark',
            timestamp=datetime.now(timezone.utc).isoformat(),
            summary_metrics=summary_metrics,
            insights=insights,
            trends=[],
            recommendations=recommendations,
            charts_generated=charts
        )
    
    def analyze_trends(self, historical_data: List[Dict[str, Any]], 
                      metric_name: str, time_window_days: int = 30) -> PerformanceTrend:
        # Analyze performance trends over time
        if len(historical_data) < 2:
            return PerformanceTrend(
                metric_name=metric_name,
                time_period=f'{time_window_days}_days',
                trend_direction='insufficient_data',
                change_percentage=0.0,
                statistical_significance=0.0,
                data_points=len(historical_data)
            )
        
        # Extract metric values and timestamps
        values = []
        timestamps = []
        
        for data_point in historical_data:
            if metric_name in data_point:
                values.append(data_point[metric_name])
                timestamps.append(data_point.get('timestamp', datetime.now().isoformat()))
        
        if len(values) < 2:
            return PerformanceTrend(
                metric_name=metric_name,
                time_period=f'{time_window_days}_days',
                trend_direction='insufficient_data',
                change_percentage=0.0,
                statistical_significance=0.0,
                data_points=len(values)
            )
        
        # Calculate trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Calculate percentage change
        first_value = values[0]
        last_value = values[-1]
        change_percentage = ((last_value - first_value) / first_value * 100) if first_value != 0 else 0
        
        # Determine trend direction
        if abs(change_percentage) < 5:  # Less than 5% change
            trend_direction = 'stable'
        elif change_percentage > 0:
            # For response time, positive change is degrading
            if 'time' in metric_name.lower() or 'latency' in metric_name.lower():
                trend_direction = 'degrading'
            else:
                trend_direction = 'improving'
        else:
            if 'time' in metric_name.lower() or 'latency' in metric_name.lower():
                trend_direction = 'improving'
            else:
                trend_direction = 'degrading'
        
        # Calculate statistical significance (simplified)
        correlation = abs(np.corrcoef(x, values)[0, 1]) if len(values) > 2 else 0
        
        return PerformanceTrend(
            metric_name=metric_name,
            time_period=f'{time_window_days}_days',
            trend_direction=trend_direction,
            change_percentage=change_percentage,
            statistical_significance=correlation,
            data_points=len(values)
        )
    
    def _generate_load_test_charts(self, result: LoadTestResult) -> List[str]:
        # Generate charts for load test results
        charts = []
        
        try:
            # Response time distribution chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Average', 'Median', 'P95', 'P99', 'Max'],
                y=[result.avg_response_time, result.median_response_time, 
                   result.p95_response_time, result.p99_response_time, result.max_response_time],
                name='Response Times',
                marker_color=['blue', 'green', 'yellow', 'orange', 'red']
            ))
            fig.update_layout(
                title=f'Response Time Distribution - {result.test_name}',
                xaxis_title='Percentile',
                yaxis_title='Response Time (seconds)',
                showlegend=False
            )
            
            chart_file = self.output_dir / f'{result.test_name}_response_times.html'
            fig.write_html(chart_file)
            charts.append(str(chart_file))
            
            # Performance summary chart
            fig2 = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Success Rate', 'Throughput', 'Response Time', 'Load Distribution'),
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "bar"}, {"type": "pie"}]]
            )
            
            # Success rate gauge
            fig2.add_trace(go.Indicator(
                mode="gauge+number",
                value=result.success_rate,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Success Rate (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 90], 'color': "lightgray"},
                                {'range': [90, 95], 'color': "yellow"},
                                {'range': [95, 100], 'color': "green"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 95}}
            ), row=1, col=1)
            
            # Throughput gauge
            fig2.add_trace(go.Indicator(
                mode="gauge+number",
                value=result.requests_per_second,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "RPS"},
                gauge={'axis': {'range': [None, result.requests_per_second * 1.5]},
                       'bar': {'color': "green"}}
            ), row=1, col=2)
            
            chart_file2 = self.output_dir / f'{result.test_name}_summary.html'
            fig2.write_html(chart_file2)
            charts.append(str(chart_file2))
            
        except Exception as e:
            logger.error(f"Failed to generate load test charts: {e}")
        
        return charts
    
    def _generate_stress_test_charts(self, result: StressTestResult) -> List[str]:
        # Generate charts for stress test results
        charts = []
        
        try:
            if result.load_phases:
                # Load progression chart
                users = [phase['users'] for phase in result.load_phases]
                success_rates = [phase['success_rate'] for phase in result.load_phases]
                rps_values = [phase['rps'] for phase in result.load_phases]
                response_times = [phase['avg_response_time'] for phase in result.load_phases]
                
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Success Rate vs Load', 'Throughput vs Load', 'Response Time vs Load'),
                    shared_xaxes=True
                )
                
                # Success rate
                fig.add_trace(go.Scatter(
                    x=users, y=success_rates,
                    mode='lines+markers',
                    name='Success Rate (%)',
                    line=dict(color='green')
                ), row=1, col=1)
                
                # Throughput
                fig.add_trace(go.Scatter(
                    x=users, y=rps_values,
                    mode='lines+markers',
                    name='RPS',
                    line=dict(color='blue')
                ), row=2, col=1)
                
                # Response time
                fig.add_trace(go.Scatter(
                    x=users, y=response_times,
                    mode='lines+markers',
                    name='Avg Response Time (s)',
                    line=dict(color='red')
                ), row=3, col=1)
                
                # Add breaking point line if detected
                if result.breaking_point:
                    for row in range(1, 4):
                        fig.add_vline(
                            x=result.breaking_point,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Breaking Point",
                            row=row, col=1
                        )
                
                # Add degradation point line if detected
                if result.degradation_point:
                    for row in range(1, 4):
                        fig.add_vline(
                            x=result.degradation_point,
                            line_dash="dash",
                            line_color="orange",
                            annotation_text="Degradation Point",
                            row=row, col=1
                        )
                
                fig.update_layout(
                    title=f'Stress Test Results - {result.test_name}',
                    height=800,
                    showlegend=False
                )
                fig.update_xaxes(title_text="Concurrent Users", row=3, col=1)
                
                chart_file = self.output_dir / f'{result.test_name}_stress_analysis.html'
                fig.write_html(chart_file)
                charts.append(str(chart_file))
                
        except Exception as e:
            logger.error(f"Failed to generate stress test charts: {e}")
        
        return charts
    
    def _generate_benchmark_charts(self, results: List[BenchmarkResult]) -> List[str]:
        # Generate charts for benchmark results
        charts = []
        
        try:
            # Performance comparison chart
            endpoints = [r.endpoint for r in results]
            baseline_times = [r.baseline_response_time for r in results]
            current_times = [r.current_response_time for r in results]
            changes = [r.performance_change_percent for r in results]
            
            fig = go.Figure(data=[
                go.Bar(name='Baseline', x=endpoints, y=baseline_times),
                go.Bar(name='Current', x=endpoints, y=current_times)
            ])
            
            fig.update_layout(
                title='Benchmark Results: Baseline vs Current Performance',
                xaxis_title='Endpoint',
                yaxis_title='Response Time (seconds)',
                barmode='group'
            )
            
            chart_file = self.output_dir / 'benchmark_comparison.html'
            fig.write_html(chart_file)
            charts.append(str(chart_file))
            
            # Performance change chart
            colors = ['red' if change > 0 else 'green' for change in changes]
            
            fig2 = go.Figure(data=[
                go.Bar(x=endpoints, y=changes, marker_color=colors)
            ])
            
            fig2.update_layout(
                title='Performance Change vs Baseline (%)',
                xaxis_title='Endpoint',
                yaxis_title='Change (%)',
                showlegend=False
            )
            
            chart_file2 = self.output_dir / 'benchmark_changes.html'
            fig2.write_html(chart_file2)
            charts.append(str(chart_file2))
            
        except Exception as e:
            logger.error(f"Failed to generate benchmark charts: {e}")
        
        return charts
    
    def export_report(self, report: PerformanceReport, format: str = 'json') -> str:
        # Export performance report in specified format
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            filename = self.output_dir / f'{report.test_type}_report_{timestamp}.json'
            with open(filename, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
        
        elif format == 'html':
            filename = self.output_dir / f'{report.test_type}_report_{timestamp}.html'
            html_content = self._generate_html_report(report)
            with open(filename, 'w') as f:
                f.write(html_content)
        
        logger.info(f"Performance report exported to {filename}")
        return str(filename)
    
    def _generate_html_report(self, report: PerformanceReport) -> str:
        # Generate HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Report - {report.test_type}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; }}
                .insight {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
                .critical {{ border-color: #e74c3c; }}
                .warning {{ border-color: #f39c12; }}
                .info {{ border-color: #3498db; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Performance Report: {report.test_type.title()}</h1>
                <p>Generated: {report.timestamp}</p>
            </div>
            
            <div class="section">
                <h2>Summary Metrics</h2>
                {self._format_metrics_html(report.summary_metrics)}
            </div>
            
            <div class="section">
                <h2>Performance Insights</h2>
                {self._format_insights_html(report.insights)}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in report.recommendations)}
                </ul>
            </div>
        </body>
        </html>
        """
        return html
    
    def _format_metrics_html(self, metrics: Dict[str, Any]) -> str:
        # Format metrics for HTML display
        html = ""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
            else:
                formatted_value = str(value)
            
            html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {formatted_value}</div>'
        
        return html
    
    def _format_insights_html(self, insights: List[PerformanceInsight]) -> str:
        # Format insights for HTML display
        html = ""
        for insight in insights:
            css_class = f"insight {insight.severity}"
            html += f"""
            <div class="{css_class}">
                <h3>{insight.title}</h3>
                <p>{insight.description}</p>
                {f'<p><strong>Recommendation:</strong> {insight.recommendation}</p>' if insight.recommendation else ''}
            </div>
            """
        
        return html