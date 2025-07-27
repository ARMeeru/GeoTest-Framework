# Advanced dashboard and visualization generator
# Creates comprehensive HTML dashboards with charts and metrics

import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import base64
import statistics
from src.monitoring import MetricsCollector, TestMetrics, SystemMetrics
from src.alerting import AlertManager, Alert, AlertSeverity

class DashboardGenerator:
    # Generates comprehensive HTML dashboards with embedded charts
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
    
    def generate_dashboard(self, output_file: Path, time_window_hours: int = 24) -> str:
        # Generate complete dashboard HTML file
        try:
            # Collect data
            dashboard_data = self._collect_dashboard_data(time_window_hours)
            
            # Generate HTML
            html_content = self._generate_html_dashboard(dashboard_data, time_window_hours)
            
            # Write to file
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(output_file)
            
        except Exception as e:
            raise Exception(f"Failed to generate dashboard: {e}")
    
    def _collect_dashboard_data(self, hours_back: int) -> Dict[str, Any]:
        # Collect all data needed for dashboard
        cutoff_time = time.time() - (hours_back * 3600)
        
        # Test metrics
        with self.metrics_collector._lock:
            test_metrics = [m for m in self.metrics_collector.test_metrics if m.start_time >= cutoff_time]
            system_metrics = [m for m in self.metrics_collector.system_metrics if m.timestamp >= cutoff_time]
        
        # Alert data
        with self.alert_manager._lock:
            alerts = [a for a in self.alert_manager.alerts if a.timestamp >= cutoff_time]
        
        return {
            'test_metrics': test_metrics,
            'system_metrics': system_metrics,
            'alerts': alerts,
            'test_summary': self.metrics_collector.get_test_summary(hours_back * 60),
            'system_summary': self.metrics_collector.get_system_summary(hours_back * 60),
            'alert_summary': self.alert_manager.get_alert_summary(hours_back),
            'generation_time': time.time()
        }
    
    def _generate_html_dashboard(self, data: Dict[str, Any], time_window_hours: int) -> str:
        # Generate complete HTML dashboard
        
        # Generate individual sections
        header_html = self._generate_header()
        css_styles = self._generate_css()
        javascript = self._generate_javascript()
        summary_cards = self._generate_summary_cards(data)
        charts_html = self._generate_charts_section(data)
        alerts_table = self._generate_alerts_table(data['alerts'])
        test_results_table = self._generate_test_results_table(data['test_metrics'])
        system_metrics_section = self._generate_system_metrics_section(data['system_metrics'])
        
        # Combine everything
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GeoTest Framework Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js"></script>
    {css_styles}
</head>
<body>
    {header_html}
    
    <div class="container">
        <!-- Summary Cards -->
        <section class="summary-section">
            <h2>Overview (Last {time_window_hours} hours)</h2>
            {summary_cards}
        </section>
        
        <!-- Charts Section -->
        <section class="charts-section">
            <h2>Performance Charts</h2>
            {charts_html}
        </section>
        
        <!-- Alerts Section -->
        <section class="alerts-section">
            <h2>Recent Alerts</h2>
            {alerts_table}
        </section>
        
        <!-- Test Results Section -->
        <section class="test-results-section">
            <h2>Test Execution Results</h2>
            {test_results_table}
        </section>
        
        <!-- System Metrics Section -->
        <section class="system-metrics-section">
            <h2>System Performance</h2>
            {system_metrics_section}
        </section>
    </div>
    
    {javascript}
</body>
</html>
"""
        return html_template
    
    def _generate_header(self) -> str:
        # Generate dashboard header
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"""
    <header class="header">
        <div class="header-content">
            <h1>GeoTest Framework Dashboard</h1>
            <div class="header-info">
                <span class="refresh-time">Last Updated: {current_time}</span>
                <button onclick="window.location.reload()" class="refresh-btn">Refresh</button>
            </div>
        </div>
    </header>
"""
    
    def _generate_css(self) -> str:
        # Generate CSS styles
        return """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            font-size: 2rem;
            font-weight: 600;
        }
        
        .header-info {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .refresh-btn {
            background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .refresh-btn:hover {
            background: rgba(255,255,255,0.3);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem 1rem;
        }
        
        section {
            background: white;
            margin: 2rem 0;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        }
        
        section h2 {
            margin-bottom: 1.5rem;
            color: #2d3748;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.5rem;
        }
        
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .summary-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .summary-card.success {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }
        
        .summary-card.warning {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }
        
        .summary-card.error {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        }
        
        .summary-card.info {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            color: #2d3748;
        }
        
        .card-title {
            font-size: 0.9rem;
            opacity: 0.9;
            margin-bottom: 0.5rem;
        }
        
        .card-value {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .card-subtitle {
            font-size: 0.8rem;
            opacity: 0.8;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
        }
        
        .chart-container {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        
        .chart-title {
            font-weight: 600;
            margin-bottom: 1rem;
            color: #495057;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        
        th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }
        
        .status-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .status-passed {
            background: #d4edda;
            color: #155724;
        }
        
        .status-failed {
            background: #f8d7da;
            color: #721c24;
        }
        
        .status-error {
            background: #fff3cd;
            color: #856404;
        }
        
        .severity-low {
            background: #d1ecf1;
            color: #0c5460;
        }
        
        .severity-medium {
            background: #fff3cd;
            color: #856404;
        }
        
        .severity-high {
            background: #f8d7da;
            color: #721c24;
        }
        
        .severity-critical {
            background: #f5c6cb;
            color: #721c24;
            font-weight: bold;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }
        
        .metric-item {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 6px;
            border-left: 4px solid #007bff;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #6c757d;
            margin-bottom: 0.25rem;
        }
        
        .metric-value {
            font-size: 1.2rem;
            font-weight: 600;
            color: #495057;
        }
        
        .no-data {
            text-align: center;
            color: #6c757d;
            font-style: italic;
            padding: 2rem;
        }
        
        @media (max-width: 768px) {
            .header-content {
                flex-direction: column;
                gap: 1rem;
            }
            
            .charts-grid {
                grid-template-columns: 1fr;
            }
            
            .summary-cards {
                grid-template-columns: 1fr;
            }
            
            table {
                font-size: 0.9rem;
            }
        }
    </style>
"""
    
    def _generate_summary_cards(self, data: Dict[str, Any]) -> str:
        # Generate summary cards section
        test_summary = data['test_summary']
        alert_summary = data['alert_summary']
        system_summary = data['system_summary']
        
        return f"""
    <div class="summary-cards">
        <div class="summary-card success">
            <div class="card-title">Test Success Rate</div>
            <div class="card-value">{test_summary.get('pass_rate', 0):.1f}%</div>
            <div class="card-subtitle">{test_summary.get('passed_tests', 0)}/{test_summary.get('total_tests', 0)} passed</div>
        </div>
        
        <div class="summary-card info">
            <div class="card-title">Average Response Time</div>
            <div class="card-value">{test_summary.get('average_response_time', 0)*1000:.0f}ms</div>
            <div class="card-subtitle">API performance</div>
        </div>
        
        <div class="summary-card warning">
            <div class="card-title">Active Alerts</div>
            <div class="card-value">{alert_summary.get('active_alerts', 0)}</div>
            <div class="card-subtitle">{alert_summary.get('total_alerts', 0)} total alerts</div>
        </div>
        
        <div class="summary-card error">
            <div class="card-title">System CPU Usage</div>
            <div class="card-value">{system_summary.get('cpu_avg', 0):.1f}%</div>
            <div class="card-subtitle">Average usage</div>
        </div>
    </div>
"""
    
    def _generate_charts_section(self, data: Dict[str, Any]) -> str:
        # Generate charts section with Chart.js
        test_metrics = data['test_metrics']
        system_metrics = data['system_metrics']
        
        # Prepare data for charts
        test_timeline_data = self._prepare_test_timeline_data(test_metrics)
        response_time_data = self._prepare_response_time_data(test_metrics)
        system_timeline_data = self._prepare_system_timeline_data(system_metrics)
        
        return f"""
    <div class="charts-grid">
        <div class="chart-container">
            <div class="chart-title">Test Execution Timeline</div>
            <canvas id="testTimelineChart" width="400" height="200"></canvas>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">API Response Times</div>
            <canvas id="responseTimeChart" width="400" height="200"></canvas>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">System CPU & Memory</div>
            <canvas id="systemMetricsChart" width="400" height="200"></canvas>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Test Results Distribution</div>
            <canvas id="testDistributionChart" width="400" height="200"></canvas>
        </div>
    </div>
    
    <script>
        const testTimelineData = {json.dumps(test_timeline_data)};
        const responseTimeData = {json.dumps(response_time_data)};
        const systemTimelineData = {json.dumps(system_timeline_data)};
    </script>
"""
    
    def _prepare_test_timeline_data(self, test_metrics: List[TestMetrics]) -> Dict[str, Any]:
        # Prepare test timeline data for Chart.js
        if not test_metrics:
            return {'labels': [], 'datasets': []}
        
        # Group tests by hour
        hourly_data = {}
        for metric in test_metrics:
            hour = datetime.fromtimestamp(metric.start_time).strftime('%H:00')
            if hour not in hourly_data:
                hourly_data[hour] = {'passed': 0, 'failed': 0, 'error': 0}
            
            if metric.status:
                hourly_data[hour][metric.status] = hourly_data[hour].get(metric.status, 0) + 1
        
        hours = sorted(hourly_data.keys())
        passed_counts = [hourly_data[h]['passed'] for h in hours]
        failed_counts = [hourly_data[h]['failed'] for h in hours]
        error_counts = [hourly_data[h]['error'] for h in hours]
        
        return {
            'labels': hours,
            'datasets': [
                {
                    'label': 'Passed',
                    'data': passed_counts,
                    'backgroundColor': 'rgba(34, 197, 94, 0.7)',
                    'borderColor': 'rgba(34, 197, 94, 1)',
                    'borderWidth': 2
                },
                {
                    'label': 'Failed',
                    'data': failed_counts,
                    'backgroundColor': 'rgba(239, 68, 68, 0.7)',
                    'borderColor': 'rgba(239, 68, 68, 1)',
                    'borderWidth': 2
                },
                {
                    'label': 'Error',
                    'data': error_counts,
                    'backgroundColor': 'rgba(245, 158, 11, 0.7)',
                    'borderColor': 'rgba(245, 158, 11, 1)',
                    'borderWidth': 2
                }
            ]
        }
    
    def _prepare_response_time_data(self, test_metrics: List[TestMetrics]) -> Dict[str, Any]:
        # Prepare response time data for Chart.js
        if not test_metrics:
            return {'labels': [], 'datasets': []}
        
        # Get response times over time
        timestamps = []
        response_times = []
        
        for metric in test_metrics:
            if metric.response_time is not None:
                timestamps.append(datetime.fromtimestamp(metric.start_time).strftime('%H:%M'))
                response_times.append(metric.response_time * 1000)  # Convert to ms
        
        return {
            'labels': timestamps[-50:],  # Last 50 data points
            'datasets': [{
                'label': 'Response Time (ms)',
                'data': response_times[-50:],
                'backgroundColor': 'rgba(59, 130, 246, 0.1)',
                'borderColor': 'rgba(59, 130, 246, 1)',
                'borderWidth': 2,
                'fill': True,
                'tension': 0.4
            }]
        }
    
    def _prepare_system_timeline_data(self, system_metrics: List[SystemMetrics]) -> Dict[str, Any]:
        # Prepare system metrics data for Chart.js
        if not system_metrics:
            return {'labels': [], 'datasets': []}
        
        timestamps = []
        cpu_data = []
        memory_data = []
        
        for metric in system_metrics:
            timestamps.append(datetime.fromtimestamp(metric.timestamp).strftime('%H:%M'))
            cpu_data.append(metric.cpu_percent)
            memory_data.append(metric.memory_percent)
        
        return {
            'labels': timestamps[-100:],  # Last 100 data points
            'datasets': [
                {
                    'label': 'CPU %',
                    'data': cpu_data[-100:],
                    'borderColor': 'rgba(239, 68, 68, 1)',
                    'backgroundColor': 'rgba(239, 68, 68, 0.1)',
                    'borderWidth': 2,
                    'fill': False,
                    'yAxisID': 'y'
                },
                {
                    'label': 'Memory %',
                    'data': memory_data[-100:],
                    'borderColor': 'rgba(34, 197, 94, 1)',
                    'backgroundColor': 'rgba(34, 197, 94, 0.1)',
                    'borderWidth': 2,
                    'fill': False,
                    'yAxisID': 'y'
                }
            ]
        }
    
    def _generate_alerts_table(self, alerts: List[Alert]) -> str:
        # Generate alerts table
        if not alerts:
            return '<div class="no-data">No alerts in the selected time period</div>'
        
        # Sort by timestamp (newest first)
        sorted_alerts = sorted(alerts, key=lambda x: x.timestamp, reverse=True)
        
        rows = []
        for alert in sorted_alerts[:20]:  # Show last 20 alerts
            timestamp = datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')
            status_class = "resolved" if alert.resolved else "active"
            severity_class = f"severity-{alert.severity.value}"
            
            rows.append(f"""
                <tr>
                    <td>{timestamp}</td>
                    <td><span class="status-badge {severity_class}">{alert.severity.value}</span></td>
                    <td>{alert.alert_type.value.replace('_', ' ').title()}</td>
                    <td>{alert.title}</td>
                    <td>{alert.source}</td>
                    <td><span class="status-badge status-{status_class}">{'Resolved' if alert.resolved else 'Active'}</span></td>
                </tr>
            """)
        
        return f"""
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Severity</th>
                        <th>Type</th>
                        <th>Title</th>
                        <th>Source</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """
    
    def _generate_test_results_table(self, test_metrics: List[TestMetrics]) -> str:
        # Generate test results table
        if not test_metrics:
            return '<div class="no-data">No test results in the selected time period</div>'
        
        # Sort by start time (newest first)
        sorted_tests = sorted(test_metrics, key=lambda x: x.start_time, reverse=True)
        
        rows = []
        for test in sorted_tests[:50]:  # Show last 50 tests
            timestamp = datetime.fromtimestamp(test.start_time).strftime('%Y-%m-%d %H:%M:%S')
            duration = f"{test.duration:.3f}s" if test.duration else "N/A"
            response_time = f"{test.response_time*1000:.0f}ms" if test.response_time else "N/A"
            status_class = f"status-{test.status}" if test.status else "status-unknown"
            
            rows.append(f"""
                <tr>
                    <td>{timestamp}</td>
                    <td>{test.test_name}</td>
                    <td>{test.api_endpoint or 'N/A'}</td>
                    <td><span class="status-badge {status_class}">{test.status or 'Unknown'}</span></td>
                    <td>{duration}</td>
                    <td>{response_time}</td>
                </tr>
            """)
        
        return f"""
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Test Name</th>
                        <th>Endpoint</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Response Time</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """
    
    def _generate_system_metrics_section(self, system_metrics: List[SystemMetrics]) -> str:
        # Generate system metrics section
        if not system_metrics:
            return '<div class="no-data">No system metrics available</div>'
        
        latest = system_metrics[-1] if system_metrics else None
        if not latest:
            return '<div class="no-data">No system metrics available</div>'
        
        return f"""
            <div class="metric-grid">
                <div class="metric-item">
                    <div class="metric-label">Current CPU Usage</div>
                    <div class="metric-value">{latest.cpu_percent:.1f}%</div>
                </div>
                
                <div class="metric-item">
                    <div class="metric-label">Current Memory Usage</div>
                    <div class="metric-value">{latest.memory_percent:.1f}%</div>
                </div>
                
                <div class="metric-item">
                    <div class="metric-label">Available Memory</div>
                    <div class="metric-value">{latest.memory_available_mb:.0f} MB</div>
                </div>
                
                <div class="metric-item">
                    <div class="metric-label">Disk Usage</div>
                    <div class="metric-value">{latest.disk_usage_percent:.1f}%</div>
                </div>
                
                <div class="metric-item">
                    <div class="metric-label">Network Connections</div>
                    <div class="metric-value">{latest.active_connections}</div>
                </div>
                
                <div class="metric-item">
                    <div class="metric-label">Load Average</div>
                    <div class="metric-value">{f"{latest.load_average:.2f}" if latest.load_average is not None else "N/A"}</div>
                </div>
            </div>
        """
    
    def _generate_javascript(self) -> str:
        # Generate JavaScript for charts
        return """
    <script>
        // Initialize charts when page loads
        document.addEventListener('DOMContentLoaded', function() {
            initializeCharts();
        });
        
        function initializeCharts() {
            // Test Timeline Chart
            const testTimelineCtx = document.getElementById('testTimelineChart').getContext('2d');
            new Chart(testTimelineCtx, {
                type: 'bar',
                data: testTimelineData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            stacked: true
                        },
                        y: {
                            stacked: true,
                            beginAtZero: true
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'top'
                        }
                    }
                }
            });
            
            // Response Time Chart
            const responseTimeCtx = document.getElementById('responseTimeChart').getContext('2d');
            new Chart(responseTimeCtx, {
                type: 'line',
                data: responseTimeData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Response Time (ms)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'top'
                        }
                    }
                }
            });
            
            // System Metrics Chart
            const systemMetricsCtx = document.getElementById('systemMetricsChart').getContext('2d');
            new Chart(systemMetricsCtx, {
                type: 'line',
                data: systemTimelineData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Percentage (%)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'top'
                        }
                    }
                }
            });
            
            // Test Distribution Chart
            const testDistributionCtx = document.getElementById('testDistributionChart').getContext('2d');
            const testCounts = calculateTestDistribution();
            new Chart(testDistributionCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Passed', 'Failed', 'Error'],
                    datasets: [{
                        data: [testCounts.passed, testCounts.failed, testCounts.error],
                        backgroundColor: [
                            'rgba(34, 197, 94, 0.8)',
                            'rgba(239, 68, 68, 0.8)',
                            'rgba(245, 158, 11, 0.8)'
                        ],
                        borderColor: [
                            'rgba(34, 197, 94, 1)',
                            'rgba(239, 68, 68, 1)',
                            'rgba(245, 158, 11, 1)'
                        ],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'right'
                        }
                    }
                }
            });
        }
        
        function calculateTestDistribution() {
            // Calculate test distribution from timeline data
            const datasets = testTimelineData.datasets;
            let passed = 0, failed = 0, error = 0;
            
            if (datasets.length >= 3) {
                passed = datasets[0].data.reduce((a, b) => a + b, 0);
                failed = datasets[1].data.reduce((a, b) => a + b, 0);
                error = datasets[2].data.reduce((a, b) => a + b, 0);
            }
            
            return { passed, failed, error };
        }
    </script>
"""