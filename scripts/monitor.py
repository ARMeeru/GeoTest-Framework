#!/usr/bin/env python3
# GeoTest Framework Monitoring CLI Tool
# Provides command-line interface for monitoring operations

import click
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring import get_metrics_collector, SystemMonitor
from src.alerting import get_alert_manager
from src.dashboard_generator import DashboardGenerator
from src.performance_testing import PerformanceTestRunner
from src.performance_analyzer import PerformanceAnalyzer
from src.load_generator import (
    AsyncLoadGenerator, ThreadBasedLoadGenerator, 
    LoadConfig, LoadPattern, STANDARD_BEHAVIORS
)


@click.group()
def cli():
    # GeoTest Framework Monitoring CLI Tool
    pass


@cli.command()
@click.option('--hours', default=24, help='Number of hours to include in dashboard')
@click.option('--output', default='reports/dashboard.html', help='Output file path')
def dashboard(hours, output):
    # Generate monitoring dashboard
    try:
        output_path = Path(output)
        output_path.parent.mkdir(exist_ok=True)
        
        metrics_collector = get_metrics_collector()
        alert_manager = get_alert_manager()
        
        generator = DashboardGenerator(metrics_collector, alert_manager)
        dashboard_file = generator.generate_dashboard(output_path, hours)
        
        click.echo(f"Dashboard generated: {dashboard_file}")
        click.echo(f"Time window: {hours} hours")
        
    except Exception as e:
        click.echo(f"Failed to generate dashboard: {e}")
        sys.exit(1)


@cli.command()
@click.option('--output-dir', default='reports', help='Output directory for metrics')
@click.option('--format', type=click.Choice(['json', 'csv']), default='json', help='Output format')
def export(output_dir, format):
    # Export metrics data
    try:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        metrics_collector = get_metrics_collector()
        files_created = metrics_collector.export_metrics(output_path)
        
        click.echo(f"Metrics exported to {output_dir}")
        click.echo(f"Files created: {len(files_created)}")
        
        for name, path in files_created.items():
            click.echo(f"   - {name}: {path}")
            
    except Exception as e:
        click.echo(f"Failed to export metrics: {e}")
        sys.exit(1)


@cli.command()
@click.option('--hours', default=1, help='Number of hours to include in summary')
def summary(hours):
    # Show metrics summary
    try:
        metrics_collector = get_metrics_collector()
        alert_manager = get_alert_manager()
        
        # Get summaries
        test_summary = metrics_collector.get_test_summary(hours * 60)
        system_summary = metrics_collector.get_system_summary(hours * 60)
        alert_summary = alert_manager.get_alert_summary(hours)
        
        click.echo(f"\nGeoTest Framework Summary (Last {hours} hours)")
        click.echo("=" * 50)
        
        # Test Summary
        click.echo(f"\nTest Execution:")
        click.echo(f"   Total Tests: {test_summary.get('total_tests', 0)}")
        click.echo(f"   Passed: {test_summary.get('passed_tests', 0)}")
        click.echo(f"   Failed: {test_summary.get('failed_tests', 0)}")
        click.echo(f"   Pass Rate: {test_summary.get('pass_rate', 0):.1f}%")
        click.echo(f"   Avg Duration: {test_summary.get('average_duration', 0):.3f}s")
        click.echo(f"   Avg Response Time: {test_summary.get('average_response_time', 0)*1000:.0f}ms")
        
        # System Summary
        click.echo(f"\nSystem Performance:")
        if 'message' not in system_summary:
            click.echo(f"   CPU Average: {system_summary.get('cpu_avg', 0):.1f}%")
            click.echo(f"   CPU Peak: {system_summary.get('cpu_max', 0):.1f}%")
            click.echo(f"   Memory Average: {system_summary.get('memory_avg', 0):.1f}%")
            click.echo(f"   Memory Peak: {system_summary.get('memory_max', 0):.1f}%")
            click.echo(f"   Samples: {system_summary.get('samples_count', 0)}")
        else:
            click.echo(f"   {system_summary['message']}")
        
        # Alert Summary
        click.echo(f"\nAlerts:")
        click.echo(f"   Total Alerts: {alert_summary.get('total_alerts', 0)}")
        click.echo(f"   Active Alerts: {alert_summary.get('active_alerts', 0)}")
        click.echo(f"   Resolved Alerts: {alert_summary.get('resolved_alerts', 0)}")
        click.echo(f"   Alert Rate: {alert_summary.get('alert_rate_per_hour', 0):.1f}/hour")
        
        # Alert Breakdown
        severity_breakdown = alert_summary.get('severity_breakdown', {})
        if any(severity_breakdown.values()):
            click.echo(f"\n   Severity Breakdown:")
            for severity, count in severity_breakdown.items():
                if count > 0:
                    click.echo(f"     {severity.title()}: {count}")
        
        click.echo()
        
    except Exception as e:
        click.echo(f"Failed to get summary: {e}")
        sys.exit(1)


@cli.command()
@click.option('--severity', type=click.Choice(['low', 'medium', 'high', 'critical']), help='Filter by severity')
@click.option('--limit', default=10, help='Maximum number of alerts to show')
def alerts(severity, limit):
    # Show recent alerts
    try:
        alert_manager = get_alert_manager()
        
        # Get alerts
        from src.alerting import AlertSeverity
        severity_filter = AlertSeverity(severity) if severity else None
        
        active_alerts = alert_manager.get_active_alerts(severity_filter)
        
        click.echo(f"\nRecent Alerts (Active: {len(active_alerts)})")
        click.echo("=" * 50)
        
        if not active_alerts:
            click.echo("No active alerts")
            return
        
        for alert in active_alerts[:limit]:
            timestamp = datetime.fromtimestamp(alert.timestamp, timezone.utc)
            click.echo(f"\n[{alert.severity.value.upper()}] {alert.title}")
            click.echo(f"   ID: {alert.alert_id}")
            click.echo(f"   Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            click.echo(f"   Source: {alert.source}")
            click.echo(f"   Message: {alert.message[:100]}...")
            
            if alert.tags:
                tags_str = ", ".join([f"{k}={v}" for k, v in alert.tags.items()])
                click.echo(f"   Tags: {tags_str}")
        
        if len(active_alerts) > limit:
            click.echo(f"\n... and {len(active_alerts) - limit} more alerts")
        
        click.echo()
        
    except Exception as e:
        click.echo(f"Failed to get alerts: {e}")
        sys.exit(1)


@cli.command()
@click.argument('alert_id')
@click.option('--resolved-by', default='cli-user', help='Who resolved the alert')
def resolve(alert_id, resolved_by):
    # Resolve an alert by ID
    try:
        alert_manager = get_alert_manager()
        
        success = alert_manager.resolve_alert(alert_id, resolved_by)
        
        if success:
            click.echo(f"Alert {alert_id} resolved by {resolved_by}")
        else:
            click.echo(f"Alert {alert_id} not found or already resolved")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Failed to resolve alert: {e}")
        sys.exit(1)


# Phase 5: Performance Testing Commands

@cli.group()
def performance():
    # Performance testing commands
    pass


@performance.command()
@click.option('--endpoint', default='all', help='API endpoint to test')
@click.option('--users', default=20, help='Number of concurrent users')
@click.option('--requests', default=5, help='Requests per user')
@click.option('--duration', default=60, help='Test duration in seconds')
@click.option('--output', default='performance_analysis/load_test', help='Output directory')
def load_test(endpoint, users, requests, duration, output):
    # Execute load test with specified parameters
    click.echo(f"Starting load test: {users} users, {requests} requests each")
    click.echo(f"Endpoint: {endpoint}, Duration: {duration}s")
    
    try:
        # Setup
        runner = PerformanceTestRunner(max_workers=users + 10)
        analyzer = PerformanceAnalyzer(Path(output))
        
        # Execute load test
        with click.progressbar(length=duration, label='Running load test') as bar:
            result = runner.load_test(
                endpoint=endpoint,
                concurrent_users=users,
                requests_per_user=requests,
                test_name=f"cli_load_test_{endpoint.replace('/', '_')}"
            )
            bar.update(duration)
        
        # Analyze results
        report = analyzer.analyze_load_test(result)
        
        # Display results
        click.echo(f"\nLoad Test Results:")
        click.echo(f"   Total Requests: {result.total_requests}")
        click.echo(f"   Success Rate: {result.success_rate:.1f}%")
        click.echo(f"   Avg Response Time: {result.avg_response_time:.3f}s")
        click.echo(f"   P95 Response Time: {result.p95_response_time:.3f}s")
        click.echo(f"   Requests/Second: {result.requests_per_second:.1f}")
        
        # Export report
        report_file = analyzer.export_report(report, format='html')
        click.echo(f"   Report: {report_file}")
        
    except Exception as e:
        click.echo(f"Load test failed: {e}")
        sys.exit(1)


@performance.command()
@click.option('--endpoint', default='all', help='API endpoint to test')
@click.option('--min-users', default=5, help='Starting number of users')
@click.option('--max-users', default=100, help='Maximum number of users')
@click.option('--step-size', default=10, help='User increment per step')
@click.option('--step-duration', default=30, help='Duration of each step in seconds')
@click.option('--output', default='performance_analysis/stress_test', help='Output directory')
def stress_test(endpoint, min_users, max_users, step_size, step_duration, output):
    # Execute stress test to find breaking point
    click.echo(f"Starting stress test: {min_users} to {max_users} users")
    click.echo(f"Endpoint: {endpoint}, Step size: {step_size}, Step duration: {step_duration}s")
    
    try:
        # Setup
        runner = PerformanceTestRunner(max_workers=max_users + 20)
        analyzer = PerformanceAnalyzer(Path(output))
        
        # Execute stress test
        estimated_duration = ((max_users - min_users) // step_size + 1) * step_duration
        with click.progressbar(length=estimated_duration, label='Running stress test') as bar:
            result = runner.stress_test(
                endpoint=endpoint,
                min_users=min_users,
                max_users=max_users,
                step_size=step_size,
                step_duration=step_duration,
                test_name=f"cli_stress_test_{endpoint.replace('/', '_')}"
            )
            bar.update(estimated_duration)
        
        # Analyze results
        report = analyzer.analyze_stress_test(result)
        
        # Display results
        click.echo(f"\nStress Test Results:")
        click.echo(f"   Max Users Tested: {result.max_users_tested}")
        click.echo(f"   Peak RPS: {result.peak_rps:.1f}")
        click.echo(f"   Peak Success Rate: {result.peak_success_rate:.1f}%")
        
        if result.degradation_point:
            click.echo(f"   Degradation Point: {result.degradation_point} users")
        
        if result.breaking_point:
            click.echo(f"   Breaking Point: {result.breaking_point} users")
        
        if result.recovery_time:
            click.echo(f"   Recovery Time: {result.recovery_time:.1f}s")
        
        # Export report
        report_file = analyzer.export_report(report, format='html')
        click.echo(f"   Report: {report_file}")
        
    except Exception as e:
        click.echo(f"Stress test failed: {e}")
        sys.exit(1)


@performance.command()
@click.option('--endpoint', default='all', help='API endpoint to benchmark')
@click.option('--iterations', default=20, help='Number of iterations')
@click.option('--baseline', default='performance_baseline/api_benchmarks.json', help='Baseline file')
@click.option('--save-baseline', is_flag=True, help='Save results as new baseline')
@click.option('--output', default='performance_analysis/benchmark', help='Output directory')
def benchmark(endpoint, iterations, baseline, save_baseline, output):
    # Execute benchmark test against baseline
    click.echo(f"Starting benchmark: {endpoint} with {iterations} iterations")
    
    try:
        # Setup
        runner = PerformanceTestRunner()
        analyzer = PerformanceAnalyzer(Path(output))
        baseline_file = Path(baseline) if not save_baseline else None
        
        # Execute benchmark
        with click.progressbar(length=iterations, label='Running benchmark') as bar:
            result = runner.benchmark_test(
                endpoint=endpoint,
                baseline_file=baseline_file,
                iterations=iterations,
                test_name=f"cli_benchmark_{endpoint.replace('/', '_')}"
            )
            bar.update(iterations)
        
        # Display results
        click.echo(f"\nBenchmark Results:")
        click.echo(f"   Endpoint: {endpoint}")
        click.echo(f"   Iterations: {result.iterations}")
        click.echo(f"   Current Response Time: {result.current_response_time:.3f}s")
        
        if result.baseline_response_time > 0:
            click.echo(f"   Baseline Response Time: {result.baseline_response_time:.3f}s")
            click.echo(f"   Performance Change: {result.performance_change_percent:+.1f}%")
            
            if result.regression_detected:
                click.echo(f"   ⚠️  REGRESSION DETECTED!")
            else:
                click.echo(f"   ✅ No regression detected")
        
        # Save baseline if requested
        if save_baseline:
            baseline_path = Path(baseline)
            runner.save_baseline([result], baseline_path)
            click.echo(f"   Baseline saved: {baseline_path}")
        
        # Analyze and export
        report = analyzer.analyze_benchmark_results([result])
        report_file = analyzer.export_report(report, format='json')
        click.echo(f"   Report: {report_file}")
        
    except Exception as e:
        click.echo(f"Benchmark failed: {e}")
        sys.exit(1)


@performance.command()
@click.option('--pattern', type=click.Choice(['constant', 'ramp_up', 'spike', 'step']), 
              default='constant', help='Load pattern type')
@click.option('--users', default=30, help='Number of concurrent users')
@click.option('--duration', default=120, help='Test duration in seconds')
@click.option('--behavior', type=click.Choice(['api_explorer', 'power_user', 'casual_user']), 
              default='api_explorer', help='User behavior pattern')
@click.option('--output', default='performance_analysis/async_load', help='Output directory')
def async_load(pattern, users, duration, behavior, output):
    # Execute async load test with specified pattern
    import asyncio
    
    click.echo(f"Starting async load test: {pattern} pattern with {users} users")
    click.echo(f"Duration: {duration}s, Behavior: {behavior}")
    
    try:
        # Setup load configuration
        config = LoadConfig(
            pattern=LoadPattern(pattern),
            duration_seconds=duration,
            min_users=5 if pattern == 'ramp_up' else users,
            max_users=users,
            ramp_duration=duration // 3 if pattern == 'ramp_up' else None,
            step_size=10 if pattern == 'step' else None,
            step_duration=30 if pattern == 'step' else None
        )
        
        # Get user behavior
        user_behavior = STANDARD_BEHAVIORS[behavior]
        
        # Execute async load test
        async def run_async_test():
            generator = AsyncLoadGenerator("https://restcountries.com/v3.1")
            return await generator.generate_load_pattern(config, user_behavior)
        
        click.echo("Executing async load test...")
        result = asyncio.run(run_async_test())
        
        # Display results
        click.echo(f"\nAsync Load Test Results:")
        click.echo(f"   Total Requests: {result.total_requests}")
        click.echo(f"   Successful: {result.successful_requests}")
        click.echo(f"   Failed: {result.failed_requests}")
        click.echo(f"   Success Rate: {result.successful_requests/result.total_requests*100:.1f}%")
        click.echo(f"   Avg Response Time: {result.avg_response_time:.3f}s")
        click.echo(f"   Requests/Second: {result.requests_per_second:.1f}")
        click.echo(f"   Peak Users: {result.peak_concurrent_users}")
        
        # Export results
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = output_dir / f"async_load_{pattern}_{behavior}.json"
        with open(result_file, 'w') as f:
            import dataclasses
            json.dump(dataclasses.asdict(result), f, indent=2, default=str)
        
        click.echo(f"   Results: {result_file}")
        
    except Exception as e:
        click.echo(f"Async load test failed: {e}")
        sys.exit(1)


@performance.command()
@click.option('--output', default='performance_analysis', help='Output directory')
def analyze(output):
    # Analyze recent performance test results
    click.echo("Analyzing recent performance test results...")
    
    try:
        output_dir = Path(output)
        
        if not output_dir.exists():
            click.echo(f"No performance analysis directory found: {output_dir}")
            return
        
        # Find recent result files
        load_files = list(output_dir.glob("**/load_test_results.json"))
        stress_files = list(output_dir.glob("**/stress_test_results.json"))
        benchmark_files = list(output_dir.glob("**/benchmark_results.json"))
        
        click.echo(f"\nFound Performance Results:")
        click.echo(f"   Load Tests: {len(load_files)}")
        click.echo(f"   Stress Tests: {len(stress_files)}")
        click.echo(f"   Benchmarks: {len(benchmark_files)}")
        
        if not any([load_files, stress_files, benchmark_files]):
            click.echo("No performance test results found to analyze")
            return
        
        # Load and summarize results
        for file_path in load_files[:3]:  # Show last 3 of each type
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            if results:
                result = results[-1]  # Latest result
                click.echo(f"\nLoad Test: {result.get('test_name', 'Unknown')}")
                click.echo(f"   Users: {result.get('concurrent_users', 0)}")
                click.echo(f"   Success Rate: {result.get('success_rate', 0):.1f}%")
                click.echo(f"   Avg Response: {result.get('avg_response_time', 0):.3f}s")
        
        for file_path in stress_files[:3]:
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            if results:
                result = results[-1]
                click.echo(f"\nStress Test: {result.get('test_name', 'Unknown')}")
                click.echo(f"   Max Users: {result.get('max_users_tested', 0)}")
                click.echo(f"   Breaking Point: {result.get('breaking_point', 'Not detected')}")
                click.echo(f"   Peak RPS: {result.get('peak_rps', 0):.1f}")
        
        for file_path in benchmark_files[:3]:
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            if results:
                result = results[-1]
                click.echo(f"\nBenchmark: {result.get('endpoint', 'Unknown')}")
                click.echo(f"   Response Time: {result.get('current_response_time', 0):.3f}s")
                change = result.get('performance_change_percent', 0)
                regression = result.get('regression_detected', False)
                status = "⚠️ REGRESSION" if regression else "✅ OK"
                click.echo(f"   Change: {change:+.1f}% {status}")
        
    except Exception as e:
        click.echo(f"Analysis failed: {e}")
        sys.exit(1)


@cli.command()
def test_notifications():
    # Test notification channels
    try:
        alert_manager = get_alert_manager()
        
        click.echo("Testing notification channels...")
        
        results = alert_manager.test_notifications()
        
        for channel_name, success in results.items():
            status = "PASS" if success else "FAIL"
            click.echo(f"   {channel_name}: {status}")
        
        if all(results.values()):
            click.echo("\nAll notification channels working")
        else:
            click.echo("\nSome notification channels failed")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Failed to test notifications: {e}")
        sys.exit(1)


@cli.command()
@click.option('--interval', default=5, help='Monitoring interval in seconds')
@click.option('--duration', default=60, help='How long to monitor in seconds')
def monitor(interval, duration):
    # Start system monitoring for specified duration
    try:
        metrics_collector = get_metrics_collector()
        system_monitor = SystemMonitor(metrics_collector, interval_seconds=interval)
        
        click.echo(f"Starting system monitoring...")
        click.echo(f"   Interval: {interval} seconds")
        click.echo(f"   Duration: {duration} seconds")
        click.echo("   Press Ctrl+C to stop early")
        
        system_monitor.start()
        
        import time
        try:
            time.sleep(duration)
        except KeyboardInterrupt:
            click.echo("\nMonitoring stopped by user")
        
        system_monitor.stop()
        
        # Show final summary
        recent_metrics = metrics_collector.get_system_summary(duration / 60)
        if 'message' not in recent_metrics:
            click.echo(f"\nMonitoring Summary:")
            click.echo(f"   CPU Average: {recent_metrics.get('cpu_avg', 0):.1f}%")
            click.echo(f"   Memory Average: {recent_metrics.get('memory_avg', 0):.1f}%")
            click.echo(f"   Samples Collected: {recent_metrics.get('samples_count', 0)}")
        
        click.echo("Monitoring completed")
        
    except Exception as e:
        click.echo(f"Monitoring failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--config-file', default='config/alerting.json', help='Alert configuration file')
def check_config(config_file):
    # Validate alert configuration
    try:
        config_path = Path(config_file)
        
        if not config_path.exists():
            click.echo(f"Configuration file not found: {config_file}")
            sys.exit(1)
        
        with open(config_path) as f:
            config = json.load(f)
        
        click.echo(f"Checking configuration: {config_file}")
        
        # Check alert rules
        rules = config.get('alert_rules', [])
        click.echo(f"\nAlert Rules: {len(rules)}")
        
        for rule in rules:
            status = "Enabled" if rule.get('enabled', True) else "Disabled"
            click.echo(f"   {rule.get('rule_name', 'Unknown')}: {status}")
        
        # Check notification channels
        channels = config.get('notification_channels', [])
        click.echo(f"\nNotification Channels: {len(channels)}")
        
        for channel in channels:
            status = "Enabled" if channel.get('config', {}).get('enabled', True) else "Disabled"
            click.echo(f"   {channel.get('name', 'Unknown')} ({channel.get('type', 'unknown')}): {status}")
        
        click.echo("\nConfiguration is valid")
        
    except json.JSONDecodeError as e:
        click.echo(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Failed to check configuration: {e}")
        sys.exit(1)


# Phase 5: Bug Tracking Commands

@cli.group()
def bug_tracking():
    # Bug tracking and failure analysis commands
    pass


@bug_tracking.command()
@click.option('--hours', default=24, help='Time window in hours for failure analysis')
def analyze_failures(hours):
    # Analyze recent test failures
    click.echo(f"Analyzing failures from the last {hours} hours...")
    
    try:
        from src.github_integration import get_bug_tracker
        from src.failure_analyzer import get_failure_analyzer
        from src.failure_reports import get_failure_report_generator
        
        bug_tracker = get_bug_tracker()
        failure_analyzer = get_failure_analyzer()
        report_generator = get_failure_report_generator()
        
        # Get failure summary
        summary = bug_tracker.get_failure_summary(hours)
        
        click.echo(f"\nFailure Summary:")
        click.echo(f"   Total Failures: {summary['total_failures']}")
        click.echo(f"   Unique Tests: {summary['unique_tests']}")
        click.echo(f"   Consecutive Failures: {summary['consecutive_failures']}")
        
        # Show failure types
        if summary['failure_types']:
            click.echo(f"\n   Failure Types:")
            for failure_type, count in summary['failure_types'].items():
                click.echo(f"     {failure_type}: {count}")
        
        # Show most common failures
        if summary['most_common_failures']:
            click.echo(f"\n   Most Common Failures:")
            for test_name, count in summary['most_common_failures'][:5]:
                click.echo(f"     {test_name}: {count} failures")
        
        # Generate comprehensive report
        click.echo(f"\nGenerating comprehensive analysis report...")
        report = report_generator.generate_comprehensive_report(hours)
        
        # Generate HTML report
        html_file = report_generator.generate_html_report(report)
        if html_file:
            click.echo(f"HTML Report: {html_file}")
        
        # Show flaky tests
        flaky_report = failure_analyzer.get_flaky_tests_report()
        if flaky_report["total_flaky_tests"] > 0:
            click.echo(f"\nFlaky Tests Detected: {flaky_report['total_flaky_tests']}")
            for test in flaky_report["flaky_tests"][:3]:
                click.echo(f"   {test['test_name']}: {test['pass_rate']:.1%} pass rate")
        
        # Show recommendations
        if report.recommendations:
            click.echo(f"\nRecommendations:")
            for rec in report.recommendations[:5]:
                click.echo(f"   - {rec}")
        
    except Exception as e:
        click.echo(f"Failed to analyze failures: {e}")
        sys.exit(1)


@bug_tracking.command()
def flaky_tests():
    # Show detailed flaky test analysis
    click.echo("Analyzing flaky test patterns...")
    
    try:
        from src.failure_analyzer import get_failure_analyzer
        
        analyzer = get_failure_analyzer()
        report = analyzer.get_flaky_tests_report()
        
        click.echo(f"\nFlaky Test Analysis:")
        click.echo(f"   Total Flaky Tests: {report['total_flaky_tests']}")
        
        if report["flaky_tests"]:
            click.echo(f"\n   Detailed Breakdown:")
            for test in report["flaky_tests"]:
                click.echo(f"\n   Test: {test['test_name']}")
                click.echo(f"     Pass Rate: {test['pass_rate']:.1%}")
                click.echo(f"     Recent Failures: {test['recent_failures']}")
                click.echo(f"     Recent Passes: {test['recent_passes']}")
                if test['failure_pattern']:
                    click.echo(f"     Pattern: {test['failure_pattern']}")
        
        if report["recommendations"]:
            click.echo(f"\n   Recommendations:")
            for rec in report["recommendations"]:
                click.echo(f"     - {rec}")
        
    except Exception as e:
        click.echo(f"Failed to analyze flaky tests: {e}")
        sys.exit(1)


@bug_tracking.command()
@click.option('--test-name', help='Test name to check failure history')
@click.option('--limit', default=10, help='Maximum number of entries to show')
def failure_history(test_name, limit):
    # Show failure history for tests
    click.echo("Checking failure history...")
    
    try:
        from src.github_integration import get_bug_tracker
        
        bug_tracker = get_bug_tracker()
        
        if test_name:
            # Show history for specific test
            matching_records = [
                record for record in bug_tracker.failure_history.values()
                if record.test_name == test_name
            ]
            
            if matching_records:
                click.echo(f"\nFailure History for '{test_name}':")
                for record in matching_records[:limit]:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.timestamp))
                    click.echo(f"   {timestamp}: {record.consecutive_failures} consecutive failures")
                    click.echo(f"     Type: {record.failure_type}")
                    click.echo(f"     Error: {record.error_message[:100]}...")
            else:
                click.echo(f"No failure history found for '{test_name}'")
        else:
            # Show all recent failures
            recent_records = sorted(
                bug_tracker.failure_history.values(),
                key=lambda x: x.timestamp,
                reverse=True
            )[:limit]
            
            if recent_records:
                click.echo(f"\nRecent Failure History:")
                for record in recent_records:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.timestamp))
                    click.echo(f"\n   {record.test_name}")
                    click.echo(f"     Last Failure: {timestamp}")
                    click.echo(f"     Consecutive: {record.consecutive_failures}")
                    click.echo(f"     Type: {record.failure_type}")
            else:
                click.echo("No failure history found")
        
    except Exception as e:
        click.echo(f"Failed to check failure history: {e}")
        sys.exit(1)


@bug_tracking.command()
def retry_stats():
    # Show retry mechanism statistics
    click.echo("Checking retry statistics...")
    
    try:
        from src.retry_manager import get_retry_manager
        
        retry_manager = get_retry_manager()
        stats = retry_manager.get_retry_statistics()
        
        click.echo(f"\nRetry Statistics Summary:")
        click.echo(f"   Functions with Retry: {stats['summary']['total_functions']}")
        click.echo(f"   Total Calls: {stats['summary']['total_calls']}")
        click.echo(f"   Total Retries: {stats['summary']['total_retries']}")
        click.echo(f"   Success Rate: {stats['summary']['overall_success_rate']:.1%}")
        
        if stats["functions"]:
            click.echo(f"\n   Per-Function Statistics:")
            for func_name, func_stats in list(stats["functions"].items())[:10]:
                click.echo(f"\n     {func_name}:")
                click.echo(f"       Total Calls: {func_stats['total_calls']}")
                click.echo(f"       Successful: {func_stats['successful_calls']}")
                click.echo(f"       Failed: {func_stats['failed_calls']}")
                click.echo(f"       Avg Time: {func_stats['avg_time']:.3f}s")
        
    except Exception as e:
        click.echo(f"Failed to get retry statistics: {e}")
        sys.exit(1)


@bug_tracking.command()
@click.option('--days', default=30, help='Days of old data to clean up')
def cleanup(days):
    # Clean up old failure data
    click.echo(f"Cleaning up failure data older than {days} days...")
    
    try:
        from src.github_integration import get_bug_tracker
        
        bug_tracker = get_bug_tracker()
        initial_count = len(bug_tracker.failure_history)
        
        bug_tracker.cleanup_old_failures(days)
        
        final_count = len(bug_tracker.failure_history)
        cleaned = initial_count - final_count
        
        click.echo(f"Cleanup completed:")
        click.echo(f"   Records before: {initial_count}")
        click.echo(f"   Records after: {final_count}")
        click.echo(f"   Records cleaned: {cleaned}")
        
    except Exception as e:
        click.echo(f"Failed to cleanup data: {e}")
        sys.exit(1)


@bug_tracking.command()
def health_check():
    # Check overall health of bug tracking system
    click.echo("Checking bug tracking system health...")
    
    try:
        from src.github_integration import get_bug_tracker
        from src.failure_analyzer import get_failure_analyzer
        from src.retry_manager import get_retry_manager
        
        # Check bug tracker
        bug_tracker = get_bug_tracker()
        github_configured = bug_tracker._is_github_configured()
        
        click.echo(f"\nBug Tracking System Health:")
        click.echo(f"   GitHub Integration: {'✅ Configured' if github_configured else '❌ Not configured'}")
        click.echo(f"   Failure Records: {len(bug_tracker.failure_history)}")
        
        # Check if daily limit is approaching
        today = time.strftime("%Y-%m-%d")
        daily_count = bug_tracker.daily_issue_count.get(today, 0)
        max_daily = bug_tracker.config["failure_tracking"]["max_issues_per_day"]
        
        limit_status = "🟡 Approaching limit" if daily_count >= max_daily * 0.8 else "✅ Normal"
        if daily_count >= max_daily:
            limit_status = "🔴 Limit reached"
        
        click.echo(f"   Daily Issue Limit: {daily_count}/{max_daily} ({limit_status})")
        
        # Check failure analyzer
        analyzer = get_failure_analyzer()
        flaky_tests = len([t for t in analyzer.known_flaky_tests.values() if t.inconsistent_results])
        
        click.echo(f"\nFailure Analysis:")
        click.echo(f"   Monitored Tests: {len(analyzer.known_flaky_tests)}")
        click.echo(f"   Flaky Tests: {flaky_tests}")
        
        # Check retry manager
        retry_manager = get_retry_manager()
        retry_stats = retry_manager.get_retry_statistics()
        
        click.echo(f"\nRetry System:")
        click.echo(f"   Functions with Retry: {retry_stats['summary']['total_functions']}")
        click.echo(f"   Overall Success Rate: {retry_stats['summary']['overall_success_rate']:.1%}")
        
        # Overall health score
        health_score = 1.0
        if not github_configured:
            health_score -= 0.3
        if daily_count >= max_daily:
            health_score -= 0.2
        if flaky_tests > 10:
            health_score -= 0.2
        if retry_stats['summary']['overall_success_rate'] < 0.8:
            health_score -= 0.3
        
        health_status = "🟢 Excellent" if health_score >= 0.9 else \
                       "🟡 Good" if health_score >= 0.7 else \
                       "🟠 Fair" if health_score >= 0.5 else "🔴 Poor"
        
        click.echo(f"\nOverall Health: {health_score:.1%} ({health_status})")
        
    except Exception as e:
        click.echo(f"Failed to check system health: {e}")
        sys.exit(1)


@bug_tracking.command()
@click.option('--format', type=click.Choice(['json', 'html']), default='html', help='Report format')
@click.option('--output', help='Output file path')
@click.option('--hours', default=168, help='Time window in hours (default: 1 week)')
def generate_report(format, output, hours):
    # Generate comprehensive failure analysis report
    click.echo(f"Generating {format.upper()} failure analysis report...")
    
    try:
        from src.failure_reports import get_failure_report_generator
        
        report_generator = get_failure_report_generator()
        
        # Generate comprehensive report
        report = report_generator.generate_comprehensive_report(hours)
        
        if format == 'html':
            if output:
                # Custom output path
                output_file = Path(output)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                html_content = report_generator._create_html_report(report)
                with open(output_file, 'w') as f:
                    f.write(html_content)
                click.echo(f"HTML report generated: {output_file}")
            else:
                # Default output path
                html_file = report_generator.generate_html_report(report)
                click.echo(f"HTML report generated: {html_file}")
        else:
            # JSON format
            if output:
                output_file = Path(output)
            else:
                output_file = Path(f"failure_report_{int(time.time())}.json")
            
            report_dict = {
                "report_id": report.report_id,
                "generated_at": report.generated_at,
                "time_period_hours": report.time_period_hours,
                "summary": report.summary,
                "recommendations": report.recommendations
            }
            
            with open(output_file, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            click.echo(f"JSON report generated: {output_file}")
        
    except Exception as e:
        click.echo(f"Failed to generate report: {e}")
        sys.exit(1)


if __name__ == '__main__':
    cli()