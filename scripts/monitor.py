#!/usr/bin/env python3
# GeoTest Framework Monitoring CLI Tool
# Provides command-line interface for monitoring operations

import click
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring import get_metrics_collector, SystemMonitor
from src.alerting import get_alert_manager
from src.dashboard_generator import DashboardGenerator


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


if __name__ == '__main__':
    cli()