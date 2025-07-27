#!/usr/bin/env python3
# Script to run tests with full monitoring capabilities enabled
# This demonstrates the complete Phase 4 monitoring integration

import sys
import subprocess
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring import get_metrics_collector, SystemMonitor
from src.alerting import get_alert_manager
from src.dashboard_generator import DashboardGenerator


def run_tests_with_monitoring():
    # Run tests with full monitoring enabled
    
    print("Starting GeoTest Framework with Phase 4 Monitoring")
    print("=" * 60)
    
    # Initialize monitoring components
    metrics_collector = get_metrics_collector()
    alert_manager = get_alert_manager()
    
    # Start system monitoring
    system_monitor = SystemMonitor(metrics_collector, interval_seconds=3)
    system_monitor.start()
    print("System monitoring started")
    
    # Record session start
    session_start = time.time()
    metrics_collector.record_custom_metric(
        "session.monitoring_demo.started",
        1,
        tags={"script": "run_tests_with_monitoring", "version": "4.0"}
    )
    
    try:
        print("\nRunning test suite with monitoring...")
        
        # Run a few different test types to demonstrate monitoring
        test_commands = [
            # Smoke tests
            ["pytest", "-m", "smoke", "-v", "--tb=short"],
            # Run a single test for demo
            ["pytest", "tests/test_countries_api.py::TestRestCountriesAPI::test_get_all_countries_returns_data", "-v"],
        ]
        
        for i, cmd in enumerate(test_commands, 1):
            print(f"\nRunning test batch {i}/{len(test_commands)}: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                # Record test batch metrics
                metrics_collector.record_custom_metric(
                    f"test_batch.{i}.exit_code",
                    result.returncode,
                    tags={"command": " ".join(cmd[:2])}
                )
                
                if result.returncode == 0:
                    print(f"Test batch {i} completed successfully")
                else:
                    print(f"Test batch {i} had issues (exit code: {result.returncode})")
                    
                    # Create alert for test failures
                    alert_manager.create_alert(
                        alert_type="test_failure",
                        severity="medium",
                        title=f"Test Batch {i} Failed",
                        message=f"Test batch {i} failed with exit code {result.returncode}\n\nCommand: {' '.join(cmd)}\n\nOutput:\n{result.stdout}\n\nErrors:\n{result.stderr}",
                        source="monitoring_demo",
                        tags={"batch": str(i), "command": " ".join(cmd[:2])}
                    )
                
                print(f"   Output lines: {len(result.stdout.splitlines())}")
                print(f"   Error lines: {len(result.stderr.splitlines())}")
                
            except subprocess.TimeoutExpired:
                print(f"Test batch {i} timed out")
                metrics_collector.record_custom_metric(
                    f"test_batch.{i}.timeout",
                    1,
                    tags={"command": " ".join(cmd[:2])}
                )
            except Exception as e:
                print(f"Test batch {i} failed: {e}")
                metrics_collector.record_custom_metric(
                    f"test_batch.{i}.error",
                    1,
                    tags={"error": str(e)[:100]}
                )
        
        print("\nGenerating enhanced reports and dashboard...")
        
        # Generate dashboard
        dashboard_generator = DashboardGenerator(metrics_collector, alert_manager)
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        dashboard_file = reports_dir / "monitoring_demo_dashboard.html"
        dashboard_generator.generate_dashboard(dashboard_file, time_window_hours=1)
        print(f"Dashboard: {dashboard_file}")
        
        # Export metrics
        files_created = metrics_collector.export_metrics(reports_dir)
        print(f"Metrics exported: {len(files_created)} files")
        
        # Show summary
        session_duration = time.time() - session_start
        test_summary = metrics_collector.get_test_summary(60)  # Last hour
        alert_summary = alert_manager.get_alert_summary(1)     # Last hour
        
        print(f"\nSession Summary:")
        print(f"   Duration: {session_duration:.2f}s")
        print(f"   Tests: {test_summary.get('total_tests', 0)} total")
        print(f"   Pass Rate: {test_summary.get('pass_rate', 0):.1f}%")
        print(f"   Alerts: {alert_summary.get('total_alerts', 0)} total")
        
        # Record session completion
        metrics_collector.record_custom_metric(
            "session.monitoring_demo.completed",
            session_duration,
            tags={"script": "run_tests_with_monitoring", "success": "true"}
        )
        
        print(f"\nMonitoring demonstration completed!")
        print(f"Check the reports/ directory for detailed results")
        print(f"Open {dashboard_file} in your browser to view the dashboard")
        
    except KeyboardInterrupt:
        print("\nMonitoring demo interrupted by user")
        metrics_collector.record_custom_metric(
            "session.monitoring_demo.interrupted",
            time.time() - session_start,
            tags={"script": "run_tests_with_monitoring"}
        )
    except Exception as e:
        print(f"\nMonitoring demo failed: {e}")
        metrics_collector.record_custom_metric(
            "session.monitoring_demo.error",
            1,
            tags={"error": str(e)[:100]}
        )
    finally:
        # Stop system monitoring
        system_monitor.stop()
        print("System monitoring stopped")


if __name__ == "__main__":
    run_tests_with_monitoring()