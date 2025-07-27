# Advanced monitoring and metrics collection system
# Provides comprehensive test execution monitoring and performance tracking

import json
import time
import psutil
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    # Single metric data point
    timestamp: float
    value: Union[float, int, str]
    tags: Optional[Dict[str, str]] = None


@dataclass
class TestMetrics:
    # Test execution metrics
    test_id: str
    test_name: str
    start_time: float
    end_time: Optional[float] = None
    status: Optional[str] = None
    duration: Optional[float] = None
    response_time: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    network_bytes: Optional[int] = None
    error_message: Optional[str] = None
    api_endpoint: Optional[str] = None
    request_size: Optional[int] = None
    response_size: Optional[int] = None


@dataclass
class SystemMetrics:
    # System performance metrics
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    network_sent_mb: float
    network_recv_mb: float
    active_connections: int
    load_average: Optional[float] = None


class MetricsCollector:
    # Collects and stores various metrics during test execution
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.test_metrics: List[TestMetrics] = []
        self.system_metrics: List[SystemMetrics] = []
        self.api_metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.custom_metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.active_tests: Dict[str, TestMetrics] = {}
        self._lock = threading.Lock()
        self._baseline_network = self._get_network_stats()
        
    def _get_network_stats(self) -> Dict[str, int]:
        # Get current network I/O statistics
        try:
            net_io = psutil.net_io_counters()
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            }
        except Exception:
            return {'bytes_sent': 0, 'bytes_recv': 0}
    
    def start_test(self, test_id: str, test_name: str, api_endpoint: str = None) -> TestMetrics:
        # Start tracking a test execution
        with self._lock:
            metrics = TestMetrics(
                test_id=test_id,
                test_name=test_name,
                start_time=time.time(),
                api_endpoint=api_endpoint
            )
            self.active_tests[test_id] = metrics
            return metrics
    
    def end_test(self, test_id: str, status: str, error_message: str = None,
                 response_time: float = None, request_size: int = None, 
                 response_size: int = None) -> Optional[TestMetrics]:
        # End test tracking and calculate final metrics
        with self._lock:
            if test_id not in self.active_tests:
                logger.warning(f"Test {test_id} not found in active tests")
                return None
            
            metrics = self.active_tests.pop(test_id)
            end_time = time.time()
            
            # Calculate final metrics
            metrics.end_time = end_time
            metrics.duration = end_time - metrics.start_time
            metrics.status = status
            metrics.error_message = error_message
            metrics.response_time = response_time
            metrics.request_size = request_size
            metrics.response_size = response_size
            
            # Capture system metrics at test end
            try:
                metrics.memory_usage = psutil.virtual_memory().percent
                metrics.cpu_usage = psutil.cpu_percent(interval=0.1)
                
                # Network usage during test
                current_network = self._get_network_stats()
                metrics.network_bytes = (
                    current_network['bytes_sent'] + current_network['bytes_recv'] -
                    self._baseline_network['bytes_sent'] - self._baseline_network['bytes_recv']
                )
            except Exception as e:
                logger.debug(f"Failed to capture system metrics: {e}")
            
            # Store completed test metrics
            self.test_metrics.append(metrics)
            
            # Maintain history limit
            if len(self.test_metrics) > self.max_history:
                self.test_metrics.pop(0)
            
            return metrics
    
    def record_api_metric(self, endpoint: str, metric_name: str, value: Union[float, int],
                         tags: Dict[str, str] = None):
        # Record API-specific metric
        with self._lock:
            metric_key = f"{endpoint}.{metric_name}"
            point = MetricPoint(
                timestamp=time.time(),
                value=value,
                tags=tags or {}
            )
            self.api_metrics[metric_key].append(point)
            
            # Maintain history limit
            if len(self.api_metrics[metric_key]) > self.max_history:
                self.api_metrics[metric_key].pop(0)
    
    def record_custom_metric(self, name: str, value: Union[float, int, str],
                           tags: Dict[str, str] = None):
        # Record custom metric
        with self._lock:
            point = MetricPoint(
                timestamp=time.time(),
                value=value,
                tags=tags or {}
            )
            self.custom_metrics[name].append(point)
            
            # Maintain history limit
            if len(self.custom_metrics[name]) > self.max_history:
                self.custom_metrics[name].pop(0)
    
    def collect_system_metrics(self):
        # Collect current system performance metrics
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network
            network = self._get_network_stats()
            
            # System load (Unix-like systems)
            load_avg = None
            try:
                load_avg = psutil.getloadavg()[0]  # 1-minute load average
            except (AttributeError, OSError):
                pass  # Not available on all systems
            
            # Active network connections (requires special permissions on some systems)
            connections = 0
            try:
                connections = len(psutil.net_connections())
            except (psutil.AccessDenied, PermissionError, OSError):
                # Fallback: estimate from process count or use 0
                connections = len(psutil.pids())
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_usage_percent=disk.percent,
                network_sent_mb=network['bytes_sent'] / 1024 / 1024,
                network_recv_mb=network['bytes_recv'] / 1024 / 1024,
                active_connections=connections,
                load_average=load_avg
            )
            
            with self._lock:
                self.system_metrics.append(metrics)
                
                # Maintain history limit
                if len(self.system_metrics) > self.max_history:
                    self.system_metrics.pop(0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return None
    
    def get_test_summary(self, time_window_minutes: int = None) -> Dict[str, Any]:
        # Get test execution summary statistics
        with self._lock:
            metrics = self.test_metrics.copy()
        
        if not metrics:
            return {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'error_tests': 0,
                'pass_rate': 0.0,
                'average_duration': 0.0,
                'average_response_time': 0.0
            }
        
        # Filter by time window if specified
        if time_window_minutes:
            cutoff_time = time.time() - (time_window_minutes * 60)
            metrics = [m for m in metrics if m.start_time >= cutoff_time]
        
        if not metrics:
            return {'message': f'No tests found in last {time_window_minutes} minutes'}
        
        # Calculate statistics
        total_tests = len(metrics)
        passed_tests = len([m for m in metrics if m.status == 'passed'])
        failed_tests = len([m for m in metrics if m.status == 'failed'])
        error_tests = len([m for m in metrics if m.status == 'error'])
        
        durations = [m.duration for m in metrics if m.duration is not None]
        response_times = [m.response_time for m in metrics if m.response_time is not None]
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'error_tests': error_tests,
            'pass_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0.0,
            'average_duration': statistics.mean(durations) if durations else 0.0,
            'median_duration': statistics.median(durations) if durations else 0.0,
            'max_duration': max(durations) if durations else 0.0,
            'average_response_time': statistics.mean(response_times) if response_times else 0.0,
            'median_response_time': statistics.median(response_times) if response_times else 0.0,
            'max_response_time': max(response_times) if response_times else 0.0,
            'time_window_minutes': time_window_minutes
        }
    
    def get_system_summary(self, time_window_minutes: int = None) -> Dict[str, Any]:
        # Get system performance summary
        with self._lock:
            metrics = self.system_metrics.copy()
        
        if not metrics:
            return {'message': 'No system metrics available'}
        
        # Filter by time window if specified
        if time_window_minutes:
            cutoff_time = time.time() - (time_window_minutes * 60)
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        if not metrics:
            return {'message': f'No system metrics found in last {time_window_minutes} minutes'}
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in metrics]
        memory_values = [m.memory_percent for m in metrics]
        
        return {
            'samples_count': len(metrics),
            'cpu_avg': statistics.mean(cpu_values),
            'cpu_max': max(cpu_values),
            'cpu_min': min(cpu_values),
            'memory_avg': statistics.mean(memory_values),
            'memory_max': max(memory_values),
            'memory_min': min(memory_values),
            'latest_load_average': metrics[-1].load_average,
            'latest_connections': metrics[-1].active_connections,
            'time_window_minutes': time_window_minutes
        }
    
    def export_metrics(self, output_dir: Path) -> Dict[str, str]:
        # Export all metrics to JSON files
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        files_created = {}
        
        try:
            # Export test metrics
            with self._lock:
                test_data = [asdict(m) for m in self.test_metrics]
                system_data = [asdict(m) for m in self.system_metrics]
                api_data = {k: [asdict(p) for p in v] for k, v in self.api_metrics.items()}
                custom_data = {k: [asdict(p) for p in v] for k, v in self.custom_metrics.items()}
            
            # Test metrics
            test_file = output_dir / "test_metrics.json"
            with open(test_file, 'w') as f:
                json.dump(test_data, f, indent=2, default=str)
            files_created['test_metrics'] = str(test_file)
            
            # System metrics
            system_file = output_dir / "system_metrics.json"
            with open(system_file, 'w') as f:
                json.dump(system_data, f, indent=2, default=str)
            files_created['system_metrics'] = str(system_file)
            
            # API metrics
            api_file = output_dir / "api_metrics.json"
            with open(api_file, 'w') as f:
                json.dump(api_data, f, indent=2, default=str)
            files_created['api_metrics'] = str(api_file)
            
            # Custom metrics
            custom_file = output_dir / "custom_metrics.json"
            with open(custom_file, 'w') as f:
                json.dump(custom_data, f, indent=2, default=str)
            files_created['custom_metrics'] = str(custom_file)
            
            # Summary report
            summary = {
                'export_timestamp': datetime.now(timezone.utc).isoformat(),
                'test_summary': self.get_test_summary(),
                'system_summary': self.get_system_summary(),
                'metrics_counts': {
                    'test_metrics': len(self.test_metrics),
                    'system_metrics': len(self.system_metrics),
                    'api_metrics': sum(len(v) for v in self.api_metrics.values()),
                    'custom_metrics': sum(len(v) for v in self.custom_metrics.values())
                }
            }
            
            summary_file = output_dir / "metrics_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            files_created['summary'] = str(summary_file)
            
            logger.info(f"Exported metrics to {len(files_created)} files in {output_dir}")
            return files_created
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise


class SystemMonitor:
    # Background system monitoring with configurable intervals
    
    def __init__(self, metrics_collector: MetricsCollector, interval_seconds: int = 5):
        self.metrics_collector = metrics_collector
        self.interval_seconds = interval_seconds
        self.running = False
        self.thread = None
    
    def start(self):
        # Start background monitoring
        if self.running:
            logger.warning("System monitor already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started system monitoring with {self.interval_seconds}s interval")
    
    def stop(self):
        # Stop background monitoring
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        logger.info("Stopped system monitoring")
    
    def _monitor_loop(self):
        # Main monitoring loop
        while self.running:
            try:
                self.metrics_collector.collect_system_metrics()
                time.sleep(self.interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.interval_seconds)


# Global metrics collector instance
_global_collector = None

def get_metrics_collector() -> MetricsCollector:
    # Get or create global metrics collector instance
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector