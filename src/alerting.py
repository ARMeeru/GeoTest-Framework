# Advanced alerting and notification system
# Provides real-time alerts for test failures and performance issues

import json
import time
import smtplib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from enum import Enum
import requests

try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
except ImportError:
    # Fallback for older Python versions
    MimeText = None
    MimeMultipart = None

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    # Alert severity levels
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    # Types of alerts
    TEST_FAILURE = "test_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SYSTEM_RESOURCE = "system_resource"
    API_ERROR_RATE = "api_error_rate"
    CUSTOM = "custom"


@dataclass
class Alert:
    # Individual alert instance
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: float
    source: str
    tags: Dict[str, str]
    resolved: bool = False
    resolved_timestamp: Optional[float] = None
    resolved_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        # Convert alert to dictionary
        return asdict(self)


@dataclass
class AlertRule:
    # Alert rule configuration
    rule_id: str
    rule_name: str
    alert_type: AlertType
    condition: str  # Python expression
    severity: AlertSeverity
    description: str
    enabled: bool = True
    cooldown_minutes: int = 15
    max_alerts_per_hour: int = 10
    tags: Optional[Dict[str, str]] = None


class NotificationChannel:
    # Base class for notification channels
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
    
    def send_notification(self, alert: Alert) -> bool:
        # Send notification for alert. Returns True if successful.
        raise NotImplementedError
    
    def test_connection(self) -> bool:
        # Test if the notification channel is working
        raise NotImplementedError


class EmailNotificationChannel(NotificationChannel):
    # Email notification channel
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email')
        self.to_emails = config.get('to_emails', [])
        self.use_tls = config.get('use_tls', True)
    
    def send_notification(self, alert: Alert) -> bool:
        # Send email notification
        if not self.enabled or not self.to_emails or not MimeText or not MimeMultipart:
            return False
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Email body
            body = f"""
GeoTest Framework Alert

Alert ID: {alert.alert_id}
Severity: {alert.severity.value.upper()}
Type: {alert.alert_type.value}
Source: {alert.source}
Timestamp: {datetime.fromtimestamp(alert.timestamp, timezone.utc).isoformat()}

Message:
{alert.message}

Tags: {json.dumps(alert.tags, indent=2)}

---
This is an automated alert from GeoTest Framework monitoring system.
            """.strip()
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent successfully: {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert {alert.alert_id}: {e}")
            return False
    
    def test_connection(self) -> bool:
        # Test email configuration
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
            return True
        except Exception as e:
            logger.error(f"Email connection test failed: {e}")
            return False


class SlackNotificationChannel(NotificationChannel):
    # Slack notification channel
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'GeoTest-Framework')
    
    def send_notification(self, alert: Alert) -> bool:
        # Send Slack notification
        if not self.enabled or not self.webhook_url:
            return False
        
        try:
            # Color coding based on severity
            color_map = {
                AlertSeverity.LOW: "#36a64f",      # Green
                AlertSeverity.MEDIUM: "#ff9f00",   # Orange
                AlertSeverity.HIGH: "#ff6b6b",     # Red
                AlertSeverity.CRITICAL: "#8b0000"  # Dark Red
            }
            
            # Create Slack message
            payload = {
                "channel": self.channel,
                "username": self.username,
                "icon_emoji": ":warning:",
                "attachments": [{
                    "color": color_map.get(alert.severity, "#808080"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Alert ID",
                            "value": alert.alert_id,
                            "short": True
                        },
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Type",
                            "value": alert.alert_type.value,
                            "short": True
                        },
                        {
                            "title": "Source",
                            "value": alert.source,
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": datetime.fromtimestamp(alert.timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": False
                        }
                    ],
                    "footer": "GeoTest Framework",
                    "ts": int(alert.timestamp)
                }]
            }
            
            # Add tags if present
            if alert.tags:
                payload["attachments"][0]["fields"].append({
                    "title": "Tags",
                    "value": ", ".join([f"{k}={v}" for k, v in alert.tags.items()]),
                    "short": False
                })
            
            # Send to Slack
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent successfully: {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert {alert.alert_id}: {e}")
            return False
    
    def test_connection(self) -> bool:
        # Test Slack webhook
        if not self.webhook_url:
            return False
        
        try:
            test_payload = {
                "channel": self.channel,
                "username": self.username,
                "text": "GeoTest Framework connection test - please ignore"
            }
            response = requests.post(self.webhook_url, json=test_payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Slack connection test failed: {e}")
            return False


class WebhookNotificationChannel(NotificationChannel):
    # Generic webhook notification channel
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.url = config.get('url')
        self.method = config.get('method', 'POST').upper()
        self.headers = config.get('headers', {})
        self.auth_token = config.get('auth_token')
        
        if self.auth_token:
            self.headers['Authorization'] = f"Bearer {self.auth_token}"
    
    def send_notification(self, alert: Alert) -> bool:
        # Send webhook notification
        if not self.enabled or not self.url:
            return False
        
        try:
            payload = {
                "alert": alert.to_dict(),
                "timestamp": datetime.fromtimestamp(alert.timestamp, timezone.utc).isoformat(),
                "source": "geotest-framework"
            }
            
            if self.method == 'POST':
                response = requests.post(self.url, json=payload, headers=self.headers, timeout=10)
            elif self.method == 'PUT':
                response = requests.put(self.url, json=payload, headers=self.headers, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {self.method}")
            
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent successfully: {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert {alert.alert_id}: {e}")
            return False
    
    def test_connection(self) -> bool:
        # Test webhook endpoint
        if not self.url:
            return False
        
        try:
            test_payload = {
                "test": True,
                "message": "GeoTest Framework connection test",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            if self.method == 'POST':
                response = requests.post(self.url, json=test_payload, headers=self.headers, timeout=10)
            else:
                response = requests.get(self.url, headers=self.headers, timeout=10)
            
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Webhook connection test failed: {e}")
            return False


class AlertManager:
    # Manages alerts, rules, and notifications
    
    def __init__(self, config_file: Optional[Path] = None):
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self.alert_history: Dict[str, List[float]] = {}  # rule_id -> alert timestamps
        self._lock = threading.Lock()
        
        if config_file and config_file.exists():
            self.load_config(config_file)
    
    def load_config(self, config_file: Path):
        # Load alerting configuration from file
        try:
            with open(config_file) as f:
                config = json.load(f)
            
            # Load alert rules
            for rule_data in config.get('alert_rules', []):
                rule = AlertRule(**rule_data)
                self.alert_rules[rule.rule_id] = rule
            
            # Load notification channels
            for channel_data in config.get('notification_channels', []):
                channel_type = channel_data.get('type')
                channel_name = channel_data.get('name')
                channel_config = channel_data.get('config', {})
                
                if channel_type == 'email':
                    channel = EmailNotificationChannel(channel_name, channel_config)
                elif channel_type == 'slack':
                    channel = SlackNotificationChannel(channel_name, channel_config)
                elif channel_type == 'webhook':
                    channel = WebhookNotificationChannel(channel_name, channel_config)
                else:
                    logger.warning(f"Unknown notification channel type: {channel_type}")
                    continue
                
                self.notification_channels[channel_name] = channel
            
            logger.info(f"Loaded {len(self.alert_rules)} rules and {len(self.notification_channels)} channels")
            
        except Exception as e:
            logger.error(f"Failed to load alerting config: {e}")
    
    def create_alert(self, alert_type: AlertType, severity: AlertSeverity, 
                    title: str, message: str, source: str = "unknown",
                    tags: Dict[str, str] = None) -> Alert:
        # Create and process a new alert
        alert_id = f"{alert_type.value}_{int(time.time() * 1000)}"
        
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            timestamp=time.time(),
            source=source,
            tags=tags or {}
        )
        
        with self._lock:
            self.alerts.append(alert)
            
            # Maintain alert history (keep last 1000)
            if len(self.alerts) > 1000:
                self.alerts.pop(0)
        
        # Send notifications
        self._send_notifications(alert)
        
        logger.info(f"Created alert {alert_id}: {title}")
        return alert
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        # Mark an alert as resolved
        with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_timestamp = time.time()
                    alert.resolved_by = resolved_by
                    logger.info(f"Resolved alert {alert_id}")
                    return True
        
        logger.warning(f"Alert {alert_id} not found or already resolved")
        return False
    
    def check_rules(self, metrics_data: Dict[str, Any]):
        # Check alert rules against current metrics
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check cooldown
                if self._is_rule_in_cooldown(rule_id, rule.cooldown_minutes):
                    continue
                
                # Check rate limiting
                if self._is_rule_rate_limited(rule_id, rule.max_alerts_per_hour):
                    continue
                
                # Evaluate rule condition
                if self._evaluate_rule_condition(rule.condition, metrics_data):
                    self._trigger_rule_alert(rule, metrics_data)
                    
            except Exception as e:
                logger.error(f"Error checking rule {rule_id}: {e}")
    
    def _is_rule_in_cooldown(self, rule_id: str, cooldown_minutes: int) -> bool:
        # Check if rule is in cooldown period
        if rule_id not in self.alert_history:
            return False
        
        last_alert_time = max(self.alert_history[rule_id]) if self.alert_history[rule_id] else 0
        cooldown_seconds = cooldown_minutes * 60
        return (time.time() - last_alert_time) < cooldown_seconds
    
    def _is_rule_rate_limited(self, rule_id: str, max_alerts_per_hour: int) -> bool:
        # Check if rule has exceeded rate limit
        if rule_id not in self.alert_history:
            return False
        
        hour_ago = time.time() - 3600
        recent_alerts = [t for t in self.alert_history[rule_id] if t > hour_ago]
        return len(recent_alerts) >= max_alerts_per_hour
    
    def _evaluate_rule_condition(self, condition: str, metrics_data: Dict[str, Any]) -> bool:
        # Safely evaluate rule condition
        try:
            # Create safe evaluation context
            safe_globals = {
                '__builtins__': {},
                'abs': abs,
                'min': min,
                'max': max,
                'len': len,
                'int': int,
                'float': float,
                'str': str,
                'bool': bool
            }
            
            # Add metrics data to context
            safe_globals.update(metrics_data)
            
            # Evaluate condition
            return bool(eval(condition, safe_globals))
            
        except Exception as e:
            logger.error(f"Failed to evaluate condition '{condition}': {e}")
            return False
    
    def _trigger_rule_alert(self, rule: AlertRule, metrics_data: Dict[str, Any]):
        # Trigger alert for rule
        # Record alert time
        if rule.rule_id not in self.alert_history:
            self.alert_history[rule.rule_id] = []
        self.alert_history[rule.rule_id].append(time.time())
        
        # Create alert
        alert = self.create_alert(
            alert_type=rule.alert_type,
            severity=rule.severity,
            title=f"Rule triggered: {rule.rule_name}",
            message=f"{rule.description}\n\nCondition: {rule.condition}\nCurrent metrics: {json.dumps(metrics_data, indent=2)}",
            source=f"rule:{rule.rule_id}",
            tags=rule.tags or {}
        )
        
        logger.warning(f"Alert rule triggered: {rule.rule_name} ({rule.rule_id})")
    
    def _send_notifications(self, alert: Alert):
        # Send alert notifications to all enabled channels
        for channel_name, channel in self.notification_channels.items():
            if not channel.enabled:
                continue
            
            try:
                success = channel.send_notification(alert)
                if success:
                    logger.info(f"Notification sent via {channel_name}")
                else:
                    logger.warning(f"Failed to send notification via {channel_name}")
            except Exception as e:
                logger.error(f"Error sending notification via {channel_name}: {e}")
    
    def get_active_alerts(self, severity_filter: AlertSeverity = None) -> List[Alert]:
        # Get all active (unresolved) alerts
        with self._lock:
            alerts = [a for a in self.alerts if not a.resolved]
            
            if severity_filter:
                alerts = [a for a in alerts if a.severity == severity_filter]
            
            return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        # Get alert summary for specified time period
        cutoff_time = time.time() - (hours_back * 3600)
        
        with self._lock:
            recent_alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        # Count by severity
        severity_counts = {sev.value: 0 for sev in AlertSeverity}
        for alert in recent_alerts:
            severity_counts[alert.severity.value] += 1
        
        # Count by type
        type_counts = {}
        for alert in recent_alerts:
            type_name = alert.alert_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Active vs resolved
        active_alerts = [a for a in recent_alerts if not a.resolved]
        resolved_alerts = [a for a in recent_alerts if a.resolved]
        
        return {
            'time_period_hours': hours_back,
            'total_alerts': len(recent_alerts),
            'active_alerts': len(active_alerts),
            'resolved_alerts': len(resolved_alerts),
            'severity_breakdown': severity_counts,
            'type_breakdown': type_counts,
            'alert_rate_per_hour': len(recent_alerts) / max(hours_back, 1)
        }
    
    def test_notifications(self) -> Dict[str, bool]:
        # Test all notification channels
        results = {}
        for channel_name, channel in self.notification_channels.items():
            try:
                results[channel_name] = channel.test_connection()
            except Exception as e:
                logger.error(f"Error testing channel {channel_name}: {e}")
                results[channel_name] = False
        return results
    
    def export_alerts(self, output_file: Path, hours_back: int = 24):
        # Export alerts to JSON file
        cutoff_time = time.time() - (hours_back * 3600)
        
        with self._lock:
            alerts_to_export = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        export_data = {
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'time_period_hours': hours_back,
            'total_alerts': len(alerts_to_export),
            'alerts': [a.to_dict() for a in alerts_to_export],
            'summary': self.get_alert_summary(hours_back)
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(alerts_to_export)} alerts to {output_file}")


# Global alert manager instance
_global_alert_manager = None

def get_alert_manager() -> AlertManager:
    # Get or create global alert manager instance
    global _global_alert_manager
    if _global_alert_manager is None:
        config_path = Path(__file__).parent.parent / "config" / "alerting.json"
        _global_alert_manager = AlertManager(config_path if config_path.exists() else None)
    return _global_alert_manager