# GeoTest Framework User Manual

Welcome to the **GeoTest Framework**. This manual will guide you through the installation, configuration, and execution of automated tests using this framework.

## 1. Prerequisites

* **Operating System**: Linux, macOS, or Windows (WSL recommended).
* **Python**: Version 3.12 or higher.
* **Docker** (Optional): For containerized execution.

## 2. Installation

It is recommended to run the framework within a virtual environment.

```bash
# 1. Clone the repository (if not already done)
# git clone <repository-url>
# cd GeoTest-Framework

# 2. Create a virtual environment
python3 -m venv venv

# 3. Activate the environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

## 3. Configuration

The framework uses JSON configuration files located in the `config/` directory (you may need to create this directory).

### 3.1. Smart Bug Tracking

To enable automated GitHub issue creation, create `config/bug_tracking.json`:

```json
{
  "github": {
    "enabled": true,
    "token": "YOUR_GITHUB_TOKEN",
    "repo_owner": "your-username",
    "repo_name": "your-repo",
    "labels": ["bug", "automated-test-failure"]
  },
  "failure_tracking": {
    "consecutive_failures_threshold": 3,
    "max_issues_per_day": 5,
    "auto_close_on_success": true
  }
}
```

### 3.2. Alerting

To configure email or Slack alerts, create `config/alerting.json`:

```json
{
  "notification_channels": [
    {
      "type": "slack",
      "name": "Team Slack",
      "config": {
        "webhook_url": "YOUR_SLACK_WEBHOOK",
        "channel": "#alerts"
      }
    }
  ],
  "alert_rules": [
    {
      "rule_id": "high_cpu",
      "condition": "cpu_percent > 90",
      "severity": "critical"
    }
  ]
}
```

## 4. Running Tests

The framework uses `pytest` as the test runner.

### 4.1. Functional Tests

Run standard API functional tests:

```bash
# Run all tests
pytest

# Run only smoke tests (quick sanity check)
pytest -m smoke

# Run regression tests
pytest -m regression
```

### 4.2. Performance Tests

The framework includes a **Unified Performance Engine** that automatically scales.

**Load Tests:**

```bash
# Run standard load tests
pytest -m performance

# The framework automatically switches to ASYNC mode if concurrent users > 50.
# You don't need to do anything special!
```

**Stress Tests:**
Find the breaking point of the API:

```bash
pytest -m stress_test
```

### 4.3. Reporting

After execution, an HTML report is generated at `reports/report.html`.
Open this file in your browser to view detailed results, graphs, and failure analysis.

## 5. Docker Execution

To run the full suite in a clean containerized environment:

```bash
# Build and run the test container
docker-compose up --build geotest

# Run specifically performance tests in Docker
docker-compose up performance-tests
```

## 6. Advanced Features

### Async Performance Mode

For high-concurrency testing (e.g., 1000+ users), the framework uses `asyncio`. This is handled automatically by the `UnifiedPerformanceEngine`. If you are writing a new test and want to force this mode:

```python
engine.load_test(..., async_mode=True)
```

### Intelligent Failure Analysis

The framework analyzes error messages to categorize them (Network Error, Assertion Failure, etc.). If a test fails consecutively 3 times (configurable), it will automatically open a GitHub Issue with a hash code to prevent duplicates.

## 7. Troubleshooting

* **"Module not found"**: Ensure you have activated the venv (`source venv/bin/activate`).
* **Performance test hangs**: If running very high loads locally, ensure your machine has enough open file descriptors (`ulimit -n`).
* **Docker network issues**: Ensure Docker is running. The framework interacts with the public internet (REST Countries API).
