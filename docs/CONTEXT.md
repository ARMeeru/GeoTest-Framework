# GeoTest Framework Context

## Project Overview

**Name**: GeoTest Framework
**Description**: An advanced API testing framework designed to demonstrate modern QA engineering practices using the REST Countries API. It encompasses functional testing, performance testing, failure analysis, monitoring, and intelligent bug tracking.
**Tech Stack**: Python 3.12+, Pytest, Docker, Asyncio, Aiohttp, GitHub REST API.

## Architecture

The framework follows a 6-phase maturity model:

1. **Foundation**: Basic functional tests using `pytest`.
2. **Data-Driven**: parametrized tests and model validation (`pydantic`).
3. **Containerization**: Docker and Docker Compose integration.
4. **Monitoring & Reporting**: System metrics collection and HTML/Allure reporting.
5. **Intelligent Bug Tracking**: Automated GitHub issue creation with de-duplication hash logic.
6. **Performance Testing**: Unified engine for load and stress testing (Synchronous & Asynchronous).

## Directory Structure

```text
GeoTest-Framework/
├── src/
│   ├── api_client.py       # REST Countries API wrapper (requests-based)
│   ├── models.py           # Pydantic data models
│   ├── monitoring.py       # generic metrics collection (psutil)
│   ├── alerting.py         # Alerting system (Email, Slack, Webhooks)
│   ├── github_integration.py # SmartBugTracker for automated issue management
│   ├── performance.py      # UnifiedPerformanceEngine (Orchestrator)
│   ├── load_generator.py   # AsyncLoadGenerator (High-concurrency engine)
│   ├── failure_analyzer.py # Logic to categorize errors (Network vs Assertion)
│   └── retry_manager.py    # Smart retry logic
├── tests/
│   ├── test_countries_api.py    # Functional tests
│   ├── test_load_performance.py # Load tests
│   ├── test_stress_performance.py # Stress tests (breaking point)
│   └── conftest.py              # Pytest fixtures
├── docs/                   # Documentation (Architecture, Performance internals)
├── reports/                # Test artifacts (HTML reports)
├── Dockerfile              # Container definition
├── docker-compose.yml      # Service orchestration
├── pytest.ini              # Pytest configuration and markers
└── requirements.txt        # Python dependencies
```

## Key Components

### 1. Performance Engine (`src/performance.py`)

* **Unified Entry Point**: `UnifiedPerformanceEngine` class.
* **Modes**:
  * **Synchronous**: Uses `ThreadPoolExecutor` for low concurrency (<50 users).
  * **Asynchronous**: Automatically switches to `AsyncLoadGenerator` (`src/load_generator.py`) for high concurrency (>50 users) using `asyncio` and `aiohttp`.
* **Capabilities**: Load Testing, Stress Testing (finding breaking points), Benchmarking.

### 2. Intelligent Bug Tracking (`src/github_integration.py`)

* **SmartBugTracker**: Analyzes failures and interacts with GitHub Issues.
* **Features**:
  * Generates `failure_hash` to group similar errors.
  * Prevents duplicate issues.
  * Rate limiting (prevent spamming API).
  * Auto-resolution: Can close issues when tests pass.

### 3. Monitoring & Alerting (`src/alerting.py`, `src/monitoring.py`)

* Real-time system metrics (CPU/RAM).
* notification channels: Email (SMTP), Slack (Webhooks), Generic Webhooks.
* Rule-based alerting with cooldowns.

## Setup & Running

**Local Setup**:

```bash
python3 -m venv venv
source venv/bin/activate  # or venv/bin/activate.fish
pip install -r requirements.txt
```

**Running Tests**:

```bash
# Run all tests
pytest

# Run specific markers
pytest -m smoke          # Quick sanity checks
pytest -m performance    # Load tests
pytest -m stress_test    # Breaking point tests

# CI/CD Mode (Docker)
docker-compose up --build
```

## Recent Engineering Improvements

* **Async Performance Refactor**: The `UnifiedPerformanceEngine` was refactored to seamlessly integrate `AsyncLoadGenerator`. This resolved a bottleneck where load tests were initially thread-bound. Now, `load_test()` can handle 1000+ concurrent users by delegating to the async engine.
