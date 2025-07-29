# GeoTest Framework

An API testing framework using REST Countries API that demonstrates modern QA engineering practices.

## Running Tests

### Phase-Wise Test Execution

#### Phase 1: Foundation Tests
```bash
# Activate virtual environment
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Basic test execution
pytest                              # All tests
pytest -v                          # Verbose output
pytest -m smoke                    # Quick validation tests
pytest -m regression               # Comprehensive test suite
pytest -m critical                 # Critical tests with alerting
pytest tests/test_countries_api.py::TestRestCountriesAPI::test_get_all_countries_returns_data  # Single test
```

#### Phase 2: Data-Driven Testing
```bash
# Data-driven tests with CSV test data
pytest tests/test_data_driven.py -v
```

#### Phase 3: Containerization
```bash
# Docker container tests
docker compose run --rm smoke-tests
docker compose run --rm regression-tests
docker compose build --no-cache    # Fresh builds when needed
```

#### Phase 4: Monitoring Integration
```bash
# Monitoring integration tests
pytest tests/test_monitoring_integration.py -v

# Run tests with full monitoring enabled
python scripts/run_tests_with_monitoring.py

# System monitoring commands
python scripts/monitor.py monitor --duration 30
python scripts/monitor.py dashboard
python scripts/monitor.py summary
python scripts/monitor.py alerts
python scripts/monitor.py export
python scripts/monitor.py check-config
python scripts/monitor.py test-notifications

# Performance tests with monitoring
pytest -m performance
```

#### Phase 5: Bug Tracking Integration
```bash
# Bug tracking tests
pytest tests/test_bug_tracking.py -v

# Bug tracking system commands
python scripts/monitor.py bug-tracking health-check
python scripts/monitor.py bug-tracking analyze-failures --hours 24
python scripts/monitor.py bug-tracking flaky-tests
python scripts/monitor.py bug-tracking retry-stats
python scripts/monitor.py bug-tracking export-failures
python scripts/monitor.py bug-tracking mttr-analysis
python scripts/monitor.py bug-tracking create-issue
python scripts/monitor.py bug-tracking config
```

#### Phase 6: Performance Testing
```bash
# Performance test execution
pytest tests/test_load_performance.py -v      # Load testing
pytest tests/test_stress_performance.py -v    # Stress testing
pytest tests/test_benchmark_performance.py -v # Benchmark testing

# Performance test markers
pytest -m performance -v
pytest -m load_testing -v
pytest -m stress_testing -v
pytest -m benchmark -v
pytest -m async_stress -v
```

### Docker Execution

```bash
# Run smoke tests in container
docker compose run --rm smoke-tests

# Run regression tests in container
docker compose run --rm regression-tests

# Run all tests in container
docker compose run --rm all-tests

# Run critical tests with enhanced alerting
docker compose run --rm critical-tests

# Run performance tests with monitoring
docker compose run --rm performance-tests

# Run monitoring integration tests
docker compose run --rm monitoring-tests

# Run Phase 5 bug tracking tests
docker compose run --rm bug-tracking-tests

# Run Phase 5 failure analysis tests
docker compose run --rm failure-analysis-tests

# Run full system integration tests
docker compose run --rm full-system-tests
```


## Architecture Overview

The GeoTest Framework implements a 6-phase architecture:

- **Phase 1**: Foundation with fast feedback (basic tests, CI/CD, Docker)
- **Phase 2**: Data-driven testing (CSV data, parameterized tests, models)
- **Phase 3**: Containerization and CI/CD (Docker Compose, Allure, health checks)
- **Phase 4**: Advanced monitoring & reporting (metrics, alerts, dashboards)
- **Phase 5**: Intelligent bug tracking (GitHub Issues API, failure analysis, retry mechanisms)
- **Phase 6**: Performance testing (load testing, stress testing, benchmarking)

Each phase builds upon the previous, creating a comprehensive testing platform suitable for enterprise environments.
