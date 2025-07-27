# GeoTest Framework

An API testing framework using REST Countries API that demonstrates modern QA engineering practices.

## Running Tests

### Local Python Execution

```bash
# Activate virtual environment first after creating and rename myenv below accordingly
source myenv/bin/activate

# Install/update dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run smoke tests only
pytest -m smoke

# Run regression tests only  
pytest -m regression

# Run critical tests only
pytest -m critical

# Run performance tests
pytest -m performance

# Run monitoring integration tests
pytest tests/test_monitoring_integration.py -v
```

### Phase 4 Monitoring Commands

```bash
# Run tests with full monitoring enabled
python scripts/run_tests_with_monitoring.py

# Start system monitoring
python scripts/monitor.py monitor --duration 30

# Generate monitoring dashboard
python scripts/monitor.py dashboard

# Show metrics summary
python scripts/monitor.py summary

# Check recent alerts
python scripts/monitor.py alerts

# Export metrics data
python scripts/monitor.py export

# Validate alert configuration
python scripts/monitor.py check-config

# Test notification channels
python scripts/monitor.py test-notifications

# View monitoring CLI help
python scripts/monitor.py --help
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
```


## Architecture Overview

The GeoTest Framework implements a 4-phase architecture:

- **Phase 1**: Foundation with fast feedback (basic tests, CI/CD, Docker)
- **Phase 2**: Data-driven testing (CSV data, parameterized tests, models)
- **Phase 3**: Containerization and CI/CD (Docker Compose, Allure, health checks)
- **Phase 4**: Advanced monitoring & reporting (metrics, alerts, dashboards)

Each phase builds upon the previous, creating a comprehensive testing platform suitable for enterprise environments.