# GeoTest Framework

An API testing framework using REST Countries API that demonstrates modern QA engineering practices.

## Running Tests

### Local Python Execution

```bash
# Activate virtual environment first
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

# Run specific test file
pytest tests/test_countries_api.py

# Run data-driven tests
pytest tests/test_data_driven.py

# Generate HTML report
pytest --html=reports/report.html --self-contained-html

# Generate JSON report
pytest --json-report --json-report-file=reports/report.json
```

### Docker Execution

```bash
# Build Docker image
docker build -t geotest-framework .

# Run smoke tests in container
docker compose run --rm smoke-tests

# Run regression tests in container
docker compose run --rm regression-tests

# Run all tests in container
docker compose run --rm all-tests

# Run specific test with Docker
docker run --rm -v $(pwd)/reports:/app/reports geotest-framework pytest -m smoke
```

### Advanced Options

```bash
# Run tests with coverage
pytest --cov=src --cov-report=html

# Run with timing information
pytest --durations=10

# Run with custom markers
pytest -m "smoke or regression"

# Run and generate all reports
pytest --html=reports/report.html --json-report --json-report-file=reports/report.json -v
```