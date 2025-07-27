# Multi-stage Docker build for GeoTest Framework
# Stage 1: Build dependencies
FROM python:3.12-slim as builder

# Install minimal build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -name "__pycache__" -type d -exec rm -rf {} + || true

# Stage 2: Runtime image
FROM python:3.12-slim as runtime

# Install only essential runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy optimized virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r geotest && useradd -r -g geotest -d /app geotest

# Set working directory and create reports
WORKDIR /app
RUN mkdir -p reports .pytest_cache/v/cache allure-results && chown -R geotest:geotest reports .pytest_cache allure-results

# Copy application code (minimal files only)
COPY --chown=geotest:geotest src/ ./src/
COPY --chown=geotest:geotest tests/ ./tests/
COPY --chown=geotest:geotest data/ ./data/
COPY --chown=geotest:geotest conftest.py ./
COPY --chown=geotest:geotest pytest.ini ./

# Create config directory and add default alerting configuration
RUN mkdir -p config && \
    echo '{"alert_rules":[],"notification_channels":[]}' > config/alerting.json && \
    chown -R geotest:geotest config

# Switch to non-root user
USER geotest

# Create writable cache directory for pytest
RUN mkdir -p /tmp/.pytest_cache && chown geotest:geotest /tmp/.pytest_cache

# Set environment variables
ENV PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('https://restcountries.com/v3.1/all', timeout=5)" || exit 1

# Default command - run smoke tests
CMD ["pytest", "-m", "smoke", "--html=reports/report.html", "--self-contained-html", "--json-report", "--json-report-file=reports/report.json", "-v"]