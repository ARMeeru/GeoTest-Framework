# GeoTest Framework Architecture

## System Overview

The GeoTest Framework is designed as a scalable, data-driven API testing system with clear separation of concerns and modular components.

```
┌─────────────────────────────────────────────────────────────────┐
│                    GeoTest Framework Architecture               │
│                        (Phase 4 Complete)                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Docker Container Layer                      │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │   Test Runner   │    │   Data Layer    │    │  Reporting  │  │
│  │                 │    │                 │    │             │  │
│  │ • pytest        │    │ • CSV Test Data │    │ • HTML      │  │
│  │ • Markers       │    │ • Pydantic      │    │ • JSON      │  │
│  │ • Fixtures      │    │ • Validation    │    │ • Dashboard │  │
│  │ • Plugins       │    │ • Test Models   │    │ • Alerts    │  │
│  └─────────┬───────┘    └─────────┬───────┘    └─────────┬───┘  │
│            │                      │                      │      │
│            │        ┌─────────────┴─────────────┐        │      │
│            └────────┤   Phase 4: Monitoring     ├────────┘      │
│                     │                           │               │
│                     │ • Real-time Metrics       │               │
│                     │ • Alert Management        │               │
│                     │ • Performance Tracking    │               │
│                     │ • System Monitoring       │               │
│                     │ • Interactive Dashboards  │               │
│                     │ • Notification Channels   │               │
│                     └─────────────┬─────────────┘               │
│                                   │                             │
│                      ┌────────────┴────────────┐                │
│                      │   Test Framework Core   │                │
│                      │                         │                │
│                      │ • RestCountriesClient   │                │
│                      │ • Request/Response Log  │                │
│                      │ • Error Handling        │                │
│                      │ • Rate Limiting         │                │
│                      └─────────────┬───────────┘                │
└────────────────────────────────────┼────────────────────────────┘
                                     │
                      ┌──────────────┴──────────────┐
                      │        External API         │
                      │                             │
                      │ • REST Countries v3.1       │
                      │ • HTTP/JSON                 │
                      │ • Rate Limited              │
                      └─────────────────────────────┘
```

## Component Details

### 1. Test Runner Layer
- **pytest**: Test execution engine with custom configuration
- **Markers**: smoke, regression, critical, performance, slow, integration, unit
- **Fixtures**: Data loading, client setup, cleanup, monitoring integration
- **Plugins**: Monitoring plugin, performance plugin for enhanced tracking
- **Parameterization**: Data-driven test execution

### 2. Data Layer (Phase 2)
- **CSV Test Data**: Human-readable test scenarios
- **Pydantic Models**: Data validation and type safety
- **Test Data Management**: Fixtures for data loading
- **Edge Case Scenarios**: Special characters, invalid codes

### 3. Test Framework Core
- **RestCountriesClient**: API wrapper with retry logic
- **Request/Response Logging**: Detailed debugging information
- **Error Handling**: Graceful failure management
- **Rate Limiting**: Respectful API usage

### 4. Reporting Layer (Phase 4 Enhanced)
- **HTML Reports**: Human-readable test results
- **JSON Reports**: Machine-readable for CI/CD
- **Interactive Dashboards**: Real-time visualization with Chart.js
- **Alert Reports**: Notification status and history
- **Performance Metrics**: Detailed timing and resource usage
- **System Monitoring**: CPU, memory, network, disk metrics

### 5. External Dependencies
- **REST Countries API**: Primary data source
- **HTTP Protocol**: RESTful communication
- **JSON Format**: Structured data exchange

## Data Flow

```
CSV Test Data → Pydantic Validation → Pytest Fixtures → Test Cases
                                                            ↓
                                               API Client (Monitored)
                                                            ↓
                                                  REST Countries API
                                                            ↓
                                              Response Processing
                                                            ↓
                                        Assertions & Logging & Metrics Collection
                                                            ↓
                        Alert Processing ← Test Reports (HTML/JSON/Dashboard)
                                ↓                          ↓
                        Notifications              System Monitoring
                    (Email/Slack/Webhook)          (CPU/Memory/Network)
```

## Phase 2 Enhancements

### Data-Driven Architecture
1. **CSV-Based Test Data**: Easily maintainable test scenarios
2. **Pydantic Models**: Type-safe data structures
3. **Parameterized Tests**: Efficient test coverage
4. **Negative Testing**: Comprehensive error scenarios

### Scalability Features
- Modular design for easy extension
- Clear separation between data, logic, and presentation
- Configurable test execution (smoke vs regression)
- Containerized deployment for consistent environments

## Phase 3: Containerization & CI/CD Foundation (Completed)

### Docker Integration
- **Multi-stage Dockerfile**: Optimized builds with runtime isolation
- **Docker Compose Services**: Organized test execution (smoke, regression, all-tests)
- **Volume Mounting**: Persistent reports and results
- **Health Checks**: API connectivity validation
- **Non-root Security**: geotest user for secure container execution

### CI/CD Ready Features
- **Containerized Execution**: Consistent environments across development/CI
- **Service Orchestration**: Multiple test configurations via Docker Compose
- **Report Generation**: HTML, JSON, and enhanced dashboard output in mounted volumes
- **Monitoring Integration**: Phase 4 monitoring system ready for CI/CD pipelines

## Phase 4: Advanced Monitoring & Reporting (Completed)

### Monitoring Architecture
- **MetricsCollector**: Thread-safe collection of test, system, API, and custom metrics
- **AlertManager**: Configurable rule-based alerting with multiple notification channels
- **DashboardGenerator**: Interactive HTML dashboards with real-time charts
- **SystemMonitor**: Continuous monitoring of system resources (CPU, memory, disk, network)
- **Performance Tracking**: Test execution timing and API response monitoring

### Key Components
1. **Real-time Metrics Collection**
   - Test execution metrics (duration, status, response times)
   - System resource monitoring (CPU, memory, network I/O)
   - API performance tracking (endpoint response times, status codes)
   - Custom business metrics with tags and metadata

2. **Intelligent Alerting System**
   - Configurable alert rules with severity levels
   - Multiple notification channels (Email, Slack, Webhooks)
   - Alert cooldown periods and automatic resolution
   - Alert history and analytics

3. **Interactive Dashboards**
   - Chart.js-powered visualizations
   - Test execution timelines and trends
   - System performance graphs
   - Alert status and history displays

4. **CLI Management Tools**
   - Monitor CLI for dashboard generation and metrics export
   - Alert management and resolution commands
   - Configuration validation and health checks
   - Continuous monitoring capabilities

### Integration Points
- **Pytest Plugin Integration**: Seamless monitoring without test code changes
- **Docker Compose Services**: New monitoring-focused test services (critical-tests, performance-tests, monitoring-tests)
- **CI/CD Ready**: Metrics export and dashboard generation for build artifacts
- **Backwards Compatible**: All existing functionality preserved

## Future Phases Integration

- **Phase 5**: GitHub Issues integration adds automation layer
- **Phase 6**: Performance testing builds on Phase 4 monitoring foundation
- **Phase 7**: Advanced features adds chaos/contract testing with monitoring
- **Phase 8**: Production polish adds optimization layer

This architecture ensures maintainability, scalability, and professional QA practices throughout all development phases. Phase 4 provides the foundation for advanced monitoring and alerting that will enhance all future phases.