# GeoTest Framework Architecture

## System Overview

The GeoTest Framework is designed as a scalable, data-driven API testing system with clear separation of concerns and modular components.

```
┌─────────────────────────────────────────────────────────────────┐
│                    GeoTest Framework Architecture               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Docker Container Layer                      │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │   Test Runner   │    │   Data Layer    │    │  Reporting  │  │
│  │                 │    │                 │    │             │  │
│  │ • pytest        │    │ • CSV Test Data │    │ • HTML      │  │
│  │ • Markers       │    │ • Pydantic      │    │ • JSON      │  │
│  │ • Fixtures      │    │ • Validation    │    │ • Allure    │  │
│  └─────────┬───────┘    └─────────┬───────┘    └─────────┬───┘  │
│            │                      │                      │      │
│            │                      │                      │      │
│            └──────────────────────┼──────────────────────┘      │
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
- **Markers**: smoke, regression, slow, integration, unit
- **Fixtures**: Data loading, client setup, cleanup
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

### 4. Reporting Layer
- **HTML Reports**: Human-readable test results
- **JSON Reports**: Machine-readable for CI/CD
- **Allure Reports**: Advanced reporting with history tracking
- **Execution Timing**: Performance monitoring

### 5. External Dependencies
- **REST Countries API**: Primary data source
- **HTTP Protocol**: RESTful communication
- **JSON Format**: Structured data exchange

## Data Flow

```
CSV Test Data → Pydantic Validation → Pytest Fixtures → Test Cases
                                                            ↓
                                                    API Client
                                                            ↓
                                                  REST Countries API
                                                            ↓
                                              Response Processing
                                                            ↓
                                               Assertions & Logging
                                                            ↓
                                              Test Reports (HTML/JSON)
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
- **Report Generation**: HTML, JSON, and Allure output in mounted volumes
- **Future Allure Service**: Ready for Phase 4 advanced reporting

## Future Phases Integration

- **Phase 4**: Allure service integration for advanced reporting
- **Phase 5**: GitHub Issues integration adds automation layer
- **Phase 6**: Performance testing adds monitoring layer
- **Phase 7**: Advanced features add chaos/contract testing
- **Phase 8**: Production polish adds optimization layer

This architecture ensures maintainability, scalability, and professional QA practices throughout all development phases.