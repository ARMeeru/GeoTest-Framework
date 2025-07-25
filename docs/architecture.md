# GeoTest Framework Architecture

## System Overview

The GeoTest Framework is designed as a scalable, data-driven API testing system with clear separation of concerns and modular components.

```
┌─────────────────────────────────────────────────────────────────┐
│                    GeoTest Framework Architecture               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Test Runner   │    │   Data Layer    │    │   Reporting     │
│                 │    │                 │    │                 │
│ • pytest        │    │ • CSV Test Data │    │ • HTML Reports  │
│ • Markers       │    │ • Pydantic      │    │ • JSON Reports  │
│ • Fixtures      │    │ • Validation    │    │ • Allure (P4)   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   Test Framework Core   │
                    │                         │
                    │ • RestCountriesClient   │
                    │ • Request/Response Log  │
                    │ • Error Handling        │
                    │ • Rate Limiting         │
                    └─────────────┬───────────┘
                                  │
                     ┌────────────┴────────────┐
                     │     External API        │
                     │                         │
                     │ • REST Countries v3.1   │
                     │ • HTTP/JSON             │
                     │ • Rate Limited          │
                     └─────────────────────────┘
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
- **Execution Timing**: Performance monitoring
- **Future**: Allure reports with history (Phase 4)

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
- Future-ready for containerization (Phase 3)

## Future Phases Integration

- **Phase 3**: Docker containers wrap entire architecture
- **Phase 4**: Allure reporting enhances reporting layer
- **Phase 5**: GitHub Issues integration adds automation layer
- **Phase 6**: Performance testing adds monitoring layer
- **Phase 7**: Advanced features add chaos/contract testing
- **Phase 8**: Production polish adds optimization layer

This architecture ensures maintainability, scalability, and professional QA practices throughout all development phases.