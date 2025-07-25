# Pytest configuration and fixtures for data-driven testing
# Provides reusable test data and client setup

import pytest
import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Iterator
from src.api_client import RestCountriesClient
from src.models import ScenarioData, CountryData, TestResult, validate_test_scenario


# Configure logging for test fixtures
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def api_client() -> Iterator[RestCountriesClient]:
    # Session-scoped API client with proper cleanup
    client = RestCountriesClient()
    logger.info("API client created for test session")
    yield client
    client.close()
    logger.info("API client closed after test session")


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    # Path to test data directory
    return Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def test_scenarios(test_data_dir: Path) -> List[ScenarioData]:
    # Load and validate test scenarios from CSV
    scenarios_file = test_data_dir / "test_scenarios.csv"
    scenarios = []
    
    try:
        with open(scenarios_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):  # Start at 2 for header
                try:
                    scenario = validate_test_scenario(row)
                    scenarios.append(scenario)
                except Exception as e:
                    logger.warning(f"Invalid test scenario at row {row_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(scenarios)} test scenarios from CSV")
        return scenarios
        
    except FileNotFoundError:
        logger.error(f"Test scenarios file not found: {scenarios_file}")
        return []
    except Exception as e:
        logger.error(f"Error loading test scenarios: {e}")
        return []


@pytest.fixture(scope="session")
def sample_countries(test_data_dir: Path) -> List[Dict[str, str]]:
    # Load sample country data for parameterized tests
    countries_file = test_data_dir / "countries_sample.csv"
    countries = []
    
    try:
        with open(countries_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            countries = list(reader)
        
        logger.info(f"Loaded {len(countries)} sample countries from CSV")
        return countries
        
    except FileNotFoundError:
        logger.error(f"Sample countries file not found: {countries_file}")
        return []
    except Exception as e:
        logger.error(f"Error loading sample countries: {e}")
        return []


@pytest.fixture
def smoke_scenarios(test_scenarios: List[ScenarioData]) -> List[ScenarioData]:
    # Filter scenarios for smoke testing
    smoke_tests = [s for s in test_scenarios if 'smoke' in (s.tags or '')]
    logger.info(f"Filtered {len(smoke_tests)} smoke test scenarios")
    return smoke_tests


@pytest.fixture
def regression_scenarios(test_scenarios: List[ScenarioData]) -> List[ScenarioData]:
    # Filter scenarios for regression testing
    regression_tests = [s for s in test_scenarios if 'regression' in (s.tags or '')]
    logger.info(f"Filtered {len(regression_tests)} regression test scenarios")
    return regression_tests


@pytest.fixture
def positive_scenarios(test_scenarios: List[ScenarioData]) -> List[ScenarioData]:
    # Filter positive test scenarios
    positive_tests = [s for s in test_scenarios if s.test_type == 'positive']
    logger.info(f"Filtered {len(positive_tests)} positive test scenarios")
    return positive_tests


@pytest.fixture
def negative_scenarios(test_scenarios: List[ScenarioData]) -> List[ScenarioData]:
    # Filter negative test scenarios
    negative_tests = [s for s in test_scenarios if s.test_type == 'negative']
    logger.info(f"Filtered {len(negative_tests)} negative test scenarios")
    return negative_tests


@pytest.fixture
def edge_case_scenarios(test_scenarios: List[ScenarioData]) -> List[ScenarioData]:
    # Filter edge case test scenarios
    edge_cases = [s for s in test_scenarios if s.test_type == 'edge_case']
    logger.info(f"Filtered {len(edge_cases)} edge case test scenarios")
    return edge_cases


@pytest.fixture
def security_scenarios(test_scenarios: List[ScenarioData]) -> List[ScenarioData]:
    # Filter security test scenarios
    security_tests = [s for s in test_scenarios if 'security' in (s.tags or '')]
    logger.info(f"Filtered {len(security_tests)} security test scenarios")
    return security_tests


@pytest.fixture
def high_priority_scenarios(test_scenarios: List[ScenarioData]) -> List[ScenarioData]:
    # Filter high priority test scenarios
    high_priority = [s for s in test_scenarios if s.priority == 'high']
    logger.info(f"Filtered {len(high_priority)} high priority test scenarios")
    return high_priority


@pytest.fixture
def request_response_capture():
    # Fixture for capturing request/response data for debugging
    captured_data = []
    
    def capture(test_id: str, request_url: str, response_status: int, 
                response_time: float, response_data: Any = None, error: str = None):
        # Capture request/response information
        capture_entry = {
            'test_id': test_id,
            'request_url': request_url,
            'response_status': response_status,
            'response_time': response_time,
            'response_data_sample': str(response_data)[:200] if response_data else None,
            'error': error,
            'timestamp': pytest.current_timestamp if hasattr(pytest, 'current_timestamp') else None
        }
        captured_data.append(capture_entry)
        logger.debug(f"Captured request/response for {test_id}")
    
    # Provide capture function to tests
    yield capture
    
    # Save captured data after test session
    if captured_data:
        try:
            reports_dir = Path(__file__).parent.parent / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            capture_file = reports_dir / "request_response_capture.json"
            with open(capture_file, 'w') as f:
                json.dump(captured_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(captured_data)} captured requests to {capture_file}")
        except Exception as e:
            logger.error(f"Failed to save captured request/response data: {e}")


@pytest.fixture
def test_result_tracker():
    # Fixture for tracking test results and performance
    test_results = []
    
    def track_result(test_id: str, status: str, execution_time: float,
                    request_url: str = None, response_status: int = None,
                    response_time: float = None, error_message: str = None):
        # Track test execution results
        result = TestResult(
            test_id=test_id,
            status=status,
            execution_time=execution_time,
            request_url=request_url,
            response_status=response_status,
            response_time=response_time,
            error_message=error_message
        )
        test_results.append(result)
        logger.debug(f"Tracked result for {test_id}: {status}")
    
    # Provide tracking function to tests
    yield track_result
    
    # Generate summary report after test session
    if test_results:
        try:
            reports_dir = Path(__file__).parent.parent / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            # Calculate summary statistics
            total_tests = len(test_results)
            passed_tests = len([r for r in test_results if r.status == 'passed'])
            failed_tests = len([r for r in test_results if r.status == 'failed'])
            avg_execution_time = sum(r.execution_time for r in test_results) / total_tests
            
            summary = {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'pass_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
                'average_execution_time': avg_execution_time,
                'test_results': [r.dict() for r in test_results]
            }
            
            summary_file = reports_dir / "test_execution_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Generated test execution summary: {passed_tests}/{total_tests} passed")
        except Exception as e:
            logger.error(f"Failed to generate test execution summary: {e}")


@pytest.fixture(autouse=True)
def test_timer():
    # Automatic fixture to time each test
    import time
    start_time = time.time()
    
    # Set current timestamp for other fixtures
    pytest.current_timestamp = start_time
    
    yield
    
    execution_time = time.time() - start_time
    pytest.last_execution_time = execution_time
    logger.debug(f"Test execution time: {execution_time:.3f}s")


# Parametrize helpers for data-driven testing
def get_country_test_data(countries: List[Dict[str, str]], max_count: int = 30) -> List[Dict[str, str]]:
    # Get subset of countries for parameterized testing
    return countries[:max_count]


def get_scenario_by_endpoint(scenarios: List[ScenarioData], endpoint: str) -> List[ScenarioData]:
    # Filter scenarios by specific endpoint
    return [s for s in scenarios if s.endpoint == endpoint]


def get_scenario_by_priority(scenarios: List[ScenarioData], priority: str) -> List[ScenarioData]:
    # Filter scenarios by priority level
    return [s for s in scenarios if s.priority == priority]