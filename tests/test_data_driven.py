# Data-driven tests using CSV test scenarios and parameterized testing
# Phase 2: Scalable test data management

import pytest
import time
import requests
from typing import List, Dict, Any
from src.api_client import RestCountriesClient
from src.models import ScenarioData, validate_country_data, validate_country_list


class TestDataDrivenScenarios:
    # Data-driven test class using CSV scenarios

    def test_smoke_scenarios_execution(self, smoke_scenarios: List[ScenarioData], 
                                     api_client: RestCountriesClient, 
                                     request_response_capture, test_result_tracker):
        # Execute all smoke test scenarios from CSV data
        failed_scenarios = []
        
        for scenario in smoke_scenarios:
            start_time = time.time()
            test_passed = False
            error_message = None
            
            try:
                # Execute API call based on scenario endpoint
                if scenario.endpoint == "alpha":
                    response_data = api_client.get_country_by_code(scenario.input_value)
                elif scenario.endpoint == "name":
                    response_data = api_client.get_country_by_name(scenario.input_value, full_text=True)
                elif scenario.endpoint == "region":
                    response_data = api_client.get_countries_by_region(scenario.input_value)
                elif scenario.endpoint == "currency":
                    response_data = api_client.get_countries_by_currency(scenario.input_value)
                elif scenario.endpoint == "capital":
                    response_data = api_client.get_countries_by_capital(scenario.input_value)
                elif scenario.endpoint == "all":
                    response_data = api_client.get_all_countries(fields=scenario.input_value)
                else:
                    continue  # Skip unsupported endpoints
                
                # Validate response based on expected result
                if scenario.expected_result == "success":
                    assert isinstance(response_data, list), f"Expected list response for {scenario.test_name}"
                    assert len(response_data) > 0, f"Expected non-empty response for {scenario.test_name}"
                    
                    # Validate data structure using Pydantic models
                    validated_countries = validate_country_list(response_data)
                    assert len(validated_countries) > 0, f"No valid countries found for {scenario.test_name}"
                    
                test_passed = True
                
            except requests.exceptions.HTTPError as e:
                if scenario.expected_result == "error":
                    test_passed = True
                    error_message = str(e)
                else:
                    error_message = f"Unexpected HTTP error: {str(e)}"
                    test_passed = False
            except Exception as e:
                error_message = f"Test execution error: {str(e)}"
                test_passed = False
            
            finally:
                execution_time = time.time() - start_time
                
                # Capture request/response data
                request_response_capture(
                    test_id=scenario.test_id,
                    request_url=f"{api_client.BASE_URL}/{scenario.endpoint}/{scenario.input_value}",
                    response_status=200 if test_passed else 400,
                    response_time=execution_time,
                    error=error_message
                )
                
                # Track test result
                test_result_tracker(
                    test_id=scenario.test_id,
                    status="passed" if test_passed else "failed",
                    execution_time=execution_time,
                    error_message=error_message
                )
                
                # Collect failed scenarios
                if not test_passed:
                    failed_scenarios.append(f"{scenario.test_name}: {error_message}")
        
        # Assert overall result
        if failed_scenarios:
            pytest.fail(f"Failed smoke scenarios: {failed_scenarios}")

    def test_negative_scenarios_execution(self, negative_scenarios: List[ScenarioData], 
                                        api_client: RestCountriesClient,
                                        request_response_capture, test_result_tracker):
        # Execute all negative test scenarios expecting errors
        failed_scenarios = []
        
        for scenario in negative_scenarios:
            start_time = time.time()
            test_passed = False
            error_message = None
            
            try:
                # Execute API call that should fail
                if scenario.endpoint == "alpha":
                    api_client.get_country_by_code(scenario.input_value)
                elif scenario.endpoint == "name":
                    if scenario.input_value == "null":
                        api_client.get_country_by_name(None)
                    elif scenario.input_value == "":
                        api_client.get_country_by_name("")
                    else:
                        api_client.get_country_by_name(scenario.input_value)
                elif scenario.endpoint == "region":
                    api_client.get_countries_by_region(scenario.input_value)
                elif scenario.endpoint == "currency":
                    api_client.get_countries_by_currency(scenario.input_value)
                else:
                    continue  # Skip unsupported endpoints
                
                # If we reach here, the API call succeeded when it should have failed
                test_passed = False
                error_message = f"Expected error for {scenario.test_name} but API call succeeded"
                
            except (requests.exceptions.HTTPError, requests.exceptions.RequestException) as e:
                test_passed = True
                error_message = f"Expected error occurred: {str(e)}"
            except (ValueError, TypeError) as e:
                test_passed = True
                error_message = f"Expected validation error: {str(e)}"
            except Exception as e:
                test_passed = False
                error_message = f"Unexpected error type: {str(e)}"
            
            finally:
                execution_time = time.time() - start_time
                
                # Capture request/response data
                request_response_capture(
                    test_id=scenario.test_id,
                    request_url=f"{api_client.BASE_URL}/{scenario.endpoint}/{scenario.input_value}",
                    response_status=404 if test_passed else 200,
                    response_time=execution_time,
                    error=error_message
                )
                
                # Track test result
                test_result_tracker(
                    test_id=scenario.test_id,
                    status="passed" if test_passed else "failed",
                    execution_time=execution_time,
                    error_message=error_message if not test_passed else None
                )
                
                # Collect failed scenarios
                if not test_passed:
                    failed_scenarios.append(f"{scenario.test_name}: {error_message}")
        
        # Assert overall result
        if failed_scenarios:
            pytest.fail(f"Failed negative scenarios: {failed_scenarios}")

    def test_edge_case_scenarios_execution(self, edge_case_scenarios: List[ScenarioData], 
                                         api_client: RestCountriesClient,
                                         request_response_capture, test_result_tracker):
        # Execute all edge case scenarios with special characters and formats
        failed_scenarios = []
        
        for scenario in edge_case_scenarios:
            start_time = time.time()
            test_passed = False
            error_message = None
            
            try:
                # Execute API call for edge cases
                if scenario.endpoint == "name":
                    response_data = api_client.get_country_by_name(scenario.input_value)
                elif scenario.endpoint == "capital":
                    response_data = api_client.get_countries_by_capital(scenario.input_value)
                elif scenario.endpoint == "alpha":
                    response_data = api_client.get_country_by_code(scenario.input_value)
                else:
                    continue  # Skip unsupported endpoints
                
                # Validate response based on expected result
                if scenario.expected_result == "success":
                    assert isinstance(response_data, list), f"Expected list response for {scenario.test_name}"
                    assert len(response_data) > 0, f"Expected non-empty response for {scenario.test_name}"
                    
                    # Validate special character handling
                    validated_countries = validate_country_list(response_data)
                    assert len(validated_countries) > 0, f"No valid countries found for {scenario.test_name}"
                    
                    test_passed = True
                else:
                    test_passed = False
                    error_message = f"Expected failure for edge case {scenario.test_name} but API succeeded"
                    
            except requests.exceptions.HTTPError as e:
                if scenario.expected_result == "error":
                    test_passed = True
                    error_message = f"Expected error for edge case: {str(e)}"
                else:
                    test_passed = False
                    error_message = f"Unexpected HTTP error in edge case: {str(e)}"
            except Exception as e:
                if scenario.expected_result == "error":
                    test_passed = True
                    error_message = f"Expected error for edge case: {str(e)}"
                else:
                    test_passed = False
                    error_message = f"Unexpected error in edge case: {str(e)}"
            
            finally:
                execution_time = time.time() - start_time
                
                # Capture request/response data
                request_response_capture(
                    test_id=scenario.test_id,
                    request_url=f"{api_client.BASE_URL}/{scenario.endpoint}/{scenario.input_value}",
                    response_status=200 if test_passed else 400,
                    response_time=execution_time,
                    error=error_message
                )
                
                # Track test result
                test_result_tracker(
                    test_id=scenario.test_id,
                    status="passed" if test_passed else "failed",
                    execution_time=execution_time,
                    error_message=error_message if not test_passed else None
                )
                
                # Collect failed scenarios
                if not test_passed:
                    failed_scenarios.append(f"{scenario.test_name}: {error_message}")
        
        # Assert overall result
        if failed_scenarios:
            pytest.fail(f"Failed edge case scenarios: {failed_scenarios}")


class TestParameterizedCountries:
    # Parameterized tests using sample country data

    def test_sample_countries_validation(self, sample_countries: List[Dict[str, str]], 
                                       api_client: RestCountriesClient):
        # Test subset of countries for validation (limit to 10 for speed)
        test_countries = sample_countries[:10]
        failed_countries = []
        
        for country in test_countries:
            try:
                country_name = country['country_name']
                expected_alpha2 = country['alpha2']
                expected_alpha3 = country['alpha3']
                
                # Get country by name
                response_data = api_client.get_country_by_name(country_name)
                
                # Validate response
                assert isinstance(response_data, list), f"Expected list response for {country_name}"
                assert len(response_data) > 0, f"No data found for {country_name}"
                
                # Find matching country (might be multiple results)
                matching_country = None
                for country_data in response_data:
                    if (country_data.get('cca2') == expected_alpha2 or 
                        country_data.get('cca3') == expected_alpha3):
                        matching_country = country_data
                        break
                
                assert matching_country is not None, f"Could not find {country_name} in response"
                
                # Validate data structure
                validated_country = validate_country_data(matching_country)
                assert validated_country.cca2 == expected_alpha2, f"Alpha-2 mismatch for {country_name}"
                assert validated_country.cca3 == expected_alpha3, f"Alpha-3 mismatch for {country_name}"
                
            except Exception as e:
                failed_countries.append(f"{country_name}: {str(e)}")
        
        # Assert overall result
        if failed_countries:
            pytest.fail(f"Failed country validations: {failed_countries}")

    def test_alpha_code_consistency(self, sample_countries: List[Dict[str, str]], 
                                  api_client: RestCountriesClient):
        # Test alpha-2 and alpha-3 code consistency for subset of countries
        test_countries = sample_countries[:8]  # Test 8 countries for speed
        failed_countries = []
        
        for country in test_countries:
            try:
                alpha2 = country['alpha2']
                alpha3 = country['alpha3']
                expected_name = country['country_name']
                
                # Test alpha-2 lookup
                response_alpha2 = api_client.get_country_by_code(alpha2)
                assert isinstance(response_alpha2, list), f"Expected list response for alpha-2 {alpha2}"
                assert len(response_alpha2) == 1, f"Expected exactly 1 result for alpha-2 {alpha2}"
                
                country_data_alpha2 = response_alpha2[0]
                validated_alpha2 = validate_country_data(country_data_alpha2)
                
                # Test alpha-3 lookup
                response_alpha3 = api_client.get_country_by_code(alpha3)
                assert isinstance(response_alpha3, list), f"Expected list response for alpha-3 {alpha3}"
                assert len(response_alpha3) == 1, f"Expected exactly 1 result for alpha-3 {alpha3}"
                
                country_data_alpha3 = response_alpha3[0]
                validated_alpha3 = validate_country_data(country_data_alpha3)
                
                # Verify both lookups return the same country
                assert validated_alpha2.cca2 == validated_alpha3.cca2, f"Alpha-2 and Alpha-3 lookups should return same country"
                assert validated_alpha2.name.common == validated_alpha3.name.common, f"Country names should match"
                
            except Exception as e:
                failed_countries.append(f"{alpha2}/{alpha3}: {str(e)}")
        
        # Assert overall result
        if failed_countries:
            pytest.fail(f"Failed alpha code consistency tests: {failed_countries}")

    @pytest.mark.regression
    def test_country_data_completeness_subset(self, sample_countries: List[Dict[str, str]], 
                                            api_client: RestCountriesClient):
        # Test data completeness for subset of sample countries
        test_countries = sample_countries[:5]  # Test 5 countries for comprehensive validation
        failed_countries = []
        
        for country in test_countries:
            try:
                alpha2 = country['alpha2']
                expected_region = country['region']
                expected_subregion = country['subregion']
                min_population = int(country['population_min'])
                
                # Get country data
                response_data = api_client.get_country_by_code(alpha2)
                country_data = response_data[0]
                validated_country = validate_country_data(country_data)
                
                # Validate regional information
                assert validated_country.region == expected_region, f"Region mismatch for {alpha2}"
                assert validated_country.subregion == expected_subregion, f"Subregion mismatch for {alpha2}"
                
                # Validate population (should be at least the minimum expected)
                if validated_country.population:
                    assert validated_country.population >= min_population, \
                        f"Population for {alpha2} is {validated_country.population}, expected at least {min_population}"
                
                # Validate required fields exist
                assert validated_country.capital is not None, f"Capital missing for {alpha2}"
                assert len(validated_country.capital) > 0, f"Capital list empty for {alpha2}"
                assert validated_country.currencies is not None, f"Currencies missing for {alpha2}"
                assert validated_country.languages is not None, f"Languages missing for {alpha2}"
                
            except Exception as e:
                failed_countries.append(f"{alpha2}: {str(e)}")
        
        # Assert overall result
        if failed_countries:
            pytest.fail(f"Failed data completeness tests: {failed_countries}")