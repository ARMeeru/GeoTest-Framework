# Test suite for REST Countries API that contains 10 passing tests and 1 intentionally failing test

import pytest
import logging
from typing import List, Dict, Any
from src.api_client import RestCountriesClient

class TestRestCountriesAPI:
    # Test cases for REST Countries API endpoints
    
    @pytest.fixture(scope="class")
    def api_client(self):
        # Fixture to provide API client instance
        client = RestCountriesClient()
        yield client
        client.close()
    
    @pytest.mark.smoke
    def test_get_all_countries_returns_data(self, api_client: RestCountriesClient):
        # Test 1: Verify all countries endpoint returns data
        # Use fields parameter to avoid 400 error
        countries = api_client.get_all_countries(fields="name,cca2,cca3,region")
        
        assert isinstance(countries, list), "Response should be a list"
        assert len(countries) > 0, "Should return at least one country"
        assert len(countries) > 190, f"Expected at least 190 countries, got {len(countries)}"
        
        # Verify first country has required fields
        first_country = countries[0]
        required_fields = ['name', 'cca2', 'cca3', 'region']
        for field in required_fields:
            assert field in first_country, f"Country should have '{field}' field"
    
    @pytest.mark.smoke
    def test_get_country_by_name_united_states(self, api_client: RestCountriesClient):
        # Test 2: Get United States by name with exact match
        # Use fullText=True to get exact match for "United States"
        countries = api_client.get_country_by_name("United States", full_text=True)
        
        assert len(countries) == 1, f"Expected 1 country, got {len(countries)}"
        country = countries[0]
        
        assert country['name']['common'] == "United States", f"Expected 'United States', got {country['name']['common']}"
        assert country['cca2'] == "US", f"Expected 'US', got {country['cca2']}"
        assert country['cca3'] == "USA", f"Expected 'USA', got {country['cca3']}"
        assert country['region'] == "Americas", f"Expected 'Americas', got {country['region']}"
    
    @pytest.mark.smoke
    def test_get_country_by_alpha2_code(self, api_client: RestCountriesClient):
        # Test 3: Get country by 2-letter alpha code
        countries = api_client.get_country_by_code("DE")
        
        assert len(countries) == 1, f"Expected 1 country, got {len(countries)}"
        country = countries[0]
        
        assert country['cca2'] == "DE", f"Expected 'DE', got {country['cca2']}"
        assert country['name']['common'] == "Germany", f"Expected 'Germany', got {country['name']['common']}"
        assert "EUR" in country.get('currencies', {}), "Germany should use EUR currency"
    
    @pytest.mark.regression
    def test_get_country_by_alpha3_code(self, api_client: RestCountriesClient):
        # Test 4: Get country by 3-letter alpha code
        countries = api_client.get_country_by_code("JPN")
        
        assert len(countries) == 1, f"Expected 1 country, got {len(countries)}"
        country = countries[0]
        
        assert country['cca3'] == "JPN", f"Expected 'JPN', got {country['cca3']}"
        assert country['name']['common'] == "Japan", f"Expected 'Japan', got {country['name']['common']}"
        assert country['region'] == "Asia", f"Expected 'Asia', got {country['region']}"
    
    @pytest.mark.regression
    def test_get_countries_by_currency_usd(self, api_client: RestCountriesClient):
        # Test 5: Get countries that use USD currency
        countries = api_client.get_countries_by_currency("USD")
        
        assert len(countries) > 0, "Should find countries using USD"
        
        # Verify at least USA is in the results
        usa_found = False
        for country in countries:
            if country['cca2'] == 'US':
                usa_found = True
                break
        
        assert usa_found, "United States should be in USD currency results"
        
        # Verify all countries have USD in their currencies
        for country in countries:
            currencies = country.get('currencies', {})
            assert 'USD' in currencies, f"Country {country['name']['common']} should have USD currency"
    
    @pytest.mark.regression
    def test_get_countries_by_language_alternative(self, api_client: RestCountriesClient):
        # Test 6: Verify English-speaking countries by checking language field (alternative to lang endpoint)
        all_countries = api_client.get_all_countries(fields="name,cca2,languages")
        
        # Filter countries that have English as a language
        english_countries = []
        for country in all_countries:
            languages = country.get('languages', {})
            if 'eng' in languages and languages['eng'] == 'English':
                english_countries.append(country)
        
        assert len(english_countries) > 0, "Should find countries using English"
        
        # Verify common English-speaking countries are present
        expected_countries = ['US', 'GB', 'CA', 'AU']
        found_codes = [country['cca2'] for country in english_countries]
        
        for code in expected_countries:
            assert code in found_codes, f"Expected {code} in English-speaking countries"
    
    @pytest.mark.regression
    def test_get_countries_by_region_europe(self, api_client: RestCountriesClient):
        # Test 7: Get countries in Europe region
        countries = api_client.get_countries_by_region("Europe")
        
        assert len(countries) > 0, "Should find countries in Europe"
        assert len(countries) > 40, f"Expected at least 40 European countries, got {len(countries)}"
        
        # Verify all countries are in Europe
        for country in countries:
            assert country['region'] == "Europe", f"Country {country['name']['common']} should be in Europe region"
        
        # Verify some major European countries are present
        expected_countries = ['DE', 'FR', 'IT', 'ES']
        found_codes = [country['cca2'] for country in countries]
        
        for code in expected_countries:
            assert code in found_codes, f"Expected {code} in European countries"
    
    @pytest.mark.regression
    def test_get_countries_by_subregion_northern_europe(self, api_client: RestCountriesClient):
        # Test 8: Get countries in Northern Europe subregion
        countries = api_client.get_countries_by_subregion("Northern Europe")
        
        assert len(countries) > 0, "Should find countries in Northern Europe"
        
        # Verify all countries are in Northern Europe subregion
        for country in countries:
            assert country['subregion'] == "Northern Europe", \
                f"Country {country['name']['common']} should be in Northern Europe subregion"
        
        # Verify some Northern European countries are present
        expected_countries = ['GB', 'SE', 'NO', 'DK']
        found_codes = [country['cca2'] for country in countries]
        
        for code in expected_countries:
            assert code in found_codes, f"Expected {code} in Northern European countries"
    
    @pytest.mark.regression
    def test_get_countries_by_capital_london(self, api_client: RestCountriesClient):
        # Test 9: Get country by capital city (London)
        countries = api_client.get_countries_by_capital("London")
        
        assert len(countries) == 1, f"Expected 1 country with capital London, got {len(countries)}"
        country = countries[0]
        
        assert country['cca2'] == "GB", f"Expected 'GB', got {country['cca2']}"
        assert country['name']['common'] == "United Kingdom", \
            f"Expected 'United Kingdom', got {country['name']['common']}"
        assert "London" in country['capital'], f"Capital should include London"
    
    @pytest.mark.regression
    def test_country_data_structure_completeness(self, api_client: RestCountriesClient):
        # Test 10: Verify country data structure has expected fields
        countries = api_client.get_country_by_name("Canada")
        
        assert len(countries) == 1, "Should find exactly one Canada"
        country = countries[0]
        
        # Check required top-level fields
        required_fields = [
            'name', 'cca2', 'cca3', 'region', 'subregion',
            'capital', 'population', 'area', 'flag', 'flags'
        ]
        
        for field in required_fields:
            assert field in country, f"Country should have '{field}' field"
        
        # Check name structure
        assert 'common' in country['name'], "Name should have 'common' field"
        assert 'official' in country['name'], "Name should have 'official' field"
        
        # Check numeric fields are positive
        assert country['population'] > 0, "Population should be positive"
        assert country['area'] > 0, "Area should be positive"
        
        # Check capital is a list
        assert isinstance(country['capital'], list), "Capital should be a list"
        assert len(country['capital']) > 0, "Capital list should not be empty"
    
    @pytest.mark.regression
    def test_intentional_failure_for_error_reporting(self, api_client: RestCountriesClient):
        # Test 11: Intentionally failing test to validate error reporting
        countries = api_client.get_country_by_name("United States", full_text=True)
        country = countries[0]
        
        # This assertion will intentionally fail to test error reporting
        assert country['name']['common'] == "Canada", \
            f"Intentional failure: Expected 'Canada' but got '{country['name']['common']}'. " \
            "This test is designed to fail to validate error reporting mechanisms."