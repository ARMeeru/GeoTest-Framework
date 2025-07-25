# REST Countries API Client Wrapper

import logging
import time
from typing import Dict, List, Optional, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RestCountriesClient:
    # Client wrapper for REST Countries API
    
    BASE_URL = "https://restcountries.com/v3.1"
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        # Initialize the REST Countries API client
        # timeout: Request timeout in seconds
        # max_retries: Maximum number of retry attempts
        self.timeout = timeout
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        self.session.headers.update({
            'User-Agent': 'GeoTest-Framework/1.0',
            'Accept': 'application/json'
        })
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        # Make HTTP request to REST Countries API
        # endpoint: API endpoint path
        # params: Query parameters
        # Returns: Response data as dictionary
        # Raises: requests.RequestException if request fails
        url = f"{self.BASE_URL}/{endpoint}"
        start_time = time.time()
        
        try:
            self.logger.info(f"Making request to: {url}")
            if params:
                self.logger.info(f"Request parameters: {params}")
                
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            self.logger.info(f"Response received in {response_time:.3f}s - Status: {response.status_code}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            self.logger.error(f"Request failed after {response_time:.3f}s: {str(e)}")
            raise
    
    def get_all_countries(self, fields: Optional[str] = None) -> List[Dict[str, Any]]:
        # Get all countries
        # fields: Comma-separated list of fields to return (e.g., "name,cca2,cca3,region")
        # If None, returns all fields
        params = {"fields": fields} if fields else None
        return self._make_request("all", params)
    
    def get_country_by_name(self, name: str, full_text: bool = False) -> List[Dict[str, Any]]:
        # Get country by name
        # name: Country name to search for
        # full_text: Whether to match full text only
        # Returns: List of matching countries
        endpoint = f"name/{name}"
        params = {"fullText": "true"} if full_text else None
        return self._make_request(endpoint, params)
    
    def get_country_by_code(self, code: str) -> List[Dict[str, Any]]:
        # Get country by alpha code (2 or 3 letters)
        # code: Country alpha code (e.g., 'US', 'USA')
        # Returns: List containing the country data
        return self._make_request(f"alpha/{code}")
    
    def get_countries_by_currency(self, currency: str) -> List[Dict[str, Any]]:
        # Get countries by currency
        # currency: Currency code (e.g., 'USD', 'EUR')
        # Returns: List of countries using the currency
        return self._make_request(f"currency/{currency}")
    
    def get_countries_by_language(self, language: str) -> List[Dict[str, Any]]:
        # Get countries by language - Note: This endpoint appears to be unavailable in v3.1
        # language: Language code (e.g., 'en', 'es')
        # Returns: List of countries using the language
        # Raises: requests.RequestException if endpoint is not available
        # Note: The /lang/{language} endpoint seems to return 404 in v3.1
        # This is kept for API completeness but may need alternative implementation
        return self._make_request(f"lang/{language}")
    
    def get_countries_by_region(self, region: str) -> List[Dict[str, Any]]:
        # Get countries by region
        # region: Region name (e.g., 'Europe', 'Asia')
        # Returns: List of countries in the region
        return self._make_request(f"region/{region}")
    
    def get_countries_by_subregion(self, subregion: str) -> List[Dict[str, Any]]:
        # Get countries by subregion
        # subregion: Subregion name (e.g., 'Northern Europe')
        # Returns: List of countries in the subregion
        return self._make_request(f"subregion/{subregion}")
    
    def get_countries_by_capital(self, capital: str) -> List[Dict[str, Any]]:
        # Get countries by capital city
        # capital: Capital city name
        # Returns: List of countries with matching capital
        return self._make_request(f"capital/{capital}")
    
    def close(self):
        # Close the session
        self.session.close()
    
    def __enter__(self):
        # Context manager entry
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Context manager exit
        self.close()