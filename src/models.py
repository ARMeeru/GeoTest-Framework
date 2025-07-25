# Pydantic models for REST Countries API data validation
# Provides type safety and data validation for API responses

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import re


class CountryName(BaseModel):
    # Country name information
    common: str = Field(..., description="Common country name")
    official: str = Field(..., description="Official country name")
    nativeName: Optional[Dict[str, Dict[str, str]]] = Field(None, description="Native names by language")


class Currency(BaseModel):
    # Currency information
    name: str = Field(..., description="Currency name")
    symbol: Optional[str] = Field(None, description="Currency symbol")


class CountryFlags(BaseModel):
    # Country flag information
    png: str = Field(..., description="PNG flag URL")
    svg: str = Field(..., description="SVG flag URL")
    alt: Optional[str] = Field(None, description="Alt text for flag")


class CountryMaps(BaseModel):
    # Map information
    googleMaps: str = Field(..., description="Google Maps URL")
    openStreetMaps: str = Field(..., description="OpenStreetMap URL")


class CarInfo(BaseModel):
    # Car/driving information
    signs: List[str] = Field(default_factory=list, description="Car signs")
    side: str = Field(..., description="Driving side (left/right)")


class CountryData(BaseModel):
    # Main country data model matching REST Countries API v3.1 response
    name: CountryName = Field(..., description="Country name information")
    cca2: str = Field(..., min_length=2, max_length=2, description="2-letter country code")
    cca3: str = Field(..., min_length=3, max_length=3, description="3-letter country code")
    ccn3: Optional[str] = Field(None, description="3-digit country code")
    region: str = Field(..., description="Country region")
    subregion: Optional[str] = Field(None, description="Country subregion")
    capital: Optional[List[str]] = Field(None, description="Capital cities")
    population: Optional[int] = Field(None, ge=0, description="Population count")
    area: Optional[float] = Field(None, ge=0, description="Area in km²")
    currencies: Optional[Dict[str, Currency]] = Field(None, description="Currencies used")
    languages: Optional[Dict[str, str]] = Field(None, description="Languages spoken")
    timezones: Optional[List[str]] = Field(None, description="Timezones")
    borders: Optional[List[str]] = Field(None, description="Bordering countries")
    flag: Optional[str] = Field(None, description="Flag emoji")
    flags: Optional[CountryFlags] = Field(None, description="Flag URLs")
    maps: Optional[CountryMaps] = Field(None, description="Map URLs")
    car: Optional[CarInfo] = Field(None, description="Car/driving info")
    independent: Optional[bool] = Field(None, description="Independence status")
    unMember: Optional[bool] = Field(None, description="UN membership status")
    landlocked: Optional[bool] = Field(None, description="Landlocked status")

    @validator('cca2')
    def validate_alpha2_code(cls, v):
        # Validate 2-letter country code format
        if not re.match(r'^[A-Z]{2}$', v):
            raise ValueError('Alpha-2 code must be exactly 2 uppercase letters')
        return v

    @validator('cca3')
    def validate_alpha3_code(cls, v):
        # Validate 3-letter country code format
        if not re.match(r'^[A-Z]{3}$', v):
            raise ValueError('Alpha-3 code must be exactly 3 uppercase letters')
        return v

    @validator('region')
    def validate_region(cls, v):
        # Validate region is one of known values
        valid_regions = {'Africa', 'Americas', 'Asia', 'Europe', 'Oceania', 'Antarctic'}
        if v not in valid_regions:
            raise ValueError(f'Region must be one of: {valid_regions}')
        return v


class ScenarioData(BaseModel):
    # Test scenario model for CSV-based test data
    test_id: str = Field(..., description="Unique test identifier")
    test_name: str = Field(..., description="Human-readable test name")
    endpoint: str = Field(..., description="API endpoint to test")
    input_value: str = Field(..., description="Input parameter value")
    expected_result: str = Field(..., description="Expected test outcome")
    test_type: str = Field(..., description="Test type (positive/negative)")
    priority: str = Field(default="medium", description="Test priority")
    tags: Optional[str] = Field(None, description="Comma-separated tags")
    notes: Optional[str] = Field(None, description="Additional notes")

    @validator('test_type')
    def validate_test_type(cls, v):
        # Validate test type
        valid_types = {'positive', 'negative', 'edge_case'}
        if v not in valid_types:
            raise ValueError(f'Test type must be one of: {valid_types}')
        return v

    @validator('priority')
    def validate_priority(cls, v):
        # Validate priority level
        valid_priorities = {'low', 'medium', 'high', 'critical'}
        if v not in valid_priorities:
            raise ValueError(f'Priority must be one of: {valid_priorities}')
        return v

    @validator('endpoint')
    def validate_endpoint(cls, v):
        # Validate endpoint format
        valid_endpoints = {
            'all', 'name', 'alpha', 'currency', 'lang', 
            'region', 'subregion', 'capital'
        }
        if v not in valid_endpoints:
            raise ValueError(f'Endpoint must be one of: {valid_endpoints}')
        return v


class ApiError(BaseModel):
    # API error response model
    status: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Error message")


class TestResult(BaseModel):
    # Test result model for capturing test outcomes
    test_id: str = Field(..., description="Test identifier")
    status: str = Field(..., description="Test status (passed/failed/skipped)")
    execution_time: float = Field(..., ge=0, description="Execution time in seconds")
    request_url: Optional[str] = Field(None, description="Request URL")
    response_status: Optional[int] = Field(None, description="HTTP response status")
    response_time: Optional[float] = Field(None, ge=0, description="API response time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    response_data: Optional[Dict[str, Any]] = Field(None, description="Response data sample")

    @validator('status')
    def validate_status(cls, v):
        # Validate test status
        valid_statuses = {'passed', 'failed', 'skipped', 'error'}
        if v not in valid_statuses:
            raise ValueError(f'Status must be one of: {valid_statuses}')
        return v


class ValidationError(BaseModel):
    # Data validation error model
    field: str = Field(..., description="Field that failed validation")
    error_type: str = Field(..., description="Type of validation error")
    message: str = Field(..., description="Validation error message")
    input_value: Optional[Any] = Field(None, description="Value that caused error")


# Utility functions for model validation
def validate_country_data(data: Dict[str, Any]) -> CountryData:
    # Validate raw API response data against CountryData model
    try:
        return CountryData(**data)
    except Exception as e:
        raise ValueError(f"Country data validation failed: {str(e)}")


def validate_test_scenario(data: Dict[str, Any]) -> ScenarioData:
    # Validate test scenario data from CSV
    try:
        return ScenarioData(**data)
    except Exception as e:
        raise ValueError(f"Test scenario validation failed: {str(e)}")


def validate_country_list(data: List[Dict[str, Any]]) -> List[CountryData]:
    # Validate list of countries from API response
    validated_countries = []
    validation_errors = []
    
    for i, country_data in enumerate(data):
        try:
            validated_countries.append(CountryData(**country_data))
        except Exception as e:
            validation_errors.append(ValidationError(
                field=f"country[{i}]",
                error_type="validation_error",
                message=str(e),
                input_value=country_data.get('name', {}).get('common', 'Unknown')
            ))
    
    if validation_errors:
        # Log validation errors but continue with valid data
        print(f"Warning: {len(validation_errors)} validation errors encountered")
        for error in validation_errors[:5]:  # Show first 5 errors
            print(f"  - {error.field}: {error.message}")
    
    return validated_countries