# Test cases demonstrating Phase 4 monitoring and alerting integration
# Shows how to use the new monitoring features in your tests

import pytest
import time
import random
from src.pytest_plugins import api_endpoint, critical, performance_test


class TestMonitoringIntegration:
    # Test cases that demonstrate the monitoring integration features
    
    @pytest.mark.smoke
    @api_endpoint("all")
    def test_get_all_countries_with_monitoring(self, api_client, metrics_collector):
        # Test getting all countries with built-in monitoring
        # This test will automatically be tracked by the monitoring system
        start_time = time.time()
        
        countries = api_client.get_all_countries(fields="name,cca2,region")
        
        # Manual metrics recording for demonstration
        response_time = time.time() - start_time
        metrics_collector.record_api_metric(
            endpoint="all",
            metric_name="custom_response_time",
            value=response_time,
            tags={"test_type": "smoke", "fields": "name,cca2,region"}
        )
        
        assert isinstance(countries, list)
        assert len(countries) > 0
        assert all('name' in country for country in countries)
        
        # Record custom business metric
        metrics_collector.record_custom_metric(
            "countries.total_count",
            len(countries),
            tags={"endpoint": "all"}
        )
    
    @pytest.mark.regression
    @critical
    @api_endpoint("alpha")
    def test_critical_country_search(self, api_client, alert_manager):
        # Critical test that will trigger high-severity alerts on failure
        # This test is marked as critical, so failures will trigger alerts
        
        # Test critical functionality using alpha code endpoint for precise lookup
        countries = api_client.get_country_by_code("US")
        
        assert len(countries) == 1
        assert countries[0]['name']['common'] == "United States"
        assert countries[0]['cca2'] == "US"
        
        # Simulate a potential failure scenario for demonstration
        # In real scenarios, this would be actual business logic
        if random.random() < 0.05:  # 5% chance of failure for demo
            pytest.fail("Simulated critical failure for alerting demonstration")
    
    @pytest.mark.performance
    @api_endpoint("region")
    def test_region_search_performance(self, api_client, performance_tracker):
        # Performance test with enhanced monitoring
        
        def search_region():
            return api_client.get_countries_by_region("Europe")
        
        # Use performance tracker fixture
        countries = performance_tracker(
            "region_search",
            search_region,
            region="Europe",
            test_type="performance"
        )
        
        # Performance assertions
        assert len(countries) > 20  # Europe should have many countries
        
        # Simulate slow operation for alerting demonstration
        if random.random() < 0.1:  # 10% chance for demo
            time.sleep(3)  # This might trigger performance alerts
    
    @pytest.mark.integration
    @api_endpoint("currency")
    def test_currency_search_with_custom_metrics(self, api_client, metrics_collector):
        # Test with custom metrics collection
        
        currencies_to_test = ["USD", "EUR", "GBP"]
        
        for currency in currencies_to_test:
            start_time = time.time()
            
            try:
                countries = api_client.get_countries_by_currency(currency)
                duration = time.time() - start_time
                
                # Record success metrics
                metrics_collector.record_custom_metric(
                    f"currency.{currency}.search_duration",
                    duration,
                    tags={"currency": currency, "status": "success"}
                )
                
                metrics_collector.record_custom_metric(
                    f"currency.{currency}.country_count",
                    len(countries),
                    tags={"currency": currency}
                )
                
                assert len(countries) > 0
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record failure metrics
                metrics_collector.record_custom_metric(
                    f"currency.{currency}.search_duration",
                    duration,
                    tags={"currency": currency, "status": "error"}
                )
                
                metrics_collector.record_custom_metric(
                    f"currency.{currency}.error_count",
                    1,
                    tags={"currency": currency, "error": str(e)[:50]}
                )
                
                raise
    
    @pytest.mark.smoke
    @api_endpoint("alpha")
    def test_country_codes_batch(self, api_client, metrics_collector):
        # Test multiple country code lookups with batch metrics
        
        country_codes = ["US", "GB", "FR", "DE", "JP"]
        successful_lookups = 0
        failed_lookups = 0
        total_response_time = 0
        
        for code in country_codes:
            start_time = time.time()
            
            try:
                countries = api_client.get_country_by_code(code)
                response_time = time.time() - start_time
                total_response_time += response_time
                
                assert len(countries) == 1
                assert countries[0]['cca2'] == code
                successful_lookups += 1
                
                # Record individual lookup metrics
                metrics_collector.record_api_metric(
                    endpoint="alpha",
                    metric_name="lookup_success",
                    value=1,
                    tags={"country_code": code}
                )
                
            except Exception as e:
                response_time = time.time() - start_time
                total_response_time += response_time
                failed_lookups += 1
                
                # Record failure metrics
                metrics_collector.record_api_metric(
                    endpoint="alpha",
                    metric_name="lookup_failure",
                    value=1,
                    tags={"country_code": code, "error": str(e)[:50]}
                )
        
        # Record batch metrics
        metrics_collector.record_custom_metric(
            "batch.country_lookups.success_rate",
            (successful_lookups / len(country_codes)) * 100,
            tags={"batch_size": str(len(country_codes))}
        )
        
        metrics_collector.record_custom_metric(
            "batch.country_lookups.average_response_time",
            total_response_time / len(country_codes),
            tags={"batch_size": str(len(country_codes))}
        )
        
        # Assert batch success criteria
        assert successful_lookups > 0
        assert (successful_lookups / len(country_codes)) >= 0.8  # 80% success rate
    
    @pytest.mark.regression
    def test_system_resource_monitoring(self, metrics_collector):
        # Test that generates system load for monitoring demonstration
        
        # Record test start
        metrics_collector.record_custom_metric(
            "test.resource_intensive.start",
            1,
            tags={"test_name": "system_resource_monitoring"}
        )
        
        # Simulate some CPU/memory intensive work
        data = []
        for i in range(100000):
            data.append(f"test_data_{i}" * 10)
        
        # Simulate processing time
        time.sleep(0.5)
        
        # Clean up
        del data
        
        # Record test completion
        metrics_collector.record_custom_metric(
            "test.resource_intensive.completed",
            1,
            tags={"test_name": "system_resource_monitoring"}
        )
        
        # This test mainly exists to generate system resource usage
        assert True
    
    def test_alert_trigger_simulation(self, alert_manager):
        # Test that demonstrates how to manually trigger alerts
        
        # Create a test alert (this would normally be triggered automatically)
        from src.alerting import AlertType, AlertSeverity
        alert = alert_manager.create_alert(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.LOW,
            title="Test Alert from Monitoring Integration",
            message="This is a demonstration alert created during testing to verify the alerting system is working correctly.",
            source="test:test_monitoring_integration",
            tags={"test_generated": "true", "environment": "test"}
        )
        
        assert alert.alert_id is not None
        assert not alert.resolved
        
        # Resolve the test alert
        success = alert_manager.resolve_alert(alert.alert_id, "automated_test")
        assert success
    
    @pytest.mark.slow
    @api_endpoint("capital")
    def test_slow_operation_for_alerting(self, api_client):
        # Test that deliberately takes time to trigger performance alerts
        
        # This test might trigger performance degradation alerts
        capitals = ["London", "Paris", "Berlin", "Madrid", "Rome"]
        
        for capital in capitals:
            # Add delay to simulate slow API responses
            time.sleep(0.3)  # This might trigger response time alerts
            
            countries = api_client.get_countries_by_capital(capital)
            assert len(countries) >= 1
            
            # Verify the capital is correct
            found_capital = False
            for country in countries:
                if country.get('capital') and capital in country['capital']:
                    found_capital = True
                    break
            
            assert found_capital, f"Capital {capital} not found in results"


class TestAdvancedMonitoringFeatures:
    # Advanced monitoring feature demonstrations
    
    def test_custom_metric_types(self, metrics_collector):
        # Test recording different types of custom metrics
        
        # Counter metrics
        metrics_collector.record_custom_metric(
            "test.counter.api_calls",
            1,
            tags={"endpoint": "test", "type": "counter"}
        )
        
        # Gauge metrics (current value)
        metrics_collector.record_custom_metric(
            "test.gauge.active_connections",
            42,
            tags={"type": "gauge"}
        )
        
        # Histogram metrics (duration/timing)
        metrics_collector.record_custom_metric(
            "test.histogram.operation_duration",
            1.234,
            tags={"operation": "data_processing", "type": "histogram"}
        )
        
        # String metrics
        metrics_collector.record_custom_metric(
            "test.info.version",
            "1.0.0",
            tags={"component": "geotest-framework", "type": "info"}
        )
        
        assert True  # Metrics recorded successfully
    
    def test_error_tracking_and_alerting(self, api_client, metrics_collector, alert_manager):
        # Test error tracking and automated alerting
        
        error_count = 0
        total_requests = 5
        
        # Simulate a series of requests with some failures
        for i in range(total_requests):
            try:
                if i == 2:  # Force an error for demonstration
                    # This will likely cause an error
                    api_client.get_country_by_name("ThisCountryDoesNotExist123")
                else:
                    api_client.get_country_by_name("Canada")
                
            except Exception as e:
                error_count += 1
                
                # Record error metrics
                metrics_collector.record_custom_metric(
                    "test.error.api_call_failed",
                    1,
                    tags={"error_type": type(e).__name__, "test_iteration": str(i)}
                )
                
                # Continue with test (don't fail on expected errors)
                if i != 2:  # Only expected error on iteration 2
                    raise
        
        # Calculate error rate
        error_rate = (error_count / total_requests) * 100
        
        metrics_collector.record_custom_metric(
            "test.error_rate.percentage",
            error_rate,
            tags={"total_requests": str(total_requests)}
        )
        
        # Verify we had the expected error
        assert error_count == 1
        assert error_rate == 20.0  # 1 out of 5 = 20%