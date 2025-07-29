# Failure categorization system for Phase 5 intelligent bug tracking
# Analyzes test failures to categorize and identify patterns

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    # Categories of test failures for intelligent classification
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error" 
    ASSERTION_FAILURE = "assertion_failure"
    TIMEOUT_ERROR = "timeout_error"
    SETUP_TEARDOWN_ERROR = "setup_teardown_error"
    FRAMEWORK_ERROR = "framework_error"
    DEPENDENCY_ERROR = "dependency_error"
    DATA_ERROR = "data_error"
    UNKNOWN = "unknown"


class FailureSeverity(Enum):
    # Severity levels for failure prioritization
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FailurePattern:
    # Pattern for identifying specific failure types
    pattern: str
    category: FailureCategory
    severity: FailureSeverity
    description: str
    retry_recommended: bool
    keywords: List[str]


@dataclass
class FailureClassification:
    # Result of failure analysis
    category: FailureCategory
    severity: FailureSeverity
    confidence: float  # 0.0 to 1.0
    description: str
    retry_recommended: bool
    similar_failures: List[str]
    root_cause_hints: List[str]


@dataclass
class FlakyTestIndicator:
    # Indicators that a test might be flaky
    test_name: str
    pass_rate: float  # 0.0 to 1.0
    failure_pattern: str
    recent_failures: int
    recent_passes: int
    inconsistent_results: bool
    last_analysis: float


class FailureAnalyzer:
    # Intelligent failure analysis and categorization system
    
    def __init__(self):
        self.failure_patterns = self._initialize_patterns()
        self.known_flaky_tests = {}  # test_name -> FlakyTestIndicator
        self.failure_history = {}  # For pattern detection
        
    def _initialize_patterns(self) -> List[FailurePattern]:
        # Initialize failure pattern definitions
        return [
            # API Errors
            FailurePattern(
                pattern=r"HTTP.*?(?:400|401|403|404|500|502|503|504)",
                category=FailureCategory.API_ERROR,
                severity=FailureSeverity.HIGH,
                description="HTTP error response from API",
                retry_recommended=True,
                keywords=["http", "status", "error", "response"]
            ),
            FailurePattern(
                pattern=r"(?i)requests\.exceptions\.(?:HTTPError|RequestException)",
                category=FailureCategory.API_ERROR,
                severity=FailureSeverity.MEDIUM,
                description="HTTP request failed",
                retry_recommended=True,
                keywords=["requests", "http", "exception"]
            ),
            FailurePattern(
                pattern=r"(?i)json\.decoder\.JSONDecodeError",
                category=FailureCategory.API_ERROR,
                severity=FailureSeverity.MEDIUM,
                description="Invalid JSON response from API",
                retry_recommended=True,
                keywords=["json", "decode", "invalid", "response"]
            ),
            
            # Network Errors
            FailurePattern(
                pattern=r"(?i)(?:connection|network).*?(?:error|failed|timeout|refused)",
                category=FailureCategory.NETWORK_ERROR,
                severity=FailureSeverity.HIGH,
                description="Network connectivity issue",
                retry_recommended=True,
                keywords=["connection", "network", "timeout", "refused"]
            ),
            FailurePattern(
                pattern=r"(?i)requests\.exceptions\.(?:ConnectionError|Timeout)",
                category=FailureCategory.NETWORK_ERROR,
                severity=FailureSeverity.HIGH,
                description="Network request failed",
                retry_recommended=True,
                keywords=["connection", "timeout", "network"]
            ),
            FailurePattern(
                pattern=r"(?i)(?:dns|hostname).*?(?:resolution|lookup).*?fail",
                category=FailureCategory.NETWORK_ERROR,
                severity=FailureSeverity.CRITICAL,
                description="DNS resolution failure",
                retry_recommended=True,
                keywords=["dns", "hostname", "resolution", "lookup"]
            ),
            
            # Assertion Failures
            FailurePattern(
                pattern=r"AssertionError",
                category=FailureCategory.ASSERTION_FAILURE,
                severity=FailureSeverity.MEDIUM,
                description="Test assertion failed",
                retry_recommended=False,
                keywords=["assertion", "assert", "expected", "actual"]
            ),
            FailurePattern(
                pattern=r"assert.*?==.*?failed",
                category=FailureCategory.ASSERTION_FAILURE,
                severity=FailureSeverity.MEDIUM,
                description="Equality assertion failed",
                retry_recommended=False,
                keywords=["assert", "equality", "comparison"]
            ),
            
            # Timeout Errors
            FailurePattern(
                pattern=r"(?i)timeout",
                category=FailureCategory.TIMEOUT_ERROR,
                severity=FailureSeverity.MEDIUM,
                description="Operation timed out",
                retry_recommended=True,
                keywords=["timeout", "time", "limit", "exceeded"]
            ),
            FailurePattern(
                pattern=r"(?i)operation.*?timed.*?out",
                category=FailureCategory.TIMEOUT_ERROR,
                severity=FailureSeverity.MEDIUM,
                description="Operation timeout",
                retry_recommended=True,
                keywords=["operation", "timed", "out"]
            ),
            
            # Setup/Teardown Errors
            FailurePattern(
                pattern=r"(?i)(?:setup|teardown|fixture).*?(?:error|failed)",
                category=FailureCategory.SETUP_TEARDOWN_ERROR,
                severity=FailureSeverity.HIGH,
                description="Test setup or teardown failed",
                retry_recommended=True,
                keywords=["setup", "teardown", "fixture", "before", "after"]
            ),
            FailurePattern(
                pattern=r"(?i)pytest.*?fixture.*?error",
                category=FailureCategory.SETUP_TEARDOWN_ERROR,
                severity=FailureSeverity.HIGH,
                description="Pytest fixture error",
                retry_recommended=True,
                keywords=["pytest", "fixture", "error"]
            ),
            
            # Framework Errors
            FailurePattern(
                pattern=r"(?i)(?:pytest|unittest).*?(?:error|exception)",
                category=FailureCategory.FRAMEWORK_ERROR,
                severity=FailureSeverity.MEDIUM,
                description="Testing framework error",
                retry_recommended=True,
                keywords=["pytest", "unittest", "framework"]
            ),
            FailurePattern(
                pattern=r"ImportError|ModuleNotFoundError",
                category=FailureCategory.FRAMEWORK_ERROR,
                severity=FailureSeverity.HIGH,
                description="Import or module error",
                retry_recommended=False,
                keywords=["import", "module", "not", "found"]
            ),
            
            # Dependency Errors
            FailurePattern(
                pattern=r"(?i)(?:dependency|package).*?(?:not.*?found|missing|unavailable)",
                category=FailureCategory.DEPENDENCY_ERROR,
                severity=FailureSeverity.HIGH,
                description="Missing dependency",
                retry_recommended=False,
                keywords=["dependency", "package", "missing", "unavailable"]
            ),
            FailurePattern(
                pattern=r"(?i)(?:docker|container).*?(?:error|failed|not.*?running)",
                category=FailureCategory.DEPENDENCY_ERROR,
                severity=FailureSeverity.HIGH,
                description="Docker/container issue",
                retry_recommended=True,
                keywords=["docker", "container", "service"]
            ),
            
            # Data Errors
            FailurePattern(
                pattern=r"(?i)(?:validation|schema).*?(?:error|failed)",
                category=FailureCategory.DATA_ERROR,
                severity=FailureSeverity.MEDIUM,
                description="Data validation error",
                retry_recommended=False,
                keywords=["validation", "schema", "data", "format"]
            ),
            FailurePattern(
                pattern=r"(?i)pydantic.*?(?:validation|error)",
                category=FailureCategory.DATA_ERROR,
                severity=FailureSeverity.MEDIUM,
                description="Pydantic validation error",
                retry_recommended=False,
                keywords=["pydantic", "validation", "model"]
            )
        ]
    
    def analyze_failure(self, test_name: str, error_message: str, traceback: str, 
                       test_duration: float = 0) -> FailureClassification:
        # Analyze a test failure and classify it
        
        # Combine error message and traceback for analysis
        full_error_text = f"{error_message}\n{traceback}"
        
        # Find matching patterns
        pattern_matches = []
        for pattern in self.failure_patterns:
            if re.search(pattern.pattern, full_error_text):
                # Calculate confidence based on keyword matches
                keyword_matches = sum(
                    1 for keyword in pattern.keywords
                    if keyword.lower() in full_error_text.lower()
                )
                confidence = min(0.9, 0.5 + (keyword_matches * 0.1))
                pattern_matches.append((pattern, confidence))
        
        # Determine best classification
        if pattern_matches:
            # Sort by confidence and severity
            pattern_matches.sort(key=lambda x: (x[1], x[0].severity.value), reverse=True)
            best_pattern, confidence = pattern_matches[0]
            
            category = best_pattern.category
            severity = best_pattern.severity
            description = best_pattern.description
            retry_recommended = best_pattern.retry_recommended
        else:
            # No pattern matched, use heuristics
            category = FailureCategory.UNKNOWN
            severity = FailureSeverity.MEDIUM
            confidence = 0.3
            description = "Unclassified failure"
            retry_recommended = True
        
        # Check for flaky test indicators
        flaky_indicator = self._check_flaky_test(test_name, full_error_text)
        if flaky_indicator and flaky_indicator.inconsistent_results:
            description += " (possibly flaky test)"
            severity = FailureSeverity.LOW  # Reduce severity for flaky tests
        
        # Generate root cause hints
        root_cause_hints = self._generate_root_cause_hints(
            category, error_message, traceback, test_duration
        )
        
        # Find similar failures
        similar_failures = self._find_similar_failures(test_name, error_message)
        
        return FailureClassification(
            category=category,
            severity=severity,
            confidence=confidence,
            description=description,
            retry_recommended=retry_recommended,
            similar_failures=similar_failures,
            root_cause_hints=root_cause_hints
        )
    
    def _check_flaky_test(self, test_name: str, error_text: str) -> Optional[FlakyTestIndicator]:
        # Check if test shows signs of being flaky
        
        # Update flaky test tracking
        if test_name not in self.known_flaky_tests:
            self.known_flaky_tests[test_name] = FlakyTestIndicator(
                test_name=test_name,
                pass_rate=1.0,
                failure_pattern="",
                recent_failures=0,
                recent_passes=0,
                inconsistent_results=False,
                last_analysis=time.time()
            )
        
        indicator = self.known_flaky_tests[test_name]
        
        # Common flaky test patterns
        flaky_patterns = [
            r"(?i)intermittent",
            r"(?i)random.*?fail",
            r"(?i)race.*?condition",
            r"(?i)timing.*?issue",
            r"(?i)occasionally.*?fail"
        ]
        
        pattern_match = any(re.search(pattern, error_text) for pattern in flaky_patterns)
        
        # Update failure tracking
        indicator.recent_failures += 1
        indicator.last_analysis = time.time()
        
        if pattern_match:
            indicator.failure_pattern = "flaky_pattern_detected"
            indicator.inconsistent_results = True
        
        # Calculate pass rate (simplified)
        total_runs = indicator.recent_failures + indicator.recent_passes
        if total_runs > 0:
            indicator.pass_rate = indicator.recent_passes / total_runs
            
            # Mark as inconsistent if pass rate is between 20% and 80%
            if 0.2 <= indicator.pass_rate <= 0.8 and total_runs >= 5:
                indicator.inconsistent_results = True
        
        return indicator
    
    def record_test_success(self, test_name: str):
        # Record a test success for flaky test detection
        if test_name not in self.known_flaky_tests:
            self.known_flaky_tests[test_name] = FlakyTestIndicator(
                test_name=test_name,
                pass_rate=1.0,
                failure_pattern="",
                recent_failures=0,
                recent_passes=1,
                inconsistent_results=False,
                last_analysis=time.time()
            )
        else:
            self.known_flaky_tests[test_name].recent_passes += 1
            self.known_flaky_tests[test_name].last_analysis = time.time()
    
    def _generate_root_cause_hints(self, category: FailureCategory, error_message: str, 
                                 traceback: str, test_duration: float) -> List[str]:
        # Generate hints for potential root causes
        hints = []
        
        if category == FailureCategory.API_ERROR:
            if "404" in error_message:
                hints.append("API endpoint may have changed or be temporarily unavailable")
            elif "500" in error_message:
                hints.append("Server-side error - check API service status")
            elif "timeout" in error_message.lower():
                hints.append("API response too slow - consider increasing timeout")
            else:
                hints.append("Check API service status and authentication")
        
        elif category == FailureCategory.NETWORK_ERROR:
            hints.extend([
                "Check internet connectivity",
                "Verify firewall settings",
                "Check DNS resolution"
            ])
        
        elif category == FailureCategory.ASSERTION_FAILURE:
            hints.extend([
                "Review test expectations vs actual API behavior",
                "Check if API response format has changed",
                "Verify test data is still valid"
            ])
        
        elif category == FailureCategory.TIMEOUT_ERROR:
            if test_duration > 30:
                hints.append("Test is taking too long - consider optimizing or increasing timeout")
            hints.extend([
                "Check system load and performance",
                "Verify network latency to API"
            ])
        
        elif category == FailureCategory.SETUP_TEARDOWN_ERROR:
            hints.extend([
                "Check test environment configuration",
                "Verify required services are running",
                "Check for resource conflicts between tests"
            ])
        
        elif category == FailureCategory.DEPENDENCY_ERROR:
            hints.extend([
                "Verify all required packages are installed",
                "Check Docker containers are running",
                "Validate environment configuration"
            ])
        
        # Add duration-based hints
        if test_duration > 60:
            hints.append("Test duration excessive - may indicate performance issue")
        elif test_duration < 0.1:
            hints.append("Test completed very quickly - may indicate early failure")
        
        return hints
    
    def _find_similar_failures(self, test_name: str, error_message: str) -> List[str]:
        # Find similar failures for pattern detection
        similar = []
        
        # This would be enhanced with actual failure history storage
        # For now, return basic similarity hints
        error_keywords = re.findall(r'\b\w+\b', error_message.lower())
        common_keywords = ['error', 'failed', 'exception', 'timeout', 'connection']
        
        significant_keywords = [kw for kw in error_keywords if kw not in common_keywords][:3]
        
        if significant_keywords:
            similar.append(f"Look for other failures containing: {', '.join(significant_keywords)}")
        
        return similar
    
    def get_flaky_tests_report(self) -> Dict[str, Any]:
        # Generate report of potentially flaky tests
        flaky_tests = [
            indicator for indicator in self.known_flaky_tests.values()
            if indicator.inconsistent_results or indicator.pass_rate < 0.8
        ]
        
        # Sort by flakiness score (lower pass rate = more flaky)
        flaky_tests.sort(key=lambda x: x.pass_rate)
        
        return {
            "total_flaky_tests": len(flaky_tests),
            "flaky_tests": [
                {
                    "test_name": test.test_name,
                    "pass_rate": test.pass_rate,
                    "recent_failures": test.recent_failures,
                    "recent_passes": test.recent_passes,
                    "failure_pattern": test.failure_pattern
                }
                for test in flaky_tests[:10]  # Top 10 flaky tests
            ],
            "recommendations": self._generate_flaky_test_recommendations(flaky_tests)
        }
    
    def _generate_flaky_test_recommendations(self, flaky_tests: List[FlakyTestIndicator]) -> List[str]:
        # Generate recommendations for handling flaky tests
        recommendations = []
        
        if len(flaky_tests) > 5:
            recommendations.append("High number of flaky tests detected - review test environment stability")
        
        very_flaky = [test for test in flaky_tests if test.pass_rate < 0.5]
        if very_flaky:
            recommendations.append(f"{len(very_flaky)} tests have very low pass rates - consider disabling or fixing")
        
        if any(test.failure_pattern == "flaky_pattern_detected" for test in flaky_tests):
            recommendations.append("Some tests show explicit flaky behavior patterns - add retry logic")
        
        if not recommendations:
            recommendations.append("Flaky test detection is working - continue monitoring")
        
        return recommendations
    
    def categorize_batch_failures(self, failures: List[Tuple[str, str, str]]) -> Dict[FailureCategory, int]:
        # Categorize multiple failures for trend analysis
        categories = {}
        
        for test_name, error_message, traceback in failures:
            classification = self.analyze_failure(test_name, error_message, traceback)
            category = classification.category
            categories[category] = categories.get(category, 0) + 1
        
        return categories
    
    def get_failure_trends(self, time_window_hours: int = 24) -> Dict[str, Any]:
        # Analyze failure trends over time window
        # This would be enhanced with actual time-series data
        
        # For now, return basic analysis based on current flaky test data
        total_tests = len(self.known_flaky_tests)
        problematic_tests = len([
            test for test in self.known_flaky_tests.values()
            if test.recent_failures > 0
        ])
        
        return {
            "time_window_hours": time_window_hours,
            "total_monitored_tests": total_tests,
            "tests_with_recent_failures": problematic_tests,
            "stability_score": 1.0 - (problematic_tests / total_tests) if total_tests > 0 else 1.0,
            "trend": "stable" if problematic_tests < total_tests * 0.1 else "concerning"
        }


def get_failure_analyzer() -> FailureAnalyzer:
    # Get global failure analyzer instance
    global _failure_analyzer_instance
    if '_failure_analyzer_instance' not in globals():
        _failure_analyzer_instance = FailureAnalyzer()
    return _failure_analyzer_instance