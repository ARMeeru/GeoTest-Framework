# Retry mechanisms for Phase 5 intelligent bug tracking
# Configurable retry logic with exponential backoff and failure type awareness

import time
import random
import logging
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import functools

from .failure_analyzer import FailureAnalyzer, FailureCategory, get_failure_analyzer

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    # Different retry strategies for various failure types
    IMMEDIATE = "immediate"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    CUSTOM = "custom"


@dataclass
class RetryConfig:
    # Configuration for retry behavior
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0
    jitter: bool = True
    skip_categories: List[FailureCategory] = None
    per_category_config: Dict[FailureCategory, 'RetryConfig'] = None


@dataclass
class RetryAttempt:
    # Record of a retry attempt
    attempt_number: int
    delay_used: float
    timestamp: float
    error_message: str
    failure_category: FailureCategory
    success: bool


@dataclass
class RetryResult:
    # Result of retry operation
    success: bool
    result: Any
    total_attempts: int
    total_time: float
    attempts: List[RetryAttempt]
    final_error: Optional[Exception]


class SmartRetryManager:
    # Intelligent retry manager with failure-aware strategies
    
    def __init__(self, default_config: RetryConfig = None):
        self.default_config = default_config or RetryConfig()
        self.failure_analyzer = get_failure_analyzer()
        self.retry_statistics = {}  # function_name -> stats
        
        # Setup category-specific configs if not provided
        if not self.default_config.per_category_config:
            self.default_config.per_category_config = self._get_default_category_configs()
        
        if not self.default_config.skip_categories:
            self.default_config.skip_categories = [
                FailureCategory.ASSERTION_FAILURE,  # Don't retry assertion failures
                FailureCategory.DATA_ERROR,          # Don't retry data validation errors
                FailureCategory.DEPENDENCY_ERROR     # Don't retry missing dependencies
            ]
    
    def _get_default_category_configs(self) -> Dict[FailureCategory, RetryConfig]:
        # Get default retry configurations for different failure categories
        return {
            FailureCategory.API_ERROR: RetryConfig(
                max_retries=5,
                base_delay=2.0,
                strategy=RetryStrategy.EXPONENTIAL,
                backoff_multiplier=1.5
            ),
            FailureCategory.NETWORK_ERROR: RetryConfig(
                max_retries=4,
                base_delay=3.0,
                strategy=RetryStrategy.EXPONENTIAL,
                backoff_multiplier=2.0
            ),
            FailureCategory.TIMEOUT_ERROR: RetryConfig(
                max_retries=3,
                base_delay=5.0,
                strategy=RetryStrategy.LINEAR,
                backoff_multiplier=1.0
            ),
            FailureCategory.SETUP_TEARDOWN_ERROR: RetryConfig(
                max_retries=2,
                base_delay=1.0,
                strategy=RetryStrategy.IMMEDIATE
            ),
            FailureCategory.FRAMEWORK_ERROR: RetryConfig(
                max_retries=2,
                base_delay=1.0,
                strategy=RetryStrategy.LINEAR
            ),
            FailureCategory.ASSERTION_FAILURE: RetryConfig(
                max_retries=0  # Never retry assertion failures
            ),
            FailureCategory.DATA_ERROR: RetryConfig(
                max_retries=0  # Never retry data validation errors
            ),
            FailureCategory.DEPENDENCY_ERROR: RetryConfig(
                max_retries=1,  # Try once in case it was temporary
                base_delay=5.0
            ),
            FailureCategory.UNKNOWN: RetryConfig(
                max_retries=2,
                base_delay=2.0,
                strategy=RetryStrategy.EXPONENTIAL
            )
        }
    
    def retry_with_analysis(self, func: Callable, *args, config: RetryConfig = None, 
                          context: str = None, **kwargs) -> RetryResult:
        # Execute function with intelligent retry based on failure analysis
        
        config = config or self.default_config
        context = context or func.__name__
        start_time = time.time()
        attempts = []
        
        for attempt_num in range(config.max_retries + 1):
            attempt_start = time.time()
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Success - record attempt and return
                attempt = RetryAttempt(
                    attempt_number=attempt_num + 1,
                    delay_used=0.0,
                    timestamp=attempt_start,
                    error_message="",
                    failure_category=FailureCategory.UNKNOWN,
                    success=True
                )
                attempts.append(attempt)
                
                total_time = time.time() - start_time
                
                # Update statistics
                self._update_retry_statistics(context, True, attempt_num + 1, total_time)
                
                return RetryResult(
                    success=True,
                    result=result,
                    total_attempts=attempt_num + 1,
                    total_time=total_time,
                    attempts=attempts,
                    final_error=None
                )
                
            except Exception as e:
                # Analyze the failure
                error_message = str(e)
                traceback_str = self._get_traceback_string(e)
                
                classification = self.failure_analyzer.analyze_failure(
                    test_name=context,
                    error_message=error_message,
                    traceback=traceback_str
                )
                
                # Record attempt
                attempt = RetryAttempt(
                    attempt_number=attempt_num + 1,
                    delay_used=0.0,
                    timestamp=attempt_start,
                    error_message=error_message,
                    failure_category=classification.category,
                    success=False
                )
                attempts.append(attempt)
                
                logger.debug(f"Attempt {attempt_num + 1} failed for {context}: "
                           f"{classification.category.value} - {error_message[:100]}")
                
                # Check if we should retry
                if not self._should_retry(classification, attempt_num, config):
                    logger.info(f"Not retrying {context} after {attempt_num + 1} attempts: "
                              f"{classification.description}")
                    break
                
                # Calculate delay for next attempt
                if attempt_num < config.max_retries:
                    delay = self._calculate_delay(attempt_num, classification.category, config)
                    attempt.delay_used = delay
                    
                    logger.info(f"Retrying {context} in {delay:.1f}s "
                              f"(attempt {attempt_num + 2}/{config.max_retries + 1})")
                    time.sleep(delay)
        
        # All attempts failed
        total_time = time.time() - start_time
        final_error = attempts[-1].error_message if attempts else "Unknown error"
        
        # Update statistics
        self._update_retry_statistics(context, False, len(attempts), total_time)
        
        logger.warning(f"All retry attempts failed for {context} after {len(attempts)} attempts")
        
        return RetryResult(
            success=False,
            result=None,
            total_attempts=len(attempts),
            total_time=total_time,
            attempts=attempts,
            final_error=Exception(final_error)
        )
    
    def _should_retry(self, classification, attempt_num: int, config: RetryConfig) -> bool:
        # Determine if we should retry based on failure analysis
        
        # Check if we've exceeded max retries
        if attempt_num >= config.max_retries:
            return False
        
        # Check if this failure category should be skipped
        if classification.category in config.skip_categories:
            logger.debug(f"Skipping retry for category: {classification.category.value}")
            return False
        
        # Don't retry if explicitly not recommended by analysis
        if not classification.retry_recommended:
            logger.debug(f"Retry not recommended for: {classification.description}")
            return False
        
        # Check category-specific config
        category_config = config.per_category_config.get(classification.category)
        if category_config and attempt_num >= category_config.max_retries:
            logger.debug(f"Category-specific retry limit reached for: {classification.category.value}")
            return False
        
        return True
    
    def _calculate_delay(self, attempt_num: int, category: FailureCategory, config: RetryConfig) -> float:
        # Calculate delay for next retry attempt
        
        # Use category-specific config if available
        category_config = config.per_category_config.get(category, config)
        
        if category_config.strategy == RetryStrategy.IMMEDIATE:
            delay = 0.0
        elif category_config.strategy == RetryStrategy.LINEAR:
            delay = category_config.base_delay * (attempt_num + 1)
        elif category_config.strategy == RetryStrategy.EXPONENTIAL:
            delay = category_config.base_delay * (category_config.backoff_multiplier ** attempt_num)
        elif category_config.strategy == RetryStrategy.FIBONACCI:
            delay = category_config.base_delay * self._fibonacci(attempt_num + 1)
        else:  # CUSTOM or fallback
            delay = category_config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, category_config.max_delay)
        
        # Add jitter if enabled
        if category_config.jitter and delay > 0:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay)  # Ensure minimum delay
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        # Calculate fibonacci number for fibonacci backoff
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    def _get_traceback_string(self, exception: Exception) -> str:
        # Get traceback string from exception
        import traceback
        return ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    
    def _update_retry_statistics(self, context: str, success: bool, attempts: int, total_time: float):
        # Update retry statistics for analysis
        if context not in self.retry_statistics:
            self.retry_statistics[context] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_attempts": 0,
                "total_time": 0.0,
                "avg_attempts_success": 0.0,
                "avg_attempts_failure": 0.0,
                "avg_time": 0.0
            }
        
        stats = self.retry_statistics[context]
        stats["total_calls"] += 1
        stats["total_attempts"] += attempts
        stats["total_time"] += total_time
        
        if success:
            stats["successful_calls"] += 1
        else:
            stats["failed_calls"] += 1
        
        # Update averages
        stats["avg_time"] = stats["total_time"] / stats["total_calls"]
        
        if stats["successful_calls"] > 0:
            # Calculate average attempts for successful calls
            success_attempts = sum(
                len(attempts) for attempts in []  # This would need to track individual call attempts
            )
            stats["avg_attempts_success"] = stats["total_attempts"] / stats["total_calls"]
        
        if stats["failed_calls"] > 0:
            stats["avg_attempts_failure"] = stats["total_attempts"] / stats["total_calls"]
    
    def get_retry_statistics(self) -> Dict[str, Any]:
        # Get retry statistics summary
        return {
            "functions": dict(self.retry_statistics),
            "summary": {
                "total_functions": len(self.retry_statistics),
                "total_calls": sum(stats["total_calls"] for stats in self.retry_statistics.values()),
                "total_retries": sum(stats["total_attempts"] - stats["total_calls"] 
                                   for stats in self.retry_statistics.values()),
                "overall_success_rate": self._calculate_overall_success_rate()
            }
        }
    
    def _calculate_overall_success_rate(self) -> float:
        # Calculate overall success rate across all functions
        total_successful = sum(stats["successful_calls"] for stats in self.retry_statistics.values())
        total_calls = sum(stats["total_calls"] for stats in self.retry_statistics.values())
        
        return (total_successful / total_calls) if total_calls > 0 else 0.0


def retry_on_failure(max_retries: int = 3, base_delay: float = 1.0, 
                    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
                    skip_categories: List[FailureCategory] = None):
    # Decorator for automatic retry with failure analysis
    
    def decorator(func: Callable) -> Callable:
        retry_manager = SmartRetryManager()
        
        config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            strategy=strategy,
            skip_categories=skip_categories or []
        )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = retry_manager.retry_with_analysis(
                func, *args, config=config, context=func.__name__, **kwargs
            )
            
            if result.success:
                return result.result
            else:
                # Re-raise the final exception
                raise result.final_error
        
        # Add retry information to function
        wrapper._retry_config = config
        wrapper._retry_manager = retry_manager
        
        return wrapper
    
    return decorator


def get_retry_manager() -> SmartRetryManager:
    # Get global retry manager instance
    global _retry_manager_instance
    if '_retry_manager_instance' not in globals():
        _retry_manager_instance = SmartRetryManager()
    return _retry_manager_instance


# Convenience decorators for common retry patterns

def retry_api_calls(max_retries: int = 5, base_delay: float = 2.0):
    # Decorator optimized for API call retries
    return retry_on_failure(
        max_retries=max_retries,
        base_delay=base_delay,
        strategy=RetryStrategy.EXPONENTIAL,
        skip_categories=[FailureCategory.ASSERTION_FAILURE, FailureCategory.DATA_ERROR]
    )


def retry_network_operations(max_retries: int = 4, base_delay: float = 3.0):
    # Decorator optimized for network operation retries
    return retry_on_failure(
        max_retries=max_retries,
        base_delay=base_delay,
        strategy=RetryStrategy.EXPONENTIAL,
        skip_categories=[FailureCategory.ASSERTION_FAILURE]
    )


def retry_flaky_tests(max_retries: int = 2, base_delay: float = 1.0):
    # Decorator for handling flaky tests
    return retry_on_failure(
        max_retries=max_retries,
        base_delay=base_delay,
        strategy=RetryStrategy.LINEAR,
        skip_categories=[FailureCategory.ASSERTION_FAILURE, FailureCategory.DATA_ERROR]
    )