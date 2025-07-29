# Advanced load generation utilities for concurrent API testing
# Provides sophisticated load patterns and user simulation capabilities

import asyncio
import aiohttp
import time
import random
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque

logger = logging.getLogger(__name__)


class LoadPattern(Enum):
    # Different load generation patterns
    CONSTANT = "constant"           # Steady concurrent users
    RAMP_UP = "ramp_up"            # Gradually increase load
    RAMP_DOWN = "ramp_down"        # Gradually decrease load
    SPIKE = "spike"                # Sudden load increase
    STEP = "step"                  # Step-wise load increases
    WAVE = "wave"                  # Sinusoidal load pattern


@dataclass
class LoadConfig:
    # Configuration for load generation
    pattern: LoadPattern
    duration_seconds: int
    min_users: int
    max_users: int
    ramp_duration: Optional[int] = None
    step_size: Optional[int] = None
    step_duration: Optional[int] = None
    requests_per_user: int = 1
    delay_between_requests: float = 0.1
    random_delay: bool = True


@dataclass
class UserBehavior:
    # Defines user behavior patterns
    name: str
    endpoints: List[str]
    weights: List[float]  # Probability weights for each endpoint
    think_time_min: float = 0.5
    think_time_max: float = 2.0
    session_duration: Optional[float] = None


@dataclass
class LoadGenerationResult:
    # Result from load generation
    config: LoadConfig
    total_requests: int
    successful_requests: int
    failed_requests: int
    actual_duration: float
    peak_concurrent_users: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    errors_by_type: Dict[str, int]
    timeline: List[Dict[str, Any]]  # Timeline of load changes


class AsyncLoadGenerator:
    # Asynchronous load generator for high-concurrency testing
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.active_sessions = 0
        self.results = []
        self._lock = threading.Lock()
        
    async def make_request(self, session: aiohttp.ClientSession, 
                          endpoint: str, user_id: int) -> Dict[str, Any]:
        # Make a single async HTTP request
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        start_time = time.time()
        
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                content = await response.text()
                end_time = time.time()
                
                result = {
                    'user_id': user_id,
                    'endpoint': endpoint,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'status_code': response.status,
                    'success': 200 <= response.status < 300,
                    'response_size': len(content),
                    'error': None
                }
                
                with self._lock:
                    self.results.append(result)
                
                return result
                
        except Exception as e:
            end_time = time.time()
            result = {
                'user_id': user_id,
                'endpoint': endpoint,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'status_code': None,
                'success': False,
                'response_size': 0,
                'error': str(e)
            }
            
            with self._lock:
                self.results.append(result)
            
            return result
    
    async def simulate_user(self, session: aiohttp.ClientSession, 
                           user_id: int, behavior: UserBehavior, 
                           duration: float = None) -> List[Dict[str, Any]]:
        # Simulate a single user's behavior
        user_results = []
        start_time = time.time()
        session_end = start_time + (duration or behavior.session_duration or 60)
        
        with self._lock:
            self.active_sessions += 1
        
        try:
            while time.time() < session_end:
                # Select endpoint based on weights
                endpoint = random.choices(behavior.endpoints, weights=behavior.weights)[0]
                
                # Make request
                result = await self.make_request(session, endpoint, user_id)
                user_results.append(result)
                
                # Think time between requests
                if behavior.random_delay:
                    think_time = random.uniform(behavior.think_time_min, behavior.think_time_max)
                else:
                    think_time = (behavior.think_time_min + behavior.think_time_max) / 2
                
                await asyncio.sleep(think_time)
                
        except asyncio.CancelledError:
            logger.debug(f"User {user_id} simulation cancelled")
        finally:
            with self._lock:
                self.active_sessions -= 1
        
        return user_results
    
    async def generate_load_pattern(self, config: LoadConfig, 
                                  behavior: UserBehavior) -> LoadGenerationResult:
        # Generate load according to specified pattern
        logger.info(f"Starting {config.pattern.value} load pattern with {config.min_users}-{config.max_users} users")
        
        start_time = time.time()
        timeline = []
        user_tasks = []
        
        # Create connector with appropriate limits
        connector = aiohttp.TCPConnector(
            limit=config.max_users * 2,
            limit_per_host=config.max_users,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:
            
            if config.pattern == LoadPattern.CONSTANT:
                # Constant load - start all users at once
                for user_id in range(config.max_users):
                    task = asyncio.create_task(
                        self.simulate_user(session, user_id, behavior, config.duration_seconds)
                    )
                    user_tasks.append(task)
                
                timeline.append({
                    'timestamp': time.time(),
                    'active_users': config.max_users,
                    'action': 'constant_load_started'
                })
                
                # Wait for completion
                await asyncio.sleep(config.duration_seconds)
                
            elif config.pattern == LoadPattern.RAMP_UP:
                # Gradual ramp up
                ramp_duration = config.ramp_duration or config.duration_seconds // 2
                user_increment = (config.max_users - config.min_users) / ramp_duration
                
                for second in range(config.duration_seconds):
                    if second < ramp_duration:
                        # Ramp up phase
                        target_users = int(config.min_users + (user_increment * second))
                    else:
                        # Sustain phase
                        target_users = config.max_users
                    
                    # Adjust active users
                    current_users = len([t for t in user_tasks if not t.done()])
                    
                    if target_users > current_users:
                        # Add more users
                        for user_id in range(current_users, target_users):
                            task = asyncio.create_task(
                                self.simulate_user(session, user_id, behavior, 
                                                 config.duration_seconds - second)
                            )
                            user_tasks.append(task)
                    
                    timeline.append({
                        'timestamp': time.time(),
                        'active_users': target_users,
                        'action': 'ramp_adjustment'
                    })
                    
                    await asyncio.sleep(1)
            
            elif config.pattern == LoadPattern.SPIKE:
                # Spike pattern - sudden increase
                spike_start = config.duration_seconds // 3
                spike_duration = config.duration_seconds // 3
                
                # Start with minimum users
                for user_id in range(config.min_users):
                    task = asyncio.create_task(
                        self.simulate_user(session, user_id, behavior, config.duration_seconds)
                    )
                    user_tasks.append(task)
                
                await asyncio.sleep(spike_start)
                
                # Add spike users
                for user_id in range(config.min_users, config.max_users):
                    task = asyncio.create_task(
                        self.simulate_user(session, user_id, behavior, spike_duration)
                    )
                    user_tasks.append(task)
                
                timeline.append({
                    'timestamp': time.time(),
                    'active_users': config.max_users,
                    'action': 'spike_started'
                })
                
                await asyncio.sleep(spike_duration)
                
                timeline.append({
                    'timestamp': time.time(),
                    'active_users': config.min_users,
                    'action': 'spike_ended'
                })
                
                # Wait for remaining duration
                await asyncio.sleep(config.duration_seconds - spike_start - spike_duration)
            
            elif config.pattern == LoadPattern.STEP:
                # Step-wise increases
                step_size = config.step_size or 10
                step_duration = config.step_duration or 30
                current_users = config.min_users
                
                while current_users <= config.max_users and time.time() - start_time < config.duration_seconds:
                    # Add users for this step
                    for user_id in range(len(user_tasks), current_users):
                        remaining_time = config.duration_seconds - (time.time() - start_time)
                        task = asyncio.create_task(
                            self.simulate_user(session, user_id, behavior, remaining_time)
                        )
                        user_tasks.append(task)
                    
                    timeline.append({
                        'timestamp': time.time(),
                        'active_users': current_users,
                        'action': f'step_to_{current_users}_users'
                    })
                    
                    await asyncio.sleep(min(step_duration, config.duration_seconds - (time.time() - start_time)))
                    current_users += step_size
            
            # Cancel any remaining tasks
            for task in user_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for all tasks to complete or be cancelled
            await asyncio.gather(*user_tasks, return_exceptions=True)
        
        actual_duration = time.time() - start_time
        
        # Analyze results
        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]
        
        response_times = [r['duration'] for r in successful]
        
        # Error analysis
        errors_by_type = {}
        for result in failed:
            error_type = result.get('error', 'unknown')
            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
        
        peak_users = max([entry['active_users'] for entry in timeline]) if timeline else 0
        
        result = LoadGenerationResult(
            config=config,
            total_requests=len(self.results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            actual_duration=actual_duration,
            peak_concurrent_users=peak_users,
            avg_response_time=sum(response_times) / len(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            requests_per_second=len(self.results) / actual_duration if actual_duration > 0 else 0,
            errors_by_type=errors_by_type,
            timeline=timeline
        )
        
        logger.info(f"Load generation completed: {len(successful)}/{len(self.results)} successful requests")
        return result


class ThreadBasedLoadGenerator:
    # Thread-based load generator for scenarios where async is not suitable
    
    def __init__(self, max_workers: int = 50):
        self.max_workers = max_workers
        self.results = []
        self._lock = threading.Lock()
    
    def make_request(self, endpoint: str, user_id: int, base_url: str) -> Dict[str, Any]:
        # Make a single HTTP request using requests library
        import requests
        
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        start_time = time.time()
        
        try:
            response = requests.get(url, timeout=30)
            end_time = time.time()
            
            result = {
                'user_id': user_id,
                'endpoint': endpoint,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'status_code': response.status_code,
                'success': 200 <= response.status_code < 300,
                'response_size': len(response.content),
                'error': None
            }
            
        except Exception as e:
            end_time = time.time()
            result = {
                'user_id': user_id,
                'endpoint': endpoint,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'status_code': None,
                'success': False,
                'response_size': 0,
                'error': str(e)
            }
        
        with self._lock:
            self.results.append(result)
        
        return result
    
    def simulate_user_thread(self, user_id: int, behavior: UserBehavior, 
                           base_url: str, duration: float) -> List[Dict[str, Any]]:
        # Simulate user in a thread
        user_results = []
        start_time = time.time()
        session_end = start_time + duration
        
        while time.time() < session_end:
            # Select endpoint
            endpoint = random.choices(behavior.endpoints, weights=behavior.weights)[0]
            
            # Make request
            result = self.make_request(endpoint, user_id, base_url)
            user_results.append(result)
            
            # Think time
            if behavior.random_delay:
                think_time = random.uniform(behavior.think_time_min, behavior.think_time_max)
            else:
                think_time = (behavior.think_time_min + behavior.think_time_max) / 2
            
            time.sleep(think_time)
        
        return user_results
    
    def generate_concurrent_load(self, base_url: str, behavior: UserBehavior, 
                               concurrent_users: int, duration: float) -> Dict[str, Any]:
        # Generate load with fixed number of concurrent users
        logger.info(f"Starting thread-based load test with {concurrent_users} users for {duration}s")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=min(concurrent_users, self.max_workers)) as executor:
            futures = []
            for user_id in range(concurrent_users):
                future = executor.submit(
                    self.simulate_user_thread, 
                    user_id, behavior, base_url, duration
                )
                futures.append(future)
            
            # Wait for all users to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"User simulation failed: {e}")
        
        actual_duration = time.time() - start_time
        
        # Analyze results
        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]
        response_times = [r['duration'] for r in successful]
        
        return {
            'total_requests': len(self.results),
            'successful_requests': len(successful),
            'failed_requests': len(failed),
            'actual_duration': actual_duration,
            'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'requests_per_second': len(self.results) / actual_duration if actual_duration > 0 else 0,
            'success_rate': len(successful) / len(self.results) * 100 if self.results else 0
        }


# Predefined user behaviors for common scenarios
STANDARD_BEHAVIORS = {
    'api_explorer': UserBehavior(
        name='api_explorer',
        endpoints=['all', 'name/germany', 'alpha/us', 'currency/usd', 'region/europe'],
        weights=[0.3, 0.2, 0.2, 0.15, 0.15],
        think_time_min=1.0,
        think_time_max=3.0
    ),
    
    'heavy_user': UserBehavior(
        name='heavy_user',
        endpoints=['all', 'region/asia', 'region/europe', 'region/americas'],
        weights=[0.4, 0.2, 0.2, 0.2],
        think_time_min=0.5,
        think_time_max=1.5
    ),
    
    'light_user': UserBehavior(
        name='light_user',
        endpoints=['name/france', 'alpha/de', 'alpha/jp'],
        weights=[0.4, 0.3, 0.3],
        think_time_min=2.0,
        think_time_max=5.0
    )
}


def create_load_config(pattern: str, duration: int, min_users: int, max_users: int, **kwargs) -> LoadConfig:
    # Helper function to create load configuration
    return LoadConfig(
        pattern=LoadPattern(pattern),
        duration_seconds=duration,
        min_users=min_users,
        max_users=max_users,
        **kwargs
    )


async def run_async_load_test(base_url: str, config: LoadConfig, 
                             behavior: UserBehavior) -> LoadGenerationResult:
    # Convenience function to run async load test
    generator = AsyncLoadGenerator(base_url)
    return await generator.generate_load_pattern(config, behavior)


def run_thread_load_test(base_url: str, behavior: UserBehavior, 
                        concurrent_users: int, duration: float) -> Dict[str, Any]:
    # Convenience function to run thread-based load test
    generator = ThreadBasedLoadGenerator()
    return generator.generate_concurrent_load(base_url, behavior, concurrent_users, duration)