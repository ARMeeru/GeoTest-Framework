# GeoTest Performance Engine Internals

## Overview

The Performance Engine ([src/performance.py](cci:7://file:///home/peter/Descargas/GeoTest-Framework-engineering-quality/src/performance.py:0:0-0:0)) serves as the central orchestration layer for all load and stress testing. It integrates metric collection, reporting, and execution strategies.

## Execution Strategies

### 1. Synchronous Thread-Based (Legacy/Low Load)

Used for functional load testing where precise sequence timing is critical over raw throughput.

- **Implementation**: `UnifiedPerformanceEngine.load_test`
- **Mechanism**: Python `ThreadPoolExecutor` + `requests` (Blocking I/O)
- **Limit**: ~50-100 concurrent users before context switching overhead degrades accuracy.

### 2. Asynchronous Event-Based (High Load)

**[RECOMMENDED INTEGRATION]**
Leverages the non-blocking I/O capabilities defined in [src/load_generator.py](cci:7://file:///home/peter/Descargas/GeoTest-Framework-engineering-quality/src/load_generator.py:0:0-0:0).

- **Implementation**: [AsyncLoadGenerator](cci:2://file:///home/peter/Descargas/GeoTest-Framework-engineering-quality/src/load_generator.py:71:0-338:21)
- **Mechanism**: `asyncio` + `aiohttp` (Single-threaded event loop)
- **Capacity**: 1000+ concurrent users per node.

## Architecture Status

**[RESOLVED]** The `UnifiedPerformanceEngine` has been successfully refactored to integrate the `AsyncLoadGenerator`. It now automatically switches to the asynchronous strategy when user load exceeds 50 concurrent users, or when explicitly requested via `async_mode=True`.

## Usage Pattern via Pytest

```python
# Synchronous (Current default)
pytest -m "performance"

# Asynchronous (Proposed Benchmark)
# Utilizes the AsyncLoadGenerator for massive scale
pytest -m "load_test" --async-mode
