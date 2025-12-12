# Performance Testing & Monitoring

This directory contains comprehensive performance testing and monitoring tools for the PULSE Trading Platform.

## Overview

The performance testing suite provides:

- **API Performance Tests**: Measure response times and throughput
- **Load Testing**: Simulate concurrent users and stress test the system
- **Performance Monitoring**: Real-time metrics collection and analysis
- **Database Optimization**: Tools for identifying and fixing performance bottlenecks

## Files

### Test Suite

- **`tests/test_performance.py`**: Comprehensive performance test suite
  - API endpoint response time tests
  - Concurrent request handling tests
  - Trading tool execution performance tests
  - Memory usage profiling tests
  - Database performance tests

### Scripts

- **`scripts/load_test.py`**: Load testing script for simulating concurrent users
- **`scripts/optimize_database.py`**: Database optimization utilities
- **`scripts/validate_performance.py`**: Validation script for performance testing suite

### Monitoring

- **`pulse_core/performance_monitor.py`**: Real-time performance monitoring module

### Mock Components

- **`production_tool_executor.py`**: Mock tool executor for testing

## Usage

### Running Performance Tests

With pytest (when dependencies are installed):

```bash
# Run all performance tests
pytest tests/test_performance.py -v -m performance

# Run specific test class
pytest tests/test_performance.py::TestAPIPerformance -v

# Run with benchmark output
pytest tests/test_performance.py -v --benchmark-only
```

Without pytest (using standalone validation):

```bash
# Validate performance testing implementation
python3 scripts/validate_performance.py
```

### Load Testing

Simulate concurrent users accessing the API:

```bash
# Basic load test (10 users for 30 seconds)
python scripts/load_test.py

# Custom load test
python scripts/load_test.py --users 100 --duration 60 --url http://127.0.0.1:8000

# Options:
#   --users N       Number of concurrent users (default: 10)
#   --duration N    Test duration in seconds (default: 30)
#   --url URL       Base URL to test (default: http://127.0.0.1:8000)
```

Expected output:
```
Starting load test: 100 users for 60 seconds
Target: http://127.0.0.1:8000
Expected requests per user: ~600
Expected total requests: ~60000
------------------------------------------------------------

============================================================
LOAD TEST RESULTS
============================================================
Total Requests: 60000
Successful: 59995 (99.99%)
Failed: 5 (0.01%)
Duration: 60.05s

Response Times:
  Min: 2.50ms
  Max: 180.30ms
  Avg: 15.20ms
  Median: 12.40ms
  95th percentile: 45.60ms
  99th percentile: 89.20ms

Throughput: 999.17 req/s
Avg per user: 600 requests
============================================================
```

### Database Optimization

Analyze and optimize database performance:

```bash
# Analyze database
python scripts/optimize_database.py --analyze

# Add recommended indexes
python scripts/optimize_database.py --add-indexes

# Run VACUUM to optimize database file
python scripts/optimize_database.py --vacuum

# Run ANALYZE to update query optimizer statistics
python scripts/optimize_database.py --analyze-stats

# Run all optimizations
python scripts/optimize_database.py --add-indexes --vacuum --analyze-stats
```

### Performance Monitoring

Use the performance monitor in your code:

```python
from pulse_core.performance_monitor import get_monitor, record_request

# Record a request
record_request("/api/v1/execute", duration=0.05, status_code=200)

# Get performance summary
monitor = get_monitor()
summary = monitor.get_summary()

print(f"Total requests: {summary['requests']['total']}")
print(f"Request rate: {summary['requests']['rate_per_second']:.2f} req/s")
print(f"Avg response time: {summary['response_times']['avg']:.2f}ms")
print(f"P95 response time: {summary['response_times']['p95']:.2f}ms")
```

## Performance Targets

The performance testing suite validates the following targets:

### API Response Times
- Health endpoint: < 50ms
- API endpoints: < 200ms (95th percentile)
- Root endpoint: < 100ms

### Concurrency
- Handle 100+ concurrent users without degradation
- Process 100+ requests per second

### Database
- Query execution: < 50ms (average)
- No memory leaks under load (< 50MB growth per 1000 operations)

### System Resources
- Memory usage stable under load
- CPU usage reasonable for workload

## CI/CD Integration

Performance tests are automatically run in the CI/CD pipeline:

```yaml
performance-test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v3
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest-benchmark
    - name: Run performance tests
      run: pytest tests/test_performance.py -v -m performance
```

## Dependencies

Required packages for performance testing:

```
pytest>=7.0.0
pytest-benchmark>=4.0.0
pytest-asyncio>=0.21.0
psutil>=5.9.0
aiohttp>=3.9.0
fastapi>=0.104.0
uvicorn>=0.24.0
httpx>=0.25.0
```

Install with:

```bash
pip install -r requirements.txt
```

## Troubleshooting

### Import Errors

If you encounter module import errors, ensure you're running from the repository root:

```bash
cd /path/to/Library-
python3 scripts/load_test.py
```

### Missing Dependencies

If dependencies are not installed:

```bash
pip install -r requirements.txt
```

### API Server Not Running

For load testing, ensure the API server is running:

```bash
# Terminal 1: Start the server
python api_server.py

# Terminal 2: Run load test
python scripts/load_test.py
```

## Contributing

When adding new performance tests:

1. Add test methods to appropriate test classes in `tests/test_performance.py`
2. Mark tests with `@pytest.mark.performance` decorator
3. Include assertions with clear failure messages
4. Document expected performance targets in docstrings
5. Update this README with new test descriptions

## License

Part of the PULSE Trading Platform project.
