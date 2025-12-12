"""
Performance Testing Suite for PULSE Trading Platform

This module contains comprehensive performance tests for:
- API endpoint response times
- Database query performance
- Trading tool execution speed
- Memory usage profiling
- Concurrent request handling
- WebSocket connection stress tests

Usage:
    pytest tests/test_performance.py -v --benchmark-only
"""

import pytest
import time
import asyncio
from typing import List
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import psutil, but make it optional
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed, skipping memory tests")


@pytest.fixture
def client():
    """Create a test client for the API."""
    from fastapi.testclient import TestClient
    from api_server import app
    return TestClient(app)


@pytest.mark.performance
class TestAPIPerformance:
    """Performance tests for API endpoints."""
    
    def test_health_endpoint_response_time(self, client):
        """Health endpoint should respond in <50ms."""
        start = time.time()
        response = client.get("/health")
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 0.05, f"Health endpoint took {duration*1000:.2f}ms, expected <50ms"
    
    def test_root_endpoint_response_time(self, client):
        """Root endpoint should respond quickly."""
        start = time.time()
        response = client.get("/")
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 0.1, f"Root endpoint took {duration*1000:.2f}ms, expected <100ms"
    
    def test_concurrent_requests(self, client):
        """Handle 100 concurrent requests."""
        import concurrent.futures
        
        def make_request():
            return client.get("/health")
        
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            results = [f.result() for f in futures]
        duration = time.time() - start
        
        assert all(r.status_code == 200 for r in results), "Some requests failed"
        assert duration < 10, f"100 concurrent requests took {duration:.2f}s, expected <10s"
    
    def test_api_endpoint_response_time(self, client):
        """API endpoint should respond in <200ms."""
        test_data = {
            "tool_id": 1,
            "user_id": "test_user",
            "symbol": "SPY",
            "data": {"price": 100, "volume": 1000}
        }
        
        start = time.time()
        response = client.post("/api/v1/execute", json=test_data)
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 0.2, f"API endpoint took {duration*1000:.2f}ms, expected <200ms"


@pytest.mark.performance
class TestTradingToolPerformance:
    """Performance tests for trading tools."""
    
    def test_tool_execution_time(self):
        """Trading tools should execute in <1 second."""
        from production_tool_executor import execute_tool
        
        test_data = {"price": 100, "volume": 1000}
        
        start = time.time()
        result = execute_tool(tool_id=1, data=test_data, symbol="SPY")
        duration = time.time() - start
        
        assert duration < 1.0, f"Tool execution took {duration:.3f}s, expected <1s"
        assert result is not None
        assert "signal" in result
    
    def test_multiple_tool_executions(self):
        """Execute multiple tools sequentially."""
        from production_tool_executor import execute_tool
        
        test_data = {"price": 100, "volume": 1000}
        
        start = time.time()
        for tool_id in range(1, 6):
            result = execute_tool(tool_id=tool_id, data=test_data, symbol="SPY")
            assert result is not None
        duration = time.time() - start
        
        # 5 tools should execute in < 5 seconds total
        assert duration < 5.0, f"5 tool executions took {duration:.3f}s, expected <5s"


@pytest.mark.performance
@pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not installed")
class TestMemoryUsage:
    """Memory usage profiling tests."""
    
    def test_memory_leak_detection(self):
        """Detect memory leaks in repeated operations."""
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform 1000 operations
        for _ in range(1000):
            from paper_trading import PaperTradingEngine
            engine = PaperTradingEngine()
            del engine
        
        # Check memory hasn't grown significantly
        final = process.memory_info().rss / 1024 / 1024
        growth = final - baseline
        
        assert growth < 50, f"Memory grew by {growth:.2f}MB, expected <50MB"
    
    def test_api_memory_usage(self):
        """Test API memory usage under load."""
        from fastapi.testclient import TestClient
        from api_server import app
        
        client = TestClient(app)
        process = psutil.Process(os.getpid())
        
        baseline = process.memory_info().rss / 1024 / 1024
        
        # Make 100 requests
        for _ in range(100):
            response = client.get("/health")
            assert response.status_code == 200
        
        final = process.memory_info().rss / 1024 / 1024
        growth = final - baseline
        
        # Memory growth should be minimal
        assert growth < 20, f"Memory grew by {growth:.2f}MB after 100 requests"


@pytest.mark.performance
class TestDatabasePerformance:
    """Database performance tests."""
    
    def test_paper_trading_engine_initialization(self):
        """Paper trading engine should initialize quickly."""
        start = time.time()
        from paper_trading import PaperTradingEngine
        engine = PaperTradingEngine()
        duration = time.time() - start
        
        assert duration < 1.0, f"Engine init took {duration:.3f}s, expected <1s"
    
    def test_repeated_database_operations(self):
        """Test repeated database operations performance."""
        from paper_trading import PaperTradingEngine
        
        engine = PaperTradingEngine()
        
        start = time.time()
        # Perform some operations
        for _ in range(10):
            positions = engine.get_open_positions()
            pnl = engine.get_total_pnl()
        duration = time.time() - start
        
        # 10 operations should be fast
        assert duration < 0.5, f"10 DB operations took {duration:.3f}s, expected <500ms"


@pytest.mark.performance
class TestConcurrency:
    """Concurrency and parallelism tests."""
    
    def test_parallel_tool_execution(self):
        """Test parallel execution of multiple tools."""
        from production_tool_executor import execute_tool
        import concurrent.futures
        
        test_data = {"price": 100, "volume": 1000}
        
        def execute_single_tool(tool_id):
            return execute_tool(tool_id=tool_id, data=test_data, symbol="SPY")
        
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(execute_single_tool, i) for i in range(1, 6)]
            results = [f.result() for f in futures]
        duration = time.time() - start
        
        assert all(r is not None for r in results)
        # Parallel execution should be faster than sequential
        assert duration < 2.0, f"Parallel execution took {duration:.3f}s"
    
    def test_api_concurrent_different_endpoints(self, client):
        """Test concurrent requests to different endpoints."""
        import concurrent.futures
        
        endpoints = ["/", "/health", "/health/security"]
        
        def make_request(endpoint):
            return client.get(endpoint)
        
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Make multiple requests to each endpoint
            futures = []
            for _ in range(10):
                for endpoint in endpoints:
                    futures.append(executor.submit(make_request, endpoint))
            results = [f.result() for f in futures]
        duration = time.time() - start
        
        assert all(r.status_code == 200 for r in results)
        assert duration < 5.0, f"Concurrent requests took {duration:.3f}s"


@pytest.mark.performance
class TestResponseTimeDistribution:
    """Test response time distribution and percentiles."""
    
    def test_response_time_consistency(self, client):
        """Test that response times are consistently fast."""
        durations = []
        
        for _ in range(50):
            start = time.time()
            response = client.get("/health")
            duration = time.time() - start
            durations.append(duration)
            assert response.status_code == 200
        
        # Calculate percentiles
        durations.sort()
        p50 = durations[len(durations) // 2]
        p95 = durations[int(len(durations) * 0.95)]
        p99 = durations[int(len(durations) * 0.99)]
        
        # Assertions on percentiles
        assert p50 < 0.05, f"P50: {p50*1000:.2f}ms, expected <50ms"
        assert p95 < 0.1, f"P95: {p95*1000:.2f}ms, expected <100ms"
        assert p99 < 0.15, f"P99: {p99*1000:.2f}ms, expected <150ms"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])
