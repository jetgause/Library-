#!/usr/bin/env python3
"""
Standalone Performance Test Validation Script

This script validates the performance testing suite without requiring pytest.
It can be run directly with Python to verify the implementation.
"""

import sys
import os
import time
import importlib.util

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_production_tool_executor():
    """Test the production tool executor module."""
    print("Testing production_tool_executor...")
    
    try:
        from production_tool_executor import execute_tool
        
        # Test basic execution
        result = execute_tool(tool_id=1, data={"price": 100, "volume": 1000}, symbol="SPY")
        assert result is not None, "execute_tool returned None"
        assert "signal" in result, "Result missing 'signal' field"
        assert "tool_id" in result, "Result missing 'tool_id' field"
        assert result["tool_id"] == 1, "Wrong tool_id returned"
        
        print("✅ production_tool_executor tests passed")
        return True
    except Exception as e:
        print(f"❌ production_tool_executor tests failed: {e}")
        return False


def test_performance_monitor():
    """Test the performance monitor module."""
    print("\nTesting performance_monitor...")
    
    try:
        # Import directly to avoid pulse_core __init__.py
        import importlib.util
        monitor_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "pulse_core", "performance_monitor.py"
        )
        
        spec = importlib.util.spec_from_file_location("performance_monitor", monitor_path)
        perf_monitor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(perf_monitor)
        
        # Test basic monitoring
        monitor = perf_monitor.PerformanceMonitor()
        
        # Record some metrics
        monitor.record_request("/test", 0.05, 200)
        monitor.record_request("/test", 0.03, 200)
        monitor.record_request("/error", 0.1, 500)
        
        # Get stats
        stats = monitor.get_response_time_stats()
        assert "avg" in stats, "Stats missing 'avg' field"
        assert "p95" in stats, "Stats missing 'p95' field"
        
        summary = monitor.get_summary()
        assert "requests" in summary, "Summary missing 'requests' field"
        assert summary["requests"]["total"] == 3, "Wrong request count"
        assert summary["requests"]["error_count"] == 1, "Wrong error count"
        
        print("✅ performance_monitor tests passed")
        return True
    except Exception as e:
        print(f"❌ performance_monitor tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_test_script():
    """Test the load test script can be imported."""
    print("\nTesting load_test script...")
    
    try:
        # Check if file exists
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "load_test.py"
        )
        
        assert os.path.exists(script_path), f"load_test.py not found at {script_path}"
        
        # Read file and check for required functions
        with open(script_path, 'r') as f:
            content = f.read()
        
        required_items = [
            "async def make_request",
            "async def user_simulation",
            "async def run_load_test",
            "import aiohttp",
            "import asyncio"
        ]
        
        for item in required_items:
            assert item in content, f"Missing required item: {item}"
        
        print("✅ load_test script tests passed")
        return True
    except Exception as e:
        print(f"❌ load_test script tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_optimizer():
    """Test the database optimizer script."""
    print("\nTesting database optimizer...")
    
    try:
        # Check if file exists
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "optimize_database.py"
        )
        
        assert os.path.exists(script_path), f"optimize_database.py not found"
        
        # Try to import it
        spec = importlib.util.spec_from_file_location("optimize_database", script_path)
        optimize_db = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(optimize_db)
        
        # Check for required class
        assert hasattr(optimize_db, "DatabaseOptimizer"), "Missing DatabaseOptimizer class"
        
        # Create instance (without connecting to DB)
        optimizer = optimize_db.DatabaseOptimizer(":memory:")
        
        print("✅ database optimizer tests passed")
        return True
    except Exception as e:
        print(f"❌ database optimizer tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_test_suite():
    """Test the performance test suite file."""
    print("\nTesting performance test suite...")
    
    try:
        # Check if file exists
        test_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests", "test_performance.py"
        )
        
        assert os.path.exists(test_path), f"test_performance.py not found"
        
        # Read the file and check for required test classes
        with open(test_path, 'r') as f:
            content = f.read()
        
        required_classes = [
            "TestAPIPerformance",
            "TestTradingToolPerformance",
            "TestMemoryUsage",
            "TestDatabasePerformance",
            "TestConcurrency"
        ]
        
        for test_class in required_classes:
            assert test_class in content, f"Missing test class: {test_class}"
        
        print("✅ performance test suite validation passed")
        return True
    except Exception as e:
        print(f"❌ performance test suite validation failed: {e}")
        return False


def test_ci_workflow():
    """Test CI workflow has performance testing job."""
    print("\nTesting CI workflow configuration...")
    
    try:
        workflow_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".github", "workflows", "ci.yml"
        )
        
        assert os.path.exists(workflow_path), "ci.yml not found"
        
        with open(workflow_path, 'r') as f:
            content = f.read()
        
        assert "performance-test:" in content, "Missing performance-test job"
        assert "pytest tests/test_performance.py" in content, "Missing performance test execution"
        
        print("✅ CI workflow validation passed")
        return True
    except Exception as e:
        print(f"❌ CI workflow validation failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("="*70)
    print("PERFORMANCE TESTING SUITE VALIDATION")
    print("="*70)
    
    tests = [
        test_production_tool_executor,
        test_performance_monitor,
        test_load_test_script,
        test_database_optimizer,
        test_performance_test_suite,
        test_ci_workflow
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✅ ALL VALIDATIONS PASSED")
        return 0
    else:
        print("\n❌ SOME VALIDATIONS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
