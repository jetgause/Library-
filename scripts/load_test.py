"""
Load Testing Script for PULSE API
Usage: python scripts/load_test.py --users 100 --duration 60

This script simulates multiple concurrent users making requests to the PULSE API
and provides detailed performance metrics.
"""
import asyncio
import aiohttp
import time
import argparse
from typing import Dict, List
import statistics
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def make_request(session: aiohttp.ClientSession, url: str) -> Dict:
    """Make a single request and return metrics."""
    start = time.time()
    try:
        async with session.get(url) as response:
            await response.read()
            return {
                "status": response.status,
                "duration": time.time() - start,
                "success": response.status == 200
            }
    except Exception as e:
        return {
            "status": 0,
            "duration": time.time() - start,
            "success": False,
            "error": str(e)
        }


async def user_simulation(user_id: int, base_url: str, duration: int) -> List[Dict]:
    """Simulate a single user making requests."""
    results = []
    async with aiohttp.ClientSession() as session:
        end_time = time.time() + duration
        
        while time.time() < end_time:
            result = await make_request(session, f"{base_url}/health")
            results.append(result)
            await asyncio.sleep(0.1)  # 10 requests per second per user
    
    return results


async def run_load_test(users: int, duration: int, base_url: str):
    """Run load test with specified parameters."""
    print(f"Starting load test: {users} users for {duration} seconds")
    print(f"Target: {base_url}")
    print(f"Expected requests per user: ~{duration * 10}")
    print(f"Expected total requests: ~{users * duration * 10}")
    print("-" * 60)
    
    start = time.time()
    
    # Run all user simulations concurrently
    tasks = [user_simulation(i, base_url, duration) for i in range(users)]
    all_results = await asyncio.gather(*tasks)
    
    # Flatten results
    results = [r for user_results in all_results for r in user_results]
    
    # Calculate metrics
    total_requests = len(results)
    successful = sum(1 for r in results if r["success"])
    failed = total_requests - successful
    durations = [r["duration"] for r in results if r["success"]]
    
    elapsed = time.time() - start
    
    print("\n" + "="*60)
    print("LOAD TEST RESULTS")
    print("="*60)
    print(f"Total Requests: {total_requests}")
    print(f"Successful: {successful} ({successful/total_requests*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total_requests*100:.1f}%)")
    print(f"Duration: {elapsed:.2f}s")
    
    if durations:
        print(f"\nResponse Times:")
        print(f"  Min: {min(durations)*1000:.2f}ms")
        print(f"  Max: {max(durations)*1000:.2f}ms")
        print(f"  Avg: {statistics.mean(durations)*1000:.2f}ms")
        print(f"  Median: {statistics.median(durations)*1000:.2f}ms")
        
        # Calculate percentiles
        sorted_durations = sorted(durations)
        p95_idx = int(len(sorted_durations) * 0.95)
        p99_idx = int(len(sorted_durations) * 0.99)
        print(f"  95th percentile: {sorted_durations[p95_idx]*1000:.2f}ms")
        print(f"  99th percentile: {sorted_durations[p99_idx]*1000:.2f}ms")
    
    print(f"\nThroughput: {total_requests/elapsed:.2f} req/s")
    print(f"Avg per user: {total_requests/users:.0f} requests")
    print("="*60)
    
    # Success criteria checks
    print("\n" + "="*60)
    print("SUCCESS CRITERIA")
    print("="*60)
    
    checks = []
    
    # Check success rate
    success_rate = successful / total_requests * 100
    checks.append(("Success rate > 99%", success_rate > 99, f"{success_rate:.1f}%"))
    
    # Check average response time
    if durations:
        avg_response = statistics.mean(durations) * 1000
        checks.append(("Avg response < 100ms", avg_response < 100, f"{avg_response:.2f}ms"))
        
        # Check 95th percentile
        p95_response = sorted_durations[p95_idx] * 1000
        checks.append(("P95 response < 200ms", p95_response < 200, f"{p95_response:.2f}ms"))
    
    # Check throughput
    throughput = total_requests / elapsed
    checks.append(("Throughput > 100 req/s", throughput > 100, f"{throughput:.2f} req/s"))
    
    for check_name, passed, value in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check_name} - {value}")
    
    all_passed = all(check[1] for check in checks)
    print("="*60)
    print(f"\nOverall: {'✅ ALL CHECKS PASSED' if all_passed else '❌ SOME CHECKS FAILED'}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PULSE API Load Testing")
    parser.add_argument("--users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_load_test(args.users, args.duration, args.url))
    except KeyboardInterrupt:
        print("\n\nLoad test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError running load test: {e}")
        sys.exit(1)
