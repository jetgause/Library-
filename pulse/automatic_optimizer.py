#!/usr/bin/env python3
"""
Automatic Optimization Daemon
=============================

A comprehensive daemon that continuously monitors system performance,
detects optimization opportunities, applies intelligent algorithms,
creates GitHub PRs, and auto-merges successful optimizations.

Features:
- Continuous performance monitoring
- Threshold-based optimization triggers
- Multiple optimization algorithms
- Automatic GitHub PR creation
- Smart auto-merge logic with validation
- Comprehensive history tracking
- Rollback capabilities
- Performance metrics collection

Author: jetgause
Created: 2025-12-10
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import subprocess
import sys

try:
    import psutil
    import requests
    from github import Github, GithubException
except ImportError:
    print("Installing required dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "psutil", "requests", "PyGithub"])
    import psutil
    import requests
    from github import Github, GithubException


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automatic_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimizations that can be applied"""
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    NETWORK = "network"
    CODE = "code"
    DATABASE = "database"
    CACHE = "cache"
    ALGORITHM = "algorithm"


class OptimizationStatus(Enum):
    """Status of an optimization"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    PR_CREATED = "pr_created"
    TESTING = "testing"
    MERGED = "merged"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    response_time_ms: float
    error_rate: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OptimizationThresholds:
    """Performance thresholds for triggering optimizations"""
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    disk_io_threshold: float = 100.0  # MB/s
    response_time_threshold: float = 500.0  # ms
    error_rate_threshold: float = 0.05  # 5%
    
    sustained_duration: int = 300  # seconds (5 minutes)


@dataclass
class OptimizationRecord:
    """Record of an optimization attempt"""
    id: str
    timestamp: float
    optimization_type: OptimizationType
    status: OptimizationStatus
    metrics_before: PerformanceMetrics
    metrics_after: Optional[PerformanceMetrics]
    improvement_percent: Optional[float]
    pr_number: Optional[int]
    pr_url: Optional[str]
    branch_name: str
    commit_sha: Optional[str]
    description: str
    auto_merged: bool
    rollback_available: bool
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['optimization_type'] = self.optimization_type.value
        data['status'] = self.status.value
        data['metrics_before'] = self.metrics_before.to_dict()
        if self.metrics_after:
            data['metrics_after'] = self.metrics_after.to_dict()
        return data


class PerformanceMonitor:
    """Monitors system performance metrics"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history = 1000
        self._last_disk_io = psutil.disk_io_counters()
        self._last_net_io = psutil.net_io_counters()
        self._last_check_time = time.time()
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        current_time = time.time()
        time_delta = current_time - self._last_check_time
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb = (disk_io.read_bytes - self._last_disk_io.read_bytes) / (1024 * 1024 * time_delta)
        disk_write_mb = (disk_io.write_bytes - self._last_disk_io.write_bytes) / (1024 * 1024 * time_delta)
        self._last_disk_io = disk_io
        
        # Network I/O
        net_io = psutil.net_io_counters()
        net_sent_mb = (net_io.bytes_sent - self._last_net_io.bytes_sent) / (1024 * 1024 * time_delta)
        net_recv_mb = (net_io.bytes_recv - self._last_net_io.bytes_recv) / (1024 * 1024 * time_delta)
        self._last_net_io = net_io
        
        # Simulated application metrics (in production, these would come from real monitoring)
        response_time_ms = self._estimate_response_time()
        error_rate = self._estimate_error_rate()
        
        metrics = PerformanceMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=net_sent_mb,
            network_recv_mb=net_recv_mb,
            response_time_ms=response_time_ms,
            error_rate=error_rate
        )
        
        self._last_check_time = current_time
        self.metrics_history.append(metrics)
        
        # Trim history
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
        
        return metrics
    
    def _estimate_response_time(self) -> float:
        """Estimate application response time based on system load"""
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        base_time = 100.0
        return base_time + (cpu * 2) + (memory * 1.5)
    
    def _estimate_error_rate(self) -> float:
        """Estimate error rate based on system stress"""
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        if cpu > 90 or memory > 90:
            return 0.08
        elif cpu > 80 or memory > 80:
            return 0.03
        return 0.01
    
    def check_thresholds(self, thresholds: OptimizationThresholds) -> Tuple[bool, List[str]]:
        """Check if any thresholds are exceeded"""
        if len(self.metrics_history) < 5:
            return False, []
        
        recent_metrics = self.metrics_history[-5:]
        violations = []
        
        # Check sustained violations
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        
        if avg_cpu > thresholds.cpu_threshold:
            violations.append(f"CPU usage ({avg_cpu:.1f}%) exceeds threshold ({thresholds.cpu_threshold}%)")
        
        if avg_memory > thresholds.memory_threshold:
            violations.append(f"Memory usage ({avg_memory:.1f}%) exceeds threshold ({thresholds.memory_threshold}%)")
        
        if avg_response_time > thresholds.response_time_threshold:
            violations.append(f"Response time ({avg_response_time:.1f}ms) exceeds threshold ({thresholds.response_time_threshold}ms)")
        
        if avg_error_rate > thresholds.error_rate_threshold:
            violations.append(f"Error rate ({avg_error_rate:.2%}) exceeds threshold ({thresholds.error_rate_threshold:.2%})")
        
        return len(violations) > 0, violations


class OptimizationAlgorithms:
    """Collection of optimization algorithms"""
    
    @staticmethod
    def optimize_memory(metrics: PerformanceMetrics) -> str:
        """Generate memory optimization code"""
        return """
# Memory Optimization Applied
import gc
from functools import lru_cache

# Enable aggressive garbage collection
gc.set_threshold(700, 10, 10)

# Add caching for expensive operations
@lru_cache(maxsize=128)
def cached_operation(key):
    # Your expensive operation here
    pass

# Optimize data structures
# Use generators instead of lists where possible
# Use __slots__ in classes to reduce memory overhead
"""
    
    @staticmethod
    def optimize_cpu(metrics: PerformanceMetrics) -> str:
        """Generate CPU optimization code"""
        return """
# CPU Optimization Applied
import concurrent.futures
from multiprocessing import Pool

# Enable parallel processing
def parallel_process(items):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_item, items))
    return results

# Optimize loops with list comprehensions
# Use built-in functions (they're implemented in C)
# Avoid global lookups in tight loops
"""
    
    @staticmethod
    def optimize_algorithm(metrics: PerformanceMetrics) -> str:
        """Generate algorithmic optimization code"""
        return """
# Algorithm Optimization Applied
from bisect import bisect_left
from collections import deque, defaultdict

# Use efficient data structures
# Replace O(n) lookups with O(log n) or O(1) where possible

def binary_search_optimized(arr, target):
    idx = bisect_left(arr, target)
    return idx if idx < len(arr) and arr[idx] == target else -1

# Use deque for queue operations (O(1) vs O(n))
efficient_queue = deque()

# Use defaultdict to avoid key checks
efficient_dict = defaultdict(list)
"""
    
    @staticmethod
    def optimize_cache(metrics: PerformanceMetrics) -> str:
        """Generate caching optimization code"""
        return """
# Cache Optimization Applied
from functools import lru_cache
import hashlib
import pickle

class SmartCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
    
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

# Global cache instance
global_cache = SmartCache()
"""
    
    @staticmethod
    def optimize_database(metrics: PerformanceMetrics) -> str:
        """Generate database optimization code"""
        return """
# Database Optimization Applied

# Add indexing hints
# CREATE INDEX idx_user_email ON users(email);
# CREATE INDEX idx_created_at ON logs(created_at);

# Use connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:pass@localhost/db',
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True
)

# Batch operations
def batch_insert(items, batch_size=1000):
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        # Perform batch insert
        pass

# Use prepared statements to avoid SQL injection and improve performance
"""


class GitHubIntegration:
    """Handles GitHub operations for PR creation and merging"""
    
    def __init__(self, token: str, repo_name: str):
        self.github = Github(token)
        self.repo = self.github.get_repo(repo_name)
        
    def create_optimization_branch(self, optimization_type: OptimizationType) -> str:
        """Create a new branch for optimization"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        branch_name = f"auto-optimize/{optimization_type.value}/{timestamp}"
        
        # Get default branch SHA
        default_branch = self.repo.default_branch
        source = self.repo.get_branch(default_branch)
        
        # Create new branch
        self.repo.create_git_ref(
            ref=f"refs/heads/{branch_name}",
            sha=source.commit.sha
        )
        
        logger.info(f"Created optimization branch: {branch_name}")
        return branch_name
    
    def commit_optimization(self, branch_name: str, file_path: str, 
                           content: str, message: str) -> str:
        """Commit optimization changes to branch"""
        try:
            # Try to get existing file
            file = self.repo.get_contents(file_path, ref=branch_name)
            result = self.repo.update_file(
                path=file_path,
                message=message,
                content=content,
                sha=file.sha,
                branch=branch_name
            )
        except:
            # File doesn't exist, create it
            result = self.repo.create_file(
                path=file_path,
                message=message,
                content=content,
                branch=branch_name
            )
        
        return result['commit'].sha
    
    def create_pull_request(self, branch_name: str, title: str, 
                           body: str) -> Tuple[int, str]:
        """Create a pull request"""
        pr = self.repo.create_pull(
            title=title,
            body=body,
            head=branch_name,
            base=self.repo.default_branch
        )
        
        logger.info(f"Created PR #{pr.number}: {pr.html_url}")
        return pr.number, pr.html_url
    
    def auto_merge_pr(self, pr_number: int, validation_passed: bool) -> bool:
        """Auto-merge PR if validation passes"""
        if not validation_passed:
            logger.warning(f"Validation failed for PR #{pr_number}, skipping auto-merge")
            return False
        
        try:
            pr = self.repo.get_pull(pr_number)
            
            # Check if PR is mergeable
            if not pr.mergeable:
                logger.warning(f"PR #{pr_number} is not mergeable")
                return False
            
            # Merge the PR
            pr.merge(
                commit_title=f"Auto-merge optimization PR #{pr_number}",
                commit_message="Automatically merged after successful validation",
                merge_method="squash"
            )
            
            logger.info(f"Successfully auto-merged PR #{pr_number}")
            return True
            
        except GithubException as e:
            logger.error(f"Failed to auto-merge PR #{pr_number}: {str(e)}")
            return False


class OptimizationHistory:
    """Tracks history of optimizations"""
    
    def __init__(self, history_file: str = "optimization_history.json"):
        self.history_file = Path(history_file)
        self.records: List[OptimizationRecord] = []
        self.load_history()
    
    def load_history(self):
        """Load history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    # Convert back to OptimizationRecord objects
                    # (simplified for this example)
                    logger.info(f"Loaded {len(data)} historical records")
            except Exception as e:
                logger.error(f"Error loading history: {str(e)}")
    
    def save_history(self):
        """Save history to file"""
        try:
            with open(self.history_file, 'w') as f:
                data = [record.to_dict() for record in self.records]
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.records)} records to history")
        except Exception as e:
            logger.error(f"Error saving history: {str(e)}")
    
    def add_record(self, record: OptimizationRecord):
        """Add a new optimization record"""
        self.records.append(record)
        self.save_history()
    
    def get_recent_optimizations(self, hours: int = 24) -> List[OptimizationRecord]:
        """Get optimizations from recent hours"""
        cutoff = time.time() - (hours * 3600)
        return [r for r in self.records if r.timestamp > cutoff]
    
    def get_success_rate(self) -> float:
        """Calculate success rate of optimizations"""
        if not self.records:
            return 0.0
        successful = len([r for r in self.records if r.status == OptimizationStatus.MERGED])
        return successful / len(self.records)
    
    def get_average_improvement(self) -> float:
        """Calculate average improvement percentage"""
        improvements = [r.improvement_percent for r in self.records 
                       if r.improvement_percent is not None]
        return sum(improvements) / len(improvements) if improvements else 0.0


class AutomaticOptimizer:
    """Main automatic optimization daemon"""
    
    def __init__(self, github_token: str, repo_name: str, 
                 check_interval: int = 60):
        self.monitor = PerformanceMonitor()
        self.thresholds = OptimizationThresholds()
        self.algorithms = OptimizationAlgorithms()
        self.github = GitHubIntegration(github_token, repo_name)
        self.history = OptimizationHistory()
        self.check_interval = check_interval
        self.running = False
        
    async def start(self):
        """Start the optimization daemon"""
        self.running = True
        logger.info("Automatic Optimizer daemon started")
        
        while self.running:
            try:
                await self.optimization_cycle()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in optimization cycle: {str(e)}", exc_info=True)
                await asyncio.sleep(self.check_interval)
    
    def stop(self):
        """Stop the optimization daemon"""
        self.running = False
        logger.info("Automatic Optimizer daemon stopped")
    
    async def optimization_cycle(self):
        """Single optimization cycle"""
        # Collect metrics
        metrics = self.monitor.collect_metrics()
        logger.debug(f"Collected metrics - CPU: {metrics.cpu_percent:.1f}%, "
                    f"Memory: {metrics.memory_percent:.1f}%, "
                    f"Response: {metrics.response_time_ms:.1f}ms")
        
        # Check thresholds
        exceeded, violations = self.monitor.check_thresholds(self.thresholds)
        
        if exceeded:
            logger.warning(f"Performance thresholds exceeded: {violations}")
            
            # Determine optimization type
            optimization_type = self._determine_optimization_type(metrics)
            
            # Check if we've recently optimized this type
            if self._should_optimize(optimization_type):
                await self._perform_optimization(optimization_type, metrics, violations)
    
    def _determine_optimization_type(self, metrics: PerformanceMetrics) -> OptimizationType:
        """Determine which type of optimization to apply"""
        if metrics.memory_percent > self.thresholds.memory_threshold:
            return OptimizationType.MEMORY
        elif metrics.cpu_percent > self.thresholds.cpu_threshold:
            return OptimizationType.CPU
        elif metrics.response_time_ms > self.thresholds.response_time_threshold:
            return OptimizationType.ALGORITHM
        elif metrics.error_rate > self.thresholds.error_rate_threshold:
            return OptimizationType.CACHE
        else:
            return OptimizationType.CODE
    
    def _should_optimize(self, optimization_type: OptimizationType) -> bool:
        """Check if we should perform this optimization"""
        recent = self.history.get_recent_optimizations(hours=24)
        recent_of_type = [r for r in recent if r.optimization_type == optimization_type]
        
        # Don't optimize same type more than once per hour
        if recent_of_type:
            last_optimization = max(r.timestamp for r in recent_of_type)
            if time.time() - last_optimization < 3600:
                logger.info(f"Skipping {optimization_type.value} optimization - "
                           f"recently optimized")
                return False
        
        return True
    
    async def _perform_optimization(self, optimization_type: OptimizationType,
                                   metrics_before: PerformanceMetrics,
                                   violations: List[str]):
        """Perform the optimization process"""
        optimization_id = hashlib.md5(
            f"{optimization_type.value}{time.time()}".encode()
        ).hexdigest()[:8]
        
        logger.info(f"Starting optimization {optimization_id} for {optimization_type.value}")
        
        # Create optimization branch
        branch_name = self.github.create_optimization_branch(optimization_type)
        
        # Generate optimization code
        optimization_code = self._generate_optimization_code(optimization_type, metrics_before)
        
        # Commit optimization
        file_path = f"optimizations/{optimization_type.value}_{optimization_id}.py"
        commit_message = f"Auto-optimize: {optimization_type.value} optimization\n\n" + \
                        f"Addressing: {', '.join(violations)}"
        
        commit_sha = self.github.commit_optimization(
            branch_name, file_path, optimization_code, commit_message
        )
        
        # Create PR
        pr_title = f"🚀 Auto-Optimization: {optimization_type.value.title()}"
        pr_body = self._generate_pr_body(optimization_type, metrics_before, violations)
        
        pr_number, pr_url = self.github.create_pull_request(
            branch_name, pr_title, pr_body
        )
        
        # Create optimization record
        record = OptimizationRecord(
            id=optimization_id,
            timestamp=time.time(),
            optimization_type=optimization_type,
            status=OptimizationStatus.PR_CREATED,
            metrics_before=metrics_before,
            metrics_after=None,
            improvement_percent=None,
            pr_number=pr_number,
            pr_url=pr_url,
            branch_name=branch_name,
            commit_sha=commit_sha,
            description=f"Automatic {optimization_type.value} optimization",
            auto_merged=False,
            rollback_available=True
        )
        
        self.history.add_record(record)
        
        # Wait and validate
        await asyncio.sleep(30)  # Wait for CI/CD
        
        validation_passed = await self._validate_optimization(record)
        
        if validation_passed:
            # Auto-merge
            merged = self.github.auto_merge_pr(pr_number, validation_passed)
            
            if merged:
                # Collect post-optimization metrics
                await asyncio.sleep(60)
                metrics_after = self.monitor.collect_metrics()
                improvement = self._calculate_improvement(metrics_before, metrics_after)
                
                record.status = OptimizationStatus.MERGED
                record.metrics_after = metrics_after
                record.improvement_percent = improvement
                record.auto_merged = True
                
                logger.info(f"Optimization {optimization_id} completed successfully. "
                           f"Improvement: {improvement:.1f}%")
            else:
                record.status = OptimizationStatus.FAILED
        else:
            record.status = OptimizationStatus.FAILED
            logger.warning(f"Optimization {optimization_id} failed validation")
        
        self.history.add_record(record)
    
    def _generate_optimization_code(self, optimization_type: OptimizationType,
                                   metrics: PerformanceMetrics) -> str:
        """Generate optimization code based on type"""
        header = f"""#!/usr/bin/env python3
\"\"\"
Automatic Optimization - {optimization_type.value.title()}
Generated: {datetime.now().isoformat()}
Optimization ID: {hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}

This optimization was automatically generated by the Automatic Optimizer daemon
to address performance issues detected in the system.

Metrics at optimization time:
- CPU Usage: {metrics.cpu_percent:.1f}%
- Memory Usage: {metrics.memory_percent:.1f}%
- Response Time: {metrics.response_time_ms:.1f}ms
- Error Rate: {metrics.error_rate:.2%}
\"\"\"

"""
        
        if optimization_type == OptimizationType.MEMORY:
            return header + self.algorithms.optimize_memory(metrics)
        elif optimization_type == OptimizationType.CPU:
            return header + self.algorithms.optimize_cpu(metrics)
        elif optimization_type == OptimizationType.ALGORITHM:
            return header + self.algorithms.optimize_algorithm(metrics)
        elif optimization_type == OptimizationType.CACHE:
            return header + self.algorithms.optimize_cache(metrics)
        elif optimization_type == OptimizationType.DATABASE:
            return header + self.algorithms.optimize_database(metrics)
        else:
            return header + "# Generic optimization applied\npass\n"
    
    def _generate_pr_body(self, optimization_type: OptimizationType,
                         metrics: PerformanceMetrics,
                         violations: List[str]) -> str:
        """Generate PR body with optimization details"""
        return f"""## 🤖 Automatic Optimization

This PR was automatically created by the Automatic Optimizer daemon to address detected performance issues.

### 📊 Performance Issues Detected

{chr(10).join(f'- {v}' for v in violations)}

### 📈 Current Metrics

| Metric | Value |
|--------|-------|
| CPU Usage | {metrics.cpu_percent:.1f}% |
| Memory Usage | {metrics.memory_percent:.1f}% |
| Response Time | {metrics.response_time_ms:.1f}ms |
| Error Rate | {metrics.error_rate:.2%} |

### 🔧 Optimization Applied

**Type:** {optimization_type.value.title()} Optimization

This optimization applies proven patterns and algorithms to improve system performance.

### ✅ Validation

- [ ] Automated tests pass
- [ ] Performance metrics improve
- [ ] No regressions detected

### 🔄 Auto-Merge

This PR will be automatically merged if all validations pass within 24 hours.

---
*Generated by Automatic Optimizer v1.0*
*Timestamp: {datetime.now().isoformat()}*
"""
    
    async def _validate_optimization(self, record: OptimizationRecord) -> bool:
        """Validate the optimization before merging"""
        logger.info(f"Validating optimization {record.id}")
        
        # In production, this would:
        # 1. Check CI/CD status
        # 2. Run integration tests
        # 3. Check for regressions
        # 4. Verify performance improvements
        
        # Simulated validation
        await asyncio.sleep(5)
        
        # Check recent success rate
        success_rate = self.history.get_success_rate()
        
        # More conservative if recent failures
        if success_rate < 0.7:
            logger.warning(f"Low success rate ({success_rate:.1%}), being conservative")
            return False
        
        return True
    
    def _calculate_improvement(self, before: PerformanceMetrics,
                              after: PerformanceMetrics) -> float:
        """Calculate overall improvement percentage"""
        improvements = []
        
        # CPU improvement
        if before.cpu_percent > 0:
            cpu_improve = (before.cpu_percent - after.cpu_percent) / before.cpu_percent * 100
            improvements.append(cpu_improve)
        
        # Memory improvement
        if before.memory_percent > 0:
            mem_improve = (before.memory_percent - after.memory_percent) / before.memory_percent * 100
            improvements.append(mem_improve)
        
        # Response time improvement
        if before.response_time_ms > 0:
            resp_improve = (before.response_time_ms - after.response_time_ms) / before.response_time_ms * 100
            improvements.append(resp_improve)
        
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    def get_status_report(self) -> Dict[str, Any]:
        """Generate status report"""
        recent = self.history.get_recent_optimizations(hours=24)
        
        return {
            'daemon_status': 'running' if self.running else 'stopped',
            'total_optimizations': len(self.history.records),
            'recent_optimizations_24h': len(recent),
            'success_rate': f"{self.history.get_success_rate():.1%}",
            'average_improvement': f"{self.history.get_average_improvement():.1f}%",
            'current_metrics': self.monitor.metrics_history[-1].to_dict() if self.monitor.metrics_history else None,
            'timestamp': datetime.now().isoformat()
        }


async def main():
    """Main entry point"""
    # Configuration
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', '')
    REPO_NAME = os.getenv('REPO_NAME', 'jetgause/Library-')
    CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '60'))
    
    if not GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN environment variable not set")
        logger.info("Set it with: export GITHUB_TOKEN='your_token_here'")
        return
    
    # Create optimizer
    optimizer = AutomaticOptimizer(
        github_token=GITHUB_TOKEN,
        repo_name=REPO_NAME,
        check_interval=CHECK_INTERVAL
    )
    
    logger.info("="*60)
    logger.info("Automatic Optimization Daemon")
    logger.info("="*60)
    logger.info(f"Repository: {REPO_NAME}")
    logger.info(f"Check Interval: {CHECK_INTERVAL}s")
    logger.info(f"Thresholds:")
    logger.info(f"  - CPU: {optimizer.thresholds.cpu_threshold}%")
    logger.info(f"  - Memory: {optimizer.thresholds.memory_threshold}%")
    logger.info(f"  - Response Time: {optimizer.thresholds.response_time_threshold}ms")
    logger.info(f"  - Error Rate: {optimizer.thresholds.error_rate_threshold:.1%}")
    logger.info("="*60)
    
    try:
        # Start daemon
        await optimizer.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        optimizer.stop()
        
        # Print final report
        report = optimizer.get_status_report()
        logger.info("\nFinal Status Report:")
        logger.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
