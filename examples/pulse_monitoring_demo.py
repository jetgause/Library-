#!/usr/bin/env python3
"""
PULSE System Monitoring Demo
=============================

This example demonstrates PULSE's monitoring and health check features:
- System health monitoring
- Performance metrics tracking
- Resource usage monitoring
- Alert generation
- Status reporting

Author: PULSE Development Team
Date: 2025-12-10
"""

import sys
import time
import random
from datetime import datetime
from typing import Dict, List, Optional

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class SystemMonitor:
    """Monitors system health and performance."""
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        self.thresholds = {
            'cpu': 80.0,
            'memory': 85.0,
            'error_rate': 5.0,
            'response_time': 2000  # ms
        }
        
    def collect_metrics(self) -> Dict:
        """Collect current system metrics."""
        # Simulate metric collection
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': random.uniform(20.0, 95.0),
            'memory_usage': random.uniform(30.0, 90.0),
            'active_tools': random.randint(10, 52),
            'requests_per_second': random.randint(50, 500),
            'average_response_time': random.randint(100, 3000),
            'error_rate': random.uniform(0.1, 10.0),
            'uptime_seconds': 86400
        }
        
        self.metrics_history.append(metrics)
        
        # Keep only last 100 metrics
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return metrics
    
    def check_health(self, metrics: Dict) -> Dict:
        """Check system health based on metrics."""
        health_status = {
            'status': 'healthy',
            'issues': [],
            'warnings': []
        }
        
        # Check CPU
        if metrics['cpu_usage'] > self.thresholds['cpu']:
            health_status['status'] = 'warning'
            health_status['warnings'].append(
                f"High CPU usage: {metrics['cpu_usage']:.1f}%"
            )
        
        # Check memory
        if metrics['memory_usage'] > self.thresholds['memory']:
            health_status['status'] = 'warning'
            health_status['warnings'].append(
                f"High memory usage: {metrics['memory_usage']:.1f}%"
            )
        
        # Check error rate
        if metrics['error_rate'] > self.thresholds['error_rate']:
            health_status['status'] = 'critical'
            health_status['issues'].append(
                f"High error rate: {metrics['error_rate']:.2f}%"
            )
        
        # Check response time
        if metrics['average_response_time'] > self.thresholds['response_time']:
            health_status['status'] = 'warning'
            health_status['warnings'].append(
                f"Slow response time: {metrics['average_response_time']}ms"
            )
        
        return health_status
    
    def generate_alert(self, alert_type: str, message: str, severity: str):
        """Generate a system alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        
        self.alerts.append(alert)
        return alert
    
    def get_summary(self) -> Dict:
        """Get monitoring summary."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]
        
        avg_cpu = sum(m['cpu_usage'] for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m['memory_usage'] for m in recent_metrics) / len(recent_metrics)
        avg_response = sum(m['average_response_time'] for m in recent_metrics) / len(recent_metrics)
        
        return {
            'total_samples': len(self.metrics_history),
            'average_cpu': avg_cpu,
            'average_memory': avg_memory,
            'average_response_time': avg_response,
            'total_alerts': len(self.alerts),
            'active_tools': recent_metrics[-1]['active_tools']
        }
    
    def get_status_emoji(self, status: str) -> str:
        """Get emoji for status."""
        emoji_map = {
            'healthy': '🟢',
            'warning': '🟡',
            'critical': '🔴'
        }
        return emoji_map.get(status, '⚪')


def run_demo():
    """Run monitoring demo."""
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print("  PULSE System Monitoring Demo")
    print(f"{'='*70}{Colors.ENDC}\n")
    
    monitor = SystemMonitor()
    
    # Demo 1: Collect and display metrics
    print(f"{Colors.BOLD}1. Collecting System Metrics{Colors.ENDC}\n")
    
    metrics = monitor.collect_metrics()
    
    print(f"  {Colors.BOLD}Current Metrics:{Colors.ENDC}")
    print(f"    • CPU Usage: {Colors.OKCYAN}{metrics['cpu_usage']:.1f}%{Colors.ENDC}")
    print(f"    • Memory Usage: {Colors.OKCYAN}{metrics['memory_usage']:.1f}%{Colors.ENDC}")
    print(f"    • Active Tools: {Colors.OKCYAN}{metrics['active_tools']}{Colors.ENDC}")
    print(f"    • Requests/sec: {Colors.OKCYAN}{metrics['requests_per_second']}{Colors.ENDC}")
    print(f"    • Avg Response: {Colors.OKCYAN}{metrics['average_response_time']}ms{Colors.ENDC}")
    print(f"    • Error Rate: {Colors.OKCYAN}{metrics['error_rate']:.2f}%{Colors.ENDC}")
    
    # Demo 2: Health check
    print(f"\n{Colors.BOLD}2. System Health Check{Colors.ENDC}\n")
    
    health = monitor.check_health(metrics)
    status_emoji = monitor.get_status_emoji(health['status'])
    
    print(f"  {status_emoji} Status: {Colors.BOLD}{health['status'].upper()}{Colors.ENDC}")
    
    if health['warnings']:
        print(f"\n  {Colors.WARNING}Warnings:{Colors.ENDC}")
        for warning in health['warnings']:
            print(f"    ⚠️  {warning}")
    
    if health['issues']:
        print(f"\n  {Colors.FAIL}Issues:{Colors.ENDC}")
        for issue in health['issues']:
            print(f"    ❌ {issue}")
    
    if not health['warnings'] and not health['issues']:
        print(f"  {Colors.OKGREEN}✓ All systems normal{Colors.ENDC}")
    
    # Demo 3: Alert generation
    print(f"\n{Colors.BOLD}3. Alert Generation{Colors.ENDC}\n")
    
    # Generate sample alerts based on health status
    if health['warnings']:
        alert = monitor.generate_alert(
            'performance',
            health['warnings'][0],
            'warning'
        )
        print(f"  {Colors.WARNING}⚠️  Alert Generated:{Colors.ENDC}")
        print(f"     Type: {alert['type']}")
        print(f"     Message: {alert['message']}")
    
    if health['issues']:
        alert = monitor.generate_alert(
            'system',
            health['issues'][0],
            'critical'
        )
        print(f"  {Colors.FAIL}🔴 Critical Alert:{Colors.ENDC}")
        print(f"     Type: {alert['type']}")
        print(f"     Message: {alert['message']}")
    
    # Demo 4: Multiple samples
    print(f"\n{Colors.BOLD}4. Collecting Multiple Samples{Colors.ENDC}\n")
    
    print(f"  Collecting 5 samples...")
    for i in range(5):
        monitor.collect_metrics()
        print(f"  {Colors.OKGREEN}✓{Colors.ENDC} Sample {i+1} collected")
        time.sleep(0.2)
    
    # Demo 5: Summary report
    print(f"\n{Colors.BOLD}5. Monitoring Summary{Colors.ENDC}\n")
    
    summary = monitor.get_summary()
    
    print(f"  Total Samples: {Colors.OKCYAN}{summary['total_samples']}{Colors.ENDC}")
    print(f"  Average CPU: {Colors.OKCYAN}{summary['average_cpu']:.1f}%{Colors.ENDC}")
    print(f"  Average Memory: {Colors.OKCYAN}{summary['average_memory']:.1f}%{Colors.ENDC}")
    print(f"  Average Response Time: {Colors.OKCYAN}{summary['average_response_time']:.0f}ms{Colors.ENDC}")
    print(f"  Total Alerts: {Colors.WARNING}{summary['total_alerts']}{Colors.ENDC}")
    print(f"  Active Tools: {Colors.OKCYAN}{summary['active_tools']}{Colors.ENDC}")
    
    # Demo 6: Threshold configuration
    print(f"\n{Colors.BOLD}6. Monitoring Thresholds{Colors.ENDC}\n")
    
    print(f"  Current Thresholds:")
    for metric, threshold in monitor.thresholds.items():
        unit = '%' if metric in ['cpu', 'memory', 'error_rate'] else 'ms'
        print(f"    • {metric}: {Colors.OKCYAN}{threshold}{unit}{Colors.ENDC}")
    
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}Demo completed successfully!{Colors.ENDC}\n")


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Demo interrupted by user{Colors.ENDC}\n")
    except Exception as e:
        print(f"\n{Colors.FAIL}Error: {e}{Colors.ENDC}\n")
        sys.exit(1)