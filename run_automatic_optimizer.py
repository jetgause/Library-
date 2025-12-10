#!/usr/bin/env python3
"""
Automatic Optimizer Daemon Runner

This script runs the automatic optimizer daemon with comprehensive features:
- Configuration management from file and environment variables
- Environment validation
- Graceful shutdown handlers
- Comprehensive logging with rotation
- Error handling and recovery
- Health monitoring

Author: jetgause
Created: 2025-12-10
"""

import os
import sys
import signal
import logging
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler
from datetime import datetime
import threading


class ConfigurationManager:
    """Manages configuration loading and validation."""
    
    DEFAULT_CONFIG = {
        'optimizer': {
            'interval_seconds': 300,
            'max_iterations': -1,  # -1 means infinite
            'batch_size': 100,
            'timeout_seconds': 3600,
        },
        'logging': {
            'level': 'INFO',
            'file': 'logs/optimizer.log',
            'max_bytes': 10485760,  # 10MB
            'backup_count': 5,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
        'monitoring': {
            'enabled': True,
            'health_check_interval': 60,
            'metrics_file': 'logs/optimizer_metrics.json',
        },
        'recovery': {
            'max_retries': 3,
            'retry_delay_seconds': 30,
            'exponential_backoff': True,
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path or os.getenv('OPTIMIZER_CONFIG', 'config/optimizer_config.json')
        self.config = self._load_config()
        self._apply_env_overrides()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    config = self._deep_merge(self.DEFAULT_CONFIG.copy(), loaded_config)
                    return config
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_path}: {e}")
                print("Using default configuration.")
        return self.DEFAULT_CONFIG.copy()
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Optimizer settings
        if os.getenv('OPTIMIZER_INTERVAL'):
            self.config['optimizer']['interval_seconds'] = int(os.getenv('OPTIMIZER_INTERVAL'))
        if os.getenv('OPTIMIZER_MAX_ITERATIONS'):
            self.config['optimizer']['max_iterations'] = int(os.getenv('OPTIMIZER_MAX_ITERATIONS'))
        if os.getenv('OPTIMIZER_BATCH_SIZE'):
            self.config['optimizer']['batch_size'] = int(os.getenv('OPTIMIZER_BATCH_SIZE'))
        
        # Logging settings
        if os.getenv('LOG_LEVEL'):
            self.config['logging']['level'] = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FILE'):
            self.config['logging']['file'] = os.getenv('LOG_FILE')
    
    def validate(self) -> bool:
        """Validate configuration values."""
        try:
            assert self.config['optimizer']['interval_seconds'] > 0, "Interval must be positive"
            assert self.config['optimizer']['batch_size'] > 0, "Batch size must be positive"
            assert self.config['optimizer']['timeout_seconds'] > 0, "Timeout must be positive"
            assert self.config['logging']['level'] in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            return True
        except AssertionError as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def get(self, *keys, default=None):
        """Get nested configuration value."""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


class EnvironmentValidator:
    """Validates the runtime environment."""
    
    @staticmethod
    def validate() -> bool:
        """Validate environment requirements."""
        checks = [
            EnvironmentValidator._check_python_version(),
            EnvironmentValidator._check_permissions(),
            EnvironmentValidator._check_dependencies(),
        ]
        return all(checks)
    
    @staticmethod
    def _check_python_version() -> bool:
        """Check Python version is 3.7+."""
        if sys.version_info < (3, 7):
            print(f"Error: Python 3.7+ required, found {sys.version_info.major}.{sys.version_info.minor}")
            return False
        return True
    
    @staticmethod
    def _check_permissions() -> bool:
        """Check file system permissions."""
        # Check log directory
        log_dir = Path('logs')
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            test_file = log_dir / '.test_write'
            test_file.touch()
            test_file.unlink()
            return True
        except PermissionError:
            print(f"Error: Insufficient permissions for log directory: {log_dir}")
            return False
    
    @staticmethod
    def _check_dependencies() -> bool:
        """Check required dependencies are available."""
        # Add any required module checks here
        return True


class OptimizerDaemon:
    """Main daemon class for the automatic optimizer."""
    
    def __init__(self, config_manager: ConfigurationManager):
        """Initialize the optimizer daemon."""
        self.config = config_manager
        self.logger = self._setup_logging()
        self.running = False
        self.shutdown_event = threading.Event()
        self.iteration_count = 0
        self.metrics = {
            'start_time': None,
            'iterations_completed': 0,
            'errors_encountered': 0,
            'last_success_time': None,
            'last_error_time': None,
        }
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Optimizer daemon initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger('OptimizerDaemon')
        logger.setLevel(getattr(logging, self.config.get('logging', 'level')))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file = self.config.get('logging', 'file')
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.config.get('logging', 'max_bytes'),
            backupCount=self.config.get('logging', 'backup_count')
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(self.config.get('logging', 'format'))
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        return logger
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        signal_name = signal.Signals(signum).name
        self.logger.warning(f"Received signal {signal_name}, initiating graceful shutdown...")
        self.shutdown()
    
    def _run_optimization(self) -> bool:
        """Execute a single optimization iteration."""
        try:
            self.logger.info(f"Starting optimization iteration {self.iteration_count + 1}")
            
            # Placeholder for actual optimization logic
            # Replace this with your actual optimizer implementation
            self.logger.debug("Processing optimization batch...")
            
            # Simulate work (replace with actual optimization code)
            time.sleep(2)
            
            self.logger.info(f"Optimization iteration {self.iteration_count + 1} completed successfully")
            self.metrics['iterations_completed'] += 1
            self.metrics['last_success_time'] = datetime.utcnow().isoformat()
            return True
            
        except Exception as e:
            self.logger.error(f"Optimization iteration failed: {e}", exc_info=True)
            self.metrics['errors_encountered'] += 1
            self.metrics['last_error_time'] = datetime.utcnow().isoformat()
            return False
    
    def _save_metrics(self):
        """Save metrics to file."""
        if self.config.get('monitoring', 'enabled'):
            try:
                metrics_file = self.config.get('monitoring', 'metrics_file')
                metrics_dir = Path(metrics_file).parent
                metrics_dir.mkdir(parents=True, exist_ok=True)
                
                with open(metrics_file, 'w') as f:
                    json.dump(self.metrics, f, indent=2)
            except Exception as e:
                self.logger.warning(f"Failed to save metrics: {e}")
    
    def _health_check(self):
        """Perform health check."""
        self.logger.debug("Health check: OK")
        # Add custom health check logic here
    
    def _monitoring_loop(self):
        """Background monitoring and health check loop."""
        interval = self.config.get('monitoring', 'health_check_interval', 60)
        while self.running and not self.shutdown_event.is_set():
            try:
                self._health_check()
                self._save_metrics()
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}", exc_info=True)
            
            self.shutdown_event.wait(interval)
    
    def run(self):
        """Main daemon run loop."""
        self.running = True
        self.metrics['start_time'] = datetime.utcnow().isoformat()
        self.logger.info("Starting optimizer daemon")
        
        # Start monitoring thread
        if self.config.get('monitoring', 'enabled'):
            monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            monitor_thread.start()
        
        max_iterations = self.config.get('optimizer', 'max_iterations')
        interval = self.config.get('optimizer', 'interval_seconds')
        max_retries = self.config.get('recovery', 'max_retries')
        retry_delay = self.config.get('recovery', 'retry_delay_seconds')
        exponential_backoff = self.config.get('recovery', 'exponential_backoff')
        
        consecutive_failures = 0
        
        try:
            while self.running:
                # Check if we've reached max iterations
                if max_iterations > 0 and self.iteration_count >= max_iterations:
                    self.logger.info(f"Reached maximum iterations ({max_iterations}), stopping")
                    break
                
                # Run optimization
                success = self._run_optimization()
                
                if success:
                    consecutive_failures = 0
                    self.iteration_count += 1
                else:
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_retries:
                        self.logger.error(f"Maximum consecutive failures reached ({max_retries}), stopping")
                        break
                    
                    # Calculate retry delay with optional exponential backoff
                    if exponential_backoff:
                        current_delay = retry_delay * (2 ** (consecutive_failures - 1))
                    else:
                        current_delay = retry_delay
                    
                    self.logger.warning(f"Retry {consecutive_failures}/{max_retries} after {current_delay}s delay")
                    if self.shutdown_event.wait(current_delay):
                        break
                    continue
                
                # Wait for next iteration
                self.logger.debug(f"Waiting {interval} seconds until next iteration")
                if self.shutdown_event.wait(interval):
                    break
                    
        except Exception as e:
            self.logger.critical(f"Fatal error in daemon loop: {e}", exc_info=True)
            raise
        finally:
            self._cleanup()
    
    def shutdown(self):
        """Gracefully shutdown the daemon."""
        self.logger.info("Shutdown initiated")
        self.running = False
        self.shutdown_event.set()
    
    def _cleanup(self):
        """Cleanup resources before exit."""
        self.logger.info("Cleaning up resources...")
        self._save_metrics()
        
        # Add any additional cleanup logic here
        
        self.logger.info(f"Daemon stopped. Total iterations: {self.iteration_count}")
        self.logger.info(f"Total errors: {self.metrics['errors_encountered']}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Automatic Optimizer Daemon Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-c', '--config',
        help='Path to configuration file',
        default=None
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Validate configuration and environment, then exit'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    print("=" * 60)
    print("Automatic Optimizer Daemon")
    print("=" * 60)
    
    # Load configuration
    print("\n[1/3] Loading configuration...")
    config_manager = ConfigurationManager(args.config)
    
    if not config_manager.validate():
        print("❌ Configuration validation failed")
        return 1
    print("✓ Configuration loaded and validated")
    
    # Validate environment
    print("\n[2/3] Validating environment...")
    if not EnvironmentValidator.validate():
        print("❌ Environment validation failed")
        return 1
    print("✓ Environment validated")
    
    if args.validate_only:
        print("\n✓ Validation complete. Exiting.")
        return 0
    
    # Run daemon
    print("\n[3/3] Starting optimizer daemon...")
    print("=" * 60)
    
    try:
        daemon = OptimizerDaemon(config_manager)
        daemon.run()
        return 0
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
