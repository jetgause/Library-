#!/usr/bin/env python3
"""
PULSE Basic Setup and Configuration
====================================

This example demonstrates the basic setup and initialization of the PULSE system.
It covers:
- System initialization
- Basic configuration
- Core component setup
- Simple tool creation

Author: PULSE Development Team
Date: 2025-12-10
"""

import sys
from datetime import datetime
from typing import Dict, Optional

# ANSI Color codes for enhanced output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class BasicPULSESetup:
    """Basic PULSE system setup and configuration."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize PULSE with optional configuration."""
        self.config = config or self._default_config()
        self.tools = {}
        self.initialized = False
        
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'system_name': 'PULSE',
            'version': '1.0.0',
            'max_tools': 100,
            'logging_enabled': True,
            'auto_optimize': True,
            'economic_tracking': True
        }
    
    def print_header(self, title: str):
        """Print formatted header."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}")
        print("=" * 70)
        print(f"  {title}")
        print("=" * 70)
        print(f"{Colors.ENDC}")
    
    def print_success(self, message: str):
        """Print success message."""
        print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")
    
    def print_info(self, message: str):
        """Print info message."""
        print(f"{Colors.OKBLUE}ℹ {message}{Colors.ENDC}")
    
    def initialize(self) -> bool:
        """Initialize the PULSE system."""
        try:
            self.print_header("Initializing PULSE System")
            
            print(f"\n{Colors.BOLD}Configuration:{Colors.ENDC}")
            for key, value in self.config.items():
                print(f"  • {key}: {Colors.OKCYAN}{value}{Colors.ENDC}")
            
            # Simulate initialization steps
            steps = [
                "Loading core modules",
                "Initializing tool registry",
                "Setting up economic calculator",
                "Configuring learning system",
                "Preparing context manager"
            ]
            
            print(f"\n{Colors.BOLD}Initialization Steps:{Colors.ENDC}")
            for step in steps:
                print(f"  {Colors.OKGREEN}✓{Colors.ENDC} {step}")
            
            self.initialized = True
            print(f"\n{Colors.OKGREEN}{Colors.BOLD}System initialized successfully!{Colors.ENDC}")
            return True
            
        except Exception as e:
            print(f"{Colors.FAIL}Initialization failed: {e}{Colors.ENDC}")
            return False
    
    def create_simple_tool(self, tool_id: str, name: str, description: str) -> Dict:
        """Create a simple tool with minimal configuration."""
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        tool = {
            'id': tool_id,
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'active': True
        }
        
        self.tools[tool_id] = tool
        return tool
    
    def get_system_info(self) -> Dict:
        """Get current system information."""
        return {
            'initialized': self.initialized,
            'config': self.config,
            'total_tools': len(self.tools),
            'system_time': datetime.now().isoformat()
        }
    
    def run_demo(self):
        """Run the basic setup demonstration."""
        self.print_header("PULSE Basic Setup Demo")
        
        print(f"{Colors.BOLD}Welcome to PULSE!{Colors.ENDC}")
        print("This demo shows basic system initialization and configuration.\n")
        
        # Initialize system
        if not self.initialize():
            return
        
        # Create a simple tool
        print(f"\n{Colors.HEADER}{Colors.BOLD}Creating a Simple Tool{Colors.ENDC}")
        tool = self.create_simple_tool(
            'hello_world',
            'Hello World Tool',
            'A simple demonstration tool'
        )
        
        self.print_success(f"Created tool: {tool['name']}")
        print(f"  ID: {tool['id']}")
        print(f"  Version: {tool['version']}")
        print(f"  Created: {tool['created_at'][:19]}")
        
        # Display system info
        print(f"\n{Colors.HEADER}{Colors.BOLD}System Information{Colors.ENDC}")
        info = self.get_system_info()
        print(f"  Status: {Colors.OKGREEN}Initialized{Colors.ENDC}")
        print(f"  Tools Registered: {Colors.OKCYAN}{info['total_tools']}{Colors.ENDC}")
        print(f"  Version: {Colors.OKCYAN}{info['config']['version']}{Colors.ENDC}")
        
        print(f"\n{Colors.OKGREEN}Basic setup complete! Explore other examples for more features.{Colors.ENDC}\n")


def main():
    """Main entry point."""
    try:
        demo = BasicPULSESetup()
        demo.run_demo()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Demo interrupted.{Colors.ENDC}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.FAIL}Error: {e}{Colors.ENDC}")
        sys.exit(1)


if __name__ == "__main__":
    main()
