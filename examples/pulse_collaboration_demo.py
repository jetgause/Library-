#!/usr/bin/env python3
"""
PULSE Tool Collaboration Demo
==============================

This example demonstrates PULSE's tool collaboration and chaining features:
- Tool composition and chaining
- Inter-tool communication
- Workflow orchestration
- Collaborative optimization

Author: PULSE Development Team
Date: 2025-12-10
"""

import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class Tool:
    """Base class for collaborative tools."""
    
    def __init__(self, tool_id: str, name: str):
        self.tool_id = tool_id
        self.name = name
        self.input_data = None
        self.output_data = None
        
    def execute(self, input_data: Any) -> Any:
        """Execute tool operation."""
        self.input_data = input_data
        # Override in subclasses
        return input_data
    
    def get_output(self) -> Any:
        """Get tool output."""
        return self.output_data


class DataCleanerTool(Tool):
    """Tool that cleans and preprocesses data."""
    
    def execute(self, input_data: Dict) -> Dict:
        """Clean input data."""
        self.input_data = input_data
        
        # Simulate data cleaning
        cleaned = {
            'records': input_data.get('records', []),
            'cleaned': True,
            'null_count': 0,
            'processed_at': datetime.now().isoformat()
        }
        
        self.output_data = cleaned
        return cleaned


class DataValidatorTool(Tool):
    """Tool that validates data quality."""
    
    def execute(self, input_data: Dict) -> Dict:
        """Validate input data."""
        self.input_data = input_data
        
        # Simulate validation
        validated = {
            **input_data,
            'validated': True,
            'quality_score': 0.95,
            'errors': [],
            'validated_at': datetime.now().isoformat()
        }
        
        self.output_data = validated
        return validated


class AnalyzerTool(Tool):
    """Tool that analyzes data."""
    
    def execute(self, input_data: Dict) -> Dict:
        """Analyze input data."""
        self.input_data = input_data
        
        # Simulate analysis
        analysis = {
            **input_data,
            'analyzed': True,
            'insights': ['Pattern detected', 'Outliers found'],
            'confidence': 0.87,
            'analyzed_at': datetime.now().isoformat()
        }
        
        self.output_data = analysis
        return analysis


class ToolChain:
    """Manages tool collaboration and chaining."""
    
    def __init__(self, chain_id: str):
        self.chain_id = chain_id
        self.tools = []
        self.results = []
        
    def add_tool(self, tool: Tool):
        """Add a tool to the chain."""
        self.tools.append(tool)
        
    def execute(self, initial_data: Any) -> Dict:
        """Execute the tool chain."""
        data = initial_data
        
        for i, tool in enumerate(self.tools):
            print(f"\n{Colors.OKCYAN}→ Executing: {tool.name}{Colors.ENDC}")
            
            # Execute tool
            result = tool.execute(data)
            
            # Store result
            self.results.append({
                'tool_id': tool.tool_id,
                'tool_name': tool.name,
                'step': i + 1,
                'output': result
            })
            
            # Pass output to next tool
            data = result
            
            print(f"  {Colors.OKGREEN}✓ Completed{Colors.ENDC}")
        
        return data
    
    def get_results(self) -> List[Dict]:
        """Get all chain execution results."""
        return self.results


class CollaborationManager:
    """Manages tool collaboration workflows."""
    
    def __init__(self):
        self.chains = {}
        self.tools = {}
        
    def register_tool(self, tool: Tool):
        """Register a tool for collaboration."""
        self.tools[tool.tool_id] = tool
        
    def create_chain(self, chain_id: str, tool_ids: List[str]) -> ToolChain:
        """Create a tool chain."""
        chain = ToolChain(chain_id)
        
        for tool_id in tool_ids:
            if tool_id in self.tools:
                chain.add_tool(self.tools[tool_id])
        
        self.chains[chain_id] = chain
        return chain
    
    def execute_chain(self, chain_id: str, initial_data: Any) -> Dict:
        """Execute a registered chain."""
        if chain_id not in self.chains:
            raise ValueError(f"Chain {chain_id} not found")
        
        return self.chains[chain_id].execute(initial_data)
    
    def get_chain_summary(self, chain_id: str) -> Dict:
        """Get summary of chain execution."""
        if chain_id not in self.chains:
            return {}
        
        chain = self.chains[chain_id]
        results = chain.get_results()
        
        return {
            'chain_id': chain_id,
            'tools_count': len(chain.tools),
            'steps_executed': len(results),
            'tool_names': [tool.name for tool in chain.tools]
        }


def run_demo():
    """Run tool collaboration demo."""
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print("  PULSE Tool Collaboration Demo")
    print(f"{'='*70}{Colors.ENDC}\n")
    
    manager = CollaborationManager()
    
    # Create and register tools
    print(f"{Colors.BOLD}1. Registering Tools{Colors.ENDC}\n")
    
    cleaner = DataCleanerTool('cleaner_01', 'Data Cleaner')
    validator = DataValidatorTool('validator_01', 'Data Validator')
    analyzer = AnalyzerTool('analyzer_01', 'Data Analyzer')
    
    manager.register_tool(cleaner)
    manager.register_tool(validator)
    manager.register_tool(analyzer)
    
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} Registered: {cleaner.name}")
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} Registered: {validator.name}")
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} Registered: {analyzer.name}")
    
    # Create tool chain
    print(f"\n{Colors.BOLD}2. Creating Tool Chain{Colors.ENDC}\n")
    
    chain = manager.create_chain(
        'data_pipeline',
        ['cleaner_01', 'validator_01', 'analyzer_01']
    )
    
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} Created chain: data_pipeline")
    print(f"  Tools in chain: {len(chain.tools)}")
    
    # Execute chain
    print(f"\n{Colors.BOLD}3. Executing Tool Chain{Colors.ENDC}")
    
    initial_data = {
        'records': [
            {'id': 1, 'value': 100},
            {'id': 2, 'value': 200},
            {'id': 3, 'value': 300}
        ],
        'source': 'input_stream'
    }
    
    print(f"\nInitial data: {len(initial_data['records'])} records")
    
    final_result = manager.execute_chain('data_pipeline', initial_data)
    
    # Show chain summary
    print(f"\n{Colors.BOLD}4. Chain Execution Summary{Colors.ENDC}\n")
    
    summary = manager.get_chain_summary('data_pipeline')
    
    print(f"  Chain ID: {Colors.OKCYAN}{summary['chain_id']}{Colors.ENDC}")
    print(f"  Steps Executed: {Colors.OKCYAN}{summary['steps_executed']}{Colors.ENDC}")
    print(f"  Tools Used:")
    for tool_name in summary['tool_names']:
        print(f"    • {tool_name}")
    
    # Show final result keys
    print(f"\n{Colors.BOLD}5. Final Result{Colors.ENDC}\n")
    print(f"  Output Keys:")
    for key in final_result.keys():
        value = final_result[key]
        if isinstance(value, (str, int, float, bool)):
            print(f"    • {key}: {Colors.OKGREEN}{value}{Colors.ENDC}")
        else:
            print(f"    • {key}: {Colors.OKCYAN}{type(value).__name__}{Colors.ENDC}")
    
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}Demo completed successfully!{Colors.ENDC}\n")


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Demo interrupted by user{Colors.ENDC}\n")
    except Exception as e:
        print(f"\n{Colors.FAIL}Error: {e}{Colors.ENDC}\n")
        sys.exit(1)