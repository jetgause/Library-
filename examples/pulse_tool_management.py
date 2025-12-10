#!/usr/bin/env python3
"""
PULSE Tool Management
=====================

This example demonstrates comprehensive tool management in PULSE including:
- Creating tools with full metadata
- Registering and organizing tools
- Updating tool properties
- Tool lifecycle management
- Taxonomy and categorization

Author: PULSE Development Team
Date: 2025-12-10
"""

import sys
from datetime import datetime
from typing import Dict, List, Optional

# ANSI Color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class ToolManager:
    """Manages tool creation, registration, and organization."""
    
    def __init__(self):
        self.tools = {}
        self.taxonomy = {
            'Data Processing': [],
            'Machine Learning': [],
            'Web Services': [],
            'Automation': [],
            'Analytics': []
        }
        self.version_history = {}
        
    def print_header(self, title: str):
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}\n  {title}\n{'=' * 70}{Colors.ENDC}")
    
    def print_success(self, message: str):
        print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")
    
    def print_info(self, message: str):
        print(f"{Colors.OKBLUE}ℹ {message}{Colors.ENDC}")
    
    def create_tool(self, tool_id: str, name: str, category: str,
                   description: str, complexity: str, 
                   tags: List[str], metadata: Optional[Dict] = None) -> Dict:
        """Create a new tool with full metadata."""
        
        tool = {
            'id': tool_id,
            'name': name,
            'category': category,
            'description': description,
            'complexity': complexity,
            'tags': tags,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'active': True,
            'metadata': metadata or {
                'author': 'PULSE Team',
                'license': 'MIT'
            }
        }
        
        self.tools[tool_id] = tool
        
        # Add to taxonomy
        if category in self.taxonomy:
            self.taxonomy[category].append(tool_id)
        
        # Initialize version history
        self.version_history[tool_id] = [{
            'version': '1.0.0',
            'date': tool['created_at'],
            'changes': 'Initial creation'
        }]
        
        return tool
    
    def update_tool(self, tool_id: str, updates: Dict) -> bool:
        """Update an existing tool's properties."""
        if tool_id not in self.tools:
            return False
        
        tool = self.tools[tool_id]
        
        for key, value in updates.items():
            if key in tool and key not in ['id', 'created_at']:
                tool[key] = value
        
        tool['updated_at'] = datetime.now().isoformat()
        return True
    
    def update_version(self, tool_id: str, new_version: str, changes: str):
        """Update tool version with changelog."""
        if tool_id not in self.tools:
            return False
        
        tool = self.tools[tool_id]
        old_version = tool['version']
        tool['version'] = new_version
        tool['updated_at'] = datetime.now().isoformat()
        
        self.version_history[tool_id].append({
            'version': new_version,
            'date': tool['updated_at'],
            'changes': changes,
            'previous_version': old_version
        })
        
        return True
    
    def get_tool(self, tool_id: str) -> Optional[Dict]:
        """Get a tool by ID."""
        return self.tools.get(tool_id)
    
    def list_tools_by_category(self, category: str) -> List[Dict]:
        """List all tools in a category."""
        if category not in self.taxonomy:
            return []
        
        return [self.tools[tid] for tid in self.taxonomy[category]]
    
    def search_tools(self, query: str) -> List[Dict]:
        """Search tools by name, description, or tags."""
        query_lower = query.lower()
        results = []
        
        for tool in self.tools.values():
            if (query_lower in tool['name'].lower() or
                query_lower in tool['description'].lower() or
                any(query_lower in tag.lower() for tag in tool['tags'])):
                results.append(tool)
        
        return results
    
    def deactivate_tool(self, tool_id: str) -> bool:
        """Deactivate a tool (soft delete)."""
        if tool_id in self.tools:
            self.tools[tool_id]['active'] = False
            self.tools[tool_id]['updated_at'] = datetime.now().isoformat()
            return True
        return False
    
    def get_version_history(self, tool_id: str) -> List[Dict]:
        """Get version history for a tool."""
        return self.version_history.get(tool_id, [])
    
    def run_demo(self):
        """Run the tool management demonstration."""
        
        self.print_header("PULSE Tool Management Demo")
        print(f"{Colors.BOLD}Demonstrating comprehensive tool management{Colors.ENDC}\n")
        
        # 1. Create Tools
        self.print_header("1. Creating Tools")
        
        tools_to_create = [
            {
                'tool_id': 'data_cleaner',
                'name': 'Data Cleaner Pro',
                'category': 'Data Processing',
                'description': 'Advanced data cleaning and normalization',
                'complexity': 'Medium',
                'tags': ['data', 'cleaning', 'etl', 'preprocessing']
            },
            {
                'tool_id': 'ml_classifier',
                'name': 'ML Classifier Suite',
                'category': 'Machine Learning',
                'description': 'Multi-algorithm classification toolkit',
                'complexity': 'High',
                'tags': ['ml', 'classification', 'ai', 'prediction']
            },
            {
                'tool_id': 'api_gateway',
                'name': 'API Gateway Manager',
                'category': 'Web Services',
                'description': 'RESTful API gateway with rate limiting',
                'complexity': 'Medium',
                'tags': ['api', 'web', 'gateway', 'rest']
            }
        ]
        
        for tool_data in tools_to_create:
            tool = self.create_tool(**tool_data)
            self.print_success(f"Created: {tool['name']} (ID: {tool['id']})")
            print(f"  Category: {Colors.OKCYAN}{tool['category']}{Colors.ENDC}")
            print(f"  Complexity: {Colors.WARNING}{tool['complexity']}{Colors.ENDC}")
            print(f"  Tags: {', '.join(tool['tags'])}\n")
        
        input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.ENDC}")
        
        # 2. View Tools by Category
        self.print_header("2. Organizing by Category")
        
        for category, tool_ids in self.taxonomy.items():
            if tool_ids:
                print(f"\n{Colors.BOLD}{Colors.OKCYAN}📁 {category}{Colors.ENDC} ({len(tool_ids)} tools)")
                for tool_id in tool_ids:
                    tool = self.tools[tool_id]
                    status = f"{Colors.OKGREEN}●{Colors.ENDC}" if tool['active'] else f"{Colors.FAIL}●{Colors.ENDC}"
                    print(f"  {status} {tool['name']} - v{tool['version']}")
        
        input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.ENDC}")
        
        # 3. Update Tool
        self.print_header("3. Updating Tool Properties")
        
        tool_id = 'data_cleaner'
        updates = {
            'description': 'Advanced data cleaning with ML-powered anomaly detection',
            'complexity': 'High'
        }
        
        if self.update_tool(tool_id, updates):
            self.print_success(f"Updated {tool_id}")
            updated_tool = self.get_tool(tool_id)
            print(f"  New description: {updated_tool['description']}")
            print(f"  New complexity: {Colors.WARNING}{updated_tool['complexity']}{Colors.ENDC}")
        
        input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.ENDC}")
        
        # 4. Version Updates
        self.print_header("4. Version Management")
        
        version_updates = [
            ('1.1.0', 'Added batch processing support'),
            ('1.2.0', 'Performance improvements'),
            ('2.0.0', 'Major rewrite with new architecture')
        ]
        
        for version, changes in version_updates:
            self.update_version(tool_id, version, changes)
            self.print_success(f"Updated to v{version}: {changes}")
        
        print(f"\n{Colors.BOLD}Version History for {tool_id}:{Colors.ENDC}")
        history = self.get_version_history(tool_id)
        for entry in history:
            print(f"  v{entry['version']} - {entry['date'][:10]}")
            print(f"    {Colors.OKBLUE}{entry['changes']}{Colors.ENDC}")
        
        input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.ENDC}")
        
        # 5. Search Tools
        self.print_header("5. Searching Tools")
        
        queries = ['data', 'ml', 'api']
        for query in queries:
            results = self.search_tools(query)
            print(f"\n{Colors.BOLD}Search for '{query}':{Colors.ENDC} {len(results)} result(s)")
            for tool in results:
                print(f"  • {tool['name']} - {tool['category']}")
        
        input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.ENDC}")
        
        # 6. Tool Details
        self.print_header("6. Detailed Tool Information")
        
        tool = self.get_tool('ml_classifier')
        if tool:
            print(f"\n{Colors.BOLD}Tool: {tool['name']}{Colors.ENDC}")
            print(f"  ID: {tool['id']}")
            print(f"  Version: {Colors.OKCYAN}v{tool['version']}{Colors.ENDC}")
            print(f"  Category: {tool['category']}")
            print(f"  Complexity: {Colors.WARNING}{tool['complexity']}{Colors.ENDC}")
            print(f"  Description: {tool['description']}")
            print(f"  Created: {tool['created_at'][:19]}")
            print(f"  Updated: {tool['updated_at'][:19]}")
            print(f"  Status: {Colors.OKGREEN}Active{Colors.ENDC}" if tool['active'] else f"{Colors.FAIL}Inactive{Colors.ENDC}")
            print(f"  Tags: {', '.join(tool['tags'])}")
            print(f"  Author: {tool['metadata']['author']}")
            print(f"  License: {tool['metadata']['license']}")
        
        # Summary
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}Tool Management Demo Complete!{Colors.ENDC}")
        print(f"\nTotal tools managed: {Colors.OKCYAN}{len(self.tools)}{Colors.ENDC}")
        print(f"Categories in use: {Colors.OKCYAN}{sum(1 for tools in self.taxonomy.values() if tools)}{Colors.ENDC}\n")


def main():
    """Main entry point."""
    try:
        manager = ToolManager()
        manager.run_demo()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Demo interrupted.{Colors.ENDC}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.FAIL}Error: {e}{Colors.ENDC}")
        sys.exit(1)


if __name__ == "__main__":
    main()
