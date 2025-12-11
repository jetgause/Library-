"""
PULSE Factory Testing Module
Unit tests for tool factory creation, management, and validation.

Created: 2025-12-10
Author: jetgause
"""

import unittest
from datetime import datetime
from typing import Dict, List, Optional, Any


class MockTool:
    """Mock Tool class for testing purposes."""
    
    def __init__(self, name: str, category: str, active: bool = True):
        self.name = name
        self.category = category
        self.active = active
        now = datetime.utcnow()
        self.created_at = now
        self.updated_at = now
    
    def __repr__(self):
        return f"MockTool(name='{self.name}', category='{self.category}', active={self.active})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation."""
        return {
            'name': self.name,
            'category': self.category,
            'active': self.active,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class ToolFactory:
    """Factory class for creating and managing tools."""
    
    VALID_CATEGORIES = ['development', 'testing', 'deployment', 'monitoring', 'analysis']
    
    def __init__(self):
        self._tools: Dict[str, MockTool] = {}
    
    def create_tool(self, name: str, category: str, active: bool = True) -> MockTool:
        """
        Create a new tool instance.
        
        Args:
            name: Unique name for the tool
            category: Tool category (must be valid)
            active: Active status (default: True)
        
        Returns:
            MockTool: Created tool instance
        
        Raises:
            ValueError: If tool already exists or category is invalid
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already exists")
        
        if category not in self.VALID_CATEGORIES:
            raise ValueError(f"Invalid category '{category}'. Must be one of: {self.VALID_CATEGORIES}")
        
        tool = MockTool(name=name, category=category, active=active)
        self._tools[name] = tool
        return tool
    
    def get_tool(self, name: str) -> Optional[MockTool]:
        """
        Retrieve a tool by name.
        
        Args:
            name: Tool name
        
        Returns:
            MockTool or None if not found
        """
        return self._tools.get(name)
    
    def delete_tool(self, name: str) -> bool:
        """
        Delete a tool by name.
        
        Args:
            name: Tool name
        
        Returns:
            bool: True if deleted, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def list_tools(self, category: Optional[str] = None) -> List[MockTool]:
        """
        List all tools or filter by category.
        
        Args:
            category: Optional category filter
        
        Returns:
            List of MockTool instances
        """
        if category is None:
            return list(self._tools.values())
        return [tool for tool in self._tools.values() if tool.category == category]
    
    def count_tools(self, category: Optional[str] = None) -> int:
        """
        Count tools, optionally filtered by category.
        
        Args:
            category: Optional category filter
        
        Returns:
            int: Number of tools
        """
        if category is None:
            return len(self._tools)
        return len([tool for tool in self._tools.values() if tool.category == category])
    
    def clear(self):
        """Clear all tools from the factory."""
        self._tools.clear()


class TestToolFactory(unittest.TestCase):
    """Test suite for ToolFactory class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.factory = ToolFactory()
        self.test_start_time = datetime.utcnow()
    
    def tearDown(self):
        """Clean up after each test method."""
        self.factory.clear()
        self.factory = None
    
    def test_tool_creation(self):
        """Test basic tool creation functionality."""
        tool = self.factory.create_tool('analyzer', 'analysis')
        
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, 'analyzer')
        self.assertEqual(tool.category, 'analysis')
        self.assertTrue(tool.active)
        self.assertIsInstance(tool.created_at, datetime)
    
    def test_duplicate_tool_handling(self):
        """Test that duplicate tool names raise ValueError."""
        self.factory.create_tool('deployer', 'deployment')
        
        with self.assertRaises(ValueError) as context:
            self.factory.create_tool('deployer', 'deployment')
        
        self.assertIn("already exists", str(context.exception))
    
    def test_invalid_category_handling(self):
        """Test that invalid categories raise ValueError."""
        invalid_categories = ['invalid', 'unknown', 'test123', '']
        
        for invalid_cat in invalid_categories:
            with self.assertRaises(ValueError) as context:
                self.factory.create_tool(f'tool_{invalid_cat}', invalid_cat)
            
            self.assertIn("Invalid category", str(context.exception))
    
    def test_tool_retrieval(self):
        """Test tool retrieval by name."""
        # Create a tool
        created_tool = self.factory.create_tool('monitor', 'monitoring')
        
        # Retrieve the tool
        retrieved_tool = self.factory.get_tool('monitor')
        
        self.assertIsNotNone(retrieved_tool)
        self.assertEqual(retrieved_tool.name, created_tool.name)
        self.assertEqual(retrieved_tool.category, created_tool.category)
        
        # Test non-existent tool
        self.assertIsNone(self.factory.get_tool('nonexistent'))
    
    def test_tool_deletion(self):
        """Test tool deletion functionality."""
        # Create and delete a tool
        self.factory.create_tool('temp_tool', 'testing')
        
        result = self.factory.delete_tool('temp_tool')
        self.assertTrue(result)
        
        # Verify deletion
        self.assertIsNone(self.factory.get_tool('temp_tool'))
        
        # Test deleting non-existent tool
        result = self.factory.delete_tool('nonexistent')
        self.assertFalse(result)
    
    def test_list_all_tools(self):
        """Test listing all tools."""
        # Create multiple tools
        self.factory.create_tool('dev_tool', 'development')
        self.factory.create_tool('test_tool', 'testing')
        self.factory.create_tool('deploy_tool', 'deployment')
        
        tools = self.factory.list_tools()
        
        self.assertEqual(len(tools), 3)
        tool_names = [tool.name for tool in tools]
        self.assertIn('dev_tool', tool_names)
        self.assertIn('test_tool', tool_names)
        self.assertIn('deploy_tool', tool_names)
    
    def test_list_tools_by_category(self):
        """Test listing tools filtered by category."""
        # Create tools in different categories
        self.factory.create_tool('dev_tool1', 'development')
        self.factory.create_tool('dev_tool2', 'development')
        self.factory.create_tool('test_tool', 'testing')
        self.factory.create_tool('monitor_tool', 'monitoring')
        
        # Test development category
        dev_tools = self.factory.list_tools(category='development')
        self.assertEqual(len(dev_tools), 2)
        
        # Test testing category
        test_tools = self.factory.list_tools(category='testing')
        self.assertEqual(len(test_tools), 1)
        self.assertEqual(test_tools[0].name, 'test_tool')
        
        # Test empty category
        analysis_tools = self.factory.list_tools(category='analysis')
        self.assertEqual(len(analysis_tools), 0)
    
    def test_count_tools(self):
        """Test tool counting functionality."""
        # Initially empty
        self.assertEqual(self.factory.count_tools(), 0)
        
        # Add tools
        self.factory.create_tool('tool1', 'development')
        self.assertEqual(self.factory.count_tools(), 1)
        
        self.factory.create_tool('tool2', 'testing')
        self.assertEqual(self.factory.count_tools(), 2)
        
        self.factory.create_tool('tool3', 'development')
        self.assertEqual(self.factory.count_tools(), 3)
        
        # Count by category
        self.assertEqual(self.factory.count_tools(category='development'), 2)
        self.assertEqual(self.factory.count_tools(category='testing'), 1)
        self.assertEqual(self.factory.count_tools(category='monitoring'), 0)
    
    def test_timestamp_verification(self):
        """Test that timestamps are correctly set on tool creation."""
        tool = self.factory.create_tool('timestamped_tool', 'analysis')
        
        # Verify timestamps exist and are datetime objects
        self.assertIsInstance(tool.created_at, datetime)
        self.assertIsInstance(tool.updated_at, datetime)
        
        # Verify timestamps are recent (within test execution time)
        time_diff = (datetime.utcnow() - tool.created_at).total_seconds()
        self.assertLess(time_diff, 5)  # Should be created within 5 seconds
        
        # Verify created_at and updated_at are initially the same
        self.assertEqual(tool.created_at, tool.updated_at)
    
    def test_active_status_default(self):
        """Test that default active status is True."""
        # Tool with default active status
        tool1 = self.factory.create_tool('default_tool', 'monitoring')
        self.assertTrue(tool1.active)
        
        # Tool with explicit active=True
        tool2 = self.factory.create_tool('active_tool', 'testing', active=True)
        self.assertTrue(tool2.active)
        
        # Tool with explicit active=False
        tool3 = self.factory.create_tool('inactive_tool', 'deployment', active=False)
        self.assertFalse(tool3.active)
    
    def test_multiple_category_testing(self):
        """Test creating tools across all valid categories."""
        categories = ToolFactory.VALID_CATEGORIES
        
        for idx, category in enumerate(categories):
            tool_name = f'{category}_tool_{idx}'
            tool = self.factory.create_tool(tool_name, category)
            
            self.assertEqual(tool.category, category)
            self.assertIsNotNone(self.factory.get_tool(tool_name))
        
        # Verify all tools were created
        self.assertEqual(self.factory.count_tools(), len(categories))
        
        # Verify each category has exactly one tool
        for category in categories:
            self.assertEqual(self.factory.count_tools(category=category), 1)
    
    def test_tool_to_dict(self):
        """Test tool dictionary conversion."""
        tool = self.factory.create_tool('dict_tool', 'analysis')
        tool_dict = tool.to_dict()
        
        self.assertIsInstance(tool_dict, dict)
        self.assertEqual(tool_dict['name'], 'dict_tool')
        self.assertEqual(tool_dict['category'], 'analysis')
        self.assertTrue(tool_dict['active'])
        self.assertIn('created_at', tool_dict)
        self.assertIn('updated_at', tool_dict)
    
    def test_tool_repr(self):
        """Test tool string representation."""
        tool = self.factory.create_tool('repr_tool', 'testing', active=False)
        repr_str = repr(tool)
        
        self.assertIn('MockTool', repr_str)
        self.assertIn('repr_tool', repr_str)
        self.assertIn('testing', repr_str)
        self.assertIn('False', repr_str)


def run_tests():
    """
    Run all tests with verbose output and summary.
    
    Returns:
        bool: True if all tests passed, False otherwise
    """
    print("=" * 70)
    print("PULSE Factory Testing Suite")
    print("=" * 70)
    print(f"Test execution started at: {datetime.utcnow().isoformat()}")
    print(f"Testing ToolFactory with {len(ToolFactory.VALID_CATEGORIES)} valid categories")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestToolFactory)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)
    
    if result.wasSuccessful():
        print("✓ All tests passed successfully!")
        print("=" * 70)
        return True
    else:
        print("✗ Some tests failed. Please review the output above.")
        print("=" * 70)
        return False


if __name__ == '__main__':
    # Run tests when executed directly
    success = run_tests()
    exit(0 if success else 1)
