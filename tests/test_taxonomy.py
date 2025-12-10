"""
Comprehensive PULSE Taxonomy Testing Module

This module provides comprehensive testing for the PULSE taxonomy system,
including category management, tool categorization, tag management, and
hierarchy navigation.

Classes:
    TaxonomyNode: Represents a node in the taxonomy hierarchy
    TaxonomyManager: Manages the entire taxonomy structure
    TestTaxonomyManager: Test suite with 25 comprehensive test cases
"""

import unittest
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from enum import Enum


class CategoryType(Enum):
    """Enumeration of category types in the taxonomy."""
    ROOT = "root"
    DOMAIN = "domain"
    SUBDOMAIN = "subdomain"
    CATEGORY = "category"
    SUBCATEGORY = "subcategory"


@dataclass
class TaxonomyNode:
    """
    Represents a node in the taxonomy hierarchy.
    
    Attributes:
        name: The name of the taxonomy node
        category_type: The type of category (root, domain, etc.)
        parent: Reference to the parent node
        children: List of child nodes
        tools: Set of tools associated with this node
        tags: Set of tags for this node
        metadata: Additional metadata for the node
    """
    name: str
    category_type: CategoryType
    parent: Optional['TaxonomyNode'] = None
    children: List['TaxonomyNode'] = field(default_factory=list)
    tools: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def add_child(self, child: 'TaxonomyNode') -> None:
        """Add a child node to this node."""
        if child not in self.children:
            self.children.append(child)
            child.parent = self
    
    def remove_child(self, child: 'TaxonomyNode') -> bool:
        """Remove a child node from this node."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            return True
        return False
    
    def add_tool(self, tool_name: str) -> None:
        """Add a tool to this node."""
        self.tools.add(tool_name)
    
    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from this node."""
        if tool_name in self.tools:
            self.tools.remove(tool_name)
            return True
        return False
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to this node."""
        self.tags.add(tag)
    
    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from this node."""
        if tag in self.tags:
            self.tags.remove(tag)
            return True
        return False
    
    def get_path(self) -> List[str]:
        """Get the full path from root to this node."""
        path = []
        current = self
        while current is not None:
            path.insert(0, current.name)
            current = current.parent
        return path
    
    def get_all_tools(self, recursive: bool = True) -> Set[str]:
        """Get all tools associated with this node and optionally its children."""
        all_tools = self.tools.copy()
        if recursive:
            for child in self.children:
                all_tools.update(child.get_all_tools(recursive=True))
        return all_tools
    
    def find_child(self, name: str) -> Optional['TaxonomyNode']:
        """Find a direct child by name."""
        for child in self.children:
            if child.name == name:
                return child
        return None


class TaxonomyManager:
    """
    Manages the entire taxonomy structure for the PULSE system.
    
    Provides functionality for:
    - Category management (add, remove, update)
    - Tool categorization
    - Tag management
    - Hierarchy navigation
    - Search and filtering
    """
    
    def __init__(self):
        """Initialize the taxonomy manager with a root node."""
        self.root = TaxonomyNode("PULSE_ROOT", CategoryType.ROOT)
        self._node_index: Dict[str, TaxonomyNode] = {"PULSE_ROOT": self.root}
        self._tool_index: Dict[str, List[TaxonomyNode]] = {}
    
    def add_category(self, name: str, category_type: CategoryType, 
                     parent_name: str = "PULSE_ROOT", 
                     tags: Optional[Set[str]] = None,
                     metadata: Optional[Dict[str, any]] = None) -> TaxonomyNode:
        """Add a new category to the taxonomy."""
        if name in self._node_index:
            raise ValueError(f"Category '{name}' already exists")
        
        parent = self._node_index.get(parent_name)
        if parent is None:
            raise ValueError(f"Parent category '{parent_name}' not found")
        
        node = TaxonomyNode(
            name=name,
            category_type=category_type,
            tags=tags or set(),
            metadata=metadata or {}
        )
        parent.add_child(node)
        self._node_index[name] = node
        return node
    
    def remove_category(self, name: str, recursive: bool = False) -> bool:
        """Remove a category from the taxonomy."""
        if name == "PULSE_ROOT":
            raise ValueError("Cannot remove root node")
        
        node = self._node_index.get(name)
        if node is None:
            return False
        
        if node.children and not recursive:
            raise ValueError(f"Category '{name}' has children. Use recursive=True to remove all.")
        
        # Remove all tools from this category
        for tool in list(node.tools):
            self.remove_tool_from_category(tool, name)
        
        # Recursively remove children
        if recursive:
            for child in list(node.children):
                self.remove_category(child.name, recursive=True)
        
        # Remove from parent
        if node.parent:
            node.parent.remove_child(node)
        
        # Remove from index
        del self._node_index[name]
        return True
    
    def get_category(self, name: str) -> Optional[TaxonomyNode]:
        """Get a category by name."""
        return self._node_index.get(name)
    
    def categorize_tool(self, tool_name: str, category_name: str) -> bool:
        """Associate a tool with a category."""
        category = self._node_index.get(category_name)
        if category is None:
            return False
        
        category.add_tool(tool_name)
        
        if tool_name not in self._tool_index:
            self._tool_index[tool_name] = []
        self._tool_index[tool_name].append(category)
        return True
    
    def remove_tool_from_category(self, tool_name: str, category_name: str) -> bool:
        """Remove a tool from a specific category."""
        category = self._node_index.get(category_name)
        if category is None:
            return False
        
        result = category.remove_tool(tool_name)
        if result and tool_name in self._tool_index:
            self._tool_index[tool_name] = [
                cat for cat in self._tool_index[tool_name] if cat != category
            ]
            if not self._tool_index[tool_name]:
                del self._tool_index[tool_name]
        return result
    
    def get_tool_categories(self, tool_name: str) -> List[TaxonomyNode]:
        """Get all categories associated with a tool."""
        return self._tool_index.get(tool_name, [])
    
    def add_tag(self, category_name: str, tag: str) -> bool:
        """Add a tag to a category."""
        category = self._node_index.get(category_name)
        if category is None:
            return False
        category.add_tag(tag)
        return True
    
    def remove_tag(self, category_name: str, tag: str) -> bool:
        """Remove a tag from a category."""
        category = self._node_index.get(category_name)
        if category is None:
            return False
        return category.remove_tag(tag)
    
    def find_categories_by_tag(self, tag: str) -> List[TaxonomyNode]:
        """Find all categories with a specific tag."""
        return [node for node in self._node_index.values() if tag in node.tags]
    
    def get_hierarchy(self, category_name: str = "PULSE_ROOT") -> Dict:
        """Get the hierarchy structure starting from a category."""
        category = self._node_index.get(category_name)
        if category is None:
            return {}
        
        return self._build_hierarchy_dict(category)
    
    def _build_hierarchy_dict(self, node: TaxonomyNode) -> Dict:
        """Helper method to build hierarchy dictionary."""
        return {
            "name": node.name,
            "type": node.category_type.value,
            "tools": list(node.tools),
            "tags": list(node.tags),
            "children": [self._build_hierarchy_dict(child) for child in node.children]
        }
    
    def search_categories(self, query: str, search_tags: bool = True) -> List[TaxonomyNode]:
        """Search for categories by name or tags."""
        results = []
        query_lower = query.lower()
        
        for node in self._node_index.values():
            if query_lower in node.name.lower():
                results.append(node)
            elif search_tags:
                for tag in node.tags:
                    if query_lower in tag.lower():
                        results.append(node)
                        break
        
        return results
    
    def get_path(self, category_name: str) -> Optional[List[str]]:
        """Get the path from root to a category."""
        category = self._node_index.get(category_name)
        if category is None:
            return None
        return category.get_path()
    
    def move_category(self, category_name: str, new_parent_name: str) -> bool:
        """Move a category to a new parent."""
        if category_name == "PULSE_ROOT":
            raise ValueError("Cannot move root node")
        
        category = self._node_index.get(category_name)
        new_parent = self._node_index.get(new_parent_name)
        
        if category is None or new_parent is None:
            return False
        
        # Check for circular reference
        current = new_parent
        while current is not None:
            if current == category:
                raise ValueError("Cannot move category to its own descendant")
            current = current.parent
        
        # Remove from old parent
        if category.parent:
            category.parent.remove_child(category)
        
        # Add to new parent
        new_parent.add_child(category)
        return True


class TestTaxonomyManager(unittest.TestCase):
    """
    Comprehensive test suite for the PULSE Taxonomy system.
    
    Contains 25 test cases covering:
    - Category management
    - Tool categorization
    - Tag management
    - Hierarchy navigation
    """
    
    def setUp(self):
        """Set up test fixtures before each test."""
        self.manager = TaxonomyManager()
    
    # Category Management Tests (1-8)
    
    def test_01_add_domain_category(self):
        """Test adding a domain-level category."""
        node = self.manager.add_category(
            "Development",
            CategoryType.DOMAIN,
            "PULSE_ROOT"
        )
        self.assertEqual(node.name, "Development")
        self.assertEqual(node.category_type, CategoryType.DOMAIN)
        self.assertEqual(node.parent, self.manager.root)
    
    def test_02_add_nested_categories(self):
        """Test adding nested category hierarchy."""
        self.manager.add_category("Development", CategoryType.DOMAIN)
        self.manager.add_category("Backend", CategoryType.SUBDOMAIN, "Development")
        self.manager.add_category("APIs", CategoryType.CATEGORY, "Backend")
        
        apis = self.manager.get_category("APIs")
        self.assertIsNotNone(apis)
        self.assertEqual(len(apis.get_path()), 4)  # ROOT -> Dev -> Backend -> APIs
    
    def test_03_duplicate_category_name(self):
        """Test that duplicate category names raise an error."""
        self.manager.add_category("Development", CategoryType.DOMAIN)
        with self.assertRaises(ValueError):
            self.manager.add_category("Development", CategoryType.DOMAIN)
    
    def test_04_add_category_invalid_parent(self):
        """Test adding a category with an invalid parent."""
        with self.assertRaises(ValueError):
            self.manager.add_category("Test", CategoryType.DOMAIN, "NonExistent")
    
    def test_05_remove_category_no_children(self):
        """Test removing a category with no children."""
        self.manager.add_category("Development", CategoryType.DOMAIN)
        result = self.manager.remove_category("Development")
        self.assertTrue(result)
        self.assertIsNone(self.manager.get_category("Development"))
    
    def test_06_remove_category_with_children_non_recursive(self):
        """Test that removing a category with children requires recursive flag."""
        self.manager.add_category("Development", CategoryType.DOMAIN)
        self.manager.add_category("Backend", CategoryType.SUBDOMAIN, "Development")
        
        with self.assertRaises(ValueError):
            self.manager.remove_category("Development", recursive=False)
    
    def test_07_remove_category_recursive(self):
        """Test recursive removal of categories."""
        self.manager.add_category("Development", CategoryType.DOMAIN)
        self.manager.add_category("Backend", CategoryType.SUBDOMAIN, "Development")
        self.manager.add_category("APIs", CategoryType.CATEGORY, "Backend")
        
        result = self.manager.remove_category("Development", recursive=True)
        self.assertTrue(result)
        self.assertIsNone(self.manager.get_category("Development"))
        self.assertIsNone(self.manager.get_category("Backend"))
        self.assertIsNone(self.manager.get_category("APIs"))
    
    def test_08_get_category_path(self):
        """Test getting the full path of a category."""
        self.manager.add_category("Development", CategoryType.DOMAIN)
        self.manager.add_category("Backend", CategoryType.SUBDOMAIN, "Development")
        
        path = self.manager.get_path("Backend")
        self.assertEqual(path, ["PULSE_ROOT", "Development", "Backend"])
    
    # Tool Categorization Tests (9-14)
    
    def test_09_categorize_tool(self):
        """Test associating a tool with a category."""
        self.manager.add_category("Development", CategoryType.DOMAIN)
        result = self.manager.categorize_tool("Python", "Development")
        self.assertTrue(result)
        
        dev = self.manager.get_category("Development")
        self.assertIn("Python", dev.tools)
    
    def test_10_categorize_tool_multiple_categories(self):
        """Test associating a tool with multiple categories."""
        self.manager.add_category("Development", CategoryType.DOMAIN)
        self.manager.add_category("DataScience", CategoryType.DOMAIN)
        
        self.manager.categorize_tool("Python", "Development")
        self.manager.categorize_tool("Python", "DataScience")
        
        categories = self.manager.get_tool_categories("Python")
        self.assertEqual(len(categories), 2)
    
    def test_11_remove_tool_from_category(self):
        """Test removing a tool from a category."""
        self.manager.add_category("Development", CategoryType.DOMAIN)
        self.manager.categorize_tool("Python", "Development")
        
        result = self.manager.remove_tool_from_category("Python", "Development")
        self.assertTrue(result)
        
        dev = self.manager.get_category("Development")
        self.assertNotIn("Python", dev.tools)
    
    def test_12_get_tool_categories(self):
        """Test retrieving all categories for a tool."""
        self.manager.add_category("Development", CategoryType.DOMAIN)
        self.manager.add_category("DataScience", CategoryType.DOMAIN)
        self.manager.add_category("ML", CategoryType.SUBDOMAIN, "DataScience")
        
        self.manager.categorize_tool("Python", "Development")
        self.manager.categorize_tool("Python", "DataScience")
        self.manager.categorize_tool("Python", "ML")
        
        categories = self.manager.get_tool_categories("Python")
        category_names = [cat.name for cat in categories]
        self.assertEqual(len(categories), 3)
        self.assertIn("Development", category_names)
        self.assertIn("DataScience", category_names)
        self.assertIn("ML", category_names)
    
    def test_13_get_all_tools_recursive(self):
        """Test getting all tools from a category and its children."""
        self.manager.add_category("Development", CategoryType.DOMAIN)
        self.manager.add_category("Backend", CategoryType.SUBDOMAIN, "Development")
        
        self.manager.categorize_tool("Python", "Development")
        self.manager.categorize_tool("Django", "Backend")
        
        dev = self.manager.get_category("Development")
        all_tools = dev.get_all_tools(recursive=True)
        self.assertEqual(len(all_tools), 2)
        self.assertIn("Python", all_tools)
        self.assertIn("Django", all_tools)
    
    def test_14_get_tools_non_recursive(self):
        """Test getting tools from only the specified category."""
        self.manager.add_category("Development", CategoryType.DOMAIN)
        self.manager.add_category("Backend", CategoryType.SUBDOMAIN, "Development")
        
        self.manager.categorize_tool("Python", "Development")
        self.manager.categorize_tool("Django", "Backend")
        
        dev = self.manager.get_category("Development")
        tools = dev.get_all_tools(recursive=False)
        self.assertEqual(len(tools), 1)
        self.assertIn("Python", tools)
        self.assertNotIn("Django", tools)
    
    # Tag Management Tests (15-18)
    
    def test_15_add_tag_to_category(self):
        """Test adding tags to a category."""
        self.manager.add_category("Development", CategoryType.DOMAIN)
        result = self.manager.add_tag("Development", "programming")
        self.assertTrue(result)
        
        dev = self.manager.get_category("Development")
        self.assertIn("programming", dev.tags)
    
    def test_16_add_multiple_tags(self):
        """Test adding multiple tags to a category."""
        self.manager.add_category("Development", CategoryType.DOMAIN, 
                                  tags={"programming", "software"})
        self.manager.add_tag("Development", "coding")
        
        dev = self.manager.get_category("Development")
        self.assertEqual(len(dev.tags), 3)
        self.assertIn("programming", dev.tags)
        self.assertIn("software", dev.tags)
        self.assertIn("coding", dev.tags)
    
    def test_17_remove_tag_from_category(self):
        """Test removing a tag from a category."""
        self.manager.add_category("Development", CategoryType.DOMAIN, 
                                  tags={"programming", "software"})
        result = self.manager.remove_tag("Development", "programming")
        self.assertTrue(result)
        
        dev = self.manager.get_category("Development")
        self.assertNotIn("programming", dev.tags)
        self.assertIn("software", dev.tags)
    
    def test_18_find_categories_by_tag(self):
        """Test finding categories by tag."""
        self.manager.add_category("Development", CategoryType.DOMAIN, 
                                  tags={"programming"})
        self.manager.add_category("DataScience", CategoryType.DOMAIN, 
                                  tags={"programming", "analytics"})
        self.manager.add_category("Design", CategoryType.DOMAIN, 
                                  tags={"creative"})
        
        results = self.manager.find_categories_by_tag("programming")
        result_names = [node.name for node in results]
        self.assertEqual(len(results), 2)
        self.assertIn("Development", result_names)
        self.assertIn("DataScience", result_names)
    
    # Hierarchy Navigation Tests (19-25)
    
    def test_19_get_hierarchy_root(self):
        """Test getting the full hierarchy from root."""
        self.manager.add_category("Development", CategoryType.DOMAIN)
        self.manager.add_category("Backend", CategoryType.SUBDOMAIN, "Development")
        
        hierarchy = self.manager.get_hierarchy("PULSE_ROOT")
        self.assertEqual(hierarchy["name"], "PULSE_ROOT")
        self.assertEqual(len(hierarchy["children"]), 1)
        self.assertEqual(hierarchy["children"][0]["name"], "Development")
    
    def test_20_get_hierarchy_subtree(self):
        """Test getting a hierarchy subtree."""
        self.manager.add_category("Development", CategoryType.DOMAIN)
        self.manager.add_category("Backend", CategoryType.SUBDOMAIN, "Development")
        self.manager.add_category("Frontend", CategoryType.SUBDOMAIN, "Development")
        
        hierarchy = self.manager.get_hierarchy("Development")
        self.assertEqual(hierarchy["name"], "Development")
        self.assertEqual(len(hierarchy["children"]), 2)
    
    def test_21_search_categories_by_name(self):
        """Test searching categories by name."""
        self.manager.add_category("Development", CategoryType.DOMAIN)
        self.manager.add_category("DataScience", CategoryType.DOMAIN)
        self.manager.add_category("DevOps", CategoryType.DOMAIN)
        
        results = self.manager.search_categories("dev")
        result_names = [node.name for node in results]
        self.assertEqual(len(results), 2)
        self.assertIn("Development", result_names)
        self.assertIn("DevOps", result_names)
    
    def test_22_search_categories_by_tag(self):
        """Test searching categories by tags."""
        self.manager.add_category("Development", CategoryType.DOMAIN, 
                                  tags={"backend-tech"})
        self.manager.add_category("DataScience", CategoryType.DOMAIN, 
                                  tags={"analytics"})
        
        results = self.manager.search_categories("backend")
        result_names = [node.name for node in results]
        self.assertEqual(len(results), 1)
        self.assertIn("Development", result_names)
    
    def test_23_move_category_to_new_parent(self):
        """Test moving a category to a different parent."""
        self.manager.add_category("Development", CategoryType.DOMAIN)
        self.manager.add_category("DataScience", CategoryType.DOMAIN)
        self.manager.add_category("Backend", CategoryType.SUBDOMAIN, "Development")
        
        result = self.manager.move_category("Backend", "DataScience")
        self.assertTrue(result)
        
        backend = self.manager.get_category("Backend")
        self.assertEqual(backend.parent.name, "DataScience")
        
        path = backend.get_path()
        self.assertEqual(path, ["PULSE_ROOT", "DataScience", "Backend"])
    
    def test_24_move_category_circular_reference(self):
        """Test that moving a category to its own descendant raises an error."""
        self.manager.add_category("Development", CategoryType.DOMAIN)
        self.manager.add_category("Backend", CategoryType.SUBDOMAIN, "Development")
        self.manager.add_category("APIs", CategoryType.CATEGORY, "Backend")
        
        with self.assertRaises(ValueError):
            self.manager.move_category("Development", "APIs")
    
    def test_25_complex_taxonomy_workflow(self):
        """Test a complex workflow combining multiple operations."""
        # Build hierarchy
        self.manager.add_category("Development", CategoryType.DOMAIN, 
                                  tags={"tech", "programming"})
        self.manager.add_category("Backend", CategoryType.SUBDOMAIN, "Development")
        self.manager.add_category("Frontend", CategoryType.SUBDOMAIN, "Development")
        self.manager.add_category("APIs", CategoryType.CATEGORY, "Backend")
        
        # Categorize tools
        self.manager.categorize_tool("Python", "Development")
        self.manager.categorize_tool("Django", "Backend")
        self.manager.categorize_tool("FastAPI", "APIs")
        self.manager.categorize_tool("React", "Frontend")
        
        # Add more tags
        self.manager.add_tag("Backend", "server-side")
        self.manager.add_tag("Frontend", "client-side")
        
        # Verify structure
        dev = self.manager.get_category("Development")
        all_tools = dev.get_all_tools(recursive=True)
        self.assertEqual(len(all_tools), 4)
        
        # Test search
        backend_results = self.manager.search_categories("backend")
        self.assertEqual(len(backend_results), 1)
        
        # Test hierarchy
        hierarchy = self.manager.get_hierarchy("Development")
        self.assertEqual(len(hierarchy["children"]), 2)
        
        # Test tool categories
        django_cats = self.manager.get_tool_categories("Django")
        self.assertEqual(len(django_cats), 1)
        self.assertEqual(django_cats[0].name, "Backend")
        
        # Test path
        api_path = self.manager.get_path("APIs")
        self.assertEqual(len(api_path), 4)
        self.assertEqual(api_path[-1], "APIs")


if __name__ == "__main__":
    # Run the test suite
    unittest.main(verbosity=2)
