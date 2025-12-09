"""
Taxonomy - Hierarchical classification system for tools
"""
from typing import Dict, Any, List, Optional, Set
from datetime import datetime


class Taxonomy:
    """Manages hierarchical tool classification and categorization"""
    
    def __init__(self):
        self.categories: Dict[str, Dict[str, Any]] = {}
        self.hierarchy: Dict[str, List[str]] = {}  # parent -> children
        self.tool_classifications: Dict[str, List[str]] = {}  # tool_id -> categories
    
    def create_category(
        self,
        category_id: str,
        name: str,
        description: str = "",
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new category in the taxonomy
        
        Args:
            category_id: Unique identifier
            name: Display name
            description: Category description
            parent_id: Parent category ID for hierarchical structure
            attributes: Additional category attributes
        """
        if category_id in self.categories:
            raise ValueError(f"Category {category_id} already exists")
        
        if parent_id and parent_id not in self.categories:
            raise ValueError(f"Parent category {parent_id} does not exist")
        
        category = {
            "id": category_id,
            "name": name,
            "description": description,
            "parent_id": parent_id,
            "attributes": attributes or {},
            "created_at": datetime.utcnow().isoformat(),
            "tool_count": 0
        }
        
        self.categories[category_id] = category
        
        # Update hierarchy
        if parent_id:
            if parent_id not in self.hierarchy:
                self.hierarchy[parent_id] = []
            self.hierarchy[parent_id].append(category_id)
        
        return category
    
    def classify_tool(self, tool_id: str, category_ids: List[str]) -> bool:
        """Assign tool to one or more categories"""
        # Validate categories exist
        for cat_id in category_ids:
            if cat_id not in self.categories:
                raise ValueError(f"Category {cat_id} does not exist")
        
        self.tool_classifications[tool_id] = category_ids
        
        # Update tool counts
        for cat_id in category_ids:
            self.categories[cat_id]["tool_count"] += 1
        
        return True
    
    def get_tool_categories(self, tool_id: str) -> List[Dict[str, Any]]:
        """Get all categories for a tool"""
        category_ids = self.tool_classifications.get(tool_id, [])
        return [self.categories[cid] for cid in category_ids if cid in self.categories]
    
    def get_category(self, category_id: str) -> Optional[Dict[str, Any]]:
        """Get category details"""
        return self.categories.get(category_id)
    
    def get_subcategories(self, category_id: str) -> List[Dict[str, Any]]:
        """Get direct children of a category"""
        child_ids = self.hierarchy.get(category_id, [])
        return [self.categories[cid] for cid in child_ids if cid in self.categories]
    
    def get_category_path(self, category_id: str) -> List[Dict[str, Any]]:
        """Get full path from root to category"""
        if category_id not in self.categories:
            return []
        
        path = []
        current_id = category_id
        
        while current_id:
            category = self.categories[current_id]
            path.insert(0, category)
            current_id = category.get("parent_id")
        
        return path
    
    def get_tools_in_category(self, category_id: str, include_subcategories: bool = False) -> List[str]:
        """Get all tools in a category"""
        tools = [
            tool_id for tool_id, cats in self.tool_classifications.items()
            if category_id in cats
        ]
        
        if include_subcategories:
            subcategories = self._get_all_descendants(category_id)
            for subcat_id in subcategories:
                sub_tools = [
                    tool_id for tool_id, cats in self.tool_classifications.items()
                    if subcat_id in cats
                ]
                tools.extend(sub_tools)
        
        return list(set(tools))  # Remove duplicates
    
    def _get_all_descendants(self, category_id: str) -> List[str]:
        """Recursively get all descendant categories"""
        descendants = []
        children = self.hierarchy.get(category_id, [])
        
        for child_id in children:
            descendants.append(child_id)
            descendants.extend(self._get_all_descendants(child_id))
        
        return descendants
    
    def search_categories(self, query: str) -> List[Dict[str, Any]]:
        """Search categories by name or description"""
        query_lower = query.lower()
        results = []
        
        for category in self.categories.values():
            if (query_lower in category["name"].lower() or
                query_lower in category.get("description", "").lower()):
                results.append(category)
        
        return results
    
    def get_root_categories(self) -> List[Dict[str, Any]]:
        """Get all top-level categories (no parent)"""
        return [
            cat for cat in self.categories.values()
            if cat.get("parent_id") is None
        ]
    
    def get_category_tree(self, root_id: Optional[str] = None) -> Dict[str, Any]:
        """Get hierarchical tree structure"""
        if root_id:
            if root_id not in self.categories:
                return {}
            root = self.categories[root_id]
        else:
            root = {"id": "root", "name": "Root", "children": []}
            root_categories = self.get_root_categories()
            for cat in root_categories:
                root["children"].append(self._build_tree(cat["id"]))
            return root
        
        return self._build_tree(root_id)
    
    def _build_tree(self, category_id: str) -> Dict[str, Any]:
        """Recursively build category tree"""
        category = self.categories[category_id].copy()
        children = self.hierarchy.get(category_id, [])
        
        if children:
            category["children"] = [self._build_tree(child_id) for child_id in children]
        else:
            category["children"] = []
        
        return category
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get taxonomy statistics"""
        total_categories = len(self.categories)
        total_tools = len(self.tool_classifications)
        root_categories = len(self.get_root_categories())
        
        # Calculate depth
        max_depth = 0
        for cat_id in self.categories:
            path = self.get_category_path(cat_id)
            max_depth = max(max_depth, len(path))
        
        # Tools per category distribution
        tools_per_category = [cat["tool_count"] for cat in self.categories.values()]
        avg_tools = sum(tools_per_category) / total_categories if total_categories > 0 else 0
        
        return {
            "total_categories": total_categories,
            "root_categories": root_categories,
            "total_tools_classified": total_tools,
            "max_depth": max_depth,
            "avg_tools_per_category": round(avg_tools, 2)
        }
    
    def validate_hierarchy(self) -> Dict[str, Any]:
        """Validate taxonomy structure for circular references"""
        errors = []
        visited = set()
        
        def has_cycle(category_id: str, path: Set[str]) -> bool:
            if category_id in path:
                return True
            
            category = self.categories.get(category_id)
            if not category:
                return False
            
            parent_id = category.get("parent_id")
            if not parent_id:
                return False
            
            return has_cycle(parent_id, path | {category_id})
        
        for cat_id in self.categories:
            if has_cycle(cat_id, set()):
                errors.append(f"Circular reference detected in category: {cat_id}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
