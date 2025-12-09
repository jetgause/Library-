"""
Tool Registry - Manages tool storage, retrieval, and lifecycle
"""
from typing import Dict, Any, List, Optional
from datetime import datetime


class ToolRegistry:
    """Central registry for managing tools and their lifecycle"""
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tags_index: Dict[str, List[str]] = {}
        self.type_index: Dict[str, List[str]] = {}
    
    def register(self, tool: Dict[str, Any]) -> bool:
        """Register a new tool or update existing"""
        tool_id = tool["id"]
        self.tools[tool_id] = tool
        
        # Index by type
        tool_type = tool.get("type")
        if tool_type:
            if tool_type not in self.type_index:
                self.type_index[tool_type] = []
            if tool_id not in self.type_index[tool_type]:
                self.type_index[tool_type].append(tool_id)
        
        # Index by tags
        tags = tool.get("metadata", {}).get("tags", [])
        for tag in tags:
            if tag not in self.tags_index:
                self.tags_index[tag] = []
            if tool_id not in self.tags_index[tag]:
                self.tags_index[tag].append(tool_id)
        
        return True
    
    def get(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a tool by ID"""
        return self.tools.get(tool_id)
    
    def find_by_type(self, tool_type: str) -> List[Dict[str, Any]]:
        """Find all tools of a specific type"""
        tool_ids = self.type_index.get(tool_type, [])
        return [self.tools[tid] for tid in tool_ids if tid in self.tools]
    
    def find_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Find all tools with a specific tag"""
        tool_ids = self.tags_index.get(tag, [])
        return [self.tools[tid] for tid in tool_ids if tid in self.tools]
    
    def search(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Advanced search with multiple criteria
        
        Query format:
        {
            "type": "code_analysis",
            "tags": ["python", "ast"],
            "min_value_score": 0.7,
            "status": "active"
        }
        """
        results = list(self.tools.values())
        
        if "type" in query:
            results = [t for t in results if t.get("type") == query["type"]]
        
        if "tags" in query:
            query_tags = set(query["tags"])
            results = [
                t for t in results
                if query_tags.issubset(set(t.get("metadata", {}).get("tags", [])))
            ]
        
        if "min_value_score" in query:
            results = [
                t for t in results
                if t.get("value_score", 0) >= query["min_value_score"]
            ]
        
        if "status" in query:
            results = [t for t in results if t.get("status") == query["status"]]
        
        return results
    
    def update_tool(self, tool_id: str, updates: Dict[str, Any]) -> bool:
        """Update tool properties"""
        if tool_id not in self.tools:
            return False
        
        self.tools[tool_id].update(updates)
        self.tools[tool_id]["updated_at"] = datetime.utcnow().isoformat()
        
        # Re-index if type or tags changed
        if "type" in updates or "metadata" in updates:
            tool = self.tools[tool_id]
            self.register(tool)  # Re-index
        
        return True
    
    def deprecate_tool(self, tool_id: str, reason: str = "") -> bool:
        """Mark a tool as deprecated"""
        return self.update_tool(tool_id, {
            "status": "deprecated",
            "deprecation_reason": reason,
            "deprecated_at": datetime.utcnow().isoformat()
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total = len(self.tools)
        by_status = {}
        by_type = {}
        
        for tool in self.tools.values():
            status = tool.get("status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1
            
            tool_type = tool.get("type", "unknown")
            by_type[tool_type] = by_type.get(tool_type, 0) + 1
        
        return {
            "total_tools": total,
            "by_status": by_status,
            "by_type": by_type,
            "total_tags": len(self.tags_index)
        }
    
    def list_all(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all tools, optionally filtered by status"""
        tools = list(self.tools.values())
        if status:
            tools = [t for t in tools if t.get("status") == status]
        return tools
