"""
Tool Factory - Creates and configures tools dynamically
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import json


class ToolFactory:
    """Factory for creating and configuring tools with metadata and versioning"""
    
    def __init__(self, registry=None):
        self.registry = registry
        self.creation_log = []
    
    def create_tool(
        self,
        tool_id: str,
        tool_type: str,
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        version: str = "1.0.0"
    ) -> Dict[str, Any]:
        """
        Create a new tool with full metadata and versioning
        
        Args:
            tool_id: Unique identifier for the tool
            tool_type: Category of tool (e.g., 'code_analysis', 'data_processor')
            parameters: Tool configuration parameters
            metadata: Additional metadata (author, description, tags)
            version: Semantic version string
        
        Returns:
            Tool definition dictionary
        """
        tool = {
            "id": tool_id,
            "type": tool_type,
            "version": version,
            "parameters": parameters,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "status": "active",
            "usage_count": 0,
            "value_score": 0.0
        }
        
        # Log creation
        self.creation_log.append({
            "tool_id": tool_id,
            "timestamp": tool["created_at"],
            "action": "created"
        })
        
        # Register if registry available
        if self.registry:
            self.registry.register(tool)
        
        return tool
    
    def clone_tool(self, source_tool: Dict[str, Any], new_id: str, modifications: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Clone an existing tool with optional modifications"""
        cloned = source_tool.copy()
        cloned["id"] = new_id
        cloned["created_at"] = datetime.utcnow().isoformat()
        cloned["parent_id"] = source_tool["id"]
        cloned["usage_count"] = 0
        
        if modifications:
            cloned["parameters"].update(modifications.get("parameters", {}))
            cloned["metadata"].update(modifications.get("metadata", {}))
        
        if self.registry:
            self.registry.register(cloned)
        
        return cloned
    
    def validate_tool(self, tool: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate tool structure and parameters"""
        errors = []
        
        required_fields = ["id", "type", "version", "parameters"]
        for field in required_fields:
            if field not in tool:
                errors.append(f"Missing required field: {field}")
        
        if "parameters" in tool and not isinstance(tool["parameters"], dict):
            errors.append("Parameters must be a dictionary")
        
        return len(errors) == 0, errors
    
    def get_creation_stats(self) -> Dict[str, Any]:
        """Get statistics about tool creation"""
        return {
            "total_created": len(self.creation_log),
            "log": self.creation_log
        }
