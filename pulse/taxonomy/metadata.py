"""
Metadata Manager for Tool Taxonomy System

This module provides the MetadataManager class for handling tool metadata,
attributes, tags, schemas, validation, search, and import/export functionality.
"""

import json
import yaml
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime
from pathlib import Path
import re
from dataclasses import dataclass, field, asdict
from enum import Enum


class MetadataType(Enum):
    """Types of metadata that can be associated with tools."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    LIST = "list"
    DICT = "dict"
    TAG = "tag"


@dataclass
class MetadataSchema:
    """Schema definition for metadata fields."""
    name: str
    type: MetadataType
    required: bool = False
    default: Any = None
    description: str = ""
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    enum_values: Optional[List[Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        data = asdict(self)
        data['type'] = self.type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetadataSchema':
        """Create schema from dictionary."""
        data = data.copy()
        data['type'] = MetadataType(data['type'])
        return cls(**data)


@dataclass
class MetadataEntry:
    """Single metadata entry with value and metadata."""
    key: str
    value: Any
    type: MetadataType
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    source: str = "manual"
    validated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary."""
        return {
            'key': self.key,
            'value': self.value,
            'type': self.type.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'source': self.source,
            'validated': self.validated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetadataEntry':
        """Create entry from dictionary."""
        data = data.copy()
        data['type'] = MetadataType(data['type'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class ValidationError(Exception):
    """Exception raised when metadata validation fails."""
    pass


class MetadataManager:
    """
    Manages tool metadata including tags, schemas, validation, search,
    and import/export functionality.
    """
    
    def __init__(self):
        """Initialize the MetadataManager."""
        self.schemas: Dict[str, MetadataSchema] = {}
        self.metadata_store: Dict[str, Dict[str, MetadataEntry]] = {}
        self.tags: Dict[str, Set[str]] = {}  # tag -> set of tool_ids
        self.tag_hierarchy: Dict[str, Set[str]] = {}  # parent_tag -> child_tags
        self._init_default_schemas()
    
    def _init_default_schemas(self):
        """Initialize default metadata schemas."""
        default_schemas = [
            MetadataSchema(
                name="name",
                type=MetadataType.STRING,
                required=True,
                description="Tool name"
            ),
            MetadataSchema(
                name="description",
                type=MetadataType.STRING,
                required=True,
                description="Tool description"
            ),
            MetadataSchema(
                name="version",
                type=MetadataType.STRING,
                description="Tool version"
            ),
            MetadataSchema(
                name="author",
                type=MetadataType.STRING,
                description="Tool author"
            ),
            MetadataSchema(
                name="created_at",
                type=MetadataType.DATETIME,
                description="Creation timestamp"
            ),
            MetadataSchema(
                name="updated_at",
                type=MetadataType.DATETIME,
                description="Last update timestamp"
            ),
            MetadataSchema(
                name="tags",
                type=MetadataType.LIST,
                default=[],
                description="Associated tags"
            ),
            MetadataSchema(
                name="category",
                type=MetadataType.STRING,
                description="Tool category"
            ),
            MetadataSchema(
                name="license",
                type=MetadataType.STRING,
                description="Tool license"
            ),
            MetadataSchema(
                name="dependencies",
                type=MetadataType.LIST,
                default=[],
                description="Tool dependencies"
            ),
            MetadataSchema(
                name="deprecated",
                type=MetadataType.BOOLEAN,
                default=False,
                description="Whether tool is deprecated"
            ),
            MetadataSchema(
                name="priority",
                type=MetadataType.INTEGER,
                default=0,
                validation_rules={'min': 0, 'max': 10},
                description="Tool priority (0-10)"
            )
        ]
        
        for schema in default_schemas:
            self.register_schema(schema)
    
    def register_schema(self, schema: MetadataSchema):
        """
        Register a metadata schema.
        
        Args:
            schema: MetadataSchema to register
        """
        self.schemas[schema.name] = schema
    
    def unregister_schema(self, schema_name: str) -> bool:
        """
        Unregister a metadata schema.
        
        Args:
            schema_name: Name of schema to unregister
            
        Returns:
            True if schema was removed, False if not found
        """
        if schema_name in self.schemas:
            del self.schemas[schema_name]
            return True
        return False
    
    def get_schema(self, schema_name: str) -> Optional[MetadataSchema]:
        """
        Get a metadata schema by name.
        
        Args:
            schema_name: Name of schema to retrieve
            
        Returns:
            MetadataSchema if found, None otherwise
        """
        return self.schemas.get(schema_name)
    
    def list_schemas(self) -> List[MetadataSchema]:
        """
        List all registered schemas.
        
        Returns:
            List of all MetadataSchema objects
        """
        return list(self.schemas.values())
    
    def validate_value(self, schema: MetadataSchema, value: Any) -> bool:
        """
        Validate a value against a schema.
        
        Args:
            schema: MetadataSchema to validate against
            value: Value to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Check enum values
        if schema.enum_values and value not in schema.enum_values:
            raise ValidationError(
                f"Value {value} not in allowed values: {schema.enum_values}"
            )
        
        # Type-specific validation
        if schema.type == MetadataType.STRING:
            if not isinstance(value, str):
                raise ValidationError(f"Expected string, got {type(value)}")
            if 'min_length' in schema.validation_rules:
                if len(value) < schema.validation_rules['min_length']:
                    raise ValidationError(
                        f"String length {len(value)} less than minimum "
                        f"{schema.validation_rules['min_length']}"
                    )
            if 'max_length' in schema.validation_rules:
                if len(value) > schema.validation_rules['max_length']:
                    raise ValidationError(
                        f"String length {len(value)} exceeds maximum "
                        f"{schema.validation_rules['max_length']}"
                    )
            if 'pattern' in schema.validation_rules:
                if not re.match(schema.validation_rules['pattern'], value):
                    raise ValidationError(
                        f"String does not match pattern "
                        f"{schema.validation_rules['pattern']}"
                    )
        
        elif schema.type == MetadataType.INTEGER:
            if not isinstance(value, int):
                raise ValidationError(f"Expected integer, got {type(value)}")
            if 'min' in schema.validation_rules:
                if value < schema.validation_rules['min']:
                    raise ValidationError(
                        f"Value {value} less than minimum "
                        f"{schema.validation_rules['min']}"
                    )
            if 'max' in schema.validation_rules:
                if value > schema.validation_rules['max']:
                    raise ValidationError(
                        f"Value {value} exceeds maximum "
                        f"{schema.validation_rules['max']}"
                    )
        
        elif schema.type == MetadataType.FLOAT:
            if not isinstance(value, (int, float)):
                raise ValidationError(f"Expected float, got {type(value)}")
            if 'min' in schema.validation_rules:
                if value < schema.validation_rules['min']:
                    raise ValidationError(
                        f"Value {value} less than minimum "
                        f"{schema.validation_rules['min']}"
                    )
            if 'max' in schema.validation_rules:
                if value > schema.validation_rules['max']:
                    raise ValidationError(
                        f"Value {value} exceeds maximum "
                        f"{schema.validation_rules['max']}"
                    )
        
        elif schema.type == MetadataType.BOOLEAN:
            if not isinstance(value, bool):
                raise ValidationError(f"Expected boolean, got {type(value)}")
        
        elif schema.type == MetadataType.LIST:
            if not isinstance(value, list):
                raise ValidationError(f"Expected list, got {type(value)}")
            if 'min_items' in schema.validation_rules:
                if len(value) < schema.validation_rules['min_items']:
                    raise ValidationError(
                        f"List length {len(value)} less than minimum "
                        f"{schema.validation_rules['min_items']}"
                    )
            if 'max_items' in schema.validation_rules:
                if len(value) > schema.validation_rules['max_items']:
                    raise ValidationError(
                        f"List length {len(value)} exceeds maximum "
                        f"{schema.validation_rules['max_items']}"
                    )
        
        elif schema.type == MetadataType.DICT:
            if not isinstance(value, dict):
                raise ValidationError(f"Expected dict, got {type(value)}")
        
        return True
    
    def set_metadata(self, tool_id: str, key: str, value: Any, 
                    source: str = "manual", validate: bool = True) -> MetadataEntry:
        """
        Set metadata for a tool.
        
        Args:
            tool_id: Tool identifier
            key: Metadata key
            value: Metadata value
            source: Source of metadata
            validate: Whether to validate against schema
            
        Returns:
            Created MetadataEntry
            
        Raises:
            ValidationError: If validation fails
        """
        # Validate if schema exists and validation requested
        if validate and key in self.schemas:
            schema = self.schemas[key]
            self.validate_value(schema, value)
            metadata_type = schema.type
        else:
            # Infer type
            metadata_type = self._infer_type(value)
        
        # Create or update entry
        if tool_id not in self.metadata_store:
            self.metadata_store[tool_id] = {}
        
        if key in self.metadata_store[tool_id]:
            entry = self.metadata_store[tool_id][key]
            entry.value = value
            entry.updated_at = datetime.utcnow()
            entry.validated = validate
        else:
            entry = MetadataEntry(
                key=key,
                value=value,
                type=metadata_type,
                source=source,
                validated=validate
            )
            self.metadata_store[tool_id][key] = entry
        
        # Handle tags specially
        if key == "tags" and isinstance(value, list):
            for tag in value:
                self.add_tag(tool_id, tag)
        
        return entry
    
    def get_metadata(self, tool_id: str, key: str) -> Optional[Any]:
        """
        Get metadata value for a tool.
        
        Args:
            tool_id: Tool identifier
            key: Metadata key
            
        Returns:
            Metadata value if found, None otherwise
        """
        if tool_id in self.metadata_store:
            if key in self.metadata_store[tool_id]:
                return self.metadata_store[tool_id][key].value
        return None
    
    def get_all_metadata(self, tool_id: str) -> Dict[str, Any]:
        """
        Get all metadata for a tool.
        
        Args:
            tool_id: Tool identifier
            
        Returns:
            Dictionary of all metadata key-value pairs
        """
        if tool_id in self.metadata_store:
            return {
                key: entry.value 
                for key, entry in self.metadata_store[tool_id].items()
            }
        return {}
    
    def delete_metadata(self, tool_id: str, key: str) -> bool:
        """
        Delete metadata for a tool.
        
        Args:
            tool_id: Tool identifier
            key: Metadata key
            
        Returns:
            True if deleted, False if not found
        """
        if tool_id in self.metadata_store:
            if key in self.metadata_store[tool_id]:
                del self.metadata_store[tool_id][key]
                return True
        return False
    
    def add_tag(self, tool_id: str, tag: str):
        """
        Add a tag to a tool.
        
        Args:
            tool_id: Tool identifier
            tag: Tag to add
        """
        if tag not in self.tags:
            self.tags[tag] = set()
        self.tags[tag].add(tool_id)
    
    def remove_tag(self, tool_id: str, tag: str) -> bool:
        """
        Remove a tag from a tool.
        
        Args:
            tool_id: Tool identifier
            tag: Tag to remove
            
        Returns:
            True if removed, False if not found
        """
        if tag in self.tags and tool_id in self.tags[tag]:
            self.tags[tag].remove(tool_id)
            if not self.tags[tag]:
                del self.tags[tag]
            return True
        return False
    
    def get_tags(self, tool_id: str) -> Set[str]:
        """
        Get all tags for a tool.
        
        Args:
            tool_id: Tool identifier
            
        Returns:
            Set of tags
        """
        return {tag for tag, tools in self.tags.items() if tool_id in tools}
    
    def get_tools_by_tag(self, tag: str) -> Set[str]:
        """
        Get all tools with a specific tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            Set of tool IDs
        """
        return self.tags.get(tag, set()).copy()
    
    def set_tag_hierarchy(self, parent_tag: str, child_tag: str):
        """
        Set a parent-child relationship between tags.
        
        Args:
            parent_tag: Parent tag
            child_tag: Child tag
        """
        if parent_tag not in self.tag_hierarchy:
            self.tag_hierarchy[parent_tag] = set()
        self.tag_hierarchy[parent_tag].add(child_tag)
    
    def get_tag_children(self, tag: str, recursive: bool = False) -> Set[str]:
        """
        Get child tags of a tag.
        
        Args:
            tag: Parent tag
            recursive: Whether to get all descendants
            
        Returns:
            Set of child tags
        """
        if tag not in self.tag_hierarchy:
            return set()
        
        children = self.tag_hierarchy[tag].copy()
        
        if recursive:
            for child in list(children):
                children.update(self.get_tag_children(child, recursive=True))
        
        return children
    
    def search(self, query: Dict[str, Any], operator: str = "AND") -> List[str]:
        """
        Search for tools by metadata criteria.
        
        Args:
            query: Dictionary of metadata key-value pairs to match
            operator: "AND" or "OR" for combining criteria
            
        Returns:
            List of matching tool IDs
        """
        if not query:
            return list(self.metadata_store.keys())
        
        matching_tools = set()
        first_criteria = True
        
        for key, value in query.items():
            tools_matching_criteria = set()
            
            for tool_id, metadata in self.metadata_store.items():
                if key in metadata:
                    entry_value = metadata[key].value
                    
                    # Handle different comparison types
                    if isinstance(value, dict) and 'operator' in value:
                        if self._compare_value(entry_value, value):
                            tools_matching_criteria.add(tool_id)
                    elif entry_value == value:
                        tools_matching_criteria.add(tool_id)
            
            # Combine results based on operator
            if first_criteria:
                matching_tools = tools_matching_criteria
                first_criteria = False
            else:
                if operator.upper() == "AND":
                    matching_tools &= tools_matching_criteria
                else:  # OR
                    matching_tools |= tools_matching_criteria
        
        return list(matching_tools)
    
    def _compare_value(self, actual_value: Any, criteria: Dict[str, Any]) -> bool:
        """Compare a value against search criteria."""
        operator = criteria['operator']
        expected_value = criteria['value']
        
        if operator == 'eq':
            return actual_value == expected_value
        elif operator == 'ne':
            return actual_value != expected_value
        elif operator == 'gt':
            return actual_value > expected_value
        elif operator == 'gte':
            return actual_value >= expected_value
        elif operator == 'lt':
            return actual_value < expected_value
        elif operator == 'lte':
            return actual_value <= expected_value
        elif operator == 'in':
            return actual_value in expected_value
        elif operator == 'contains':
            return expected_value in actual_value
        elif operator == 'startswith':
            return str(actual_value).startswith(str(expected_value))
        elif operator == 'endswith':
            return str(actual_value).endswith(str(expected_value))
        
        return False
    
    def export_metadata(self, tool_id: str, format: str = "json") -> str:
        """
        Export metadata for a tool.
        
        Args:
            tool_id: Tool identifier
            format: Export format ("json" or "yaml")
            
        Returns:
            Exported metadata as string
        """
        metadata = self.get_all_metadata(tool_id)
        
        if format.lower() == "json":
            return json.dumps(metadata, indent=2, default=str)
        elif format.lower() == "yaml":
            return yaml.dump(metadata, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_metadata(self, tool_id: str, data: str, 
                       format: str = "json", validate: bool = True):
        """
        Import metadata for a tool.
        
        Args:
            tool_id: Tool identifier
            data: Metadata string to import
            format: Import format ("json" or "yaml")
            validate: Whether to validate imported data
        """
        if format.lower() == "json":
            metadata = json.loads(data)
        elif format.lower() == "yaml":
            metadata = yaml.safe_load(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        for key, value in metadata.items():
            self.set_metadata(tool_id, key, value, 
                            source="import", validate=validate)
    
    def export_all_metadata(self, format: str = "json") -> str:
        """
        Export all metadata for all tools.
        
        Args:
            format: Export format ("json" or "yaml")
            
        Returns:
            Exported metadata as string
        """
        all_metadata = {
            tool_id: self.get_all_metadata(tool_id)
            for tool_id in self.metadata_store.keys()
        }
        
        if format.lower() == "json":
            return json.dumps(all_metadata, indent=2, default=str)
        elif format.lower() == "yaml":
            return yaml.dump(all_metadata, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def bulk_update_metadata(self, updates: Dict[str, Dict[str, Any]], 
                            validate: bool = True) -> Dict[str, int]:
        """
        Update metadata for multiple tools.
        
        Args:
            updates: Dictionary mapping tool_id to metadata updates
            validate: Whether to validate updates
            
        Returns:
            Dictionary with counts of successful and failed updates
        """
        results = {"successful": 0, "failed": 0}
        
        for tool_id, metadata in updates.items():
            try:
                for key, value in metadata.items():
                    self.set_metadata(tool_id, key, value, validate=validate)
                results["successful"] += 1
            except Exception:
                results["failed"] += 1
        
        return results
    
    def _infer_type(self, value: Any) -> MetadataType:
        """Infer MetadataType from a value."""
        if isinstance(value, bool):
            return MetadataType.BOOLEAN
        elif isinstance(value, int):
            return MetadataType.INTEGER
        elif isinstance(value, float):
            return MetadataType.FLOAT
        elif isinstance(value, str):
            return MetadataType.STRING
        elif isinstance(value, list):
            return MetadataType.LIST
        elif isinstance(value, dict):
            return MetadataType.DICT
        elif isinstance(value, datetime):
            return MetadataType.DATETIME
        else:
            return MetadataType.STRING
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about metadata.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_tools": len(self.metadata_store),
            "total_schemas": len(self.schemas),
            "total_tags": len(self.tags),
            "total_metadata_entries": sum(
                len(entries) for entries in self.metadata_store.values()
            ),
            "tags_by_count": {
                tag: len(tools) for tag, tools in self.tags.items()
            },
            "most_common_tags": sorted(
                self.tags.items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            )[:10]
        }
