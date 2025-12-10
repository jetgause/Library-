"""
Pulse Core Models Module

Comprehensive Pydantic models for task management, analytics, notifications,
and health monitoring with full validation and database ORM support.

Author: jetgause
Created: 2025-12-10
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator
import re
import json


# ==================== ENUMS ====================

class TaskStatus(str, Enum):
    """Task status enumeration"""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Task priority enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TaskCategory(str, Enum):
    """Task category enumeration"""
    WORK = "work"
    PERSONAL = "personal"
    SHOPPING = "shopping"
    HEALTH = "health"
    FINANCE = "finance"
    OTHER = "other"


class NotificationType(str, Enum):
    """Notification type enumeration"""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


# ==================== VALIDATION UTILITIES ====================

class ValidationUtils:
    """Utility class for common validation functions"""
    
    # XSS prevention pattern - blocks common XSS attack vectors
    XSS_PATTERN = re.compile(
        r'(<script[^>]*>.*?</script>|'
        r'<iframe[^>]*>.*?</iframe>|'
        r'javascript:|'
        r'on\w+\s*=|'
        r'<object[^>]*>|'
        r'<embed[^>]*>|'
        r'<applet[^>]*>)',
        re.IGNORECASE | re.DOTALL
    )
    
    MAX_TAGS = 20
    MAX_METADATA_SIZE = 10 * 1024  # 10KB in bytes
    
    @staticmethod
    def check_xss(value: str, field_name: str = "field") -> str:
        """Check for XSS attack patterns in string"""
        if ValidationUtils.XSS_PATTERN.search(value):
            raise ValueError(
                f"{field_name} contains potentially dangerous content (XSS detected)"
            )
        return value
    
    @staticmethod
    def validate_tags(tags: Optional[List[str]]) -> Optional[List[str]]:
        """Validate tags list"""
        if tags is None:
            return tags
        
        if len(tags) > ValidationUtils.MAX_TAGS:
            raise ValueError(
                f"Maximum {ValidationUtils.MAX_TAGS} tags allowed, got {len(tags)}"
            )
        
        # Check each tag for XSS
        validated_tags = []
        for tag in tags:
            if not isinstance(tag, str):
                raise ValueError(f"All tags must be strings, got {type(tag)}")
            validated_tags.append(ValidationUtils.check_xss(tag.strip(), "tag"))
        
        return validated_tags
    
    @staticmethod
    def validate_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate metadata dictionary size"""
        if metadata is None:
            return metadata
        
        # Convert to JSON to check size
        try:
            metadata_json = json.dumps(metadata)
            metadata_size = len(metadata_json.encode('utf-8'))
            
            if metadata_size > ValidationUtils.MAX_METADATA_SIZE:
                raise ValueError(
                    f"Metadata exceeds maximum size of {ValidationUtils.MAX_METADATA_SIZE} bytes "
                    f"(current: {metadata_size} bytes)"
                )
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid metadata format: {str(e)}")
        
        return metadata


# ==================== TASK MODELS ====================

class TaskCreate(BaseModel):
    """Model for creating a new task"""
    
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Task title (1-200 characters)"
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Task description (max 2000 characters)"
    )
    status: TaskStatus = Field(
        default=TaskStatus.TODO,
        description="Task status"
    )
    priority: TaskPriority = Field(
        default=TaskPriority.MEDIUM,
        description="Task priority level"
    )
    category: TaskCategory = Field(
        default=TaskCategory.OTHER,
        description="Task category"
    )
    due_date: Optional[datetime] = Field(
        None,
        description="Task due date (ISO 8601 format)"
    )
    tags: Optional[List[str]] = Field(
        default_factory=list,
        description=f"Task tags (max {ValidationUtils.MAX_TAGS})"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description=f"Additional metadata (max {ValidationUtils.MAX_METADATA_SIZE} bytes)"
    )
    assigned_to: Optional[str] = Field(
        None,
        max_length=100,
        description="User assigned to this task"
    )
    
    @validator('title')
    def validate_title(cls, v):
        """Validate title for XSS attacks"""
        if not v or not v.strip():
            raise ValueError("Title cannot be empty or whitespace only")
        return ValidationUtils.check_xss(v.strip(), "title")
    
    @validator('description')
    def validate_description(cls, v):
        """Validate description for XSS attacks"""
        if v is not None and v.strip():
            return ValidationUtils.check_xss(v.strip(), "description")
        return v
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate tags list"""
        return ValidationUtils.validate_tags(v)
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """Validate metadata size"""
        return ValidationUtils.validate_metadata(v)
    
    @validator('due_date')
    def validate_due_date(cls, v):
        """Validate due date is not in the past"""
        if v is not None and v < datetime.utcnow():
            raise ValueError("Due date cannot be in the past")
        return v
    
    @validator('assigned_to')
    def validate_assigned_to(cls, v):
        """Validate assigned_to field"""
        if v is not None and v.strip():
            return ValidationUtils.check_xss(v.strip(), "assigned_to")
        return v
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "title": "Complete project proposal",
                "description": "Finalize and submit the Q1 project proposal document",
                "status": "todo",
                "priority": "high",
                "category": "work",
                "due_date": "2025-12-15T17:00:00Z",
                "tags": ["project", "urgent", "q1"],
                "metadata": {
                    "project_id": "PRJ-2025-001",
                    "estimated_hours": 8,
                    "department": "Engineering"
                },
                "assigned_to": "jetgause"
            }
        }


class TaskUpdate(BaseModel):
    """Model for updating an existing task"""
    
    title: Optional[str] = Field(
        None,
        min_length=1,
        max_length=200,
        description="Task title (1-200 characters)"
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Task description (max 2000 characters)"
    )
    status: Optional[TaskStatus] = Field(
        None,
        description="Task status"
    )
    priority: Optional[TaskPriority] = Field(
        None,
        description="Task priority level"
    )
    category: Optional[TaskCategory] = Field(
        None,
        description="Task category"
    )
    due_date: Optional[datetime] = Field(
        None,
        description="Task due date (ISO 8601 format)"
    )
    tags: Optional[List[str]] = Field(
        None,
        description=f"Task tags (max {ValidationUtils.MAX_TAGS})"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description=f"Additional metadata (max {ValidationUtils.MAX_METADATA_SIZE} bytes)"
    )
    assigned_to: Optional[str] = Field(
        None,
        max_length=100,
        description="User assigned to this task"
    )
    
    @validator('title')
    def validate_title(cls, v):
        """Validate title for XSS attacks"""
        if v is not None:
            if not v.strip():
                raise ValueError("Title cannot be empty or whitespace only")
            return ValidationUtils.check_xss(v.strip(), "title")
        return v
    
    @validator('description')
    def validate_description(cls, v):
        """Validate description for XSS attacks"""
        if v is not None and v.strip():
            return ValidationUtils.check_xss(v.strip(), "description")
        return v
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate tags list"""
        return ValidationUtils.validate_tags(v)
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """Validate metadata size"""
        return ValidationUtils.validate_metadata(v)
    
    @validator('due_date')
    def validate_due_date(cls, v):
        """Validate due date is not in the past"""
        if v is not None and v < datetime.utcnow():
            raise ValueError("Due date cannot be in the past")
        return v
    
    @validator('assigned_to')
    def validate_assigned_to(cls, v):
        """Validate assigned_to field"""
        if v is not None and v.strip():
            return ValidationUtils.check_xss(v.strip(), "assigned_to")
        return v
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "status": "in_progress",
                "priority": "urgent",
                "metadata": {
                    "progress_percentage": 45,
                    "last_updated_by": "jetgause"
                }
            }
        }


class TaskResponse(BaseModel):
    """Model for task response"""
    
    id: str = Field(..., description="Unique task identifier")
    title: str = Field(..., description="Task title")
    description: Optional[str] = Field(None, description="Task description")
    status: TaskStatus = Field(..., description="Task status")
    priority: TaskPriority = Field(..., description="Task priority level")
    category: TaskCategory = Field(..., description="Task category")
    due_date: Optional[datetime] = Field(None, description="Task due date")
    tags: List[str] = Field(default_factory=list, description="Task tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    assigned_to: Optional[str] = Field(None, description="User assigned to this task")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    created_by: str = Field(..., description="User who created the task")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    class Config:
        use_enum_values = True
        orm_mode = True
        json_schema_extra = {
            "example": {
                "id": "task_2025_001",
                "title": "Complete project proposal",
                "description": "Finalize and submit the Q1 project proposal document",
                "status": "in_progress",
                "priority": "high",
                "category": "work",
                "due_date": "2025-12-15T17:00:00Z",
                "tags": ["project", "urgent", "q1"],
                "metadata": {
                    "project_id": "PRJ-2025-001",
                    "estimated_hours": 8,
                    "progress_percentage": 45
                },
                "assigned_to": "jetgause",
                "created_at": "2025-12-10T04:36:24Z",
                "updated_at": "2025-12-10T04:36:24Z",
                "created_by": "jetgause",
                "completed_at": None
            }
        }


class TaskListResponse(BaseModel):
    """Model for paginated task list response"""
    
    tasks: List[TaskResponse] = Field(..., description="List of tasks")
    total: int = Field(..., ge=0, description="Total number of tasks")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=100, description="Number of items per page")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "tasks": [
                    {
                        "id": "task_2025_001",
                        "title": "Complete project proposal",
                        "description": "Finalize and submit the Q1 project proposal",
                        "status": "in_progress",
                        "priority": "high",
                        "category": "work",
                        "due_date": "2025-12-15T17:00:00Z",
                        "tags": ["project", "urgent"],
                        "metadata": {},
                        "assigned_to": "jetgause",
                        "created_at": "2025-12-10T04:36:24Z",
                        "updated_at": "2025-12-10T04:36:24Z",
                        "created_by": "jetgause",
                        "completed_at": None
                    }
                ],
                "total": 42,
                "page": 1,
                "page_size": 20,
                "total_pages": 3,
                "has_next": True,
                "has_prev": False
            }
        }


# ==================== ANALYTICS MODELS ====================

class TaskStatistics(BaseModel):
    """Statistics for task counts by status"""
    
    total: int = Field(..., ge=0, description="Total number of tasks")
    todo: int = Field(..., ge=0, description="Number of TODO tasks")
    in_progress: int = Field(..., ge=0, description="Number of IN_PROGRESS tasks")
    completed: int = Field(..., ge=0, description="Number of COMPLETED tasks")
    cancelled: int = Field(..., ge=0, description="Number of CANCELLED tasks")
    overdue: int = Field(..., ge=0, description="Number of overdue tasks")


class PriorityDistribution(BaseModel):
    """Distribution of tasks by priority"""
    
    low: int = Field(..., ge=0, description="Number of LOW priority tasks")
    medium: int = Field(..., ge=0, description="Number of MEDIUM priority tasks")
    high: int = Field(..., ge=0, description="Number of HIGH priority tasks")
    urgent: int = Field(..., ge=0, description="Number of URGENT priority tasks")


class CategoryDistribution(BaseModel):
    """Distribution of tasks by category"""
    
    work: int = Field(..., ge=0, description="Number of WORK tasks")
    personal: int = Field(..., ge=0, description="Number of PERSONAL tasks")
    shopping: int = Field(..., ge=0, description="Number of SHOPPING tasks")
    health: int = Field(..., ge=0, description="Number of HEALTH tasks")
    finance: int = Field(..., ge=0, description="Number of FINANCE tasks")
    other: int = Field(..., ge=0, description="Number of OTHER tasks")


class AnalyticsResponse(BaseModel):
    """Model for analytics response"""
    
    statistics: TaskStatistics = Field(..., description="Task statistics")
    priority_distribution: PriorityDistribution = Field(..., description="Priority distribution")
    category_distribution: CategoryDistribution = Field(..., description="Category distribution")
    completion_rate: float = Field(..., ge=0.0, le=100.0, description="Task completion rate (%)")
    average_completion_time: Optional[float] = Field(
        None,
        ge=0.0,
        description="Average task completion time in hours"
    )
    most_used_tags: List[Dict[str, Union[str, int]]] = Field(
        default_factory=list,
        description="Most frequently used tags"
    )
    generated_at: datetime = Field(..., description="Analytics generation timestamp")
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "statistics": {
                    "total": 100,
                    "todo": 25,
                    "in_progress": 30,
                    "completed": 40,
                    "cancelled": 5,
                    "overdue": 8
                },
                "priority_distribution": {
                    "low": 20,
                    "medium": 45,
                    "high": 25,
                    "urgent": 10
                },
                "category_distribution": {
                    "work": 50,
                    "personal": 25,
                    "shopping": 10,
                    "health": 8,
                    "finance": 5,
                    "other": 2
                },
                "completion_rate": 40.0,
                "average_completion_time": 24.5,
                "most_used_tags": [
                    {"tag": "urgent", "count": 15},
                    {"tag": "project", "count": 12},
                    {"tag": "review", "count": 8}
                ],
                "generated_at": "2025-12-10T04:36:24Z"
            }
        }


# ==================== NOTIFICATION MODELS ====================

class NotificationResponse(BaseModel):
    """Model for notification response"""
    
    id: str = Field(..., description="Unique notification identifier")
    type: NotificationType = Field(..., description="Notification type")
    title: str = Field(..., min_length=1, max_length=200, description="Notification title")
    message: str = Field(..., min_length=1, max_length=1000, description="Notification message")
    task_id: Optional[str] = Field(None, description="Associated task ID")
    user_id: str = Field(..., description="Target user ID")
    is_read: bool = Field(default=False, description="Read status")
    created_at: datetime = Field(..., description="Creation timestamp")
    read_at: Optional[datetime] = Field(None, description="Read timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        use_enum_values = True
        orm_mode = True
        json_schema_extra = {
            "example": {
                "id": "notif_2025_001",
                "type": "warning",
                "title": "Task Due Soon",
                "message": "Your task 'Complete project proposal' is due in 2 hours",
                "task_id": "task_2025_001",
                "user_id": "jetgause",
                "is_read": False,
                "created_at": "2025-12-10T04:36:24Z",
                "read_at": None,
                "metadata": {
                    "priority": "high",
                    "due_date": "2025-12-15T17:00:00Z"
                }
            }
        }


# ==================== HEALTH & ERROR MODELS ====================

class HealthResponse(BaseModel):
    """Model for health check response"""
    
    status: str = Field(..., description="Service status (healthy, degraded, unhealthy)")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Health check timestamp")
    uptime_seconds: float = Field(..., ge=0.0, description="Service uptime in seconds")
    database: Dict[str, Any] = Field(..., description="Database health status")
    memory: Dict[str, Any] = Field(..., description="Memory usage information")
    tasks_count: int = Field(..., ge=0, description="Total tasks in database")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2025-12-10T04:36:24Z",
                "uptime_seconds": 86400.0,
                "database": {
                    "status": "connected",
                    "latency_ms": 5.2
                },
                "memory": {
                    "used_mb": 256.5,
                    "available_mb": 1024.0,
                    "usage_percent": 25.05
                },
                "tasks_count": 100
            }
        }


class ErrorResponse(BaseModel):
    """Model for error response"""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., ge=400, le=599, description="HTTP status code")
    timestamp: datetime = Field(..., description="Error timestamp")
    path: Optional[str] = Field(None, description="Request path that caused the error")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data",
                "detail": "Title contains potentially dangerous content (XSS detected)",
                "status_code": 422,
                "timestamp": "2025-12-10T04:36:24Z",
                "path": "/api/v1/tasks",
                "request_id": "req_abc123xyz"
            }
        }


# ==================== DATABASE MODELS ====================

class TaskDB(BaseModel):
    """Database model for Task entity with ORM support"""
    
    id: str = Field(..., description="Unique task identifier")
    title: str = Field(..., description="Task title")
    description: Optional[str] = Field(None, description="Task description")
    status: str = Field(..., description="Task status")
    priority: str = Field(..., description="Task priority level")
    category: str = Field(..., description="Task category")
    due_date: Optional[datetime] = Field(None, description="Task due date")
    tags: List[str] = Field(default_factory=list, description="Task tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    assigned_to: Optional[str] = Field(None, description="User assigned to this task")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    created_by: str = Field(..., description="User who created the task")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    is_deleted: bool = Field(default=False, description="Soft delete flag")
    deleted_at: Optional[datetime] = Field(None, description="Deletion timestamp")
    version: int = Field(default=1, ge=1, description="Optimistic locking version")
    
    class Config:
        orm_mode = True
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "id": "task_2025_001",
                "title": "Complete project proposal",
                "description": "Finalize and submit the Q1 project proposal document",
                "status": "in_progress",
                "priority": "high",
                "category": "work",
                "due_date": "2025-12-15T17:00:00Z",
                "tags": ["project", "urgent", "q1"],
                "metadata": {
                    "project_id": "PRJ-2025-001",
                    "estimated_hours": 8
                },
                "assigned_to": "jetgause",
                "created_at": "2025-12-10T04:36:24Z",
                "updated_at": "2025-12-10T04:36:24Z",
                "created_by": "jetgause",
                "completed_at": None,
                "is_deleted": False,
                "deleted_at": None,
                "version": 1
            }
        }


class NotificationDB(BaseModel):
    """Database model for Notification entity with ORM support"""
    
    id: str = Field(..., description="Unique notification identifier")
    type: str = Field(..., description="Notification type")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    task_id: Optional[str] = Field(None, description="Associated task ID")
    user_id: str = Field(..., description="Target user ID")
    is_read: bool = Field(default=False, description="Read status")
    created_at: datetime = Field(..., description="Creation timestamp")
    read_at: Optional[datetime] = Field(None, description="Read timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    is_deleted: bool = Field(default=False, description="Soft delete flag")
    deleted_at: Optional[datetime] = Field(None, description="Deletion timestamp")
    expires_at: Optional[datetime] = Field(None, description="Notification expiration timestamp")
    
    class Config:
        orm_mode = True
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "id": "notif_2025_001",
                "type": "warning",
                "title": "Task Due Soon",
                "message": "Your task 'Complete project proposal' is due in 2 hours",
                "task_id": "task_2025_001",
                "user_id": "jetgause",
                "is_read": False,
                "created_at": "2025-12-10T04:36:24Z",
                "read_at": None,
                "metadata": {
                    "priority": "high",
                    "due_date": "2025-12-15T17:00:00Z"
                },
                "is_deleted": False,
                "deleted_at": None,
                "expires_at": "2025-12-17T04:36:24Z"
            }
        }


# ==================== FILTER & QUERY MODELS ====================

class TaskFilterParams(BaseModel):
    """Model for task filtering parameters"""
    
    status: Optional[TaskStatus] = Field(None, description="Filter by status")
    priority: Optional[TaskPriority] = Field(None, description="Filter by priority")
    category: Optional[TaskCategory] = Field(None, description="Filter by category")
    assigned_to: Optional[str] = Field(None, description="Filter by assigned user")
    created_by: Optional[str] = Field(None, description="Filter by creator")
    tags: Optional[List[str]] = Field(None, description="Filter by tags (OR operation)")
    due_before: Optional[datetime] = Field(None, description="Filter by due date before")
    due_after: Optional[datetime] = Field(None, description="Filter by due date after")
    search: Optional[str] = Field(None, max_length=200, description="Search in title and description")
    include_completed: bool = Field(default=True, description="Include completed tasks")
    include_cancelled: bool = Field(default=False, description="Include cancelled tasks")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    sort_by: str = Field(default="created_at", description="Sort field")
    sort_order: str = Field(default="desc", regex="^(asc|desc)$", description="Sort order")
    
    @validator('search')
    def validate_search(cls, v):
        """Validate search query for XSS"""
        if v is not None and v.strip():
            return ValidationUtils.check_xss(v.strip(), "search")
        return v
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "status": "in_progress",
                "priority": "high",
                "category": "work",
                "assigned_to": "jetgause",
                "tags": ["urgent", "project"],
                "due_before": "2025-12-31T23:59:59Z",
                "search": "proposal",
                "include_completed": False,
                "page": 1,
                "page_size": 20,
                "sort_by": "due_date",
                "sort_order": "asc"
            }
        }


# ==================== BULK OPERATIONS ====================

class BulkTaskUpdate(BaseModel):
    """Model for bulk task update operations"""
    
    task_ids: List[str] = Field(..., min_items=1, max_items=100, description="List of task IDs to update")
    updates: TaskUpdate = Field(..., description="Updates to apply to all tasks")
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "task_ids": ["task_2025_001", "task_2025_002", "task_2025_003"],
                "updates": {
                    "status": "in_progress",
                    "priority": "high"
                }
            }
        }


class BulkOperationResponse(BaseModel):
    """Model for bulk operation response"""
    
    success_count: int = Field(..., ge=0, description="Number of successful operations")
    failure_count: int = Field(..., ge=0, description="Number of failed operations")
    total: int = Field(..., ge=0, description="Total number of operations")
    errors: List[Dict[str, str]] = Field(default_factory=list, description="List of errors")
    updated_ids: List[str] = Field(default_factory=list, description="List of successfully updated IDs")
    timestamp: datetime = Field(..., description="Operation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success_count": 2,
                "failure_count": 1,
                "total": 3,
                "errors": [
                    {
                        "task_id": "task_2025_003",
                        "error": "Task not found"
                    }
                ],
                "updated_ids": ["task_2025_001", "task_2025_002"],
                "timestamp": "2025-12-10T04:36:24Z"
            }
        }
