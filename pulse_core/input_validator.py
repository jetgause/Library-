"""
Input Validation Utilities for Pulse Core

Comprehensive input validation and sanitization utilities including:
- SQL injection detection
- XSS (Cross-Site Scripting) prevention
- Path traversal protection
- Command injection detection
- Request body size validation
- Task ID format validation
- Query parameter sanitization
- Pagination validation
- FastAPI dependency injection functions

Author: jetgause
Created: 2025-12-10
"""

import re
from typing import Optional, Any, Dict, List
from fastapi import HTTPException, Request, Query, Path as FastAPIPath
from pydantic import BaseModel, validator, Field
import html


# ============================================================================
# Constants and Patterns
# ============================================================================

# SQL Injection Detection Patterns
SQL_INJECTION_PATTERNS = [
    r"(\bOR\b|\bAND\b).*[=<>].*",  # OR/AND with comparison operators
    r";\s*(DROP|DELETE|UPDATE|INSERT|CREATE|ALTER|EXEC|EXECUTE)\b",
    r"UNION\s+SELECT",
    r"--\s*$",  # SQL comments
    r"/\*.*\*/",  # Multi-line SQL comments
    r"'\s*(OR|AND)\s*'?\d*'?\s*=\s*'?\d*'?",
    r"xp_cmdshell",
    r"WAITFOR\s+DELAY",
    r"BENCHMARK\s*\(",
    r"SLEEP\s*\(",
]

# XSS Detection Patterns
XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"on\w+\s*=",  # Event handlers like onclick, onerror, etc.
    r"<iframe[^>]*>",
    r"<object[^>]*>",
    r"<embed[^>]*>",
    r"<applet[^>]*>",
    r"<meta[^>]*>",
    r"<link[^>]*>",
    r"<style[^>]*>.*?</style>",
    r"eval\s*\(",
    r"expression\s*\(",
]

# Path Traversal Patterns
PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"\.\.",
    r"\.\\",
    r"%2e%2e",
    r"%252e%252e",
    r"..%2f",
    r"..%5c",
]

# Command Injection Patterns
COMMAND_INJECTION_PATTERNS = [
    r"[;&|`$]",
    r"\$\(.*\)",
    r"`.*`",
    r"\|\|",
    r"&&",
    r"\n",
    r"\r",
]

# Task ID Format (UUID v4)
TASK_ID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"

# Maximum sizes
MAX_REQUEST_BODY_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_STRING_LENGTH = 10000
MAX_QUERY_PARAM_LENGTH = 1000
MAX_PAGE_SIZE = 1000
DEFAULT_PAGE_SIZE = 50


# ============================================================================
# Validation Functions
# ============================================================================

def detect_sql_injection(input_string: str) -> bool:
    """
    Detect potential SQL injection attempts in input string.
    
    Args:
        input_string: String to check for SQL injection patterns
        
    Returns:
        True if SQL injection pattern detected, False otherwise
    """
    if not isinstance(input_string, str):
        return False
    
    input_upper = input_string.upper()
    
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, input_upper, re.IGNORECASE):
            return True
    
    return False


def detect_xss(input_string: str) -> bool:
    """
    Detect potential XSS (Cross-Site Scripting) attempts in input string.
    
    Args:
        input_string: String to check for XSS patterns
        
    Returns:
        True if XSS pattern detected, False otherwise
    """
    if not isinstance(input_string, str):
        return False
    
    for pattern in XSS_PATTERNS:
        if re.search(pattern, input_string, re.IGNORECASE):
            return True
    
    return False


def detect_path_traversal(input_string: str) -> bool:
    """
    Detect potential path traversal attempts in input string.
    
    Args:
        input_string: String to check for path traversal patterns
        
    Returns:
        True if path traversal pattern detected, False otherwise
    """
    if not isinstance(input_string, str):
        return False
    
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if re.search(pattern, input_string, re.IGNORECASE):
            return True
    
    return False


def detect_command_injection(input_string: str) -> bool:
    """
    Detect potential command injection attempts in input string.
    
    Args:
        input_string: String to check for command injection patterns
        
    Returns:
        True if command injection pattern detected, False otherwise
    """
    if not isinstance(input_string, str):
        return False
    
    for pattern in COMMAND_INJECTION_PATTERNS:
        if re.search(pattern, input_string):
            return True
    
    return False


def validate_task_id(task_id: str) -> bool:
    """
    Validate task ID format (UUID v4).
    
    Args:
        task_id: Task ID string to validate
        
    Returns:
        True if valid UUID v4 format, False otherwise
    """
    if not isinstance(task_id, str):
        return False
    
    return bool(re.match(TASK_ID_PATTERN, task_id.lower()))


def sanitize_string(input_string: str, max_length: int = MAX_STRING_LENGTH) -> str:
    """
    Sanitize input string by escaping HTML and limiting length.
    
    Args:
        input_string: String to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not isinstance(input_string, str):
        raise ValueError("Input must be a string")
    
    # Truncate to max length
    sanitized = input_string[:max_length]
    
    # Escape HTML entities
    sanitized = html.escape(sanitized)
    
    return sanitized


def sanitize_query_param(param: Optional[str]) -> Optional[str]:
    """
    Sanitize query parameter.
    
    Args:
        param: Query parameter value
        
    Returns:
        Sanitized parameter or None
    """
    if param is None:
        return None
    
    if not isinstance(param, str):
        param = str(param)
    
    # Truncate to max length
    if len(param) > MAX_QUERY_PARAM_LENGTH:
        param = param[:MAX_QUERY_PARAM_LENGTH]
    
    # Check for malicious patterns
    if detect_sql_injection(param):
        raise HTTPException(status_code=400, detail="Invalid input: SQL injection detected")
    
    if detect_xss(param):
        raise HTTPException(status_code=400, detail="Invalid input: XSS pattern detected")
    
    if detect_path_traversal(param):
        raise HTTPException(status_code=400, detail="Invalid input: Path traversal detected")
    
    if detect_command_injection(param):
        raise HTTPException(status_code=400, detail="Invalid input: Command injection detected")
    
    return param


def validate_pagination(page: int, page_size: int) -> tuple[int, int]:
    """
    Validate pagination parameters.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        
    Returns:
        Tuple of validated (page, page_size)
        
    Raises:
        HTTPException: If pagination parameters are invalid
    """
    if page < 1:
        raise HTTPException(status_code=400, detail="Page number must be >= 1")
    
    if page_size < 1:
        raise HTTPException(status_code=400, detail="Page size must be >= 1")
    
    if page_size > MAX_PAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Page size must be <= {MAX_PAGE_SIZE}"
        )
    
    return page, page_size


def validate_string_input(
    input_string: str,
    field_name: str = "input",
    min_length: int = 0,
    max_length: int = MAX_STRING_LENGTH,
    check_sql_injection: bool = True,
    check_xss: bool = True,
    check_path_traversal: bool = False,
    check_command_injection: bool = False,
) -> str:
    """
    Comprehensive string input validation.
    
    Args:
        input_string: String to validate
        field_name: Name of the field (for error messages)
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        check_sql_injection: Whether to check for SQL injection
        check_xss: Whether to check for XSS
        check_path_traversal: Whether to check for path traversal
        check_command_injection: Whether to check for command injection
        
    Returns:
        Validated string
        
    Raises:
        HTTPException: If validation fails
    """
    if not isinstance(input_string, str):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be a string"
        )
    
    if len(input_string) < min_length:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be at least {min_length} characters"
        )
    
    if len(input_string) > max_length:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be at most {max_length} characters"
        )
    
    if check_sql_injection and detect_sql_injection(input_string):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name}: SQL injection pattern detected"
        )
    
    if check_xss and detect_xss(input_string):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name}: XSS pattern detected"
        )
    
    if check_path_traversal and detect_path_traversal(input_string):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name}: Path traversal pattern detected"
        )
    
    if check_command_injection and detect_command_injection(input_string):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name}: Command injection pattern detected"
        )
    
    return input_string


# ============================================================================
# FastAPI Dependency Injection Functions
# ============================================================================

async def validate_request_body_size(request: Request):
    """
    FastAPI dependency to validate request body size.
    
    Raises:
        HTTPException: If request body exceeds maximum size
    """
    content_length = request.headers.get("content-length")
    
    if content_length:
        content_length = int(content_length)
        if content_length > MAX_REQUEST_BODY_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE} bytes"
            )


def get_validated_task_id(
    task_id: str = FastAPIPath(..., description="Task ID (UUID v4 format)")
) -> str:
    """
    FastAPI dependency to validate task ID format.
    
    Args:
        task_id: Task ID from path parameter
        
    Returns:
        Validated task ID
        
    Raises:
        HTTPException: If task ID format is invalid
    """
    if not validate_task_id(task_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid task ID format. Expected UUID v4 format"
        )
    
    return task_id


def get_pagination_params(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="Items per page")
) -> Dict[str, int]:
    """
    FastAPI dependency to get and validate pagination parameters.
    
    Args:
        page: Page number
        page_size: Items per page
        
    Returns:
        Dictionary with validated pagination parameters and offset
    """
    page, page_size = validate_pagination(page, page_size)
    offset = (page - 1) * page_size
    
    return {
        "page": page,
        "page_size": page_size,
        "offset": offset,
        "limit": page_size
    }


def get_search_query(
    q: Optional[str] = Query(None, max_length=MAX_QUERY_PARAM_LENGTH, description="Search query")
) -> Optional[str]:
    """
    FastAPI dependency to get and sanitize search query parameter.
    
    Args:
        q: Search query string
        
    Returns:
        Sanitized search query or None
    """
    if q is None:
        return None
    
    return sanitize_query_param(q)


def get_sort_params(
    sort_by: Optional[str] = Query(None, max_length=100, description="Field to sort by"),
    sort_order: Optional[str] = Query("asc", regex="^(asc|desc)$", description="Sort order")
) -> Dict[str, Optional[str]]:
    """
    FastAPI dependency to get and validate sort parameters.
    
    Args:
        sort_by: Field name to sort by
        sort_order: Sort order (asc or desc)
        
    Returns:
        Dictionary with validated sort parameters
    """
    if sort_by:
        # Validate sort_by to only allow alphanumeric and underscore
        if not re.match(r'^[a-zA-Z0-9_]+$', sort_by):
            raise HTTPException(
                status_code=400,
                detail="Invalid sort_by parameter. Only alphanumeric characters and underscores allowed"
            )
    
    return {
        "sort_by": sort_by,
        "sort_order": sort_order
    }


# ============================================================================
# Pydantic Models for Request Validation
# ============================================================================

class PaginationParams(BaseModel):
    """Pydantic model for pagination parameters."""
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="Items per page")
    
    @validator("page", "page_size")
    def validate_positive(cls, v):
        if v < 1:
            raise ValueError("Must be a positive integer")
        return v
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        """Get limit for database queries."""
        return self.page_size


class SearchParams(BaseModel):
    """Pydantic model for search parameters."""
    q: Optional[str] = Field(None, max_length=MAX_QUERY_PARAM_LENGTH, description="Search query")
    
    @validator("q")
    def validate_search_query(cls, v):
        if v is None:
            return v
        
        if detect_sql_injection(v):
            raise ValueError("Invalid search query: SQL injection pattern detected")
        
        if detect_xss(v):
            raise ValueError("Invalid search query: XSS pattern detected")
        
        return v


class SortParams(BaseModel):
    """Pydantic model for sort parameters."""
    sort_by: Optional[str] = Field(None, max_length=100, description="Field to sort by")
    sort_order: str = Field("asc", regex="^(asc|desc)$", description="Sort order")
    
    @validator("sort_by")
    def validate_sort_by(cls, v):
        if v is None:
            return v
        
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError("Only alphanumeric characters and underscores allowed")
        
        return v


class FilterParams(BaseModel):
    """Pydantic model for filter parameters."""
    status: Optional[str] = Field(None, max_length=50)
    category: Optional[str] = Field(None, max_length=50)
    date_from: Optional[str] = Field(None, max_length=50)
    date_to: Optional[str] = Field(None, max_length=50)
    
    @validator("status", "category", "date_from", "date_to")
    def validate_filter_value(cls, v):
        if v is None:
            return v
        
        if detect_sql_injection(v):
            raise ValueError("Invalid filter value: SQL injection pattern detected")
        
        if detect_xss(v):
            raise ValueError("Invalid filter value: XSS pattern detected")
        
        return v


# ============================================================================
# Utility Classes
# ============================================================================

class InputValidator:
    """
    Comprehensive input validator class for various validation scenarios.
    """
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format."""
        pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number format (basic validation)."""
        # Remove common formatting characters
        cleaned = re.sub(r'[\s\-\(\)\.]', '', phone)
        # Check if it contains only digits and optional + at start
        pattern = r'^\+?\d{10,15}$'
        return bool(re.match(pattern, cleaned))
    
    @staticmethod
    def validate_alphanumeric(text: str, allow_spaces: bool = False) -> bool:
        """Validate alphanumeric text."""
        if allow_spaces:
            pattern = r'^[a-zA-Z0-9\s]+$'
        else:
            pattern = r'^[a-zA-Z0-9]+$'
        return bool(re.match(pattern, text))
    
    @staticmethod
    def validate_filename(filename: str) -> bool:
        """Validate filename (no path traversal, reasonable characters)."""
        # Check for path traversal
        if detect_path_traversal(filename):
            return False
        
        # Only allow alphanumeric, dash, underscore, and dot
        pattern = r'^[a-zA-Z0-9._-]+$'
        return bool(re.match(pattern, filename))


# ============================================================================
# Export all public functions and classes
# ============================================================================

__all__ = [
    # Constants
    "MAX_REQUEST_BODY_SIZE",
    "MAX_STRING_LENGTH",
    "MAX_QUERY_PARAM_LENGTH",
    "MAX_PAGE_SIZE",
    "DEFAULT_PAGE_SIZE",
    
    # Detection functions
    "detect_sql_injection",
    "detect_xss",
    "detect_path_traversal",
    "detect_command_injection",
    
    # Validation functions
    "validate_task_id",
    "sanitize_string",
    "sanitize_query_param",
    "validate_pagination",
    "validate_string_input",
    "sanitize_input",
    
    # FastAPI dependencies
    "validate_request_body_size",
    "get_validated_task_id",
    "get_pagination_params",
    "get_search_query",
    "get_sort_params",
    
    # Pydantic models
    "PaginationParams",
    "SearchParams",
    "SortParams",
    "FilterParams",
    
    # Utility classes
    "InputValidator",
]


# ============================================================================
# SIMPLIFIED API FOR TESTS
# ============================================================================

def sanitize_input(text: str) -> str:
    """Sanitize input text by removing dangerous content."""
    # Remove script tags (including variations with spaces before >)
    # This regex handles: <script>, <script >, <script  >, etc.
    text = re.sub(r'<script[^>]*>.*?</script\s*>', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Remove other dangerous tags
    text = re.sub(r'<(iframe|object|embed|applet)[^>]*>', '', text, flags=re.IGNORECASE)
    # Remove javascript: protocol
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    # Remove event handlers
    text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
    return text
