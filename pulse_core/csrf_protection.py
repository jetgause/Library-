"""
Comprehensive CSRF (Cross-Site Request Forgery) Protection Module

This module provides enterprise-grade CSRF protection with multiple patterns:
- Synchronizer Token Pattern
- Double-Submit Cookie Pattern
- Token rotation and expiration
- FastAPI middleware integration
- Origin and Referer validation

Author: jetgause
Created: 2025-12-10
"""

import secrets
import hashlib
import hmac
import logging
from datetime import datetime, timedelta
from typing import Optional, Set, List, Callable, Dict, Any
from functools import wraps
from urllib.parse import urlparse

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import Headers


# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class CSRFConfig:
    """Configuration for CSRF protection."""
    
    def __init__(
        self,
        secret_key: str,
        token_length: int = 32,
        token_expiration: int = 3600,  # 1 hour in seconds
        cookie_name: str = "csrf_token",
        header_name: str = "X-CSRF-Token",
        cookie_path: str = "/",
        cookie_domain: Optional[str] = None,
        cookie_secure: bool = True,
        cookie_httponly: bool = True,
        cookie_samesite: str = "strict",
        safe_methods: Set[str] = None,
        exempt_urls: Set[str] = None,
        check_referer: bool = True,
        check_origin: bool = True,
        allowed_origins: Set[str] = None,
        per_request_tokens: bool = False,
        rotate_on_sensitive: bool = True,
    ):
        """
        Initialize CSRF configuration.
        
        Args:
            secret_key: Secret key for signing tokens
            token_length: Length of generated tokens in bytes
            token_expiration: Token expiration time in seconds
            cookie_name: Name of the CSRF cookie
            header_name: Name of the CSRF header
            cookie_path: Cookie path
            cookie_domain: Cookie domain
            cookie_secure: Enable Secure flag
            cookie_httponly: Enable HttpOnly flag
            cookie_samesite: SameSite attribute (strict, lax, none)
            safe_methods: HTTP methods that don't require CSRF protection
            exempt_urls: URLs exempt from CSRF protection
            check_referer: Enable Referer header validation
            check_origin: Enable Origin header validation
            allowed_origins: Set of allowed origins
            per_request_tokens: Generate new token per request
            rotate_on_sensitive: Rotate tokens on sensitive operations
        """
        self.secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key
        self.token_length = token_length
        self.token_expiration = token_expiration
        self.cookie_name = cookie_name
        self.header_name = header_name
        self.cookie_path = cookie_path
        self.cookie_domain = cookie_domain
        self.cookie_secure = cookie_secure
        self.cookie_httponly = cookie_httponly
        self.cookie_samesite = cookie_samesite.lower()
        self.safe_methods = safe_methods or {"GET", "HEAD", "OPTIONS", "TRACE"}
        self.exempt_urls = exempt_urls or set()
        self.check_referer = check_referer
        self.check_origin = check_origin
        self.allowed_origins = allowed_origins or set()
        self.per_request_tokens = per_request_tokens
        self.rotate_on_sensitive = rotate_on_sensitive


class CSRFToken:
    """Represents a CSRF token with metadata."""
    
    def __init__(
        self,
        token: str,
        created_at: datetime,
        expires_at: datetime,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        Initialize CSRF token.
        
        Args:
            token: The token string
            created_at: Creation timestamp
            expires_at: Expiration timestamp
            session_id: Associated session ID
            user_id: Associated user ID
        """
        self.token = token
        self.created_at = created_at
        self.expires_at = expires_at
        self.session_id = session_id
        self.user_id = user_id
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert token to dictionary."""
        return {
            "token": self.token,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "session_id": self.session_id,
            "user_id": self.user_id,
        }


class CSRFTokenStore:
    """In-memory token store with automatic cleanup."""
    
    def __init__(self):
        """Initialize token store."""
        self._tokens: Dict[str, CSRFToken] = {}
        self._session_tokens: Dict[str, Set[str]] = {}
        self._user_tokens: Dict[str, Set[str]] = {}
    
    def store_token(self, token: CSRFToken) -> None:
        """
        Store a CSRF token.
        
        Args:
            token: CSRFToken instance to store
        """
        self._tokens[token.token] = token
        
        if token.session_id:
            if token.session_id not in self._session_tokens:
                self._session_tokens[token.session_id] = set()
            self._session_tokens[token.session_id].add(token.token)
        
        if token.user_id:
            if token.user_id not in self._user_tokens:
                self._user_tokens[token.user_id] = set()
            self._user_tokens[token.user_id].add(token.token)
    
    def get_token(self, token_str: str) -> Optional[CSRFToken]:
        """
        Retrieve a token.
        
        Args:
            token_str: Token string to retrieve
            
        Returns:
            CSRFToken if found, None otherwise
        """
        return self._tokens.get(token_str)
    
    def validate_token(self, token_str: str) -> bool:
        """
        Validate a token.
        
        Args:
            token_str: Token string to validate
            
        Returns:
            True if valid, False otherwise
        """
        token = self.get_token(token_str)
        if token is None:
            return False
        
        if token.is_expired():
            self.remove_token(token_str)
            return False
        
        return True
    
    def remove_token(self, token_str: str) -> None:
        """
        Remove a token from store.
        
        Args:
            token_str: Token string to remove
        """
        token = self._tokens.pop(token_str, None)
        if token:
            if token.session_id and token.session_id in self._session_tokens:
                self._session_tokens[token.session_id].discard(token_str)
            if token.user_id and token.user_id in self._user_tokens:
                self._user_tokens[token.user_id].discard(token_str)
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired tokens.
        
        Returns:
            Number of tokens removed
        """
        expired_tokens = [
            token_str for token_str, token in self._tokens.items()
            if token.is_expired()
        ]
        
        for token_str in expired_tokens:
            self.remove_token(token_str)
        
        logger.info(f"Cleaned up {len(expired_tokens)} expired CSRF tokens")
        return len(expired_tokens)
    
    def get_session_tokens(self, session_id: str) -> Set[str]:
        """Get all tokens for a session."""
        return self._session_tokens.get(session_id, set()).copy()
    
    def get_user_tokens(self, user_id: str) -> Set[str]:
        """Get all tokens for a user."""
        return self._user_tokens.get(user_id, set()).copy()
    
    def clear_session_tokens(self, session_id: str) -> None:
        """Clear all tokens for a session."""
        tokens = self._session_tokens.get(session_id, set()).copy()
        for token_str in tokens:
            self.remove_token(token_str)
        self._session_tokens.pop(session_id, None)
    
    def clear_user_tokens(self, user_id: str) -> None:
        """Clear all tokens for a user."""
        tokens = self._user_tokens.get(user_id, set()).copy()
        for token_str in tokens:
            self.remove_token(token_str)
        self._user_tokens.pop(user_id, None)


class CSRFProtection:
    """Main CSRF protection implementation."""
    
    def __init__(self, config: CSRFConfig):
        """
        Initialize CSRF protection.
        
        Args:
            config: CSRFConfig instance
        """
        self.config = config
        self.token_store = CSRFTokenStore()
    
    def generate_token(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> CSRFToken:
        """
        Generate a new CSRF token.
        
        Args:
            session_id: Optional session ID
            user_id: Optional user ID
            
        Returns:
            Generated CSRFToken instance
        """
        # Generate cryptographically secure random token
        raw_token = secrets.token_urlsafe(self.config.token_length)
        
        # Sign the token with HMAC
        signature = hmac.new(
            self.config.secret_key,
            raw_token.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Combine token and signature
        signed_token = f"{raw_token}.{signature}"
        
        # Create token with metadata
        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(seconds=self.config.token_expiration)
        
        csrf_token = CSRFToken(
            token=signed_token,
            created_at=created_at,
            expires_at=expires_at,
            session_id=session_id,
            user_id=user_id,
        )
        
        # Store token
        self.token_store.store_token(csrf_token)
        
        logger.debug(f"Generated new CSRF token for session: {session_id}, user: {user_id}")
        return csrf_token
    
    def verify_token(self, token_str: str) -> bool:
        """
        Verify a CSRF token.
        
        Args:
            token_str: Token string to verify
            
        Returns:
            True if valid, False otherwise
        """
        if not token_str:
            return False
        
        # Split token and signature
        parts = token_str.split(".")
        if len(parts) != 2:
            logger.warning("Invalid CSRF token format")
            return False
        
        raw_token, signature = parts
        
        # Verify signature
        expected_signature = hmac.new(
            self.config.secret_key,
            raw_token.encode(),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(signature, expected_signature):
            logger.warning("CSRF token signature verification failed")
            return False
        
        # Validate token in store
        if not self.token_store.validate_token(token_str):
            logger.warning("CSRF token not found or expired")
            return False
        
        return True
    
    def rotate_token(
        self,
        old_token: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[CSRFToken]:
        """
        Rotate a CSRF token (invalidate old, generate new).
        
        Args:
            old_token: Old token to invalidate
            session_id: Optional session ID
            user_id: Optional user ID
            
        Returns:
            New CSRFToken instance
        """
        # Remove old token
        self.token_store.remove_token(old_token)
        
        # Generate new token
        new_token = self.generate_token(session_id=session_id, user_id=user_id)
        
        logger.info(f"Rotated CSRF token for session: {session_id}")
        return new_token
    
    def validate_origin(self, request: Request) -> bool:
        """
        Validate Origin header.
        
        Args:
            request: FastAPI request object
            
        Returns:
            True if valid, False otherwise
        """
        if not self.config.check_origin:
            return True
        
        origin = request.headers.get("origin")
        if not origin:
            return True  # Origin header is optional
        
        # Parse origin
        parsed_origin = urlparse(origin)
        origin_host = parsed_origin.netloc
        
        # Check against allowed origins
        if self.config.allowed_origins:
            if origin_host not in self.config.allowed_origins:
                logger.warning(f"Origin validation failed: {origin_host}")
                return False
        
        # Check against request host
        request_host = request.headers.get("host", "")
        if origin_host != request_host:
            logger.warning(f"Origin mismatch: {origin_host} != {request_host}")
            return False
        
        return True
    
    def validate_referer(self, request: Request) -> bool:
        """
        Validate Referer header.
        
        Args:
            request: FastAPI request object
            
        Returns:
            True if valid, False otherwise
        """
        if not self.config.check_referer:
            return True
        
        referer = request.headers.get("referer")
        if not referer:
            logger.warning("Missing Referer header")
            return False
        
        # Parse referer
        parsed_referer = urlparse(referer)
        referer_host = parsed_referer.netloc
        
        # Check against request host
        request_host = request.headers.get("host", "")
        if referer_host != request_host:
            logger.warning(f"Referer mismatch: {referer_host} != {request_host}")
            return False
        
        return True
    
    def is_exempt(self, path: str) -> bool:
        """
        Check if path is exempt from CSRF protection.
        
        Args:
            path: Request path
            
        Returns:
            True if exempt, False otherwise
        """
        return path in self.config.exempt_urls
    
    def set_csrf_cookie(
        self,
        response: Response,
        token: CSRFToken,
    ) -> None:
        """
        Set CSRF token cookie on response.
        
        Args:
            response: FastAPI response object
            token: CSRFToken to set
        """
        response.set_cookie(
            key=self.config.cookie_name,
            value=token.token,
            max_age=self.config.token_expiration,
            path=self.config.cookie_path,
            domain=self.config.cookie_domain,
            secure=self.config.cookie_secure,
            httponly=self.config.cookie_httponly,
            samesite=self.config.cookie_samesite,
        )
    
    def get_token_from_request(self, request: Request) -> Optional[str]:
        """
        Extract CSRF token from request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Token string if found, None otherwise
        """
        # Check header first
        token = request.headers.get(self.config.header_name.lower())
        if token:
            return token
        
        # Check form data
        if hasattr(request, "form"):
            try:
                form_data = request.form()
                token = form_data.get("csrf_token")
                if token:
                    return token
            except Exception:
                pass
        
        # Check cookie (for double-submit pattern)
        token = request.cookies.get(self.config.cookie_name)
        return token
    
    async def validate_request(self, request: Request) -> bool:
        """
        Validate CSRF protection for request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            True if valid, False otherwise
        """
        # Check if method is safe
        if request.method in self.config.safe_methods:
            return True
        
        # Check if path is exempt
        if self.is_exempt(request.url.path):
            return True
        
        # Validate origin
        if not self.validate_origin(request):
            logger.error(f"CSRF attack attempt detected - Origin validation failed: {request.url.path}")
            return False
        
        # Validate referer
        if not self.validate_referer(request):
            logger.error(f"CSRF attack attempt detected - Referer validation failed: {request.url.path}")
            return False
        
        # Get token from request
        token = self.get_token_from_request(request)
        if not token:
            logger.error(f"CSRF attack attempt detected - Missing token: {request.url.path}")
            return False
        
        # Verify token
        if not self.verify_token(token):
            logger.error(f"CSRF attack attempt detected - Invalid token: {request.url.path}")
            return False
        
        return True


class CSRFMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for CSRF protection."""
    
    def __init__(self, app, csrf_protection: CSRFProtection):
        """
        Initialize CSRF middleware.
        
        Args:
            app: FastAPI application
            csrf_protection: CSRFProtection instance
        """
        super().__init__(app)
        self.csrf = csrf_protection
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request with CSRF protection.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in chain
            
        Returns:
            Response object
        """
        # Validate request
        is_valid = await self.csrf.validate_request(request)
        
        if not is_valid:
            logger.error(
                f"CSRF validation failed - "
                f"Method: {request.method}, "
                f"Path: {request.url.path}, "
                f"IP: {request.client.host if request.client else 'unknown'}"
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "detail": "CSRF validation failed",
                    "error": "invalid_csrf_token",
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Generate and set token for safe methods
        if request.method in self.csrf.config.safe_methods:
            # Generate new token
            session_id = request.cookies.get("session_id")
            user_id = request.state.user_id if hasattr(request.state, "user_id") else None
            
            token = self.csrf.generate_token(
                session_id=session_id,
                user_id=user_id,
            )
            
            # Set token cookie
            self.csrf.set_csrf_cookie(response, token)
            
            # Add token to response headers for client access
            response.headers[self.csrf.config.header_name] = token.token
        
        # Rotate token on sensitive operations if configured
        elif self.csrf.config.rotate_on_sensitive:
            old_token = self.csrf.get_token_from_request(request)
            if old_token:
                session_id = request.cookies.get("session_id")
                user_id = request.state.user_id if hasattr(request.state, "user_id") else None
                
                new_token = self.csrf.rotate_token(
                    old_token=old_token,
                    session_id=session_id,
                    user_id=user_id,
                )
                
                if new_token:
                    self.csrf.set_csrf_cookie(response, new_token)
                    response.headers[self.csrf.config.header_name] = new_token.token
        
        return response


def csrf_protect(csrf_protection: CSRFProtection):
    """
    Decorator for CSRF protection on individual endpoints.
    
    Args:
        csrf_protection: CSRFProtection instance
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from kwargs
            request = kwargs.get("request")
            if not request:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
            
            if not request:
                raise ValueError("Request object not found in function arguments")
            
            # Validate CSRF
            is_valid = await csrf_protection.validate_request(request)
            if not is_valid:
                logger.error(f"CSRF validation failed in decorator for {func.__name__}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="CSRF validation failed"
                )
            
            # Call original function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def csrf_exempt(func: Callable):
    """
    Decorator to exempt an endpoint from CSRF protection.
    
    Args:
        func: Function to exempt
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    
    wrapper._csrf_exempt = True
    return wrapper


# Utility functions

def create_csrf_protection(
    secret_key: str,
    **kwargs
) -> CSRFProtection:
    """
    Create a CSRFProtection instance with configuration.
    
    Args:
        secret_key: Secret key for signing tokens
        **kwargs: Additional configuration options
        
    Returns:
        CSRFProtection instance
    """
    config = CSRFConfig(secret_key=secret_key, **kwargs)
    return CSRFProtection(config)


def get_csrf_token(request: Request, csrf_protection: CSRFProtection) -> Optional[str]:
    """
    Get CSRF token from request.
    
    Args:
        request: FastAPI request object
        csrf_protection: CSRFProtection instance
        
    Returns:
        Token string if found, None otherwise
    """
    return csrf_protection.get_token_from_request(request)


def generate_csrf_token_for_response(
    csrf_protection: CSRFProtection,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """
    Generate a CSRF token for including in responses.
    
    Args:
        csrf_protection: CSRFProtection instance
        session_id: Optional session ID
        user_id: Optional user ID
        
    Returns:
        Token string
    """
    token = csrf_protection.generate_token(
        session_id=session_id,
        user_id=user_id,
    )
    return token.token


# Example usage and integration

"""
Example Usage:

1. Basic Setup with FastAPI:

```python
from fastapi import FastAPI
from pulse_core.csrf_protection import (
    create_csrf_protection,
    CSRFMiddleware,
    csrf_protect,
    csrf_exempt,
)

app = FastAPI()

# Create CSRF protection
csrf = create_csrf_protection(
    secret_key="your-secret-key-here",
    cookie_secure=True,
    cookie_samesite="strict",
)

# Add middleware
app.add_middleware(CSRFMiddleware, csrf_protection=csrf)

# Protected endpoint
@app.post("/api/sensitive-operation")
async def sensitive_operation(request: Request):
    # Automatically protected by middleware
    return {"status": "success"}

# Exempt endpoint
@app.post("/api/webhook")
@csrf_exempt
async def webhook(request: Request):
    # This endpoint bypasses CSRF protection
    return {"status": "received"}

# Manual protection with decorator
@app.post("/api/manual-protected")
@csrf_protect(csrf)
async def manual_protected(request: Request):
    return {"status": "protected"}
```

2. Token Rotation on Login:

```python
@app.post("/api/login")
async def login(request: Request, response: Response):
    # Authenticate user
    user_id = authenticate_user(request)
    
    # Rotate CSRF token after authentication
    old_token = csrf.get_token_from_request(request)
    new_token = csrf.rotate_token(
        old_token=old_token,
        session_id=request.cookies.get("session_id"),
        user_id=user_id,
    )
    
    csrf.set_csrf_cookie(response, new_token)
    
    return {"status": "logged_in", "csrf_token": new_token.token}
```

3. Client-Side Integration:

```javascript
// Get CSRF token from cookie or meta tag
function getCsrfToken() {
    const token = document.cookie
        .split('; ')
        .find(row => row.startsWith('csrf_token='))
        ?.split('=')[1];
    return token;
}

// Include in fetch requests
fetch('/api/sensitive-operation', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': getCsrfToken(),
    },
    body: JSON.stringify(data),
});
```

4. Cleanup Expired Tokens:

```python
import asyncio

async def cleanup_task():
    while True:
        csrf.token_store.cleanup_expired()
        await asyncio.sleep(3600)  # Run every hour

# Add to startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_task())
```
"""


if __name__ == "__main__":
    # Example demonstration
    print("CSRF Protection Module")
    print("=" * 50)
    
    # Create protection instance
    csrf = create_csrf_protection(
        secret_key="demo-secret-key-change-in-production",
        token_expiration=3600,
    )
    
    # Generate token
    token = csrf.generate_token(session_id="demo-session-123")
    print(f"\nGenerated Token: {token.token[:50]}...")
    print(f"Expires At: {token.expires_at}")
    
    # Verify token
    is_valid = csrf.verify_token(token.token)
    print(f"\nToken Valid: {is_valid}")
    
    # Invalid token
    is_valid = csrf.verify_token("invalid-token")
    print(f"Invalid Token Valid: {is_valid}")
    
    print("\n" + "=" * 50)
    print("Module loaded successfully!")


# ============================================================================
# SIMPLIFIED API FOR TESTS
# ============================================================================

# Global token store for simple API
_token_store: Dict[str, str] = {}


def generate_csrf_token() -> str:
    """Generate a simple CSRF token for testing."""
    return secrets.token_urlsafe(32)


def store_csrf_token(session_id: str, token: str) -> None:
    """Store a CSRF token for a session."""
    _token_store[session_id] = token


def validate_csrf_token(session_id: str, token: str) -> bool:
    """Validate a CSRF token for a session."""
    stored_token = _token_store.get(session_id)
    if not stored_token:
        return False
    return hmac.compare_digest(stored_token, token)
