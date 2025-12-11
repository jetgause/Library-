"""
Pulse Core Security Utilities
==============================

Comprehensive security module providing:
- Input validation and sanitization
- Rate limiting
- CSRF protection
- XSS prevention
- SQL injection prevention
- Secure headers
- Encryption helpers
- Audit logging

Author: jetgause
Created: 2025-12-11
"""

import re
import hashlib
import hmac
import secrets
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Pattern, Union
from functools import wraps
from datetime import datetime, timedelta
from collections import defaultdict
import html
import json
import base64
from urllib.parse import quote, unquote

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

class SecurityConfig:
    """Security configuration constants"""
    
    # Rate Limiting
    RATE_LIMIT_WINDOW = 60  # seconds
    RATE_LIMIT_MAX_REQUESTS = 100
    
    # CSRF
    CSRF_TOKEN_LENGTH = 32
    CSRF_TOKEN_EXPIRY = 3600  # seconds
    
    # Encryption
    PBKDF2_ITERATIONS = 100000
    SALT_LENGTH = 32
    
    # Input Validation
    MAX_INPUT_LENGTH = 10000
    ALLOWED_TAGS = ['p', 'br', 'strong', 'em', 'u', 'a', 'ul', 'ol', 'li']
    
    # Headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'",
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
    }


# ============================================================================
# AUDIT LOGGING
# ============================================================================

class AuditLogger:
    """Comprehensive audit logging for security events"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.logger = logging.getLogger('pulse_security_audit')
        self.logger.setLevel(logging.INFO)
        
        if log_file:
            handler = logging.FileHandler(log_file)
        else:
            handler = logging.StreamHandler()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_event(self, event_type: str, details: Dict[str, Any], 
                  severity: str = 'INFO', user_id: Optional[str] = None):
        """Log a security event"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'details': details,
            'severity': severity
        }
        
        log_message = json.dumps(log_data)
        
        if severity == 'CRITICAL':
            self.logger.critical(log_message)
        elif severity == 'ERROR':
            self.logger.error(log_message)
        elif severity == 'WARNING':
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def log_authentication(self, user_id: str, success: bool, ip_address: str):
        """Log authentication attempt"""
        self.log_event(
            'AUTHENTICATION',
            {
                'success': success,
                'ip_address': ip_address
            },
            severity='WARNING' if not success else 'INFO',
            user_id=user_id
        )
    
    def log_access(self, user_id: str, resource: str, action: str, granted: bool):
        """Log access control event"""
        self.log_event(
            'ACCESS_CONTROL',
            {
                'resource': resource,
                'action': action,
                'granted': granted
            },
            severity='WARNING' if not granted else 'INFO',
            user_id=user_id
        )
    
    def log_data_modification(self, user_id: str, resource: str, operation: str):
        """Log data modification"""
        self.log_event(
            'DATA_MODIFICATION',
            {
                'resource': resource,
                'operation': operation
            },
            user_id=user_id
        )
    
    def log_security_violation(self, violation_type: str, details: Dict[str, Any], 
                              user_id: Optional[str] = None):
        """Log security violation"""
        self.log_event(
            'SECURITY_VIOLATION',
            {
                'violation_type': violation_type,
                **details
            },
            severity='ERROR',
            user_id=user_id
        )


# Global audit logger instance
audit_logger = AuditLogger()


# ============================================================================
# INPUT VALIDATION
# ============================================================================

class InputValidator:
    """Comprehensive input validation"""
    
    # Common regex patterns
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    URL_PATTERN = re.compile(r'^https?://[^\s/$.?#].[^\s]*$', re.IGNORECASE)
    USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{3,32}$')
    PHONE_PATTERN = re.compile(r'^\+?1?\d{9,15}$')
    IPV4_PATTERN = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
    UUID_PATTERN = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        re.compile(r"(\bUNION\b.*\bSELECT\b)", re.IGNORECASE),
        re.compile(r"(\bSELECT\b.*\bFROM\b)", re.IGNORECASE),
        re.compile(r"(\bINSERT\b.*\bINTO\b)", re.IGNORECASE),
        re.compile(r"(\bDELETE\b.*\bFROM\b)", re.IGNORECASE),
        re.compile(r"(\bDROP\b.*\bTABLE\b)", re.IGNORECASE),
        re.compile(r"(--|\#|\/\*|\*\/)", re.IGNORECASE),
        re.compile(r"(\bOR\b.*=.*)", re.IGNORECASE),
        re.compile(r"(\bAND\b.*=.*)", re.IGNORECASE),
    ]
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address"""
        if not email or len(email) > 254:
            return False
        return bool(InputValidator.EMAIL_PATTERN.match(email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL"""
        if not url or len(url) > 2048:
            return False
        return bool(InputValidator.URL_PATTERN.match(url))
    
    @staticmethod
    def validate_username(username: str) -> bool:
        """Validate username"""
        return bool(InputValidator.USERNAME_PATTERN.match(username))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number"""
        return bool(InputValidator.PHONE_PATTERN.match(phone))
    
    @staticmethod
    def validate_ipv4(ip: str) -> bool:
        """Validate IPv4 address"""
        if not InputValidator.IPV4_PATTERN.match(ip):
            return False
        octets = ip.split('.')
        return all(0 <= int(octet) <= 255 for octet in octets)
    
    @staticmethod
    def validate_uuid(uuid_string: str) -> bool:
        """Validate UUID"""
        return bool(InputValidator.UUID_PATTERN.match(uuid_string))
    
    @staticmethod
    def validate_length(value: str, min_length: int = 0, 
                       max_length: int = SecurityConfig.MAX_INPUT_LENGTH) -> bool:
        """Validate string length"""
        return min_length <= len(value) <= max_length
    
    @staticmethod
    def validate_numeric(value: str, min_val: Optional[float] = None, 
                        max_val: Optional[float] = None) -> bool:
        """Validate numeric value"""
        try:
            num = float(value)
            if min_val is not None and num < min_val:
                return False
            if max_val is not None and num > max_val:
                return False
            return True
        except ValueError:
            return False
    
    @staticmethod
    def check_sql_injection(value: str) -> bool:
        """Check for SQL injection patterns"""
        for pattern in InputValidator.SQL_INJECTION_PATTERNS:
            if pattern.search(value):
                audit_logger.log_security_violation(
                    'SQL_INJECTION_ATTEMPT',
                    {'pattern': pattern.pattern, 'value': value[:100]}
                )
                return True
        return False
    
    @staticmethod
    def validate_safe_string(value: str, allow_special: bool = False) -> bool:
        """Validate string contains only safe characters"""
        if allow_special:
            pattern = r'^[a-zA-Z0-9\s\-_.,!?@#$%&*()+=]*$'
        else:
            pattern = r'^[a-zA-Z0-9\s\-_]*$'
        return bool(re.match(pattern, value))


# ============================================================================
# INPUT SANITIZATION
# ============================================================================

class InputSanitizer:
    """Input sanitization and escaping"""
    
    @staticmethod
    def sanitize_html(text: str, allowed_tags: Optional[List[str]] = None) -> str:
        """Sanitize HTML input to prevent XSS"""
        if allowed_tags is None:
            allowed_tags = SecurityConfig.ALLOWED_TAGS
        
        # Remove all HTML tags except allowed ones
        cleaned = html.escape(text)
        
        # Allow specific safe tags
        for tag in allowed_tags:
            cleaned = cleaned.replace(f'&lt;{tag}&gt;', f'<{tag}>')
            cleaned = cleaned.replace(f'&lt;/{tag}&gt;', f'</{tag}>')
        
        return cleaned
    
    @staticmethod
    def strip_html(text: str) -> str:
        """Remove all HTML tags"""
        return html.escape(text)
    
    @staticmethod
    def sanitize_sql(value: str) -> str:
        """Sanitize SQL input (use parameterized queries instead when possible)"""
        # Remove common SQL injection characters
        sanitized = value.replace("'", "''")
        sanitized = sanitized.replace(";", "")
        sanitized = sanitized.replace("--", "")
        sanitized = sanitized.replace("/*", "")
        sanitized = sanitized.replace("*/", "")
        return sanitized
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        # Remove path separators and special characters
        sanitized = re.sub(r'[^\w\s.-]', '', filename)
        sanitized = sanitized.replace('..', '')
        sanitized = sanitized.strip('. ')
        return sanitized or 'unnamed'
    
    @staticmethod
    def sanitize_url(url: str) -> str:
        """Sanitize URL"""
        # Basic URL encoding
        return quote(url, safe=':/?#[]@!$&\'()*+,;=')
    
    @staticmethod
    def sanitize_json(data: Any) -> str:
        """Safely encode JSON"""
        return json.dumps(data, ensure_ascii=True, indent=None, separators=(',', ':'))
    
    @staticmethod
    def remove_null_bytes(text: str) -> str:
        """Remove null bytes from string"""
        return text.replace('\x00', '')
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in string"""
        return ' '.join(text.split())


# ============================================================================
# XSS PREVENTION
# ============================================================================

class XSSProtection:
    """Cross-Site Scripting (XSS) prevention"""
    
    XSS_PATTERNS = [
        re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'on\w+\s*=', re.IGNORECASE),
        re.compile(r'<iframe[^>]*>', re.IGNORECASE),
        re.compile(r'<object[^>]*>', re.IGNORECASE),
        re.compile(r'<embed[^>]*>', re.IGNORECASE),
        re.compile(r'<applet[^>]*>', re.IGNORECASE),
    ]
    
    @staticmethod
    def detect_xss(text: str) -> bool:
        """Detect potential XSS attempts"""
        for pattern in XSSProtection.XSS_PATTERNS:
            if pattern.search(text):
                audit_logger.log_security_violation(
                    'XSS_ATTEMPT',
                    {'pattern': pattern.pattern, 'value': text[:100]}
                )
                return True
        return False
    
    @staticmethod
    def clean_xss(text: str) -> str:
        """Remove XSS patterns from text"""
        cleaned = text
        for pattern in XSSProtection.XSS_PATTERNS:
            cleaned = pattern.sub('', cleaned)
        return InputSanitizer.sanitize_html(cleaned)
    
    @staticmethod
    def escape_js_string(text: str) -> str:
        """Escape string for safe use in JavaScript"""
        escape_chars = {
            '\\': '\\\\',
            '"': '\\"',
            "'": "\\'",
            '\n': '\\n',
            '\r': '\\r',
            '\t': '\\t',
            '<': '\\x3C',
            '>': '\\x3E',
            '&': '\\x26',
        }
        return ''.join(escape_chars.get(c, c) for c in text)


# ============================================================================
# CSRF PROTECTION
# ============================================================================

class CSRFProtection:
    """Cross-Site Request Forgery (CSRF) protection"""
    
    def __init__(self):
        self.tokens: Dict[str, Dict[str, Any]] = {}
    
    def generate_token(self, session_id: str) -> str:
        """Generate CSRF token for session"""
        token = secrets.token_urlsafe(SecurityConfig.CSRF_TOKEN_LENGTH)
        self.tokens[session_id] = {
            'token': token,
            'created_at': time.time()
        }
        return token
    
    def validate_token(self, session_id: str, token: str) -> bool:
        """Validate CSRF token"""
        if session_id not in self.tokens:
            return False
        
        stored_data = self.tokens[session_id]
        
        # Check expiry
        if time.time() - stored_data['created_at'] > SecurityConfig.CSRF_TOKEN_EXPIRY:
            del self.tokens[session_id]
            return False
        
        # Constant-time comparison
        is_valid = hmac.compare_digest(stored_data['token'], token)
        
        if not is_valid:
            audit_logger.log_security_violation(
                'CSRF_TOKEN_INVALID',
                {'session_id': session_id}
            )
        
        return is_valid
    
    def remove_token(self, session_id: str):
        """Remove CSRF token"""
        if session_id in self.tokens:
            del self.tokens[session_id]
    
    def cleanup_expired_tokens(self):
        """Remove expired tokens"""
        current_time = time.time()
        expired = [
            sid for sid, data in self.tokens.items()
            if current_time - data['created_at'] > SecurityConfig.CSRF_TOKEN_EXPIRY
        ]
        for sid in expired:
            del self.tokens[sid]


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Rate limiting for API endpoints"""
    
    def __init__(self, max_requests: int = SecurityConfig.RATE_LIMIT_MAX_REQUESTS,
                 window: int = SecurityConfig.RATE_LIMIT_WINDOW):
        self.max_requests = max_requests
        self.window = window
        self.requests: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        current_time = time.time()
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if current_time - req_time < self.window
        ]
        
        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            audit_logger.log_security_violation(
                'RATE_LIMIT_EXCEEDED',
                {'identifier': identifier, 'limit': self.max_requests}
            )
            return False
        
        self.requests[identifier].append(current_time)
        return True
    
    def reset(self, identifier: str):
        """Reset rate limit for identifier"""
        if identifier in self.requests:
            del self.requests[identifier]
    
    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests for identifier"""
        current_time = time.time()
        recent_requests = [
            req_time for req_time in self.requests.get(identifier, [])
            if current_time - req_time < self.window
        ]
        return max(0, self.max_requests - len(recent_requests))


def rate_limit(max_requests: int = SecurityConfig.RATE_LIMIT_MAX_REQUESTS,
               window: int = SecurityConfig.RATE_LIMIT_WINDOW,
               identifier_func: Optional[Callable] = None):
    """Decorator for rate limiting"""
    limiter = RateLimiter(max_requests, window)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get identifier (default to function name)
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            else:
                identifier = func.__name__
            
            if not limiter.is_allowed(identifier):
                raise PermissionError(f"Rate limit exceeded for {identifier}")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# ============================================================================
# ENCRYPTION HELPERS
# ============================================================================

class EncryptionHelper:
    """Encryption and hashing utilities"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography package required for encryption")
        
        self.master_key = master_key or Fernet.generate_key()
        self.cipher = Fernet(self.master_key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        encrypted = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        decoded = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.cipher.decrypt(decoded)
        return decrypted.decode()
    
    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> tuple:
        """Hash password with PBKDF2"""
        if salt is None:
            salt = secrets.token_bytes(SecurityConfig.SALT_LENGTH)
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=SecurityConfig.PBKDF2_ITERATIONS,
            backend=default_backend()
        )
        key = kdf.derive(password.encode())
        return base64.b64encode(key).decode(), base64.b64encode(salt).decode()
    
    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """Verify password against hash"""
        try:
            salt_bytes = base64.b64decode(salt.encode())
            new_hash, _ = EncryptionHelper.hash_password(password, salt_bytes)
            return hmac.compare_digest(new_hash, hashed)
        except Exception:
            return False
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_data(data: str, algorithm: str = 'sha256') -> str:
        """Hash data with specified algorithm"""
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(data.encode())
        return hash_obj.hexdigest()
    
    @staticmethod
    def create_hmac(data: str, key: str, algorithm: str = 'sha256') -> str:
        """Create HMAC for data"""
        return hmac.new(
            key.encode(),
            data.encode(),
            algorithm
        ).hexdigest()
    
    @staticmethod
    def verify_hmac(data: str, signature: str, key: str) -> bool:
        """Verify HMAC signature"""
        expected = EncryptionHelper.create_hmac(data, key)
        return hmac.compare_digest(expected, signature)


# ============================================================================
# SECURE HEADERS
# ============================================================================

class SecureHeaders:
    """Security headers management"""
    
    @staticmethod
    def get_default_headers() -> Dict[str, str]:
        """Get default security headers"""
        return SecurityConfig.SECURITY_HEADERS.copy()
    
    @staticmethod
    def add_headers_to_response(headers: Dict[str, str],
                               additional: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Add security headers to response headers"""
        secure_headers = SecureHeaders.get_default_headers()
        if additional:
            secure_headers.update(additional)
        headers.update(secure_headers)
        return headers
    
    @staticmethod
    def create_csp(directives: Dict[str, str]) -> str:
        """Create Content Security Policy header value"""
        return '; '.join(f"{key} {value}" for key, value in directives.items())
    
    @staticmethod
    def create_cors_headers(allowed_origins: List[str],
                          allowed_methods: List[str] = None,
                          allowed_headers: List[str] = None,
                          max_age: int = 86400) -> Dict[str, str]:
        """Create CORS headers"""
        if allowed_methods is None:
            allowed_methods = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
        if allowed_headers is None:
            allowed_headers = ['Content-Type', 'Authorization']
        
        return {
            'Access-Control-Allow-Origin': ', '.join(allowed_origins),
            'Access-Control-Allow-Methods': ', '.join(allowed_methods),
            'Access-Control-Allow-Headers': ', '.join(allowed_headers),
            'Access-Control-Max-Age': str(max_age)
        }


# ============================================================================
# SECURE SESSION MANAGEMENT
# ============================================================================

class SessionManager:
    """Secure session management"""
    
    def __init__(self, session_timeout: int = 1800):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = session_timeout
    
    def create_session(self, user_id: str, data: Optional[Dict[str, Any]] = None) -> str:
        """Create new session"""
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            'user_id': user_id,
            'created_at': time.time(),
            'last_activity': time.time(),
            'data': data or {}
        }
        audit_logger.log_event('SESSION_CREATED', {'session_id': session_id}, user_id=user_id)
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """Validate session"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        current_time = time.time()
        
        # Check timeout
        if current_time - session['last_activity'] > self.session_timeout:
            self.destroy_session(session_id)
            return False
        
        # Update activity
        session['last_activity'] = current_time
        return True
    
    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        if self.validate_session(session_id):
            return self.sessions[session_id]['data']
        return None
    
    def update_session_data(self, session_id: str, data: Dict[str, Any]):
        """Update session data"""
        if self.validate_session(session_id):
            self.sessions[session_id]['data'].update(data)
    
    def destroy_session(self, session_id: str):
        """Destroy session"""
        if session_id in self.sessions:
            user_id = self.sessions[session_id].get('user_id')
            del self.sessions[session_id]
            audit_logger.log_event('SESSION_DESTROYED', {'session_id': session_id}, user_id=user_id)
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = time.time()
        expired = [
            sid for sid, session in self.sessions.items()
            if current_time - session['last_activity'] > self.session_timeout
        ]
        for sid in expired:
            self.destroy_session(sid)


# ============================================================================
# SECURITY MIDDLEWARE
# ============================================================================

def security_middleware(
    validate_input: bool = True,
    check_xss: bool = True,
    check_sql_injection: bool = True,
    rate_limit_enabled: bool = True,
    csrf_protection: bool = True
):
    """Comprehensive security middleware decorator"""
    
    rate_limiter = RateLimiter() if rate_limit_enabled else None
    csrf = CSRFProtection() if csrf_protection else None
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Rate limiting
            if rate_limiter:
                identifier = kwargs.get('user_id', func.__name__)
                if not rate_limiter.is_allowed(str(identifier)):
                    raise PermissionError("Rate limit exceeded")
            
            # Input validation
            if validate_input or check_xss or check_sql_injection:
                for arg in args:
                    if isinstance(arg, str):
                        if check_xss and XSSProtection.detect_xss(arg):
                            raise ValueError("XSS pattern detected")
                        if check_sql_injection and InputValidator.check_sql_injection(arg):
                            raise ValueError("SQL injection pattern detected")
                
                for value in kwargs.values():
                    if isinstance(value, str):
                        if check_xss and XSSProtection.detect_xss(value):
                            raise ValueError("XSS pattern detected")
                        if check_sql_injection and InputValidator.check_sql_injection(value):
                            raise ValueError("SQL injection pattern detected")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_api_key() -> str:
    """Generate secure API key"""
    return EncryptionHelper.generate_secure_token(48)


def mask_sensitive_data(data: str, visible_chars: int = 4, mask_char: str = '*') -> str:
    """Mask sensitive data (e.g., credit cards, passwords)"""
    if len(data) <= visible_chars:
        return mask_char * len(data)
    return data[:visible_chars] + mask_char * (len(data) - visible_chars)


def is_safe_redirect(url: str, allowed_domains: List[str]) -> bool:
    """Check if redirect URL is safe"""
    from urllib.parse import urlparse
    
    try:
        parsed = urlparse(url)
        if not parsed.netloc:  # Relative URL
            return True
        return any(parsed.netloc.endswith(domain) for domain in allowed_domains)
    except Exception:
        return False


def sanitize_dict(data: Dict[str, Any], sensitive_keys: List[str] = None) -> Dict[str, Any]:
    """Sanitize dictionary by removing/masking sensitive keys"""
    if sensitive_keys is None:
        sensitive_keys = ['password', 'token', 'secret', 'api_key', 'private_key']
    
    sanitized = {}
    for key, value in data.items():
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            sanitized[key] = '***REDACTED***'
        elif isinstance(value, dict):
            sanitized[key] = sanitize_dict(value, sensitive_keys)
        else:
            sanitized[key] = value
    
    return sanitized


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'SecurityConfig',
    'AuditLogger',
    'InputValidator',
    'InputSanitizer',
    'XSSProtection',
    'CSRFProtection',
    'RateLimiter',
    'EncryptionHelper',
    'SecureHeaders',
    'SessionManager',
    'audit_logger',
    'rate_limit',
    'security_middleware',
    'generate_api_key',
    'mask_sensitive_data',
    'is_safe_redirect',
    'sanitize_dict',
]
