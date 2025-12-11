"""
Security Hardening Module for Pulse Core
Comprehensive security features including API key management, IP filtering,
brute force protection, JWT blacklisting, password policies, file upload validation,
and security event monitoring.

Author: jetgause
Created: 2025-12-11
"""

import hashlib
import hmac
import secrets
import re
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import ipaddress
import mimetypes
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Security event types for monitoring"""
    API_KEY_CREATED = "api_key_created"
    API_KEY_ROTATED = "api_key_rotated"
    API_KEY_REVOKED = "api_key_revoked"
    API_KEY_INVALID = "api_key_invalid"
    IP_BLOCKED = "ip_blocked"
    IP_WHITELISTED = "ip_whitelisted"
    LOGIN_FAILED = "login_failed"
    LOGIN_SUCCESS = "login_success"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    JWT_BLACKLISTED = "jwt_blacklisted"
    PASSWORD_POLICY_VIOLATION = "password_policy_violation"
    FILE_UPLOAD_REJECTED = "file_upload_rejected"
    FILE_UPLOAD_ACCEPTED = "file_upload_accepted"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_type: SecurityEventType
    timestamp: datetime
    ip_address: Optional[str] = None
    user_id: Optional[str] = None
    details: Dict = field(default_factory=dict)
    severity: str = "INFO"  # INFO, WARNING, CRITICAL


@dataclass
class APIKey:
    """API Key data structure"""
    key_id: str
    key_hash: str
    user_id: str
    created_at: datetime
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    permissions: List[str] = field(default_factory=list)
    rotation_count: int = 0


@dataclass
class PasswordPolicy:
    """Password policy configuration"""
    min_length: int = 12
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digits: bool = True
    require_special_chars: bool = True
    special_chars: str = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    max_age_days: int = 90
    prevent_reuse_count: int = 5
    lockout_threshold: int = 5
    lockout_duration_minutes: int = 30


class APIKeyManager:
    """Manages API keys with rotation and expiration"""
    
    def __init__(self, key_length: int = 32):
        self.key_length = key_length
        self.keys: Dict[str, APIKey] = {}
        self.key_lookup: Dict[str, str] = {}  # raw_key -> key_id
        
    def generate_key(self, user_id: str, permissions: List[str] = None,
                    expires_in_days: Optional[int] = None) -> Tuple[str, str]:
        """
        Generate a new API key for a user
        Returns: (key_id, raw_key)
        """
        key_id = f"key_{secrets.token_urlsafe(16)}"
        raw_key = secrets.token_urlsafe(self.key_length)
        key_hash = self._hash_key(raw_key)
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            permissions=permissions or []
        )
        
        self.keys[key_id] = api_key
        self.key_lookup[raw_key] = key_id
        
        logger.info(f"Generated new API key {key_id} for user {user_id}")
        return key_id, raw_key
    
    def validate_key(self, raw_key: str) -> Optional[APIKey]:
        """Validate an API key and return the key object if valid"""
        key_id = self.key_lookup.get(raw_key)
        if not key_id:
            logger.warning("Invalid API key attempted")
            return None
        
        api_key = self.keys.get(key_id)
        if not api_key or not api_key.is_active:
            return None
        
        # Check expiration
        if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
            logger.warning(f"Expired API key {key_id} used")
            return None
        
        # Update last used timestamp
        api_key.last_used = datetime.utcnow()
        return api_key
    
    def rotate_key(self, key_id: str) -> Optional[Tuple[str, str]]:
        """Rotate an existing API key"""
        old_key = self.keys.get(key_id)
        if not old_key:
            return None
        
        # Revoke old key
        old_key.is_active = False
        
        # Generate new key
        new_key_id, raw_key = self.generate_key(
            user_id=old_key.user_id,
            permissions=old_key.permissions,
            expires_in_days=(old_key.expires_at - datetime.utcnow()).days if old_key.expires_at else None
        )
        
        self.keys[new_key_id].rotation_count = old_key.rotation_count + 1
        logger.info(f"Rotated API key {key_id} to {new_key_id}")
        return new_key_id, raw_key
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key"""
        if key_id in self.keys:
            self.keys[key_id].is_active = False
            logger.info(f"Revoked API key {key_id}")
            return True
        return False
    
    def list_user_keys(self, user_id: str) -> List[APIKey]:
        """List all active keys for a user"""
        return [key for key in self.keys.values() 
                if key.user_id == user_id and key.is_active]
    
    @staticmethod
    def _hash_key(raw_key: str) -> str:
        """Hash an API key for storage"""
        return hashlib.sha256(raw_key.encode()).hexdigest()


class IPFilter:
    """IP whitelist/blacklist management"""
    
    def __init__(self):
        self.whitelist: Set[str] = set()
        self.blacklist: Set[str] = set()
        self.whitelist_networks: List[ipaddress.IPv4Network] = []
        self.blacklist_networks: List[ipaddress.IPv4Network] = []
    
    def add_to_whitelist(self, ip_or_network: str):
        """Add IP or network to whitelist"""
        try:
            network = ipaddress.ip_network(ip_or_network, strict=False)
            if network.num_addresses == 1:
                self.whitelist.add(str(network.network_address))
            else:
                self.whitelist_networks.append(network)
            logger.info(f"Added {ip_or_network} to whitelist")
        except ValueError as e:
            logger.error(f"Invalid IP/network format: {e}")
    
    def add_to_blacklist(self, ip_or_network: str):
        """Add IP or network to blacklist"""
        try:
            network = ipaddress.ip_network(ip_or_network, strict=False)
            if network.num_addresses == 1:
                self.blacklist.add(str(network.network_address))
            else:
                self.blacklist_networks.append(network)
            logger.info(f"Added {ip_or_network} to blacklist")
        except ValueError as e:
            logger.error(f"Invalid IP/network format: {e}")
    
    def remove_from_whitelist(self, ip_or_network: str):
        """Remove IP or network from whitelist"""
        self.whitelist.discard(ip_or_network)
        # Remove from networks
        self.whitelist_networks = [
            net for net in self.whitelist_networks 
            if str(net) != ip_or_network
        ]
    
    def remove_from_blacklist(self, ip_or_network: str):
        """Remove IP or network from blacklist"""
        self.blacklist.discard(ip_or_network)
        self.blacklist_networks = [
            net for net in self.blacklist_networks 
            if str(net) != ip_or_network
        ]
    
    def is_allowed(self, ip_address: str) -> bool:
        """Check if an IP address is allowed"""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            # Check blacklist first (blacklist takes precedence)
            if ip_address in self.blacklist:
                return False
            
            for network in self.blacklist_networks:
                if ip in network:
                    return False
            
            # If whitelist is empty, allow all (except blacklisted)
            if not self.whitelist and not self.whitelist_networks:
                return True
            
            # Check whitelist
            if ip_address in self.whitelist:
                return True
            
            for network in self.whitelist_networks:
                if ip in network:
                    return True
            
            return False
        except ValueError:
            logger.error(f"Invalid IP address: {ip_address}")
            return False


class BruteForceProtection:
    """Brute force protection with account lockout"""
    
    def __init__(self, policy: PasswordPolicy):
        self.policy = policy
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.locked_accounts: Dict[str, datetime] = {}
    
    def record_failed_attempt(self, identifier: str) -> bool:
        """
        Record a failed login attempt
        Returns True if account should be locked
        """
        now = datetime.utcnow()
        
        # Clean old attempts
        self.failed_attempts[identifier] = [
            attempt for attempt in self.failed_attempts[identifier]
            if now - attempt < timedelta(minutes=self.policy.lockout_duration_minutes)
        ]
        
        self.failed_attempts[identifier].append(now)
        
        if len(self.failed_attempts[identifier]) >= self.policy.lockout_threshold:
            self.lock_account(identifier)
            return True
        
        return False
    
    def record_successful_attempt(self, identifier: str):
        """Record a successful login and clear failed attempts"""
        if identifier in self.failed_attempts:
            del self.failed_attempts[identifier]
        if identifier in self.locked_accounts:
            del self.locked_accounts[identifier]
    
    def lock_account(self, identifier: str):
        """Lock an account"""
        self.locked_accounts[identifier] = datetime.utcnow()
        logger.warning(f"Account locked: {identifier}")
    
    def unlock_account(self, identifier: str):
        """Manually unlock an account"""
        if identifier in self.locked_accounts:
            del self.locked_accounts[identifier]
        if identifier in self.failed_attempts:
            del self.failed_attempts[identifier]
        logger.info(f"Account unlocked: {identifier}")
    
    def is_locked(self, identifier: str) -> bool:
        """Check if an account is locked"""
        if identifier not in self.locked_accounts:
            return False
        
        locked_at = self.locked_accounts[identifier]
        lockout_duration = timedelta(minutes=self.policy.lockout_duration_minutes)
        
        if datetime.utcnow() - locked_at > lockout_duration:
            # Auto-unlock after duration
            self.unlock_account(identifier)
            return False
        
        return True
    
    def get_remaining_lockout_time(self, identifier: str) -> Optional[int]:
        """Get remaining lockout time in seconds"""
        if identifier not in self.locked_accounts:
            return None
        
        locked_at = self.locked_accounts[identifier]
        lockout_duration = timedelta(minutes=self.policy.lockout_duration_minutes)
        remaining = lockout_duration - (datetime.utcnow() - locked_at)
        
        return max(0, int(remaining.total_seconds()))


class JWTBlacklist:
    """JWT token blacklist for logout and revocation"""
    
    def __init__(self, cleanup_interval: int = 3600):
        self.blacklisted_tokens: Dict[str, datetime] = {}
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
    
    def add_token(self, token_jti: str, expires_at: datetime):
        """Add a token to the blacklist"""
        self.blacklisted_tokens[token_jti] = expires_at
        logger.info(f"Token {token_jti[:8]}... blacklisted")
        
        # Periodic cleanup
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._cleanup_expired()
    
    def is_blacklisted(self, token_jti: str) -> bool:
        """Check if a token is blacklisted"""
        if token_jti not in self.blacklisted_tokens:
            return False
        
        # Check if token has expired
        if datetime.utcnow() > self.blacklisted_tokens[token_jti]:
            del self.blacklisted_tokens[token_jti]
            return False
        
        return True
    
    def remove_token(self, token_jti: str):
        """Remove a token from blacklist"""
        if token_jti in self.blacklisted_tokens:
            del self.blacklisted_tokens[token_jti]
    
    def _cleanup_expired(self):
        """Remove expired tokens from blacklist"""
        now = datetime.utcnow()
        expired = [jti for jti, exp in self.blacklisted_tokens.items() if exp < now]
        
        for jti in expired:
            del self.blacklisted_tokens[jti]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired tokens from blacklist")
        
        self.last_cleanup = time.time()


class PasswordValidator:
    """Password policy enforcement"""
    
    def __init__(self, policy: PasswordPolicy):
        self.policy = policy
        self.password_history: Dict[str, List[str]] = defaultdict(list)
    
    def validate_password(self, password: str, user_id: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Validate a password against the policy
        Returns: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check length
        if len(password) < self.policy.min_length:
            errors.append(f"Password must be at least {self.policy.min_length} characters long")
        
        # Check uppercase
        if self.policy.require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        # Check lowercase
        if self.policy.require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        # Check digits
        if self.policy.require_digits and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        # Check special characters
        if self.policy.require_special_chars:
            if not any(c in self.policy.special_chars for c in password):
                errors.append(f"Password must contain at least one special character: {self.policy.special_chars}")
        
        # Check against password history
        if user_id and self._is_password_reused(user_id, password):
            errors.append(f"Password cannot be one of the last {self.policy.prevent_reuse_count} passwords")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"Password validation failed for user {user_id}: {', '.join(errors)}")
        
        return is_valid, errors
    
    def add_to_history(self, user_id: str, password_hash: str):
        """Add a password hash to user's history"""
        history = self.password_history[user_id]
        history.append(password_hash)
        
        # Keep only the last N passwords
        if len(history) > self.policy.prevent_reuse_count:
            self.password_history[user_id] = history[-self.policy.prevent_reuse_count:]
    
    def _is_password_reused(self, user_id: str, password: str) -> bool:
        """Check if password is in user's history"""
        password_hash = self._hash_password(password)
        return password_hash in self.password_history.get(user_id, [])
    
    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash a password for comparison"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def generate_strong_password(length: int = 16) -> str:
        """Generate a strong random password"""
        import string
        chars = string.ascii_letters + string.digits + "!@#$%^&*()_+-="
        return ''.join(secrets.choice(chars) for _ in range(length))


class FileUploadValidator:
    """Secure file upload validation"""
    
    def __init__(self, max_file_size: int = 10 * 1024 * 1024):  # 10MB default
        self.max_file_size = max_file_size
        self.allowed_extensions: Set[str] = set()
        self.allowed_mime_types: Set[str] = set()
        self.blocked_extensions: Set[str] = {
            'exe', 'bat', 'cmd', 'com', 'pif', 'scr', 'vbs', 'js', 'jar',
            'sh', 'ps1', 'app', 'deb', 'rpm'
        }
    
    def add_allowed_extension(self, extension: str):
        """Add an allowed file extension"""
        self.allowed_extensions.add(extension.lower().lstrip('.'))
    
    def add_allowed_mime_type(self, mime_type: str):
        """Add an allowed MIME type"""
        self.allowed_mime_types.add(mime_type.lower())
    
    def validate_file(self, filename: str, file_content: bytes, 
                     declared_mime_type: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Validate a file upload
        Returns: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check file size
        if len(file_content) > self.max_file_size:
            errors.append(f"File size exceeds maximum allowed size of {self.max_file_size} bytes")
        
        # Check filename
        if not self._is_safe_filename(filename):
            errors.append("Invalid filename: contains unsafe characters")
        
        # Get file extension
        extension = Path(filename).suffix.lstrip('.').lower()
        
        # Check blocked extensions
        if extension in self.blocked_extensions:
            errors.append(f"File extension '.{extension}' is not allowed for security reasons")
        
        # Check allowed extensions
        if self.allowed_extensions and extension not in self.allowed_extensions:
            errors.append(f"File extension '.{extension}' is not in the allowed list")
        
        # Validate MIME type
        if declared_mime_type:
            if self.allowed_mime_types and declared_mime_type not in self.allowed_mime_types:
                errors.append(f"MIME type '{declared_mime_type}' is not allowed")
        
        # Check for null bytes
        if b'\x00' in file_content:
            errors.append("File contains null bytes and may be malicious")
        
        # Basic magic byte validation
        if not self._validate_magic_bytes(filename, file_content):
            errors.append("File content does not match its extension")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"File upload validation failed for {filename}: {', '.join(errors)}")
        
        return is_valid, errors
    
    @staticmethod
    def _is_safe_filename(filename: str) -> bool:
        """Check if filename is safe"""
        # Disallow path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return False
        
        # Check for valid characters
        safe_pattern = re.compile(r'^[a-zA-Z0-9_\-. ]+$')
        return bool(safe_pattern.match(filename))
    
    def _validate_magic_bytes(self, filename: str, content: bytes) -> bool:
        """Validate file magic bytes against extension"""
        if len(content) < 4:
            return True  # Too small to validate
        
        extension = Path(filename).suffix.lstrip('.').lower()
        magic_bytes = content[:8]
        
        # Common file signatures
        signatures = {
            'pdf': [b'%PDF'],
            'png': [b'\x89PNG'],
            'jpg': [b'\xff\xd8\xff'],
            'jpeg': [b'\xff\xd8\xff'],
            'gif': [b'GIF87a', b'GIF89a'],
            'zip': [b'PK\x03\x04'],
            'docx': [b'PK\x03\x04'],  # Office files are ZIP-based
            'xlsx': [b'PK\x03\x04'],
        }
        
        if extension in signatures:
            return any(magic_bytes.startswith(sig) for sig in signatures[extension])
        
        return True  # Allow if no signature defined


class SecurityEventMonitor:
    """Security event monitoring and logging system"""
    
    def __init__(self, max_events: int = 10000):
        self.events: List[SecurityEvent] = []
        self.max_events = max_events
        self.event_handlers: Dict[SecurityEventType, List[callable]] = defaultdict(list)
    
    def log_event(self, event: SecurityEvent):
        """Log a security event"""
        self.events.append(event)
        
        # Trim old events if needed
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        # Log to standard logger
        log_message = (
            f"Security Event: {event.event_type.value} | "
            f"IP: {event.ip_address} | User: {event.user_id} | "
            f"Details: {event.details}"
        )
        
        if event.severity == "CRITICAL":
            logger.critical(log_message)
        elif event.severity == "WARNING":
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Call registered handlers
        for handler in self.event_handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    def register_handler(self, event_type: SecurityEventType, handler: callable):
        """Register a handler for specific event types"""
        self.event_handlers[event_type].append(handler)
    
    def get_events(self, event_type: Optional[SecurityEventType] = None,
                   user_id: Optional[str] = None,
                   ip_address: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   limit: int = 100) -> List[SecurityEvent]:
        """Query security events with filters"""
        filtered_events = self.events
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        if ip_address:
            filtered_events = [e for e in filtered_events if e.ip_address == ip_address]
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        return filtered_events[-limit:]
    
    def get_statistics(self) -> Dict:
        """Get security event statistics"""
        stats = {
            'total_events': len(self.events),
            'by_type': defaultdict(int),
            'by_severity': defaultdict(int),
            'recent_critical': []
        }
        
        for event in self.events:
            stats['by_type'][event.event_type.value] += 1
            stats['by_severity'][event.severity] += 1
            
            if event.severity == "CRITICAL":
                stats['recent_critical'].append({
                    'type': event.event_type.value,
                    'timestamp': event.timestamp.isoformat(),
                    'ip': event.ip_address,
                    'user': event.user_id
                })
        
        # Keep only last 10 critical events
        stats['recent_critical'] = stats['recent_critical'][-10:]
        
        return dict(stats)


class SecurityHardeningManager:
    """Main security hardening manager integrating all components"""
    
    def __init__(self, password_policy: Optional[PasswordPolicy] = None):
        self.password_policy = password_policy or PasswordPolicy()
        self.api_key_manager = APIKeyManager()
        self.ip_filter = IPFilter()
        self.brute_force_protection = BruteForceProtection(self.password_policy)
        self.jwt_blacklist = JWTBlacklist()
        self.password_validator = PasswordValidator(self.password_policy)
        self.file_upload_validator = FileUploadValidator()
        self.event_monitor = SecurityEventMonitor()
    
    def validate_request(self, ip_address: str, api_key: Optional[str] = None,
                        jwt_token_jti: Optional[str] = None) -> Tuple[bool, str]:
        """
        Validate an incoming request
        Returns: (is_valid, reason)
        """
        # Check IP filter
        if not self.ip_filter.is_allowed(ip_address):
            self.event_monitor.log_event(SecurityEvent(
                event_type=SecurityEventType.IP_BLOCKED,
                timestamp=datetime.utcnow(),
                ip_address=ip_address,
                severity="WARNING"
            ))
            return False, "IP address not allowed"
        
        # Check API key if provided
        if api_key:
            key_obj = self.api_key_manager.validate_key(api_key)
            if not key_obj:
                self.event_monitor.log_event(SecurityEvent(
                    event_type=SecurityEventType.API_KEY_INVALID,
                    timestamp=datetime.utcnow(),
                    ip_address=ip_address,
                    severity="WARNING"
                ))
                return False, "Invalid API key"
        
        # Check JWT blacklist if token provided
        if jwt_token_jti and self.jwt_blacklist.is_blacklisted(jwt_token_jti):
            return False, "Token has been revoked"
        
        return True, "Request validated"
    
    def handle_login_attempt(self, identifier: str, ip_address: str, 
                            success: bool, user_id: Optional[str] = None):
        """Handle a login attempt"""
        if self.brute_force_protection.is_locked(identifier):
            remaining = self.brute_force_protection.get_remaining_lockout_time(identifier)
            self.event_monitor.log_event(SecurityEvent(
                event_type=SecurityEventType.LOGIN_FAILED,
                timestamp=datetime.utcnow(),
                ip_address=ip_address,
                user_id=user_id,
                details={'reason': 'account_locked', 'remaining_seconds': remaining},
                severity="WARNING"
            ))
            return False, f"Account locked. Try again in {remaining} seconds"
        
        if success:
            self.brute_force_protection.record_successful_attempt(identifier)
            self.event_monitor.log_event(SecurityEvent(
                event_type=SecurityEventType.LOGIN_SUCCESS,
                timestamp=datetime.utcnow(),
                ip_address=ip_address,
                user_id=user_id,
                severity="INFO"
            ))
            return True, "Login successful"
        else:
            locked = self.brute_force_protection.record_failed_attempt(identifier)
            
            if locked:
                self.event_monitor.log_event(SecurityEvent(
                    event_type=SecurityEventType.ACCOUNT_LOCKED,
                    timestamp=datetime.utcnow(),
                    ip_address=ip_address,
                    user_id=user_id,
                    details={'failed_attempts': self.password_policy.lockout_threshold},
                    severity="CRITICAL"
                ))
                return False, "Account locked due to too many failed attempts"
            
            self.event_monitor.log_event(SecurityEvent(
                event_type=SecurityEventType.LOGIN_FAILED,
                timestamp=datetime.utcnow(),
                ip_address=ip_address,
                user_id=user_id,
                severity="WARNING"
            ))
            return False, "Invalid credentials"
    
    def get_security_status(self) -> Dict:
        """Get overall security status"""
        return {
            'active_api_keys': len([k for k in self.api_key_manager.keys.values() if k.is_active]),
            'whitelisted_ips': len(self.ip_filter.whitelist) + len(self.ip_filter.whitelist_networks),
            'blacklisted_ips': len(self.ip_filter.blacklist) + len(self.ip_filter.blacklist_networks),
            'locked_accounts': len(self.brute_force_protection.locked_accounts),
            'blacklisted_tokens': len(self.jwt_blacklist.blacklisted_tokens),
            'password_policy': {
                'min_length': self.password_policy.min_length,
                'lockout_threshold': self.password_policy.lockout_threshold,
                'lockout_duration_minutes': self.password_policy.lockout_duration_minutes
            },
            'event_statistics': self.event_monitor.get_statistics()
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize security manager
    policy = PasswordPolicy(
        min_length=12,
        lockout_threshold=5,
        lockout_duration_minutes=30
    )
    
    security_manager = SecurityHardeningManager(password_policy=policy)
    
    # Configure IP whitelist
    security_manager.ip_filter.add_to_whitelist("192.168.1.0/24")
    security_manager.ip_filter.add_to_whitelist("10.0.0.100")
    
    # Generate API key
    key_id, raw_key = security_manager.api_key_manager.generate_key(
        user_id="user123",
        permissions=["read", "write"],
        expires_in_days=90
    )
    print(f"Generated API Key: {raw_key}")
    
    # Validate password
    test_password = "SecureP@ssw0rd123"
    is_valid, errors = security_manager.password_validator.validate_password(test_password)
    print(f"Password valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    # Configure file upload validator
    security_manager.file_upload_validator.add_allowed_extension("pdf")
    security_manager.file_upload_validator.add_allowed_extension("png")
    security_manager.file_upload_validator.add_allowed_mime_type("application/pdf")
    security_manager.file_upload_validator.add_allowed_mime_type("image/png")
    
    # Test request validation
    is_valid, reason = security_manager.validate_request(
        ip_address="192.168.1.50",
        api_key=raw_key
    )
    print(f"Request valid: {is_valid} - {reason}")
    
    # Get security status
    status = security_manager.get_security_status()
    print(f"\nSecurity Status:")
    print(f"Active API Keys: {status['active_api_keys']}")
    print(f"Whitelisted IPs: {status['whitelisted_ips']}")
    print(f"Total Events: {status['event_statistics']['total_events']}")
