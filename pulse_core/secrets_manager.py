"""
Pulse Core Security - Secrets Manager
Security Patch 17/23

Comprehensive secrets management system with:
- Environment variable validation
- Secret encryption at rest using Fernet
- Secret rotation with zero-downtime support
- Role-based secret access control (RBAC)
- Audit logging for secret access
- Development vs production environment handling
- .env file security validation and scanning
- Secret detection in code repositories

Author: jetgause
Created: 2025-12-11
"""

import os
import re
import json
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from threading import Lock
import warnings

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    warnings.warn("cryptography package not installed. Encryption features disabled.")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class SecretRole(Enum):
    """Role-based access levels"""
    ADMIN = "admin"
    DEVELOPER = "developer"
    SERVICE = "service"
    READONLY = "readonly"


class SecretSensitivity(Enum):
    """Secret sensitivity levels"""
    CRITICAL = "critical"  # Production DB passwords, API keys
    HIGH = "high"          # Service credentials
    MEDIUM = "medium"      # Internal API tokens
    LOW = "low"            # Non-sensitive config


@dataclass
class SecretMetadata:
    """Metadata for a secret"""
    key: str
    created_at: datetime
    updated_at: datetime
    last_rotated: datetime
    rotation_interval: timedelta
    sensitivity: SecretSensitivity
    allowed_roles: Set[SecretRole]
    allowed_environments: Set[Environment]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    version: int = 1
    previous_versions: List[str] = field(default_factory=list)


@dataclass
class AuditLogEntry:
    """Audit log entry for secret access"""
    timestamp: datetime
    action: str
    secret_key: str
    user: str
    role: SecretRole
    environment: Environment
    success: bool
    ip_address: Optional[str] = None
    details: Optional[str] = None


class SecretsEncryption:
    """Handle encryption and decryption of secrets"""
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize encryption handler
        
        Args:
            master_key: Master encryption key (auto-generated if not provided)
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography package required for encryption")
        
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = Fernet.generate_key()
        
        self.fernet = Fernet(self.master_key)
        self._key_cache = {}
        self._cache_lock = Lock()
    
    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext secret"""
        try:
            encrypted = self.fernet.encrypt(plaintext.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, ciphertext: str) -> str:
        """Decrypt encrypted secret"""
        try:
            decrypted = self.fernet.decrypt(ciphertext.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def rotate_master_key(self, new_master_key: str) -> Fernet:
        """Rotate the master encryption key"""
        new_fernet = Fernet(new_master_key.encode())
        return new_fernet
    
    @staticmethod
    def generate_key() -> str:
        """Generate a new encryption key"""
        return Fernet.generate_key().decode()
    
    @staticmethod
    def derive_key_from_password(password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2"""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(password.encode())


class SecretDetector:
    """Detect secrets in code and configuration files"""
    
    # Common secret patterns
    PATTERNS = {
        'aws_key': r'AKIA[0-9A-Z]{16}',
        'aws_secret': r'aws[_\-]?secret[_\-]?access[_\-]?key',
        'github_token': r'ghp_[a-zA-Z0-9]{36}',
        'generic_api_key': r'api[_\-]?key[\s:=]+[\'"][a-zA-Z0-9]{32,}[\'"]',
        'password': r'password[\s:=]+[\'"][^\'"]{8,}[\'"]',
        'private_key': r'-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----',
        'jwt': r'eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}',
        'connection_string': r'(mongodb|mysql|postgresql):\/\/[^\s]+',
        'slack_token': r'xox[baprs]-[0-9a-zA-Z]{10,}',
        'stripe_key': r'sk_live_[0-9a-zA-Z]{24,}',
        'google_api': r'AIza[0-9A-Za-z_-]{35}',
        'heroku_api': r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',
        'mailgun_api': r'key-[0-9a-zA-Z]{32}',
        'generic_secret': r'secret[\s:=]+[\'"][^\'"]{16,}[\'"]',
    }
    
    # Entropy threshold for detecting random strings (potential secrets)
    ENTROPY_THRESHOLD = 4.5
    
    @classmethod
    def scan_text(cls, text: str) -> List[Dict[str, Any]]:
        """
        Scan text for potential secrets
        
        Returns:
            List of detected secrets with type and position
        """
        findings = []
        
        for secret_type, pattern in cls.PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                findings.append({
                    'type': secret_type,
                    'value': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'line': text[:match.start()].count('\n') + 1
                })
        
        # Check for high-entropy strings
        high_entropy = cls._find_high_entropy_strings(text)
        findings.extend(high_entropy)
        
        return findings
    
    @classmethod
    def scan_file(cls, filepath: Path) -> List[Dict[str, Any]]:
        """Scan a file for potential secrets"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                findings = cls.scan_text(content)
                for finding in findings:
                    finding['file'] = str(filepath)
                return findings
        except Exception as e:
            logger.error(f"Error scanning file {filepath}: {e}")
            return []
    
    @classmethod
    def scan_directory(cls, directory: Path, exclude_patterns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Recursively scan directory for secrets"""
        if exclude_patterns is None:
            exclude_patterns = [
                r'\.git/',
                r'\.venv/',
                r'node_modules/',
                r'__pycache__/',
                r'\.pyc$',
                r'\.log$',
            ]
        
        findings = []
        for filepath in directory.rglob('*'):
            if filepath.is_file():
                # Check exclusion patterns
                if any(re.search(pattern, str(filepath)) for pattern in exclude_patterns):
                    continue
                
                file_findings = cls.scan_file(filepath)
                findings.extend(file_findings)
        
        return findings
    
    @staticmethod
    def _find_high_entropy_strings(text: str) -> List[Dict[str, Any]]:
        """Find high-entropy strings that might be secrets"""
        findings = []
        # Look for quoted strings or assignment values
        pattern = r'["\']([a-zA-Z0-9+/=]{32,})["\']'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            value = match.group(1)
            entropy = SecretDetector._calculate_entropy(value)
            
            if entropy >= SecretDetector.ENTROPY_THRESHOLD:
                findings.append({
                    'type': 'high_entropy_string',
                    'value': value,
                    'start': match.start(),
                    'end': match.end(),
                    'line': text[:match.start()].count('\n') + 1,
                    'entropy': entropy
                })
        
        return findings
    
    @staticmethod
    def _calculate_entropy(string: str) -> float:
        """Calculate Shannon entropy of a string"""
        if not string:
            return 0.0
        
        entropy = 0.0
        for char in set(string):
            p_char = string.count(char) / len(string)
            entropy -= p_char * (p_char and hashlib.sha256(str(p_char).encode()).digest()[0] / 256)
        
        # Simplified entropy calculation
        char_counts = {}
        for char in string:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        entropy = 0.0
        for count in char_counts.values():
            p = count / len(string)
            entropy -= p * (p.bit_length() if p > 0 else 0)
        
        return entropy


class EnvFileValidator:
    """Validate and secure .env files"""
    
    REQUIRED_PERMISSIONS = 0o600  # Read/write for owner only
    
    @staticmethod
    def validate_env_file(filepath: Path) -> Dict[str, Any]:
        """
        Validate .env file security
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'secrets_found': []
        }
        
        if not filepath.exists():
            results['errors'].append(f"File not found: {filepath}")
            results['valid'] = False
            return results
        
        # Check file permissions
        stat_info = filepath.stat()
        if stat_info.st_mode & 0o777 != EnvFileValidator.REQUIRED_PERMISSIONS:
            results['warnings'].append(
                f"Insecure permissions: {oct(stat_info.st_mode & 0o777)}. "
                f"Should be {oct(EnvFileValidator.REQUIRED_PERMISSIONS)}"
            )
        
        # Scan for secrets
        secrets_found = SecretDetector.scan_file(filepath)
        if secrets_found:
            results['secrets_found'] = secrets_found
            results['warnings'].append(f"Found {len(secrets_found)} potential secrets")
        
        # Check for common issues
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Check format
                    if '=' not in line:
                        results['warnings'].append(f"Line {i}: Invalid format (missing '=')")
                        continue
                    
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Check for empty values
                    if not value:
                        results['warnings'].append(f"Line {i}: Empty value for {key}")
                    
                    # Check for unquoted values with spaces
                    if ' ' in value and not (value.startswith('"') and value.endswith('"')):
                        results['warnings'].append(f"Line {i}: Unquoted value with spaces")
        
        except Exception as e:
            results['errors'].append(f"Error reading file: {e}")
            results['valid'] = False
        
        return results
    
    @staticmethod
    def secure_env_file(filepath: Path) -> bool:
        """Set secure permissions on .env file"""
        try:
            filepath.chmod(EnvFileValidator.REQUIRED_PERMISSIONS)
            logger.info(f"Secured {filepath} with permissions {oct(EnvFileValidator.REQUIRED_PERMISSIONS)}")
            return True
        except Exception as e:
            logger.error(f"Failed to secure {filepath}: {e}")
            return False


class SecretsManager:
    """
    Comprehensive secrets management system
    """
    
    def __init__(
        self,
        environment: Environment,
        encryption_key: Optional[str] = None,
        storage_path: Optional[Path] = None,
        enable_audit_log: bool = True
    ):
        """
        Initialize Secrets Manager
        
        Args:
            environment: Current environment (dev, staging, prod)
            encryption_key: Master encryption key
            storage_path: Path to store encrypted secrets
            enable_audit_log: Enable audit logging
        """
        self.environment = environment
        self.enable_audit_log = enable_audit_log
        self.storage_path = storage_path or Path.home() / '.pulse_secrets'
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption
        if CRYPTO_AVAILABLE:
            self.encryption = SecretsEncryption(encryption_key)
        else:
            self.encryption = None
            logger.warning("Encryption disabled - cryptography package not available")
        
        # Secret storage
        self.secrets: Dict[str, str] = {}
        self.metadata: Dict[str, SecretMetadata] = {}
        self.audit_log: List[AuditLogEntry] = []
        
        # Access control
        self.current_user = os.getenv('USER', 'unknown')
        self.current_role = SecretRole.DEVELOPER
        
        # Thread safety
        self._lock = Lock()
        
        # Load secrets from storage
        self._load_secrets()
    
    def set_secret(
        self,
        key: str,
        value: str,
        sensitivity: SecretSensitivity = SecretSensitivity.MEDIUM,
        allowed_roles: Optional[Set[SecretRole]] = None,
        allowed_environments: Optional[Set[Environment]] = None,
        rotation_interval: timedelta = timedelta(days=90)
    ) -> bool:
        """
        Store a secret with encryption and metadata
        
        Args:
            key: Secret identifier
            value: Secret value
            sensitivity: Sensitivity level
            allowed_roles: Roles that can access this secret
            allowed_environments: Environments where secret is valid
            rotation_interval: How often secret should be rotated
        
        Returns:
            Success status
        """
        with self._lock:
            try:
                # Encrypt if available
                if self.encryption:
                    encrypted_value = self.encryption.encrypt(value)
                else:
                    encrypted_value = value
                    logger.warning(f"Storing secret '{key}' without encryption")
                
                # Create or update metadata
                now = datetime.utcnow()
                if key in self.metadata:
                    # Update existing secret
                    meta = self.metadata[key]
                    meta.previous_versions.append(self.secrets[key])
                    meta.updated_at = now
                    meta.version += 1
                else:
                    # Create new secret
                    meta = SecretMetadata(
                        key=key,
                        created_at=now,
                        updated_at=now,
                        last_rotated=now,
                        rotation_interval=rotation_interval,
                        sensitivity=sensitivity,
                        allowed_roles=allowed_roles or {SecretRole.ADMIN, SecretRole.DEVELOPER},
                        allowed_environments=allowed_environments or set(Environment)
                    )
                    self.metadata[key] = meta
                
                self.secrets[key] = encrypted_value
                
                # Persist to disk
                self._save_secrets()
                
                # Audit log
                self._log_access('set_secret', key, True)
                
                logger.info(f"Secret '{key}' stored successfully (version {meta.version})")
                return True
            
            except Exception as e:
                logger.error(f"Failed to store secret '{key}': {e}")
                self._log_access('set_secret', key, False, str(e))
                return False
    
    def get_secret(
        self,
        key: str,
        role: Optional[SecretRole] = None,
        environment: Optional[Environment] = None
    ) -> Optional[str]:
        """
        Retrieve a secret with access control
        
        Args:
            key: Secret identifier
            role: User role (uses current_role if not specified)
            environment: Environment (uses current environment if not specified)
        
        Returns:
            Decrypted secret value or None
        """
        with self._lock:
            role = role or self.current_role
            environment = environment or self.environment
            
            # Check if secret exists
            if key not in self.secrets:
                self._log_access('get_secret', key, False, 'Secret not found')
                return None
            
            # Check access control
            meta = self.metadata[key]
            
            if role not in meta.allowed_roles:
                logger.warning(f"Access denied: Role {role} cannot access '{key}'")
                self._log_access('get_secret', key, False, f'Role {role} not allowed')
                return None
            
            if environment not in meta.allowed_environments:
                logger.warning(f"Access denied: Environment {environment} cannot access '{key}'")
                self._log_access('get_secret', key, False, f'Environment {environment} not allowed')
                return None
            
            try:
                # Decrypt if needed
                encrypted_value = self.secrets[key]
                if self.encryption:
                    value = self.encryption.decrypt(encrypted_value)
                else:
                    value = encrypted_value
                
                # Update metadata
                meta.access_count += 1
                meta.last_accessed = datetime.utcnow()
                
                # Check rotation
                if self._needs_rotation(meta):
                    logger.warning(f"Secret '{key}' needs rotation (last rotated {meta.last_rotated})")
                
                self._log_access('get_secret', key, True)
                return value
            
            except Exception as e:
                logger.error(f"Failed to retrieve secret '{key}': {e}")
                self._log_access('get_secret', key, False, str(e))
                return None
    
    def rotate_secret(
        self,
        key: str,
        new_value: str,
        zero_downtime: bool = True
    ) -> bool:
        """
        Rotate a secret with optional zero-downtime support
        
        Args:
            key: Secret identifier
            new_value: New secret value
            zero_downtime: Keep old version temporarily for zero-downtime rotation
        
        Returns:
            Success status
        """
        with self._lock:
            if key not in self.secrets:
                logger.error(f"Cannot rotate non-existent secret '{key}'")
                return False
            
            try:
                meta = self.metadata[key]
                
                # Store old version if zero-downtime
                if zero_downtime:
                    old_key = f"{key}_v{meta.version}"
                    self.secrets[old_key] = self.secrets[key]
                    logger.info(f"Old version stored as '{old_key}' for zero-downtime rotation")
                
                # Set new secret
                success = self.set_secret(
                    key,
                    new_value,
                    meta.sensitivity,
                    meta.allowed_roles,
                    meta.allowed_environments,
                    meta.rotation_interval
                )
                
                if success:
                    meta.last_rotated = datetime.utcnow()
                    self._log_access('rotate_secret', key, True)
                    logger.info(f"Secret '{key}' rotated successfully")
                
                return success
            
            except Exception as e:
                logger.error(f"Failed to rotate secret '{key}': {e}")
                self._log_access('rotate_secret', key, False, str(e))
                return False
    
    def delete_secret(self, key: str) -> bool:
        """Delete a secret"""
        with self._lock:
            if key not in self.secrets:
                return False
            
            try:
                del self.secrets[key]
                del self.metadata[key]
                self._save_secrets()
                self._log_access('delete_secret', key, True)
                logger.info(f"Secret '{key}' deleted")
                return True
            except Exception as e:
                logger.error(f"Failed to delete secret '{key}': {e}")
                return False
    
    def list_secrets(self, role: Optional[SecretRole] = None) -> List[str]:
        """List all accessible secret keys"""
        role = role or self.current_role
        return [
            key for key, meta in self.metadata.items()
            if role in meta.allowed_roles and self.environment in meta.allowed_environments
        ]
    
    def get_secret_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a secret"""
        if key not in self.metadata:
            return None
        
        meta = self.metadata[key]
        return {
            'key': meta.key,
            'created_at': meta.created_at.isoformat(),
            'updated_at': meta.updated_at.isoformat(),
            'last_rotated': meta.last_rotated.isoformat(),
            'last_accessed': meta.last_accessed.isoformat() if meta.last_accessed else None,
            'rotation_interval_days': meta.rotation_interval.days,
            'needs_rotation': self._needs_rotation(meta),
            'sensitivity': meta.sensitivity.value,
            'access_count': meta.access_count,
            'version': meta.version,
            'allowed_roles': [role.value for role in meta.allowed_roles],
            'allowed_environments': [env.value for env in meta.allowed_environments]
        }
    
    def get_secrets_needing_rotation(self) -> List[str]:
        """Get list of secrets that need rotation"""
        return [
            key for key, meta in self.metadata.items()
            if self._needs_rotation(meta)
        ]
    
    def export_audit_log(self, filepath: Optional[Path] = None) -> str:
        """Export audit log to JSON"""
        filepath = filepath or self.storage_path / f'audit_log_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.json'
        
        log_data = [
            {
                'timestamp': entry.timestamp.isoformat(),
                'action': entry.action,
                'secret_key': entry.secret_key,
                'user': entry.user,
                'role': entry.role.value,
                'environment': entry.environment.value,
                'success': entry.success,
                'ip_address': entry.ip_address,
                'details': entry.details
            }
            for entry in self.audit_log
        ]
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Audit log exported to {filepath}")
        return str(filepath)
    
    def validate_environment_variables(self, required_vars: List[str]) -> Dict[str, Any]:
        """
        Validate that required environment variables are set
        
        Args:
            required_vars: List of required environment variable names
        
        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'missing': [],
            'empty': [],
            'found': []
        }
        
        for var in required_vars:
            value = os.getenv(var)
            if value is None:
                results['missing'].append(var)
                results['valid'] = False
            elif not value:
                results['empty'].append(var)
                results['valid'] = False
            else:
                results['found'].append(var)
        
        return results
    
    def _needs_rotation(self, meta: SecretMetadata) -> bool:
        """Check if secret needs rotation"""
        time_since_rotation = datetime.utcnow() - meta.last_rotated
        return time_since_rotation >= meta.rotation_interval
    
    def _log_access(
        self,
        action: str,
        secret_key: str,
        success: bool,
        details: Optional[str] = None
    ):
        """Log secret access to audit log"""
        if not self.enable_audit_log:
            return
        
        entry = AuditLogEntry(
            timestamp=datetime.utcnow(),
            action=action,
            secret_key=secret_key,
            user=self.current_user,
            role=self.current_role,
            environment=self.environment,
            success=success,
            details=details
        )
        self.audit_log.append(entry)
        
        # Keep audit log size manageable
        if len(self.audit_log) > 10000:
            self.export_audit_log()
            self.audit_log = self.audit_log[-1000:]
    
    def _save_secrets(self):
        """Persist secrets to disk"""
        data = {
            'secrets': self.secrets,
            'metadata': {
                key: {
                    'key': meta.key,
                    'created_at': meta.created_at.isoformat(),
                    'updated_at': meta.updated_at.isoformat(),
                    'last_rotated': meta.last_rotated.isoformat(),
                    'rotation_interval_days': meta.rotation_interval.days,
                    'sensitivity': meta.sensitivity.value,
                    'allowed_roles': [role.value for role in meta.allowed_roles],
                    'allowed_environments': [env.value for env in meta.allowed_environments],
                    'access_count': meta.access_count,
                    'last_accessed': meta.last_accessed.isoformat() if meta.last_accessed else None,
                    'version': meta.version,
                    'previous_versions': meta.previous_versions
                }
                for key, meta in self.metadata.items()
            }
        }
        
        filepath = self.storage_path / f'secrets_{self.environment.value}.json'
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Secure the file
        filepath.chmod(0o600)
    
    def _load_secrets(self):
        """Load secrets from disk"""
        filepath = self.storage_path / f'secrets_{self.environment.value}.json'
        if not filepath.exists():
            return
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.secrets = data.get('secrets', {})
            
            # Reconstruct metadata
            for key, meta_dict in data.get('metadata', {}).items():
                self.metadata[key] = SecretMetadata(
                    key=meta_dict['key'],
                    created_at=datetime.fromisoformat(meta_dict['created_at']),
                    updated_at=datetime.fromisoformat(meta_dict['updated_at']),
                    last_rotated=datetime.fromisoformat(meta_dict['last_rotated']),
                    rotation_interval=timedelta(days=meta_dict['rotation_interval_days']),
                    sensitivity=SecretSensitivity(meta_dict['sensitivity']),
                    allowed_roles={SecretRole(role) for role in meta_dict['allowed_roles']},
                    allowed_environments={Environment(env) for env in meta_dict['allowed_environments']},
                    access_count=meta_dict.get('access_count', 0),
                    last_accessed=datetime.fromisoformat(meta_dict['last_accessed']) if meta_dict.get('last_accessed') else None,
                    version=meta_dict.get('version', 1),
                    previous_versions=meta_dict.get('previous_versions', [])
                )
            
            logger.info(f"Loaded {len(self.secrets)} secrets from storage")
        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")


# Example usage and integration
if __name__ == "__main__":
    print("=" * 70)
    print("Pulse Core Security - Secrets Manager")
    print("Security Patch 17/23")
    print("=" * 70)
    
    # Initialize secrets manager
    manager = SecretsManager(
        environment=Environment.DEVELOPMENT,
        enable_audit_log=True
    )
    
    print("\n[+] Secrets Manager initialized")
    
    # Set some example secrets
    manager.set_secret(
        'database_password',
        'super_secure_password_123',
        sensitivity=SecretSensitivity.CRITICAL,
        allowed_roles={SecretRole.ADMIN, SecretRole.SERVICE},
        allowed_environments={Environment.PRODUCTION}
    )
    
    manager.set_secret(
        'api_key',
        'sk_test_1234567890abcdef',
        sensitivity=SecretSensitivity.HIGH,
        rotation_interval=timedelta(days=30)
    )
    
    print("[+] Example secrets stored")
    
    # Demonstrate secret detection
    print("\n[+] Running secret detection scan...")
    test_code = '''
    API_KEY = "sk_live_1234567890abcdef"
    password = "my_secret_password"
    aws_key = "AKIAIOSFODNN7EXAMPLE"
    '''
    
    findings = SecretDetector.scan_text(test_code)
    print(f"    Found {len(findings)} potential secrets in test code")
    for finding in findings:
        print(f"    - {finding['type']} at line {finding['line']}")
    
    # Validate environment variables
    print("\n[+] Validating environment variables...")
    validation = manager.validate_environment_variables(['USER', 'HOME', 'NONEXISTENT_VAR'])
    print(f"    Found: {validation['found']}")
    print(f"    Missing: {validation['missing']}")
    
    # List accessible secrets
    print("\n[+] Accessible secrets:")
    for key in manager.list_secrets():
        info = manager.get_secret_info(key)
        print(f"    - {key} (sensitivity: {info['sensitivity']}, version: {info['version']})")
    
    # Export audit log
    log_file = manager.export_audit_log()
    print(f"\n[+] Audit log exported to: {log_file}")
    
    print("\n[+] Secrets Manager demonstration complete")
    print("=" * 70)
