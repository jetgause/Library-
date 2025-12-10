"""
Encryption and Data Protection Module
=====================================

Enterprise-grade encryption utilities for protecting sensitive data:
- AES-256-GCM encryption for data at rest
- RSA encryption for asymmetric operations
- Password hashing with bcrypt
- Key derivation with PBKDF2
- Secure secret management
- Field-level encryption
- Database column encryption helpers
- Configuration encryption

Author: jetgause
Created: 2025-12-10
Version: 1.0.0
"""

import os
import base64
import secrets
import hashlib
import hmac
import json
from typing import Optional, Dict, Any, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
import bcrypt

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    AES_256_GCM = "aes-256-gcm"
    FERNET = "fernet"
    RSA_OAEP = "rsa-oaep"


class KeySize(Enum):
    """Encryption key sizes."""
    AES_128 = 16
    AES_192 = 24
    AES_256 = 32
    RSA_2048 = 2048
    RSA_4096 = 4096


class EncryptionConfig:
    """Configuration for encryption operations."""
    
    def __init__(
        self,
        master_key: Optional[bytes] = None,
        key_rotation_days: int = 90,
        pbkdf2_iterations: int = 100000,
        default_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
        key_size: KeySize = KeySize.AES_256,
    ):
        """
        Initialize encryption configuration.
        
        Args:
            master_key: Master encryption key (32 bytes for AES-256)
            key_rotation_days: Days before key rotation required
            pbkdf2_iterations: PBKDF2 iteration count
            default_algorithm: Default encryption algorithm
            key_size: Default key size
        """
        self.master_key = master_key or self._generate_master_key()
        self.key_rotation_days = key_rotation_days
        self.pbkdf2_iterations = pbkdf2_iterations
        self.default_algorithm = default_algorithm
        self.key_size = key_size
    
    @staticmethod
    def _generate_master_key() -> bytes:
        """Generate a cryptographically secure master key."""
        return secrets.token_bytes(32)
    
    @staticmethod
    def from_env(env_var: str = "ENCRYPTION_MASTER_KEY") -> 'EncryptionConfig':
        """
        Load configuration from environment variable.
        
        Args:
            env_var: Environment variable name
            
        Returns:
            EncryptionConfig instance
        """
        key_b64 = os.getenv(env_var)
        if key_b64:
            master_key = base64.b64decode(key_b64)
            return EncryptionConfig(master_key=master_key)
        
        logger.warning(f"No master key found in {env_var}, generating new key")
        return EncryptionConfig()


class AESEncryption:
    """AES-256-GCM encryption implementation."""
    
    def __init__(self, key: bytes):
        """
        Initialize AES encryption.
        
        Args:
            key: 32-byte encryption key for AES-256
        """
        if len(key) != 32:
            raise ValueError("AES-256 requires a 32-byte key")
        self.key = key
        self.aesgcm = AESGCM(key)
    
    def encrypt(self, plaintext: Union[str, bytes], associated_data: Optional[bytes] = None) -> Dict[str, str]:
        """
        Encrypt data using AES-256-GCM.
        
        Args:
            plaintext: Data to encrypt
            associated_data: Optional additional authenticated data
            
        Returns:
            Dictionary with ciphertext, nonce, and tag (base64 encoded)
        """
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        
        # Generate random nonce (12 bytes for GCM)
        nonce = secrets.token_bytes(12)
        
        # Encrypt
        ciphertext = self.aesgcm.encrypt(nonce, plaintext, associated_data)
        
        return {
            "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
            "nonce": base64.b64encode(nonce).decode('utf-8'),
            "algorithm": EncryptionAlgorithm.AES_256_GCM.value,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def decrypt(self, encrypted_data: Dict[str, str], associated_data: Optional[bytes] = None) -> bytes:
        """
        Decrypt AES-256-GCM encrypted data.
        
        Args:
            encrypted_data: Dictionary with ciphertext and nonce
            associated_data: Optional additional authenticated data
            
        Returns:
            Decrypted plaintext bytes
        """
        ciphertext = base64.b64decode(encrypted_data["ciphertext"])
        nonce = base64.b64decode(encrypted_data["nonce"])
        
        try:
            plaintext = self.aesgcm.decrypt(nonce, ciphertext, associated_data)
            return plaintext
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise ValueError("Decryption failed - invalid ciphertext or key")
    
    @staticmethod
    def generate_key() -> bytes:
        """Generate a random AES-256 key."""
        return secrets.token_bytes(32)


class RSAEncryption:
    """RSA asymmetric encryption implementation."""
    
    def __init__(self, key_size: int = 2048):
        """
        Initialize RSA encryption.
        
        Args:
            key_size: RSA key size in bits (2048 or 4096)
        """
        self.key_size = key_size
        self.private_key = None
        self.public_key = None
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate RSA keypair.
        
        Returns:
            Tuple of (private_key_pem, public_key_pem)
        """
        # Generate private key
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size,
            backend=default_backend()
        )
        
        # Get public key
        self.public_key = self.private_key.public_key()
        
        # Serialize keys to PEM format
        private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def load_private_key(self, private_key_pem: bytes, password: Optional[bytes] = None):
        """Load private key from PEM."""
        self.private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=password,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
    
    def load_public_key(self, public_key_pem: bytes):
        """Load public key from PEM."""
        self.public_key = serialization.load_pem_public_key(
            public_key_pem,
            backend=default_backend()
        )
    
    def encrypt(self, plaintext: Union[str, bytes]) -> bytes:
        """
        Encrypt data with RSA public key.
        
        Args:
            plaintext: Data to encrypt
            
        Returns:
            Encrypted ciphertext
        """
        if not self.public_key:
            raise ValueError("Public key not loaded")
        
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        
        ciphertext = self.public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return ciphertext
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        """
        Decrypt data with RSA private key.
        
        Args:
            ciphertext: Encrypted data
            
        Returns:
            Decrypted plaintext
        """
        if not self.private_key:
            raise ValueError("Private key not loaded")
        
        plaintext = self.private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return plaintext


class PasswordHasher:
    """Secure password hashing with bcrypt."""
    
    def __init__(self, rounds: int = 12):
        """
        Initialize password hasher.
        
        Args:
            rounds: Bcrypt cost factor (4-31, default 12)
        """
        self.rounds = rounds
    
    def hash(self, password: str) -> str:
        """
        Hash a password using bcrypt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password (base64 encoded)
        """
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt(rounds=self.rounds)
        hashed = bcrypt.hashpw(password_bytes, salt)
        return base64.b64encode(hashed).decode('utf-8')
    
    def verify(self, password: str, hashed: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            password: Plain text password
            hashed: Hashed password (base64 encoded)
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            password_bytes = password.encode('utf-8')
            hashed_bytes = base64.b64decode(hashed)
            return bcrypt.checkpw(password_bytes, hashed_bytes)
        except Exception as e:
            logger.error(f"Password verification failed: {str(e)}")
            return False
    
    def needs_rehash(self, hashed: str) -> bool:
        """Check if password hash needs to be updated with new cost factor."""
        try:
            hashed_bytes = base64.b64decode(hashed)
            # Extract cost factor from bcrypt hash
            cost = int(hashed_bytes[4:6])
            return cost < self.rounds
        except Exception:
            return True


class KeyDerivation:
    """Key derivation functions."""
    
    @staticmethod
    def derive_key(
        password: str,
        salt: Optional[bytes] = None,
        iterations: int = 100000,
        key_length: int = 32
    ) -> Tuple[bytes, bytes]:
        """
        Derive encryption key from password using PBKDF2.
        
        Args:
            password: Password to derive from
            salt: Salt value (generated if not provided)
            iterations: PBKDF2 iteration count
            key_length: Desired key length in bytes
            
        Returns:
            Tuple of (derived_key, salt)
        """
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        
        key = kdf.derive(password.encode('utf-8'))
        return key, salt
    
    @staticmethod
    def derive_multiple_keys(
        master_key: bytes,
        key_count: int,
        key_length: int = 32
    ) -> list:
        """
        Derive multiple keys from a master key.
        
        Args:
            master_key: Master key
            key_count: Number of keys to derive
            key_length: Length of each derived key
            
        Returns:
            List of derived keys
        """
        keys = []
        for i in range(key_count):
            # Use counter as salt
            salt = i.to_bytes(8, 'big')
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                iterations=10000,
                backend=default_backend()
            )
            key = kdf.derive(master_key)
            keys.append(key)
        
        return keys


class FieldEncryption:
    """Field-level encryption for database columns."""
    
    def __init__(self, config: EncryptionConfig):
        """
        Initialize field encryption.
        
        Args:
            config: EncryptionConfig instance
        """
        self.config = config
        self.aes = AESEncryption(config.master_key)
    
    def encrypt_field(self, value: Any, field_name: str = "") -> str:
        """
        Encrypt a single field value.
        
        Args:
            value: Value to encrypt
            field_name: Field name for associated data
            
        Returns:
            Base64 encoded encrypted value
        """
        if value is None:
            return None
        
        # Convert to string
        if not isinstance(value, str):
            value = json.dumps(value)
        
        # Use field name as associated data for additional security
        associated_data = field_name.encode('utf-8') if field_name else None
        
        # Encrypt
        encrypted = self.aes.encrypt(value, associated_data)
        
        # Return as JSON string
        return json.dumps(encrypted)
    
    def decrypt_field(self, encrypted_value: str, field_name: str = "") -> Any:
        """
        Decrypt a field value.
        
        Args:
            encrypted_value: Encrypted value (JSON string)
            field_name: Field name for associated data
            
        Returns:
            Decrypted value
        """
        if not encrypted_value:
            return None
        
        # Parse encrypted data
        encrypted_data = json.loads(encrypted_value)
        
        # Use field name as associated data
        associated_data = field_name.encode('utf-8') if field_name else None
        
        # Decrypt
        plaintext_bytes = self.aes.decrypt(encrypted_data, associated_data)
        plaintext = plaintext_bytes.decode('utf-8')
        
        # Try to parse as JSON
        try:
            return json.loads(plaintext)
        except json.JSONDecodeError:
            return plaintext
    
    def encrypt_dict(self, data: Dict[str, Any], fields: list) -> Dict[str, Any]:
        """
        Encrypt specified fields in a dictionary.
        
        Args:
            data: Dictionary with data
            fields: List of field names to encrypt
            
        Returns:
            Dictionary with encrypted fields
        """
        encrypted_data = data.copy()
        
        for field in fields:
            if field in encrypted_data:
                encrypted_data[field] = self.encrypt_field(
                    encrypted_data[field],
                    field_name=field
                )
        
        return encrypted_data
    
    def decrypt_dict(self, data: Dict[str, Any], fields: list) -> Dict[str, Any]:
        """
        Decrypt specified fields in a dictionary.
        
        Args:
            data: Dictionary with encrypted data
            fields: List of field names to decrypt
            
        Returns:
            Dictionary with decrypted fields
        """
        decrypted_data = data.copy()
        
        for field in fields:
            if field in decrypted_data:
                decrypted_data[field] = self.decrypt_field(
                    decrypted_data[field],
                    field_name=field
                )
        
        return decrypted_data


class SecretManager:
    """Secure secret management."""
    
    def __init__(self, config: EncryptionConfig):
        """
        Initialize secret manager.
        
        Args:
            config: EncryptionConfig instance
        """
        self.config = config
        self.field_encryption = FieldEncryption(config)
        self._secrets: Dict[str, Dict[str, Any]] = {}
    
    def store_secret(
        self,
        name: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store an encrypted secret.
        
        Args:
            name: Secret name
            value: Secret value
            metadata: Optional metadata
            
        Returns:
            Secret ID
        """
        secret_id = hashlib.sha256(name.encode()).hexdigest()[:16]
        
        # Encrypt value
        encrypted_value = self.field_encryption.encrypt_field(value, field_name=name)
        
        # Store secret
        self._secrets[secret_id] = {
            "name": name,
            "encrypted_value": encrypted_value,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "accessed_at": None,
            "access_count": 0,
        }
        
        logger.info(f"Stored secret: {name} (ID: {secret_id})")
        return secret_id
    
    def get_secret(self, name: str) -> Optional[str]:
        """
        Retrieve and decrypt a secret.
        
        Args:
            name: Secret name
            
        Returns:
            Decrypted secret value
        """
        secret_id = hashlib.sha256(name.encode()).hexdigest()[:16]
        
        if secret_id not in self._secrets:
            return None
        
        secret = self._secrets[secret_id]
        
        # Update access tracking
        secret["accessed_at"] = datetime.utcnow().isoformat()
        secret["access_count"] += 1
        
        # Decrypt and return
        decrypted_value = self.field_encryption.decrypt_field(
            secret["encrypted_value"],
            field_name=name
        )
        
        return decrypted_value
    
    def delete_secret(self, name: str) -> bool:
        """
        Delete a secret.
        
        Args:
            name: Secret name
            
        Returns:
            True if deleted, False if not found
        """
        secret_id = hashlib.sha256(name.encode()).hexdigest()[:16]
        
        if secret_id in self._secrets:
            del self._secrets[secret_id]
            logger.info(f"Deleted secret: {name}")
            return True
        
        return False
    
    def list_secrets(self) -> list:
        """
        List all secret names (not values).
        
        Returns:
            List of secret names
        """
        return [secret["name"] for secret in self._secrets.values()]
    
    def rotate_secret(self, name: str, new_value: str) -> bool:
        """
        Rotate a secret with a new value.
        
        Args:
            name: Secret name
            new_value: New secret value
            
        Returns:
            True if rotated, False if not found
        """
        secret_id = hashlib.sha256(name.encode()).hexdigest()[:16]
        
        if secret_id not in self._secrets:
            return False
        
        # Encrypt new value
        encrypted_value = self.field_encryption.encrypt_field(new_value, field_name=name)
        
        # Update secret
        self._secrets[secret_id]["encrypted_value"] = encrypted_value
        self._secrets[secret_id]["rotated_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Rotated secret: {name}")
        return True


class EncryptedConfig:
    """Encrypted configuration management."""
    
    def __init__(self, config: EncryptionConfig):
        """
        Initialize encrypted config.
        
        Args:
            config: EncryptionConfig instance
        """
        self.field_encryption = FieldEncryption(config)
        self.config_data: Dict[str, Any] = {}
    
    def set(self, key: str, value: Any, encrypted: bool = True):
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            encrypted: Whether to encrypt the value
        """
        if encrypted:
            self.config_data[key] = {
                "value": self.field_encryption.encrypt_field(value, field_name=key),
                "encrypted": True,
            }
        else:
            self.config_data[key] = {
                "value": value,
                "encrypted": False,
            }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        if key not in self.config_data:
            return default
        
        entry = self.config_data[key]
        
        if entry["encrypted"]:
            return self.field_encryption.decrypt_field(entry["value"], field_name=key)
        
        return entry["value"]
    
    def save_to_file(self, filepath: str):
        """Save encrypted configuration to file."""
        with open(filepath, 'w') as f:
            json.dump(self.config_data, f, indent=2)
        logger.info(f"Saved encrypted config to {filepath}")
    
    def load_from_file(self, filepath: str):
        """Load encrypted configuration from file."""
        with open(filepath, 'r') as f:
            self.config_data = json.load(f)
        logger.info(f"Loaded encrypted config from {filepath}")


# Utility functions

def generate_encryption_key() -> str:
    """Generate a base64-encoded encryption key."""
    key = secrets.token_bytes(32)
    return base64.b64encode(key).decode('utf-8')


def encrypt_string(plaintext: str, key: bytes) -> str:
    """Quick encryption of a string."""
    aes = AESEncryption(key)
    encrypted = aes.encrypt(plaintext)
    return json.dumps(encrypted)


def decrypt_string(encrypted: str, key: bytes) -> str:
    """Quick decryption of a string."""
    aes = AESEncryption(key)
    encrypted_data = json.loads(encrypted)
    plaintext_bytes = aes.decrypt(encrypted_data)
    return plaintext_bytes.decode('utf-8')


# Example usage
if __name__ == "__main__":
    print("Encryption Module")
    print("=" * 60)
    
    # Generate master key
    print("\n1. Generating master encryption key...")
    config = EncryptionConfig()
    master_key_b64 = base64.b64encode(config.master_key).decode('utf-8')
    print(f"Master Key (base64): {master_key_b64[:40]}...")
    
    # AES encryption
    print("\n2. AES-256-GCM Encryption:")
    aes = AESEncryption(config.master_key)
    plaintext = "Sensitive data: Credit Card 4111-1111-1111-1111"
    encrypted = aes.encrypt(plaintext)
    print(f"Plaintext: {plaintext}")
    print(f"Encrypted: {encrypted['ciphertext'][:40]}...")
    decrypted = aes.decrypt(encrypted)
    print(f"Decrypted: {decrypted.decode('utf-8')}")
    
    # Password hashing
    print("\n3. Password Hashing:")
    hasher = PasswordHasher(rounds=12)
    password = "SecurePassword123!"
    hashed = hasher.hash(password)
    print(f"Password: {password}")
    print(f"Hashed: {hashed[:40]}...")
    print(f"Verification: {hasher.verify(password, hashed)}")
    
    # Field encryption
    print("\n4. Field-Level Encryption:")
    field_enc = FieldEncryption(config)
    user_data = {
        "username": "john_doe",
        "email": "john@example.com",
        "ssn": "123-45-6789",
        "credit_card": "4111-1111-1111-1111"
    }
    encrypted_data = field_enc.encrypt_dict(user_data, ["ssn", "credit_card"])
    print(f"Original: {user_data}")
    print(f"Encrypted SSN: {encrypted_data['ssn'][:40]}...")
    decrypted_data = field_enc.decrypt_dict(encrypted_data, ["ssn", "credit_card"])
    print(f"Decrypted: {decrypted_data}")
    
    # Secret management
    print("\n5. Secret Management:")
    secret_mgr = SecretManager(config)
    secret_mgr.store_secret("api_key", "sk-1234567890abcdef")
    secret_mgr.store_secret("db_password", "super_secret_password")
    print(f"Stored secrets: {secret_mgr.list_secrets()}")
    api_key = secret_mgr.get_secret("api_key")
    print(f"Retrieved API key: {api_key}")
    
    print("\n" + "=" * 60)
    print("Encryption module loaded successfully!")
