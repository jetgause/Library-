"""
PULSE Platform Security Test Suite
===================================

Comprehensive security testing module covering:
- Configuration security (SECRET_KEY, CORS, API binding)
- Input validation (SQL injection, XSS, sanitization)
- Rate limiting
- Encryption (password hashing, token generation)
- Security headers
- CSRF protection
- Session security
- API key security
- Authentication

Created: 2025-12-11
Author: jetgause
"""

import pytest
import os
import sys
import tempfile
import secrets
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestConfigurationSecurity:
    """Test security-related configuration."""
    
    def test_secret_key_required(self):
        """Verify SECRET_KEY is required."""
        with patch.dict(os.environ, {}, clear=True):
            # Load .env first if exists
            from dotenv import load_dotenv
            # Clear the environment variable
            if 'SECRET_KEY' in os.environ:
                del os.environ['SECRET_KEY']
            
            with pytest.raises(SystemExit) as exc_info:
                # Reimport config to trigger validation
                import importlib
                import config
                importlib.reload(config)
            
            assert exc_info.value.code == 1
    
    def test_secret_key_minimum_length(self):
        """Enforce SECRET_KEY minimum 32 characters."""
        short_key = "tooshort"
        
        with patch.dict(os.environ, {'SECRET_KEY': short_key}):
            with pytest.raises(SystemExit) as exc_info:
                import importlib
                import config
                importlib.reload(config)
            
            assert exc_info.value.code == 1
    
    def test_secret_key_rejects_weak_defaults(self):
        """Block weak/default keys."""
        weak_keys = [
            "your-secret-key-change-in-production",
            "change-this-in-production",
            "secret",
            "password",
            "secret-key"
        ]
        
        for weak_key in weak_keys:
            with patch.dict(os.environ, {'SECRET_KEY': weak_key}):
                with pytest.raises(SystemExit) as exc_info:
                    import importlib
                    import config
                    importlib.reload(config)
                
                assert exc_info.value.code == 1
    
    def test_cors_defaults_to_localhost(self):
        """Verify CORS defaults are secure."""
        # Generate a valid SECRET_KEY for this test
        valid_key = secrets.token_urlsafe(32)
        
        with patch.dict(os.environ, {'SECRET_KEY': valid_key}, clear=True):
            import importlib
            import config
            importlib.reload(config)
            
            # Check default origins are localhost only
            assert 'localhost' in str(config.ALLOWED_ORIGINS).lower() or \
                   '127.0.0.1' in str(config.ALLOWED_ORIGINS)
            # Ensure no wildcard by default
            assert '*' not in config.ALLOWED_ORIGINS
    
    def test_api_host_defaults_secure(self):
        """Verify API binding is localhost by default."""
        valid_key = secrets.token_urlsafe(32)
        
        with patch.dict(os.environ, {'SECRET_KEY': valid_key}, clear=True):
            import importlib
            import config
            importlib.reload(config)
            
            assert config.API_HOST in ['127.0.0.1', 'localhost']


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================

class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_sql_injection_prevention(self):
        """Test SQL injection detection."""
        from pulse_core.input_validator import detect_sql_injection
        
        # Malicious inputs
        malicious_inputs = [
            "admin' OR '1'='1",
            "'; DROP TABLE users; --",
            "1 UNION SELECT * FROM passwords",
            "admin'--",
            "1' OR 1=1--"
        ]
        
        for malicious in malicious_inputs:
            assert detect_sql_injection(malicious), \
                f"Failed to detect SQL injection: {malicious}"
    
    def test_xss_detection(self):
        """Test XSS attack detection."""
        from pulse_core.input_validator import detect_xss
        
        # XSS payloads
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='javascript:alert(1)'></iframe>",
            "<body onload=alert('XSS')>"
        ]
        
        for payload in xss_payloads:
            assert detect_xss(payload), \
                f"Failed to detect XSS: {payload}"
    
    def test_input_sanitization(self):
        """Test dangerous input is sanitized."""
        from pulse_core.input_validator import sanitize_input
        
        dangerous_input = "<script>alert('XSS')</script>Hello World"
        sanitized = sanitize_input(dangerous_input)
        
        # Should remove script tags
        assert "<script>" not in sanitized
        assert "Hello World" in sanitized


# ============================================================================
# RATE LIMITING TESTS
# ============================================================================

class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limiter_tracks_requests(self):
        """Verify request tracking."""
        from pulse_core.security import RateLimiter
        
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        client_id = "test_client_123"
        
        # Should allow first 5 requests
        for i in range(5):
            assert limiter.is_allowed(client_id), \
                f"Request {i+1} should be allowed"
        
        # 6th request should be denied
        assert not limiter.is_allowed(client_id), \
            "Request should be rate limited"
    
    def test_rate_limiter_resets_window(self):
        """Verify time window reset."""
        from pulse_core.security import RateLimiter
        
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        client_id = "test_client_456"
        
        # Use up requests
        assert limiter.is_allowed(client_id)
        assert limiter.is_allowed(client_id)
        assert not limiter.is_allowed(client_id)
        
        # Wait for window to reset
        time.sleep(1.1)
        
        # Should be allowed again
        assert limiter.is_allowed(client_id)


# ============================================================================
# ENCRYPTION TESTS
# ============================================================================

class TestEncryption:
    """Test encryption functionality."""
    
    def test_password_hashing(self):
        """Verify secure password hashing."""
        from pulse_core.encryption import hash_password, verify_password
        
        password = "MySecureP@ssw0rd123"
        hashed = hash_password(password)
        
        # Hash should be different from password
        assert hashed != password
        
        # Should verify correctly
        assert verify_password(password, hashed)
        
        # Wrong password should fail
        assert not verify_password("WrongPassword", hashed)
    
    def test_secure_token_generation(self):
        """Test random token generation."""
        from pulse_core.security import generate_secure_token
        
        token1 = generate_secure_token()
        token2 = generate_secure_token()
        
        # Tokens should be unique
        assert token1 != token2
        
        # Should be of adequate length (at least 32 chars)
        assert len(token1) >= 32
        assert len(token2) >= 32


# ============================================================================
# SECURITY HEADERS TESTS
# ============================================================================

class TestSecurityHeaders:
    """Test security headers."""
    
    def test_security_headers_present(self):
        """Verify all required headers."""
        from pulse_core.security import SecurityConfig
        
        required_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection',
            'Strict-Transport-Security',
            'Content-Security-Policy'
        ]
        
        for header in required_headers:
            assert header in SecurityConfig.SECURITY_HEADERS, \
                f"Missing security header: {header}"
    
    def test_cors_headers_restrictive(self):
        """Ensure no wildcard CORS in production."""
        valid_key = secrets.token_urlsafe(32)
        
        # Test production mode
        with patch.dict(os.environ, {
            'SECRET_KEY': valid_key,
            'ENVIRONMENT': 'production',
            'ALLOWED_ORIGINS': '*'
        }):
            with pytest.raises(SystemExit):
                import importlib
                import config
                importlib.reload(config)


# ============================================================================
# CSRF PROTECTION TESTS
# ============================================================================

class TestCSRFProtection:
    """Test CSRF protection."""
    
    def test_csrf_token_generation(self):
        """Test CSRF token creation."""
        from pulse_core.csrf_protection import generate_csrf_token
        
        token1 = generate_csrf_token()
        token2 = generate_csrf_token()
        
        # Tokens should be unique
        assert token1 != token2
        
        # Should be adequate length
        assert len(token1) >= 32
    
    def test_csrf_token_validation(self):
        """Test token validation."""
        from pulse_core.csrf_protection import (
            generate_csrf_token,
            validate_csrf_token,
            store_csrf_token
        )
        
        session_id = "test_session_123"
        token = generate_csrf_token()
        
        # Store token
        store_csrf_token(session_id, token)
        
        # Should validate correctly
        assert validate_csrf_token(session_id, token)
        
        # Wrong token should fail
        assert not validate_csrf_token(session_id, "wrong_token")


# ============================================================================
# SESSION SECURITY TESTS
# ============================================================================

class TestSessionSecurity:
    """Test session security."""
    
    def test_session_creation(self):
        """Test secure session creation."""
        from pulse_core.security import create_session
        
        user_id = "user_123"
        session = create_session(user_id)
        
        assert 'session_id' in session
        assert 'user_id' in session
        assert session['user_id'] == user_id
        assert 'created_at' in session
    
    def test_session_expiration(self):
        """Test session timeout."""
        from pulse_core.security import create_session, is_session_valid
        
        user_id = "user_456"
        session = create_session(user_id, ttl_seconds=1)
        
        # Should be valid immediately
        assert is_session_valid(session)
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert not is_session_valid(session)


# ============================================================================
# API KEY SECURITY TESTS
# ============================================================================

class TestAPIKeySecurity:
    """Test API key security."""
    
    def test_api_key_generation(self):
        """Test secure API key generation."""
        from pulse_core.security import generate_api_key
        
        key1 = generate_api_key()
        key2 = generate_api_key()
        
        # Keys should be unique
        assert key1 != key2
        
        # Should start with a prefix for identification
        assert key1.startswith('pk_') or len(key1) >= 32
        assert key2.startswith('pk_') or len(key2) >= 32
    
    def test_sensitive_data_masking(self):
        """Test data masking in logs."""
        from pulse_core.security import mask_sensitive_data
        
        data = {
            'api_key': 'pk_test_1234567890abcdef',
            'password': 'MySecretPassword',
            'credit_card': '4111-1111-1111-1111',
            'safe_data': 'This is public'
        }
        
        masked = mask_sensitive_data(data)
        
        # Sensitive fields should be masked
        assert 'pk_test_1234567890abcdef' not in str(masked)
        assert 'MySecretPassword' not in str(masked)
        
        # Safe data should remain
        assert masked['safe_data'] == 'This is public'


# ============================================================================
# AUTHENTICATION TESTS
# ============================================================================

class TestAuthentication:
    """Test authentication functionality."""
    
    def test_password_strength_validation(self):
        """Test password requirements."""
        from pulse_core.security import validate_password_strength
        
        # Weak passwords should fail
        weak_passwords = [
            "short",
            "alllowercase",
            "ALLUPPERCASE",
            "NoNumbers!",
            "NoSpecial123"
        ]
        
        for weak in weak_passwords:
            result = validate_password_strength(weak)
            assert not result['valid'], \
                f"Weak password should fail: {weak}"
    
    def test_strong_password_accepted(self):
        """Verify strong passwords work."""
        from pulse_core.security import validate_password_strength
        
        strong_passwords = [
            "MyP@ssw0rd123!",
            "C0mpl3x!P@ssw0rd",
            "Str0ng&Secure#2024"
        ]
        
        for strong in strong_passwords:
            result = validate_password_strength(strong)
            assert result['valid'], \
                f"Strong password should pass: {strong}"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSecurityIntegration:
    """Integration tests for security features."""
    
    def test_full_security_stack(self):
        """Test multiple security features together."""
        from pulse_core.security import (
            RateLimiter,
            generate_secure_token,
            create_session
        )
        from pulse_core.input_validator import sanitize_input
        
        # Rate limiter
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        
        # Create session
        user_id = "test_user_789"
        session = create_session(user_id)
        
        # Generate token
        token = generate_secure_token()
        
        # Sanitize input
        unsafe_input = "<script>alert('test')</script>Safe text"
        safe_input = sanitize_input(unsafe_input)
        
        # All should work together
        assert limiter.is_allowed(user_id)
        assert session['user_id'] == user_id
        assert len(token) >= 32
        assert "<script>" not in safe_input


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
