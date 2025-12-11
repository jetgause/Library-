"""
TLS/SSL Certificate Management System - Security Patch 18/23
=============================================================

Complete certificate management with Let's Encrypt integration,
automatic renewal, and HTTPS enforcement.

Author: jetgause
Date: 2025-12-11
Version: 1.0.0
"""

import ssl
import socket
import logging
import subprocess
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CertificateInfo:
    """Certificate information dataclass."""
    common_name: str
    issuer: str
    valid_from: datetime
    valid_to: datetime
    days_until_expiry: int
    serial_number: str
    fingerprint: str
    is_self_signed: bool
    subject_alt_names: List[str]

    def is_expired(self) -> bool:
        return self.days_until_expiry < 0

    def is_expiring_soon(self, threshold_days: int = 30) -> bool:
        return 0 <= self.days_until_expiry <= threshold_days


class TLSManager:
    """Comprehensive TLS/SSL Certificate Management System."""

    SECURE_CIPHERS = ':'.join([
        'ECDHE-ECDSA-AES256-GCM-SHA384',
        'ECDHE-RSA-AES256-GCM-SHA384',
        'ECDHE-ECDSA-CHACHA20-POLY1305',
        'ECDHE-RSA-CHACHA20-POLY1305',
        'ECDHE-ECDSA-AES128-GCM-SHA256',
        'ECDHE-RSA-AES128-GCM-SHA256',
        '!aNULL', '!eNULL', '!EXPORT', '!DES', '!RC4', '!MD5', '!PSK', '!SRP', '!CAMELLIA'
    ])

    def __init__(self, cert_dir: str = "/etc/letsencrypt/live"):
        self.cert_dir = Path(cert_dir)
        self.cert_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TLSManager initialized with cert_dir: {self.cert_dir}")

    def validate_certificate(self, cert_path: str, key_path: str) -> Tuple[bool, str]:
        """Validate certificate and key pair."""
        try:
            cert_path = Path(cert_path)
            key_path = Path(key_path)

            if not cert_path.exists():
                return False, f"Certificate file not found: {cert_path}"
            if not key_path.exists():
                return False, f"Key file not found: {key_path}"

            with open(cert_path, 'rb') as f:
                cert_data = f.read()
                cert = x509.load_pem_x509_certificate(cert_data, default_backend())

            with open(key_path, 'rb') as f:
                key_data = f.read()
                key = serialization.load_pem_private_key(key_data, password=None, backend=default_backend())

            now = datetime.utcnow()
            if cert.not_valid_before > now:
                return False, f"Certificate not yet valid. Valid from: {cert.not_valid_before}"
            if cert.not_valid_after < now:
                return False, f"Certificate expired on: {cert.not_valid_after}"

            cert_public_key = cert.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            key_public_key = key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            if cert_public_key != key_public_key:
                return False, "Private key does not match certificate"

            days_remaining = (cert.not_valid_after - now).days
            logger.info(f"Certificate valid. Expires in {days_remaining} days")
            
            return True, f"Certificate valid. Expires in {days_remaining} days"

        except Exception as e:
            logger.error(f"Certificate validation error: {str(e)}")
            return False, f"Validation error: {str(e)}"

    def generate_self_signed_cert(self, domain: str, days_valid: int = 365) -> Tuple[str, str]:
        """Generate self-signed certificate for development."""
        try:
            logger.info(f"Generating self-signed certificate for {domain}")
            
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"State"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, u"City"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Organization"),
                x509.NameAttribute(NameOID.COMMON_NAME, domain),
            ])
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=days_valid)
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(domain),
                    x509.DNSName(f"*.{domain}"),
                ]),
                critical=False,
            ).sign(private_key, hashes.SHA256(), default_backend())
            
            cert_path = self.cert_dir / f"{domain}.crt"
            key_path = self.cert_dir / f"{domain}.key"
            
            with open(cert_path, 'wb') as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            
            with open(key_path, 'wb') as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            key_path.chmod(0o600)
            cert_path.chmod(0o644)
            
            logger.info(f"Self-signed certificate generated: {cert_path}")
            return str(cert_path), str(key_path)

        except Exception as e:
            logger.error(f"Failed to generate self-signed certificate: {str(e)}")
            raise

    def request_lets_encrypt_cert(self, domain: str, email: str, webroot: str) -> bool:
        """Request Let's Encrypt certificate using certbot."""
        try:
            logger.info(f"Requesting Let's Encrypt certificate for {domain}")
            
            cmd = [
                'certbot', 'certonly',
                '--webroot',
                '-w', webroot,
                '-d', domain,
                '--email', email,
                '--agree-tos',
                '--non-interactive',
                '--expand'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"Successfully obtained Let's Encrypt certificate for {domain}")
                return True
            else:
                logger.error(f"Certbot failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Failed to request Let's Encrypt certificate: {str(e)}")
            return False

    def setup_auto_renewal(self, cron_schedule: str = "0 0,12 * * *") -> bool:
        """Setup automatic certificate renewal with cron."""
        try:
            logger.info("Setting up automatic certificate renewal")
            
            renewal_script = Path("/usr/local/bin/renew-certificates.sh")
            script_content = """#!/bin/bash
LOG_FILE="/var/log/cert-renewal.log"
echo "$(date): Starting certificate renewal" >> "$LOG_FILE"
certbot renew --quiet --deploy-hook "systemctl reload nginx" >> "$LOG_FILE" 2>&1
if [ $? -eq 0 ]; then
    echo "$(date): Certificate renewal successful" >> "$LOG_FILE"
else
    echo "$(date): Certificate renewal failed" >> "$LOG_FILE"
    exit 1
fi
"""
            
            with open(renewal_script, 'w') as f:
                f.write(script_content)
            
            renewal_script.chmod(0o755)
            
            try:
                result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
                existing_crontab = result.stdout
            except subprocess.CalledProcessError:
                existing_crontab = ""
            
            if str(renewal_script) not in existing_crontab:
                new_crontab = existing_crontab + f"{cron_schedule} {renewal_script}\n"
                subprocess.run(['crontab', '-'], input=new_crontab, text=True, check=True)
                logger.info("Auto-renewal cron job installed")
            else:
                logger.info("Auto-renewal cron job already exists")
            
            return True

        except Exception as e:
            logger.error(f"Failed to setup auto-renewal: {str(e)}")
            return False

    def get_ssl_context(self, cert_path: str, key_path: str) -> ssl.SSLContext:
        """Create secure SSL context with TLS 1.2+ enforcement."""
        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            
            context.options |= ssl.OP_NO_SSLv2
            context.options |= ssl.OP_NO_SSLv3
            context.options |= ssl.OP_NO_TLSv1
            context.options |= ssl.OP_NO_TLSv1_1
            
            context.set_ciphers(self.SECURE_CIPHERS)
            context.load_cert_chain(cert_path, key_path)
            
            context.options |= ssl.OP_CIPHER_SERVER_PREFERENCE
            context.options |= ssl.OP_SINGLE_DH_USE
            context.options |= ssl.OP_SINGLE_ECDH_USE
            
            logger.info("Secure SSL context created")
            return context

        except Exception as e:
            logger.error(f"Failed to create SSL context: {str(e)}")
            raise


class HTTPSRedirectMiddleware:
    """ASGI middleware for automatic HTTP to HTTPS redirection."""

    def __init__(self, app, hsts_max_age: int = 31536000):
        self.app = app
        self.hsts_header = f"max-age={hsts_max_age}; includeSubDomains; preload"

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http' and scope.get('scheme') == 'http':
            host = dict(scope['headers']).get(b'host', b'localhost').decode()
            path = scope.get('path', '/')
            query = scope.get('query_string', b'').decode()
            
            redirect_url = f"https://{host}{path}"
            if query:
                redirect_url += f"?{query}"
            
            await send({
                'type': 'http.response.start',
                'status': 301,
                'headers': [(b'location', redirect_url.encode()), (b'content-length', b'0')],
            })
            await send({'type': 'http.response.body', 'body': b''})
            return
        
        async def send_with_hsts(message):
            if message['type'] == 'http.response.start':
                headers = list(message.get('headers', []))
                headers.append((b'strict-transport-security', self.hsts_header.encode()))
                message['headers'] = headers
            await send(message)
        
        await self.app(scope, receive, send_with_hsts)


if __name__ == "__main__":
    tls_manager = TLSManager(cert_dir="/etc/ssl/certs")
    cert_path, key_path = tls_manager.generate_self_signed_cert("example.com", days_valid=365)
    print(f"Generated certificate: {cert_path}")
    is_valid, message = tls_manager.validate_certificate(cert_path, key_path)
    print(f"Certificate validation: {message}")
