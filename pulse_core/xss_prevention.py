"""
XSS Prevention Module
====================

Comprehensive Cross-Site Scripting (XSS) prevention module providing:
- XSS detection patterns and validation
- Multi-context sanitization (HTML, JavaScript, CSS, URL, JSON)
- Content Security Policy (CSP) builder
- Dangerous tag and attribute detection
- Rich text sanitization with allowlists
- FastAPI integration helpers

Author: jetgause
Created: 2025-12-10
Version: 1.0.0
"""

import re
import html
import json
import urllib.parse
from typing import Dict, List, Optional, Set, Union, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
import hashlib
import secrets


class SanitizationContext(Enum):
    """Enumeration of different sanitization contexts."""
    HTML = "html"
    HTML_ATTRIBUTE = "html_attribute"
    JAVASCRIPT = "javascript"
    CSS = "css"
    URL = "url"
    JSON = "json"
    RICH_TEXT = "rich_text"


@dataclass
class XSSPattern:
    """Represents an XSS attack pattern."""
    name: str
    pattern: re.Pattern
    severity: str  # 'high', 'medium', 'low'
    description: str


@dataclass
class SanitizationResult:
    """Result of a sanitization operation."""
    sanitized: str
    was_modified: bool
    detected_threats: List[str] = field(default_factory=list)
    context: Optional[SanitizationContext] = None


class XSSDetector:
    """Detects potential XSS attack patterns in user input."""
    
    # Comprehensive XSS detection patterns
    PATTERNS = [
        XSSPattern(
            name="script_tag",
            pattern=re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            severity="high",
            description="Script tag injection"
        ),
        XSSPattern(
            name="script_tag_unclosed",
            pattern=re.compile(r'<script[^>]*>', re.IGNORECASE),
            severity="high",
            description="Unclosed script tag"
        ),
        XSSPattern(
            name="javascript_protocol",
            pattern=re.compile(r'javascript\s*:', re.IGNORECASE),
            severity="high",
            description="JavaScript protocol in URL"
        ),
        XSSPattern(
            name="data_protocol",
            pattern=re.compile(r'data\s*:[^,]*,', re.IGNORECASE),
            severity="medium",
            description="Data protocol (potential XSS vector)"
        ),
        XSSPattern(
            name="vbscript_protocol",
            pattern=re.compile(r'vbscript\s*:', re.IGNORECASE),
            severity="high",
            description="VBScript protocol"
        ),
        XSSPattern(
            name="on_event_handler",
            pattern=re.compile(r'on\w+\s*=', re.IGNORECASE),
            severity="high",
            description="Event handler attribute (onclick, onerror, etc.)"
        ),
        XSSPattern(
            name="iframe_tag",
            pattern=re.compile(r'<iframe[^>]*>', re.IGNORECASE),
            severity="medium",
            description="Iframe tag injection"
        ),
        XSSPattern(
            name="object_tag",
            pattern=re.compile(r'<object[^>]*>', re.IGNORECASE),
            severity="medium",
            description="Object tag injection"
        ),
        XSSPattern(
            name="embed_tag",
            pattern=re.compile(r'<embed[^>]*>', re.IGNORECASE),
            severity="medium",
            description="Embed tag injection"
        ),
        XSSPattern(
            name="base_tag",
            pattern=re.compile(r'<base[^>]*>', re.IGNORECASE),
            severity="high",
            description="Base tag injection"
        ),
        XSSPattern(
            name="meta_tag",
            pattern=re.compile(r'<meta[^>]*>', re.IGNORECASE),
            severity="medium",
            description="Meta tag injection"
        ),
        XSSPattern(
            name="link_tag",
            pattern=re.compile(r'<link[^>]*>', re.IGNORECASE),
            severity="medium",
            description="Link tag injection"
        ),
        XSSPattern(
            name="style_tag",
            pattern=re.compile(r'<style[^>]*>.*?</style>', re.IGNORECASE | re.DOTALL),
            severity="medium",
            description="Style tag injection"
        ),
        XSSPattern(
            name="expression",
            pattern=re.compile(r'expression\s*\(', re.IGNORECASE),
            severity="high",
            description="CSS expression injection"
        ),
        XSSPattern(
            name="import_css",
            pattern=re.compile(r'@import', re.IGNORECASE),
            severity="medium",
            description="CSS import directive"
        ),
        XSSPattern(
            name="svg_tag",
            pattern=re.compile(r'<svg[^>]*>', re.IGNORECASE),
            severity="medium",
            description="SVG tag (can contain scripts)"
        ),
        XSSPattern(
            name="xml_entities",
            pattern=re.compile(r'&#?\w+;'),
            severity="low",
            description="XML/HTML entities (potential encoding attack)"
        ),
    ]
    
    @classmethod
    def detect(cls, text: str) -> List[XSSPattern]:
        """
        Detect XSS patterns in the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected XSS patterns
        """
        detected = []
        for pattern in cls.PATTERNS:
            if pattern.pattern.search(text):
                detected.append(pattern)
        return detected
    
    @classmethod
    def is_safe(cls, text: str, severity_threshold: str = "low") -> bool:
        """
        Check if text is safe based on severity threshold.
        
        Args:
            text: Text to check
            severity_threshold: Minimum severity to flag ('low', 'medium', 'high')
            
        Returns:
            True if safe, False otherwise
        """
        severity_levels = {"low": 0, "medium": 1, "high": 2}
        threshold = severity_levels.get(severity_threshold, 0)
        
        detected = cls.detect(text)
        for pattern in detected:
            if severity_levels.get(pattern.severity, 0) >= threshold:
                return False
        return True


class DangerousElements:
    """Defines dangerous HTML tags and attributes."""
    
    # Tags that should always be removed
    DANGEROUS_TAGS = {
        'script', 'iframe', 'object', 'embed', 'applet', 'meta', 'link',
        'base', 'form', 'input', 'button', 'textarea', 'select', 'option',
        'frameset', 'frame', 'bgsound', 'video', 'audio', 'canvas'
    }
    
    # Attributes that should always be removed
    DANGEROUS_ATTRIBUTES = {
        'onload', 'onerror', 'onclick', 'ondblclick', 'onmousedown',
        'onmouseup', 'onmouseover', 'onmousemove', 'onmouseout',
        'onmouseenter', 'onmouseleave', 'onchange', 'onsubmit',
        'onkeydown', 'onkeypress', 'onkeyup', 'onfocus', 'onblur',
        'oncontextmenu', 'oninput', 'oninvalid', 'onreset', 'onsearch',
        'onselect', 'onabort', 'oncanplay', 'oncanplaythrough',
        'oncuechange', 'ondurationchange', 'onemptied', 'onended',
        'onloadeddata', 'onloadedmetadata', 'onloadstart', 'onpause',
        'onplay', 'onplaying', 'onprogress', 'onratechange', 'onseeked',
        'onseeking', 'onstalled', 'onsuspend', 'ontimeupdate',
        'onvolumechange', 'onwaiting', 'ontoggle', 'onwheel',
        'oncopy', 'oncut', 'onpaste', 'onanimationstart',
        'onanimationend', 'onanimationiteration', 'ontransitionend',
        'formaction', 'action', 'data'
    }
    
    # Safe tags for rich text
    SAFE_TAGS = {
        'p', 'br', 'strong', 'em', 'u', 'strike', 'b', 'i',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote',
        'ul', 'ol', 'li', 'dl', 'dt', 'dd', 'table', 'thead',
        'tbody', 'tr', 'th', 'td', 'a', 'img', 'code', 'pre',
        'span', 'div', 'hr'
    }
    
    # Safe attributes for rich text
    SAFE_ATTRIBUTES = {
        'href', 'src', 'alt', 'title', 'class', 'id', 'name',
        'width', 'height', 'align', 'colspan', 'rowspan'
    }
    
    @classmethod
    def is_dangerous_tag(cls, tag: str) -> bool:
        """Check if a tag is dangerous."""
        return tag.lower() in cls.DANGEROUS_TAGS
    
    @classmethod
    def is_dangerous_attribute(cls, attr: str) -> bool:
        """Check if an attribute is dangerous."""
        return attr.lower() in cls.DANGEROUS_ATTRIBUTES
    
    @classmethod
    def is_safe_tag(cls, tag: str) -> bool:
        """Check if a tag is safe for rich text."""
        return tag.lower() in cls.SAFE_TAGS
    
    @classmethod
    def is_safe_attribute(cls, attr: str) -> bool:
        """Check if an attribute is safe for rich text."""
        return attr.lower() in cls.SAFE_ATTRIBUTES


class Sanitizer:
    """Multi-context sanitizer for preventing XSS attacks."""
    
    @staticmethod
    def sanitize_html(text: str, strip_tags: bool = True) -> SanitizationResult:
        """
        Sanitize HTML content.
        
        Args:
            text: HTML text to sanitize
            strip_tags: If True, remove all HTML tags; if False, escape them
            
        Returns:
            SanitizationResult with sanitized content
        """
        original = text
        detected = [p.name for p in XSSDetector.detect(text)]
        
        if strip_tags:
            # Remove all HTML tags
            sanitized = re.sub(r'<[^>]+>', '', text)
        else:
            # Escape HTML entities
            sanitized = html.escape(text, quote=True)
        
        return SanitizationResult(
            sanitized=sanitized,
            was_modified=(original != sanitized),
            detected_threats=detected,
            context=SanitizationContext.HTML
        )
    
    @staticmethod
    def sanitize_html_attribute(text: str) -> SanitizationResult:
        """
        Sanitize HTML attribute values.
        
        Args:
            text: Attribute value to sanitize
            
        Returns:
            SanitizationResult with sanitized content
        """
        original = text
        detected = [p.name for p in XSSDetector.detect(text)]
        
        # Escape quotes and special characters
        sanitized = html.escape(text, quote=True)
        # Remove any javascript: protocols
        sanitized = re.sub(r'javascript\s*:', '', sanitized, flags=re.IGNORECASE)
        # Remove any data: protocols
        sanitized = re.sub(r'data\s*:', '', sanitized, flags=re.IGNORECASE)
        
        return SanitizationResult(
            sanitized=sanitized,
            was_modified=(original != sanitized),
            detected_threats=detected,
            context=SanitizationContext.HTML_ATTRIBUTE
        )
    
    @staticmethod
    def sanitize_javascript(text: str) -> SanitizationResult:
        """
        Sanitize text for use in JavaScript context.
        
        Args:
            text: Text to sanitize
            
        Returns:
            SanitizationResult with sanitized content
        """
        original = text
        detected = [p.name for p in XSSDetector.detect(text)]
        
        # Encode as JSON string (handles escaping)
        sanitized = json.dumps(text)[1:-1]  # Remove outer quotes
        
        return SanitizationResult(
            sanitized=sanitized,
            was_modified=(original != sanitized),
            detected_threats=detected,
            context=SanitizationContext.JAVASCRIPT
        )
    
    @staticmethod
    def sanitize_css(text: str) -> SanitizationResult:
        """
        Sanitize CSS content.
        
        Args:
            text: CSS text to sanitize
            
        Returns:
            SanitizationResult with sanitized content
        """
        original = text
        detected = [p.name for p in XSSDetector.detect(text)]
        
        # Remove dangerous CSS constructs
        sanitized = re.sub(r'expression\s*\(.*?\)', '', text, flags=re.IGNORECASE)
        sanitized = re.sub(r'@import', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'javascript\s*:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'behaviour\s*:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'-moz-binding\s*:', '', sanitized, flags=re.IGNORECASE)
        
        return SanitizationResult(
            sanitized=sanitized,
            was_modified=(original != sanitized),
            detected_threats=detected,
            context=SanitizationContext.CSS
        )
    
    @staticmethod
    def sanitize_url(url: str, allowed_schemes: Optional[Set[str]] = None) -> SanitizationResult:
        """
        Sanitize URL.
        
        Args:
            url: URL to sanitize
            allowed_schemes: Set of allowed URL schemes (default: http, https)
            
        Returns:
            SanitizationResult with sanitized content
        """
        if allowed_schemes is None:
            allowed_schemes = {'http', 'https', 'mailto', 'ftp'}
        
        original = url
        detected = [p.name for p in XSSDetector.detect(url)]
        
        try:
            parsed = urllib.parse.urlparse(url)
            
            # Check if scheme is allowed
            if parsed.scheme and parsed.scheme.lower() not in allowed_schemes:
                sanitized = ""
            else:
                # Reconstruct URL safely
                sanitized = urllib.parse.urlunparse(parsed)
        except Exception:
            # If parsing fails, return empty string
            sanitized = ""
            detected.append("invalid_url")
        
        return SanitizationResult(
            sanitized=sanitized,
            was_modified=(original != sanitized),
            detected_threats=detected,
            context=SanitizationContext.URL
        )
    
    @staticmethod
    def sanitize_json(data: Any) -> SanitizationResult:
        """
        Sanitize data for JSON output.
        
        Args:
            data: Data to sanitize
            
        Returns:
            SanitizationResult with sanitized JSON string
        """
        try:
            # Encode to JSON safely
            sanitized = json.dumps(data, ensure_ascii=True)
            original_str = str(data)
            detected = []
            
            return SanitizationResult(
                sanitized=sanitized,
                was_modified=(original_str != sanitized),
                detected_threats=detected,
                context=SanitizationContext.JSON
            )
        except Exception as e:
            return SanitizationResult(
                sanitized="{}",
                was_modified=True,
                detected_threats=["json_encoding_error"],
                context=SanitizationContext.JSON
            )


class RichTextSanitizer:
    """Sanitizer for rich text that preserves safe HTML tags."""
    
    def __init__(
        self,
        allowed_tags: Optional[Set[str]] = None,
        allowed_attributes: Optional[Set[str]] = None
    ):
        """
        Initialize rich text sanitizer.
        
        Args:
            allowed_tags: Set of allowed HTML tags
            allowed_attributes: Set of allowed HTML attributes
        """
        self.allowed_tags = allowed_tags or DangerousElements.SAFE_TAGS
        self.allowed_attributes = allowed_attributes or DangerousElements.SAFE_ATTRIBUTES
    
    def sanitize(self, html_content: str) -> SanitizationResult:
        """
        Sanitize rich text HTML content.
        
        Args:
            html_content: HTML content to sanitize
            
        Returns:
            SanitizationResult with sanitized content
        """
        original = html_content
        detected = [p.name for p in XSSDetector.detect(html_content)]
        
        # Remove dangerous tags
        sanitized = self._remove_dangerous_tags(html_content)
        
        # Remove dangerous attributes
        sanitized = self._remove_dangerous_attributes(sanitized)
        
        # Sanitize URLs in href and src attributes
        sanitized = self._sanitize_urls(sanitized)
        
        return SanitizationResult(
            sanitized=sanitized,
            was_modified=(original != sanitized),
            detected_threats=detected,
            context=SanitizationContext.RICH_TEXT
        )
    
    def _remove_dangerous_tags(self, html: str) -> str:
        """Remove dangerous HTML tags."""
        # Remove dangerous tags and their content
        for tag in DangerousElements.DANGEROUS_TAGS:
            # Remove opening and closing tags with content
            html = re.sub(
                f'<{tag}[^>]*>.*?</{tag}>',
                '',
                html,
                flags=re.IGNORECASE | re.DOTALL
            )
            # Remove self-closing tags
            html = re.sub(
                f'<{tag}[^>]*/>',
                '',
                html,
                flags=re.IGNORECASE
            )
        
        # Remove tags not in allowed list
        def replace_tag(match):
            tag = match.group(1).lower().split()[0]
            if tag.startswith('/'):
                tag = tag[1:]
            if tag in self.allowed_tags:
                return match.group(0)
            return ''
        
        html = re.sub(r'<(/?\w+)([^>]*)>', replace_tag, html)
        return html
    
    def _remove_dangerous_attributes(self, html: str) -> str:
        """Remove dangerous HTML attributes."""
        def clean_tag(match):
            tag = match.group(1)
            attrs = match.group(2)
            
            if not attrs:
                return f'<{tag}>'
            
            # Parse and filter attributes
            safe_attrs = []
            attr_pattern = re.compile(r'(\w+)\s*=\s*["\']([^"\']*)["\']')
            
            for attr_match in attr_pattern.finditer(attrs):
                attr_name = attr_match.group(1).lower()
                attr_value = attr_match.group(2)
                
                if attr_name in self.allowed_attributes:
                    # Escape attribute value
                    attr_value = html.escape(attr_value, quote=True)
                    safe_attrs.append(f'{attr_name}="{attr_value}"')
            
            if safe_attrs:
                return f'<{tag} {" ".join(safe_attrs)}>'
            return f'<{tag}>'
        
        html = re.sub(r'<(\w+)([^>]*)>', clean_tag, html)
        return html
    
    def _sanitize_urls(self, html: str) -> str:
        """Sanitize URLs in href and src attributes."""
        def sanitize_url_attr(match):
            attr_name = match.group(1)
            url = match.group(2)
            
            result = Sanitizer.sanitize_url(url)
            if result.sanitized:
                return f'{attr_name}="{result.sanitized}"'
            return ''
        
        html = re.sub(
            r'(href|src)\s*=\s*["\']([^"\']*)["\']',
            sanitize_url_attr,
            html,
            flags=re.IGNORECASE
        )
        return html


@dataclass
class CSPDirective:
    """Content Security Policy directive."""
    name: str
    values: List[str]
    
    def __str__(self) -> str:
        """Convert directive to CSP string format."""
        return f"{self.name} {' '.join(self.values)}"


class CSPBuilder:
    """Content Security Policy builder."""
    
    def __init__(self):
        """Initialize CSP builder with default secure directives."""
        self.directives: Dict[str, CSPDirective] = {}
        self.nonces: Dict[str, str] = {}
        
    def default_strict(self) -> 'CSPBuilder':
        """Set strict default CSP directives."""
        self.add_directive("default-src", ["'self'"])
        self.add_directive("script-src", ["'self'"])
        self.add_directive("style-src", ["'self'"])
        self.add_directive("img-src", ["'self'", "data:", "https:"])
        self.add_directive("font-src", ["'self'"])
        self.add_directive("connect-src", ["'self'"])
        self.add_directive("frame-ancestors", ["'none'"])
        self.add_directive("base-uri", ["'self'"])
        self.add_directive("form-action", ["'self'"])
        return self
    
    def add_directive(self, name: str, values: List[str]) -> 'CSPBuilder':
        """
        Add or update a CSP directive.
        
        Args:
            name: Directive name (e.g., 'script-src')
            values: List of directive values
            
        Returns:
            Self for chaining
        """
        if name in self.directives:
            self.directives[name].values.extend(values)
        else:
            self.directives[name] = CSPDirective(name, values)
        return self
    
    def allow_inline_scripts(self, use_nonce: bool = True) -> 'CSPBuilder':
        """
        Allow inline scripts with nonce or unsafe-inline.
        
        Args:
            use_nonce: If True, use nonce; if False, use 'unsafe-inline'
            
        Returns:
            Self for chaining
        """
        if use_nonce:
            nonce = self._generate_nonce()
            self.nonces['script'] = nonce
            self.add_directive("script-src", [f"'nonce-{nonce}'"])
        else:
            self.add_directive("script-src", ["'unsafe-inline'"])
        return self
    
    def allow_inline_styles(self, use_nonce: bool = True) -> 'CSPBuilder':
        """
        Allow inline styles with nonce or unsafe-inline.
        
        Args:
            use_nonce: If True, use nonce; if False, use 'unsafe-inline'
            
        Returns:
            Self for chaining
        """
        if use_nonce:
            nonce = self._generate_nonce()
            self.nonces['style'] = nonce
            self.add_directive("style-src", [f"'nonce-{nonce}'"])
        else:
            self.add_directive("style-src", ["'unsafe-inline'"])
        return self
    
    def allow_eval(self) -> 'CSPBuilder':
        """Allow eval() in scripts (not recommended)."""
        self.add_directive("script-src", ["'unsafe-eval'"])
        return self
    
    def allow_domain(self, directive: str, domain: str) -> 'CSPBuilder':
        """
        Allow specific domain for a directive.
        
        Args:
            directive: CSP directive name
            domain: Domain to allow
            
        Returns:
            Self for chaining
        """
        self.add_directive(directive, [domain])
        return self
    
    def upgrade_insecure_requests(self) -> 'CSPBuilder':
        """Enable upgrade-insecure-requests directive."""
        self.directives["upgrade-insecure-requests"] = CSPDirective(
            "upgrade-insecure-requests", []
        )
        return self
    
    def block_all_mixed_content(self) -> 'CSPBuilder':
        """Enable block-all-mixed-content directive."""
        self.directives["block-all-mixed-content"] = CSPDirective(
            "block-all-mixed-content", []
        )
        return self
    
    def _generate_nonce(self) -> str:
        """Generate a cryptographically secure nonce."""
        random_bytes = secrets.token_bytes(16)
        return hashlib.sha256(random_bytes).hexdigest()[:32]
    
    def get_nonce(self, nonce_type: str) -> Optional[str]:
        """
        Get generated nonce for script or style.
        
        Args:
            nonce_type: 'script' or 'style'
            
        Returns:
            Nonce string or None
        """
        return self.nonces.get(nonce_type)
    
    def build(self) -> str:
        """
        Build CSP header value.
        
        Returns:
            CSP header string
        """
        directives_str = []
        for directive in self.directives.values():
            if directive.values:
                directives_str.append(str(directive))
            else:
                directives_str.append(directive.name)
        return "; ".join(directives_str)
    
    def build_header(self) -> Dict[str, str]:
        """
        Build CSP as HTTP header dictionary.
        
        Returns:
            Dictionary with Content-Security-Policy header
        """
        return {"Content-Security-Policy": self.build()}


# FastAPI Integration Helpers

try:
    from fastapi import Request, Response
    from fastapi.responses import JSONResponse
    from functools import wraps
    
    class XSSProtection:
        """FastAPI middleware and decorators for XSS protection."""
        
        @staticmethod
        def sanitize_input(
            context: SanitizationContext = SanitizationContext.HTML
        ) -> Callable:
            """
            Decorator to sanitize request input.
            
            Args:
                context: Sanitization context to use
                
            Returns:
                Decorator function
            """
            def decorator(func: Callable) -> Callable:
                @wraps(func)
                async def wrapper(*args, **kwargs):
                    # Sanitize string arguments
                    sanitized_kwargs = {}
                    for key, value in kwargs.items():
                        if isinstance(value, str):
                            if context == SanitizationContext.HTML:
                                result = Sanitizer.sanitize_html(value)
                            elif context == SanitizationContext.URL:
                                result = Sanitizer.sanitize_url(value)
                            elif context == SanitizationContext.JAVASCRIPT:
                                result = Sanitizer.sanitize_javascript(value)
                            else:
                                result = Sanitizer.sanitize_html(value)
                            sanitized_kwargs[key] = result.sanitized
                        else:
                            sanitized_kwargs[key] = value
                    
                    return await func(*args, **sanitized_kwargs)
                return wrapper
            return decorator
        
        @staticmethod
        def add_security_headers(
            response: Response,
            csp: Optional[CSPBuilder] = None
        ) -> Response:
            """
            Add security headers to response.
            
            Args:
                response: FastAPI response object
                csp: Optional CSP builder
                
            Returns:
                Response with security headers
            """
            # Add XSS protection headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            
            # Add CSP if provided
            if csp:
                response.headers.update(csp.build_header())
            
            return response
        
        @staticmethod
        async def csp_middleware(request: Request, call_next: Callable) -> Response:
            """
            Middleware to add CSP headers to all responses.
            
            Args:
                request: FastAPI request
                call_next: Next middleware/handler
                
            Returns:
                Response with CSP headers
            """
            response = await call_next(request)
            
            # Build strict CSP
            csp = CSPBuilder().default_strict()
            response.headers.update(csp.build_header())
            
            return response
        
        @staticmethod
        def validate_input(
            severity_threshold: str = "medium"
        ) -> Callable:
            """
            Decorator to validate input for XSS patterns.
            
            Args:
                severity_threshold: Minimum severity to reject
                
            Returns:
                Decorator function
            """
            def decorator(func: Callable) -> Callable:
                @wraps(func)
                async def wrapper(*args, **kwargs):
                    # Check string arguments for XSS
                    for key, value in kwargs.items():
                        if isinstance(value, str):
                            if not XSSDetector.is_safe(value, severity_threshold):
                                return JSONResponse(
                                    status_code=400,
                                    content={
                                        "error": "Invalid input detected",
                                        "message": f"Parameter '{key}' contains potentially malicious content"
                                    }
                                )
                    
                    return await func(*args, **kwargs)
                return wrapper
            return decorator

except ImportError:
    # FastAPI not installed, skip integration helpers
    pass


# Utility Functions

def quick_sanitize(
    text: str,
    context: SanitizationContext = SanitizationContext.HTML
) -> str:
    """
    Quick sanitization utility function.
    
    Args:
        text: Text to sanitize
        context: Sanitization context
        
    Returns:
        Sanitized text
    """
    if context == SanitizationContext.HTML:
        return Sanitizer.sanitize_html(text).sanitized
    elif context == SanitizationContext.HTML_ATTRIBUTE:
        return Sanitizer.sanitize_html_attribute(text).sanitized
    elif context == SanitizationContext.JAVASCRIPT:
        return Sanitizer.sanitize_javascript(text).sanitized
    elif context == SanitizationContext.CSS:
        return Sanitizer.sanitize_css(text).sanitized
    elif context == SanitizationContext.URL:
        return Sanitizer.sanitize_url(text).sanitized
    elif context == SanitizationContext.JSON:
        return Sanitizer.sanitize_json(text).sanitized
    elif context == SanitizationContext.RICH_TEXT:
        return RichTextSanitizer().sanitize(text).sanitized
    return text


def is_xss_safe(text: str, severity: str = "low") -> bool:
    """
    Quick check if text is XSS-safe.
    
    Args:
        text: Text to check
        severity: Severity threshold
        
    Returns:
        True if safe, False otherwise
    """
    return XSSDetector.is_safe(text, severity)


def build_csp_header(strict: bool = True) -> Dict[str, str]:
    """
    Build a CSP header dictionary.
    
    Args:
        strict: If True, use strict CSP; if False, use relaxed CSP
        
    Returns:
        Dictionary with CSP header
    """
    builder = CSPBuilder()
    if strict:
        builder.default_strict()
    else:
        builder.add_directive("default-src", ["'self'", "'unsafe-inline'"])
    
    return builder.build_header()


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Detect XSS patterns
    print("=== XSS Detection ===")
    malicious_input = '<script>alert("XSS")</script>'
    patterns = XSSDetector.detect(malicious_input)
    print(f"Input: {malicious_input}")
    print(f"Detected patterns: {[p.name for p in patterns]}")
    print(f"Is safe: {XSSDetector.is_safe(malicious_input)}\n")
    
    # Example 2: Sanitize HTML
    print("=== HTML Sanitization ===")
    html_input = '<p onclick="alert()">Hello <script>alert("XSS")</script></p>'
    result = Sanitizer.sanitize_html(html_input)
    print(f"Original: {html_input}")
    print(f"Sanitized: {result.sanitized}")
    print(f"Threats: {result.detected_threats}\n")
    
    # Example 3: Sanitize URL
    print("=== URL Sanitization ===")
    url_input = 'javascript:alert("XSS")'
    result = Sanitizer.sanitize_url(url_input)
    print(f"Original: {url_input}")
    print(f"Sanitized: {result.sanitized}")
    print(f"Threats: {result.detected_threats}\n")
    
    # Example 4: Rich text sanitization
    print("=== Rich Text Sanitization ===")
    rich_text = '<p>Safe text</p><script>alert("XSS")</script><a href="javascript:void(0)">Link</a>'
    sanitizer = RichTextSanitizer()
    result = sanitizer.sanitize(rich_text)
    print(f"Original: {rich_text}")
    print(f"Sanitized: {result.sanitized}\n")
    
    # Example 5: Build CSP
    print("=== Content Security Policy ===")
    csp = (CSPBuilder()
           .default_strict()
           .allow_inline_scripts(use_nonce=True)
           .allow_domain("img-src", "https://cdn.example.com")
           .upgrade_insecure_requests())
    print(f"CSP Header: {csp.build()}")
    print(f"Script nonce: {csp.get_nonce('script')}\n")
    
    # Example 6: Quick utilities
    print("=== Quick Utilities ===")
    text = '<img src="x" onerror="alert(1)">'
    sanitized = quick_sanitize(text)
    print(f"Quick sanitize: {sanitized}")
    print(f"Is XSS safe: {is_xss_safe(text)}")
