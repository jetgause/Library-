"""
Comprehensive JWT Authentication System
========================================
Features:
- JWT token-based authentication
- Password hashing with bcrypt
- Role-based access control (Admin, User, Guest)
- Session management with login attempt tracking
- Account lockout after 5 failed attempts (15-minute duration)
- Password strength validation
- FastAPI security dependencies

Created: 2025-12-10
Author: jetgause
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union
from enum import Enum
import re
import secrets

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, validator, Field
import bcrypt


# ========================
# Configuration
# ========================

SECRET_KEY = secrets.token_urlsafe(32)  # Generate secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 15


# ========================
# Password Context
# ========================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ========================
# Enums
# ========================

class UserRole(str, Enum):
    """User role enumeration for RBAC"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class TokenType(str, Enum):
    """Token type enumeration"""
    ACCESS = "access"
    REFRESH = "refresh"


# ========================
# Data Models
# ========================

class User(BaseModel):
    """User model with authentication attributes"""
    id: str
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    hashed_password: str
    role: UserRole = UserRole.USER
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    class Config:
        use_enum_values = True


class UserCreate(BaseModel):
    """User creation model with password validation"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = Field(None, max_length=100)
    password: str = Field(..., min_length=8, max_length=128)
    role: UserRole = UserRole.USER
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format"""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username must contain only letters, numbers, underscores, and hyphens')
        return v
    
    @validator('password')
    def validate_password_strength(cls, v):
        """Validate password strength requirements"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        
        return v


class UserUpdate(BaseModel):
    """User update model"""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserInDB(User):
    """User model for database storage"""
    pass


class Token(BaseModel):
    """JWT token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token payload data"""
    username: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[UserRole] = None
    token_type: TokenType = TokenType.ACCESS


class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str


class LoginSession(BaseModel):
    """Login session tracking model"""
    user_id: str
    username: str
    failed_attempts: int = 0
    last_failed_attempt: Optional[datetime] = None
    locked_until: Optional[datetime] = None
    last_success: Optional[datetime] = None
    ip_address: Optional[str] = None


# ========================
# Session Management
# ========================

class SessionManager:
    """Manages user login sessions and lockout logic"""
    
    def __init__(self):
        self.sessions: Dict[str, LoginSession] = {}
    
    def get_session(self, username: str) -> LoginSession:
        """Get or create session for username"""
        if username not in self.sessions:
            self.sessions[username] = LoginSession(
                user_id="",  # Will be set on successful login
                username=username
            )
        return self.sessions[username]
    
    def is_locked(self, username: str) -> bool:
        """Check if account is locked"""
        session = self.get_session(username)
        if session.locked_until:
            if datetime.utcnow() < session.locked_until:
                return True
            else:
                # Lockout expired, reset
                self.reset_session(username)
        return False
    
    def record_failed_attempt(self, username: str, ip_address: Optional[str] = None):
        """Record a failed login attempt"""
        session = self.get_session(username)
        session.failed_attempts += 1
        session.last_failed_attempt = datetime.utcnow()
        session.ip_address = ip_address
        
        if session.failed_attempts >= MAX_LOGIN_ATTEMPTS:
            session.locked_until = datetime.utcnow() + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
    
    def record_success(self, username: str, user_id: str):
        """Record a successful login"""
        session = self.get_session(username)
        session.user_id = user_id
        session.failed_attempts = 0
        session.last_failed_attempt = None
        session.locked_until = None
        session.last_success = datetime.utcnow()
    
    def reset_session(self, username: str):
        """Reset session for username"""
        if username in self.sessions:
            del self.sessions[username]
    
    def get_lockout_time_remaining(self, username: str) -> Optional[int]:
        """Get remaining lockout time in seconds"""
        session = self.get_session(username)
        if session.locked_until:
            remaining = (session.locked_until - datetime.utcnow()).total_seconds()
            return max(0, int(remaining))
        return None


# Global session manager instance
session_manager = SessionManager()


# ========================
# Password Utilities
# ========================

class PasswordManager:
    """Password hashing and verification using bcrypt"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    
    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, List[str]]:
        """
        Validate password strength and return validation status and errors
        
        Returns:
            tuple: (is_valid, error_messages)
        """
        errors = []
        
        if len(password) < 8:
            errors.append('Password must be at least 8 characters long')
        
        if not re.search(r'[A-Z]', password):
            errors.append('Password must contain at least one uppercase letter')
        
        if not re.search(r'[a-z]', password):
            errors.append('Password must contain at least one lowercase letter')
        
        if not re.search(r'\d', password):
            errors.append('Password must contain at least one digit')
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append('Password must contain at least one special character')
        
        return (len(errors) == 0, errors)


password_manager = PasswordManager()


# ========================
# JWT Token Management
# ========================

class TokenManager:
    """JWT token creation and validation"""
    
    @staticmethod
    def create_access_token(
        data: dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "token_type": TokenType.ACCESS.value
        })
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(data: dict) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "token_type": TokenType.REFRESH.value
        })
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def create_token_pair(user: User) -> Token:
        """Create access and refresh token pair"""
        token_data = {
            "sub": user.username,
            "user_id": user.id,
            "role": user.role
        }
        
        access_token = TokenManager.create_access_token(token_data)
        refresh_token = TokenManager.create_refresh_token(token_data)
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    @staticmethod
    def decode_token(token: str) -> TokenData:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            role: str = payload.get("role")
            token_type: str = payload.get("token_type")
            
            if username is None or user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload"
                )
            
            return TokenData(
                username=username,
                user_id=user_id,
                role=role,
                token_type=token_type
            )
        
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Could not validate credentials: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"}
            )


token_manager = TokenManager()


# ========================
# In-Memory User Store (Replace with Database)
# ========================

class UserStore:
    """In-memory user storage - Replace with actual database"""
    
    def __init__(self):
        self.users: Dict[str, UserInDB] = {}
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin_user = UserInDB(
            id="admin-001",
            username="admin",
            email="admin@example.com",
            full_name="System Administrator",
            hashed_password=password_manager.hash_password("Admin@123"),
            role=UserRole.ADMIN,
            is_active=True,
            is_verified=True
        )
        self.users[admin_user.username] = admin_user
    
    def get_user(self, username: str) -> Optional[UserInDB]:
        """Get user by username"""
        return self.users.get(username)
    
    def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """Get user by ID"""
        for user in self.users.values():
            if user.id == user_id:
                return user
        return None
    
    def create_user(self, user_create: UserCreate) -> UserInDB:
        """Create new user"""
        if user_create.username in self.users:
            raise ValueError("Username already exists")
        
        user_id = f"user-{secrets.token_urlsafe(8)}"
        hashed_password = password_manager.hash_password(user_create.password)
        
        user = UserInDB(
            id=user_id,
            username=user_create.username,
            email=user_create.email,
            full_name=user_create.full_name,
            hashed_password=hashed_password,
            role=user_create.role
        )
        
        self.users[user.username] = user
        return user
    
    def update_user(self, username: str, user_update: UserUpdate) -> Optional[UserInDB]:
        """Update existing user"""
        user = self.get_user(username)
        if not user:
            return None
        
        update_data = user_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)
        
        user.updated_at = datetime.utcnow()
        return user
    
    def delete_user(self, username: str) -> bool:
        """Delete user"""
        if username in self.users:
            del self.users[username]
            return True
        return False


user_store = UserStore()


# ========================
# Authentication Functions
# ========================

def authenticate_user(username: str, password: str, ip_address: Optional[str] = None) -> Optional[UserInDB]:
    """
    Authenticate user with username and password
    Includes lockout logic for failed attempts
    """
    # Check if account is locked
    if session_manager.is_locked(username):
        remaining = session_manager.get_lockout_time_remaining(username)
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=f"Account locked due to multiple failed login attempts. Try again in {remaining} seconds."
        )
    
    user = user_store.get_user(username)
    
    # User not found or password incorrect
    if not user or not password_manager.verify_password(password, user.hashed_password):
        session_manager.record_failed_attempt(username, ip_address)
        
        # Check if this attempt triggered a lockout
        if session_manager.is_locked(username):
            remaining = session_manager.get_lockout_time_remaining(username)
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail=f"Account locked due to multiple failed login attempts. Try again in {remaining} seconds."
            )
        
        session = session_manager.get_session(username)
        attempts_left = MAX_LOGIN_ATTEMPTS - session.failed_attempts
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Incorrect username or password. {attempts_left} attempts remaining."
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    # Successful authentication
    session_manager.record_success(username, user.id)
    user.last_login = datetime.utcnow()
    
    return user


# ========================
# FastAPI Security Dependencies
# ========================

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> UserInDB:
    """
    FastAPI dependency to get current authenticated user
    """
    token = credentials.credentials
    token_data = token_manager.decode_token(token)
    
    # Verify it's an access token
    if token_data.token_type != TokenType.ACCESS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type"
        )
    
    user = user_store.get_user(token_data.username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return user


async def get_current_active_user(
    current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    """
    FastAPI dependency to get current active user
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


# ========================
# Role-Based Access Control
# ========================

class RoleChecker:
    """Dependency class for role-based access control"""
    
    def __init__(self, allowed_roles: List[UserRole]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, user: UserInDB = Depends(get_current_user)) -> UserInDB:
        if user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted. Required roles: {[r.value for r in self.allowed_roles]}"
            )
        return user


# Convenience role checker instances
require_admin = RoleChecker([UserRole.ADMIN])
require_user = RoleChecker([UserRole.USER, UserRole.ADMIN])
require_any = RoleChecker([UserRole.GUEST, UserRole.USER, UserRole.ADMIN])


# ========================
# Authentication Endpoints (Example Usage)
# ========================

"""
Example FastAPI endpoint usage:

from fastapi import FastAPI, Request
from pulse_core.auth import *

app = FastAPI()


@app.post("/auth/register", response_model=User)
async def register(user_create: UserCreate):
    '''Register new user'''
    try:
        user = user_store.create_user(user_create)
        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/login", response_model=Token)
async def login(login_request: LoginRequest, request: Request):
    '''Login and get access token'''
    client_ip = request.client.host
    user = authenticate_user(
        login_request.username,
        login_request.password,
        ip_address=client_ip
    )
    return token_manager.create_token_pair(user)


@app.post("/auth/refresh", response_model=Token)
async def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    '''Refresh access token using refresh token'''
    token = credentials.credentials
    token_data = token_manager.decode_token(token)
    
    if token_data.token_type != TokenType.REFRESH:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type. Refresh token required."
        )
    
    user = user_store.get_user(token_data.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return token_manager.create_token_pair(user)


@app.get("/auth/me", response_model=User)
async def get_me(current_user: UserInDB = Depends(get_current_active_user)):
    '''Get current user information'''
    return current_user


@app.get("/admin/users")
async def list_users(admin_user: UserInDB = Depends(require_admin)):
    '''Admin only: List all users'''
    return {"users": list(user_store.users.values())}


@app.get("/protected")
async def protected_route(current_user: UserInDB = Depends(require_user)):
    '''Protected route requiring user or admin role'''
    return {"message": f"Hello {current_user.username}!", "role": current_user.role}


@app.get("/public")
async def public_route(current_user: UserInDB = Depends(require_any)):
    '''Public route accessible by any authenticated user'''
    return {"message": "This is a public endpoint"}
"""


# ========================
# Utility Functions
# ========================

def get_password_hash(password: str) -> str:
    """Convenience function to hash password"""
    return password_manager.hash_password(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Convenience function to verify password"""
    return password_manager.verify_password(plain_password, hashed_password)


def create_user(username: str, email: str, password: str, role: UserRole = UserRole.USER) -> UserInDB:
    """Convenience function to create user"""
    user_create = UserCreate(
        username=username,
        email=email,
        password=password,
        role=role
    )
    return user_store.create_user(user_create)


# ========================
# Module Exports
# ========================

__all__ = [
    # Enums
    'UserRole',
    'TokenType',
    
    # Models
    'User',
    'UserCreate',
    'UserUpdate',
    'UserInDB',
    'Token',
    'TokenData',
    'LoginRequest',
    'LoginSession',
    
    # Managers
    'SessionManager',
    'PasswordManager',
    'TokenManager',
    'UserStore',
    
    # Instances
    'session_manager',
    'password_manager',
    'token_manager',
    'user_store',
    
    # Auth Functions
    'authenticate_user',
    'get_current_user',
    'get_current_active_user',
    
    # RBAC
    'RoleChecker',
    'require_admin',
    'require_user',
    'require_any',
    
    # Utilities
    'get_password_hash',
    'verify_password',
    'create_user',
    
    # Configuration
    'SECRET_KEY',
    'ALGORITHM',
    'ACCESS_TOKEN_EXPIRE_MINUTES',
    'REFRESH_TOKEN_EXPIRE_DAYS',
    'MAX_LOGIN_ATTEMPTS',
    'LOCKOUT_DURATION_MINUTES',
]
