"""
Pulse Core Database Module
==========================

Comprehensive database layer with SQL injection prevention, secure ORM models,
query sanitization, connection pooling, and transaction management.

Features:
- SQLAlchemy ORM for safe database operations
- Parameterized queries with injection prevention
- Connection pooling with configurable limits
- QuerySanitizer class for input validation
- Secure User, Task, and Session models
- LIKE pattern sanitization
- ORDER BY whitelisting
- Dangerous SQL pattern detection
- Soft delete support
- Database indexes for performance
- Transaction management with automatic rollback
- Async context managers
- Query logging for security monitoring

Author: jetgause
Created: 2025-12-10
"""

import re
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Set, Union
from contextlib import asynccontextmanager, contextmanager
from enum import Enum

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    Index,
    event,
    MetaData,
)
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import (
    declarative_base,
    relationship,
    sessionmaker,
    Session,
)
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.sql import text
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

# Configure logging for security monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security logger for audit trail
security_logger = logging.getLogger('security.database')
security_logger.setLevel(logging.INFO)

# Create base class for declarative models
Base = declarative_base()
metadata = MetaData()


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class UserRole(str, Enum):
    """User role enumeration"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# Whitelist for ORDER BY columns to prevent SQL injection
ALLOWED_ORDER_BY_COLUMNS = {
    'users': {'id', 'username', 'email', 'created_at', 'updated_at'},
    'tasks': {'id', 'title', 'status', 'priority', 'created_at', 'updated_at', 'due_date'},
    'sessions': {'id', 'user_id', 'created_at', 'expires_at'},
}

# Dangerous SQL patterns to detect and block
DANGEROUS_SQL_PATTERNS = [
    r'\b(DROP|DELETE|TRUNCATE|ALTER|CREATE|EXEC|EXECUTE)\b',
    r'--',  # SQL comment
    r'/\*.*?\*/',  # Multi-line comment
    r';.*',  # Multiple statements
    r'\bUNION\b.*\bSELECT\b',
    r'\bINTO\b.*\bOUTFILE\b',
    r'\bLOAD_FILE\b',
    r'xp_cmdshell',
]

DANGEROUS_SQL_REGEX = re.compile('|'.join(DANGEROUS_SQL_PATTERNS), re.IGNORECASE)


# ============================================================================
# QUERY SANITIZER CLASS
# ============================================================================

class QuerySanitizer:
    """
    Comprehensive query sanitization and validation class.
    
    Provides methods to sanitize user inputs, validate ORDER BY clauses,
    escape LIKE patterns, and detect dangerous SQL patterns.
    """
    
    @staticmethod
    def sanitize_like_pattern(pattern: str, escape_char: str = '\\') -> str:
        """
        Sanitize LIKE pattern to prevent SQL injection.
        
        Args:
            pattern: The pattern to sanitize
            escape_char: The escape character to use
            
        Returns:
            Sanitized pattern safe for LIKE queries
        """
        if not isinstance(pattern, str):
            raise ValueError("Pattern must be a string")
        
        # Escape special LIKE characters
        pattern = pattern.replace(escape_char, escape_char + escape_char)
        pattern = pattern.replace('%', escape_char + '%')
        pattern = pattern.replace('_', escape_char + '_')
        
        security_logger.info(f"Sanitized LIKE pattern: {pattern}")
        return pattern
    
    @staticmethod
    def validate_order_by(table_name: str, column: str, direction: str = 'ASC') -> tuple:
        """
        Validate ORDER BY clause against whitelist.
        
        Args:
            table_name: Name of the table
            column: Column name to order by
            direction: Sort direction (ASC or DESC)
            
        Returns:
            Tuple of (validated_column, validated_direction)
            
        Raises:
            ValueError: If column or direction is invalid
        """
        # Validate table exists in whitelist
        if table_name not in ALLOWED_ORDER_BY_COLUMNS:
            raise ValueError(f"Table '{table_name}' not allowed for ordering")
        
        # Validate column exists in whitelist
        if column not in ALLOWED_ORDER_BY_COLUMNS[table_name]:
            raise ValueError(
                f"Column '{column}' not allowed for ordering in table '{table_name}'"
            )
        
        # Validate direction
        direction = direction.upper()
        if direction not in ('ASC', 'DESC'):
            raise ValueError(f"Invalid sort direction: {direction}")
        
        security_logger.info(
            f"Validated ORDER BY: {table_name}.{column} {direction}"
        )
        return column, direction
    
    @staticmethod
    def detect_sql_injection(user_input: str) -> bool:
        """
        Detect potential SQL injection attempts.
        
        Args:
            user_input: User-provided input to check
            
        Returns:
            True if dangerous patterns detected, False otherwise
        """
        if not isinstance(user_input, str):
            return False
        
        if DANGEROUS_SQL_REGEX.search(user_input):
            security_logger.warning(
                f"Potential SQL injection detected: {user_input[:100]}"
            )
            return True
        
        return False
    
    @staticmethod
    def sanitize_input(user_input: str, max_length: int = 1000) -> str:
        """
        Sanitize general user input.
        
        Args:
            user_input: Input to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized input
            
        Raises:
            ValueError: If input contains dangerous patterns or exceeds max length
        """
        if not isinstance(user_input, str):
            raise ValueError("Input must be a string")
        
        if len(user_input) > max_length:
            raise ValueError(f"Input exceeds maximum length of {max_length}")
        
        if QuerySanitizer.detect_sql_injection(user_input):
            raise ValueError("Input contains potentially dangerous SQL patterns")
        
        return user_input.strip()


# ============================================================================
# DATABASE MODELS
# ============================================================================

class User(Base):
    """
    User model with security features and soft delete support.
    """
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)  # Store hashed passwords only
    role = Column(String(20), default=UserRole.USER.value, nullable=False)
    
    # Soft delete support
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Security fields
    failed_login_attempts = Column(Integer, default=0)
    account_locked_until = Column(DateTime, nullable=True)
    
    # Relationships
    tasks = relationship('Task', back_populates='user', lazy='dynamic')
    sessions = relationship('Session', back_populates='user', lazy='dynamic')
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_user_email_deleted', 'email', 'is_deleted'),
        Index('idx_user_username_deleted', 'username', 'is_deleted'),
    )
    
    def soft_delete(self):
        """Perform soft delete on user"""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
        security_logger.info(f"User {self.id} soft deleted")
    
    def restore(self):
        """Restore soft deleted user"""
        self.is_deleted = False
        self.deleted_at = None
        security_logger.info(f"User {self.id} restored")
    
    def lock_account(self, duration_minutes: int = 30):
        """Lock user account for specified duration"""
        self.account_locked_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
        security_logger.warning(f"User {self.id} account locked for {duration_minutes} minutes")
    
    def is_account_locked(self) -> bool:
        """Check if account is currently locked"""
        if self.account_locked_until and self.account_locked_until > datetime.utcnow():
            return True
        return False
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"


class Task(Base):
    """
    Task model with relationships and soft delete support.
    """
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(20), default=TaskStatus.PENDING.value, nullable=False)
    priority = Column(Integer, default=0)
    
    # Soft delete support
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    due_date = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship('User', back_populates='tasks')
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_task_user_status', 'user_id', 'status', 'is_deleted'),
        Index('idx_task_due_date', 'due_date'),
    )
    
    def soft_delete(self):
        """Perform soft delete on task"""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
        security_logger.info(f"Task {self.id} soft deleted")
    
    def restore(self):
        """Restore soft deleted task"""
        self.is_deleted = False
        self.deleted_at = None
        security_logger.info(f"Task {self.id} restored")
    
    def mark_completed(self):
        """Mark task as completed"""
        self.status = TaskStatus.COMPLETED.value
        self.completed_at = datetime.utcnow()
    
    def __repr__(self):
        return f"<Task(id={self.id}, title='{self.title}', status='{self.status}')>"


class Session(Base):
    """
    Session model for user authentication tracking.
    """
    __tablename__ = 'sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(255), nullable=True)
    
    # Session lifecycle
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    last_activity = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    user = relationship('User', back_populates='sessions')
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_session_user_active', 'user_id', 'is_active'),
        Index('idx_session_token_active', 'session_token', 'is_active'),
    )
    
    def is_valid(self) -> bool:
        """Check if session is valid and not expired"""
        return self.is_active and self.expires_at > datetime.utcnow()
    
    def invalidate(self):
        """Invalidate session"""
        self.is_active = False
        security_logger.info(f"Session {self.id} invalidated")
    
    def refresh_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()
    
    def __repr__(self):
        return f"<Session(id={self.id}, user_id={self.user_id}, valid={self.is_valid()})>"


# ============================================================================
# DATABASE CONNECTION AND POOLING
# ============================================================================

class DatabaseConfig:
    """Database configuration with secure defaults"""
    
    def __init__(
        self,
        database_url: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False,
        echo_pool: bool = False,
    ):
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.echo = echo
        self.echo_pool = echo_pool


class DatabaseManager:
    """
    Database manager with connection pooling and transaction management.
    
    Provides both synchronous and asynchronous database operations with
    automatic rollback on errors and comprehensive logging.
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None
        self._initialized = False
    
    def initialize(self):
        """Initialize synchronous database engine and session factory"""
        if self._initialized:
            logger.warning("DatabaseManager already initialized")
            return
        
        try:
            # Create engine with connection pooling
            self.engine = create_engine(
                self.config.database_url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo,
                echo_pool=self.config.echo_pool,
            )
            
            # Create session factory
            self.session_factory = sessionmaker(
                bind=self.engine,
                expire_on_commit=False,
            )
            
            # Create tables
            Base.metadata.create_all(self.engine)
            
            # Set up event listeners for query logging
            self._setup_event_listeners()
            
            self._initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def initialize_async(self):
        """Initialize asynchronous database engine and session factory"""
        try:
            # Convert database URL for async
            async_url = self.config.database_url.replace(
                'postgresql://', 'postgresql+asyncpg://'
            ).replace(
                'mysql://', 'mysql+aiomysql://'
            )
            
            # Create async engine
            self.async_engine = create_async_engine(
                async_url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo,
            )
            
            # Create async session factory
            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                expire_on_commit=False,
                class_=AsyncSession,
            )
            
            # Create tables
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Async database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize async database: {e}")
            raise
    
    def _setup_event_listeners(self):
        """Set up SQLAlchemy event listeners for query logging"""
        
        @event.listens_for(self.engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Log queries before execution"""
            security_logger.debug(f"Query: {statement}")
            security_logger.debug(f"Parameters: {parameters}")
        
        @event.listens_for(self.engine, "handle_error")
        def handle_error(exception_context):
            """Log database errors"""
            security_logger.error(
                f"Database error: {exception_context.original_exception}"
            )
    
    @contextmanager
    def get_session(self) -> Session:
        """
        Context manager for database sessions with automatic rollback.
        
        Usage:
            with db_manager.get_session() as session:
                user = session.query(User).filter_by(id=1).first()
        """
        if not self._initialized:
            raise RuntimeError("DatabaseManager not initialized. Call initialize() first.")
        
        session = self.session_factory()
        try:
            yield session
            session.commit()
            logger.debug("Session committed successfully")
        except Exception as e:
            session.rollback()
            logger.error(f"Session rollback due to error: {e}")
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncSession:
        """
        Async context manager for database sessions with automatic rollback.
        
        Usage:
            async with db_manager.get_async_session() as session:
                result = await session.execute(select(User).filter_by(id=1))
                user = result.scalar_one_or_none()
        """
        if not self.async_session_factory:
            raise RuntimeError("Async database not initialized. Call initialize_async() first.")
        
        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
            logger.debug("Async session committed successfully")
        except Exception as e:
            await session.rollback()
            logger.error(f"Async session rollback due to error: {e}")
            raise
        finally:
            await session.close()
    
    def close(self):
        """Close database connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database engine disposed")
        
        if self.async_engine:
            asyncio.create_task(self.async_engine.dispose())
            logger.info("Async database engine disposed")


# ============================================================================
# SECURE QUERY HELPERS
# ============================================================================

class SecureQueryBuilder:
    """
    Builder for creating secure parameterized queries.
    """
    
    def __init__(self, session: Session):
        self.session = session
        self.sanitizer = QuerySanitizer()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Safely retrieve user by username using parameterized query.
        
        Args:
            username: Username to search for
            
        Returns:
            User object or None
        """
        try:
            username = self.sanitizer.sanitize_input(username, max_length=50)
            return self.session.query(User).filter(
                User.username == username,
                User.is_deleted == False
            ).first()
        except Exception as e:
            logger.error(f"Error retrieving user by username: {e}")
            return None
    
    def search_tasks_by_title(
        self,
        user_id: int,
        title_pattern: str,
        order_by: str = 'created_at',
        direction: str = 'DESC'
    ) -> List[Task]:
        """
        Search tasks by title using safe LIKE query.
        
        Args:
            user_id: User ID to filter by
            title_pattern: Pattern to search for
            order_by: Column to order by
            direction: Sort direction
            
        Returns:
            List of matching tasks
        """
        try:
            # Sanitize LIKE pattern
            safe_pattern = self.sanitizer.sanitize_like_pattern(title_pattern)
            
            # Validate ORDER BY
            col, dir = self.sanitizer.validate_order_by('tasks', order_by, direction)
            
            # Build query with parameterized LIKE
            query = self.session.query(Task).filter(
                Task.user_id == user_id,
                Task.is_deleted == False,
                Task.title.like(f'%{safe_pattern}%', escape='\\')
            )
            
            # Apply ordering
            order_column = getattr(Task, col)
            if dir == 'DESC':
                query = query.order_by(order_column.desc())
            else:
                query = query.order_by(order_column.asc())
            
            return query.all()
            
        except Exception as e:
            logger.error(f"Error searching tasks: {e}")
            return []
    
    def get_active_sessions(self, user_id: int) -> List[Session]:
        """
        Get all active sessions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of active sessions
        """
        return self.session.query(Session).filter(
            Session.user_id == user_id,
            Session.is_active == True,
            Session.expires_at > datetime.utcnow()
        ).all()
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            count = self.session.query(Session).filter(
                Session.expires_at < datetime.utcnow()
            ).update({'is_active': False})
            
            self.session.commit()
            logger.info(f"Cleaned up {count} expired sessions")
            return count
            
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
            self.session.rollback()
            return 0


# ============================================================================
# TRANSACTION HELPERS
# ============================================================================

class TransactionManager:
    """
    Helper class for managing complex transactions with automatic rollback.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def create_user_with_session(
        self,
        username: str,
        email: str,
        password_hash: str,
        session_token: str,
        ip_address: str = None,
        user_agent: str = None,
    ) -> tuple[User, Session]:
        """
        Create user and initial session in a single transaction.
        
        Args:
            username: User's username
            email: User's email
            password_hash: Hashed password
            session_token: Session token
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Tuple of (User, Session)
        """
        with self.db_manager.get_session() as session:
            try:
                # Create user
                user = User(
                    username=username,
                    email=email,
                    password_hash=password_hash,
                )
                session.add(user)
                session.flush()  # Get user ID without committing
                
                # Create session
                user_session = Session(
                    user_id=user.id,
                    session_token=session_token,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    expires_at=datetime.utcnow() + timedelta(days=7),
                )
                session.add(user_session)
                
                # Commit transaction
                session.commit()
                
                security_logger.info(f"Created user {user.id} with session {user_session.id}")
                return user, user_session
                
            except IntegrityError as e:
                logger.error(f"Integrity error creating user: {e}")
                raise ValueError("Username or email already exists")
            except Exception as e:
                logger.error(f"Error creating user with session: {e}")
                raise


# ============================================================================
# EXAMPLE USAGE AND UTILITIES
# ============================================================================

def example_usage():
    """Example usage of the database module"""
    
    # Configuration
    config = DatabaseConfig(
        database_url="postgresql://user:pass@localhost/dbname",
        pool_size=5,
        max_overflow=10,
    )
    
    # Initialize database manager
    db_manager = DatabaseManager(config)
    db_manager.initialize()
    
    # Use context manager for queries
    with db_manager.get_session() as session:
        # Create secure query builder
        query_builder = SecureQueryBuilder(session)
        
        # Safe user lookup
        user = query_builder.get_user_by_username("john_doe")
        
        # Safe task search with LIKE
        tasks = query_builder.search_tasks_by_title(
            user_id=user.id,
            title_pattern="important",
            order_by="due_date",
            direction="ASC"
        )
    
    # Transaction management
    transaction_mgr = TransactionManager(db_manager)
    user, session = transaction_mgr.create_user_with_session(
        username="new_user",
        email="user@example.com",
        password_hash="hashed_password_here",
        session_token="secure_token",
        ip_address="192.168.1.1",
    )
    
    # Cleanup
    db_manager.close()


if __name__ == "__main__":
    logger.info("Pulse Core Database Module loaded")
    logger.info("For usage examples, call example_usage()")
