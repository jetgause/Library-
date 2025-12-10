"""
PulseEngine - Core Business Logic Implementation
A comprehensive task management engine with analytics, notifications, and background processing.

Author: jetgause
Created: 2025-12-10
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from collections import defaultdict
import uuid


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Enumeration for task status types."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Enumeration for task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class NotificationType(Enum):
    """Enumeration for notification types."""
    TASK_CREATED = "task_created"
    TASK_UPDATED = "task_updated"
    TASK_COMPLETED = "task_completed"
    TASK_OVERDUE = "task_overdue"
    TASK_DUE_SOON = "task_due_soon"
    TASK_DELETED = "task_deleted"


@dataclass
class Task:
    """Task data model with comprehensive attributes."""
    id: str
    title: str
    description: str
    status: TaskStatus
    priority: TaskPriority
    created_at: datetime
    updated_at: datetime
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        data = asdict(self)
        data['status'] = self.status.value
        data['priority'] = self.priority.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.due_date:
            data['due_date'] = self.due_date.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data

    def is_overdue(self) -> bool:
        """Check if task is overdue."""
        if self.due_date and self.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
            return datetime.utcnow() > self.due_date
        return False


@dataclass
class Notification:
    """Notification data model."""
    id: str
    notification_type: NotificationType
    task_id: str
    message: str
    created_at: datetime
    read: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to dictionary representation."""
        return {
            'id': self.id,
            'notification_type': self.notification_type.value,
            'task_id': self.task_id,
            'message': self.message,
            'created_at': self.created_at.isoformat(),
            'read': self.read,
            'metadata': self.metadata
        }


class InMemoryStorage:
    """In-memory storage implementation for tasks and notifications."""

    def __init__(self):
        """Initialize storage with empty collections."""
        self.tasks: Dict[str, Task] = {}
        self.notifications: Dict[str, Notification] = {}
        self._lock = asyncio.Lock()
        logger.info("InMemoryStorage initialized")

    async def save_task(self, task: Task) -> None:
        """Save or update a task in storage."""
        async with self._lock:
            self.tasks[task.id] = task
            logger.debug(f"Task saved: {task.id}")

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by ID."""
        async with self._lock:
            task = self.tasks.get(task_id)
            if task:
                logger.debug(f"Task retrieved: {task_id}")
            else:
                logger.warning(f"Task not found: {task_id}")
            return task

    async def delete_task(self, task_id: str) -> bool:
        """Delete a task from storage."""
        async with self._lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                logger.info(f"Task deleted: {task_id}")
                return True
            logger.warning(f"Attempted to delete non-existent task: {task_id}")
            return False

    async def get_all_tasks(self) -> List[Task]:
        """Retrieve all tasks."""
        async with self._lock:
            tasks = list(self.tasks.values())
            logger.debug(f"Retrieved {len(tasks)} tasks")
            return tasks

    async def save_notification(self, notification: Notification) -> None:
        """Save a notification in storage."""
        async with self._lock:
            self.notifications[notification.id] = notification
            logger.debug(f"Notification saved: {notification.id}")

    async def get_notifications(self, task_id: Optional[str] = None, 
                                unread_only: bool = False) -> List[Notification]:
        """Retrieve notifications with optional filters."""
        async with self._lock:
            notifications = list(self.notifications.values())
            
            if task_id:
                notifications = [n for n in notifications if n.task_id == task_id]
            
            if unread_only:
                notifications = [n for n in notifications if not n.read]
            
            logger.debug(f"Retrieved {len(notifications)} notifications")
            return notifications

    async def mark_notification_read(self, notification_id: str) -> bool:
        """Mark a notification as read."""
        async with self._lock:
            if notification_id in self.notifications:
                self.notifications[notification_id].read = True
                logger.debug(f"Notification marked as read: {notification_id}")
                return True
            return False

    async def clear_all(self) -> None:
        """Clear all storage (for testing/reset purposes)."""
        async with self._lock:
            self.tasks.clear()
            self.notifications.clear()
            logger.warning("All storage cleared")


class AnalyticsEngine:
    """Analytics generation engine for task metrics and insights."""

    def __init__(self, storage: InMemoryStorage):
        """Initialize analytics engine with storage reference."""
        self.storage = storage
        logger.info("AnalyticsEngine initialized")

    async def generate_task_summary(self) -> Dict[str, Any]:
        """Generate comprehensive task summary statistics."""
        tasks = await self.storage.get_all_tasks()
        
        summary = {
            'total_tasks': len(tasks),
            'by_status': defaultdict(int),
            'by_priority': defaultdict(int),
            'overdue_count': 0,
            'completed_count': 0,
            'completion_rate': 0.0,
            'average_completion_time': None,
            'generated_at': datetime.utcnow().isoformat()
        }

        completion_times = []

        for task in tasks:
            summary['by_status'][task.status.value] += 1
            summary['by_priority'][task.priority.value] += 1
            
            if task.is_overdue():
                summary['overdue_count'] += 1
            
            if task.status == TaskStatus.COMPLETED:
                summary['completed_count'] += 1
                if task.completed_at:
                    duration = (task.completed_at - task.created_at).total_seconds()
                    completion_times.append(duration)

        if len(tasks) > 0:
            summary['completion_rate'] = (summary['completed_count'] / len(tasks)) * 100

        if completion_times:
            avg_seconds = sum(completion_times) / len(completion_times)
            summary['average_completion_time'] = f"{avg_seconds / 3600:.2f} hours"

        logger.info(f"Analytics summary generated: {summary['total_tasks']} tasks analyzed")
        return dict(summary)

    async def get_task_trends(self, days: int = 7) -> Dict[str, Any]:
        """Analyze task trends over specified time period."""
        tasks = await self.storage.get_all_tasks()
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        recent_tasks = [t for t in tasks if t.created_at >= cutoff_date]
        completed_recent = [t for t in recent_tasks if t.status == TaskStatus.COMPLETED]
        
        trends = {
            'period_days': days,
            'tasks_created': len(recent_tasks),
            'tasks_completed': len(completed_recent),
            'tasks_per_day': len(recent_tasks) / days if days > 0 else 0,
            'daily_breakdown': defaultdict(int),
            'analyzed_at': datetime.utcnow().isoformat()
        }

        for task in recent_tasks:
            day_key = task.created_at.strftime('%Y-%m-%d')
            trends['daily_breakdown'][day_key] += 1

        logger.info(f"Task trends analyzed for {days} days")
        return dict(trends)

    async def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Generate statistics for a specific user."""
        tasks = await self.storage.get_all_tasks()
        user_tasks = [t for t in tasks if t.assigned_to == user_id]
        
        stats = {
            'user_id': user_id,
            'total_assigned': len(user_tasks),
            'completed': len([t for t in user_tasks if t.status == TaskStatus.COMPLETED]),
            'in_progress': len([t for t in user_tasks if t.status == TaskStatus.IN_PROGRESS]),
            'overdue': len([t for t in user_tasks if t.is_overdue()]),
            'by_priority': defaultdict(int),
            'generated_at': datetime.utcnow().isoformat()
        }

        for task in user_tasks:
            stats['by_priority'][task.priority.value] += 1

        logger.info(f"User statistics generated for: {user_id}")
        return dict(stats)


class NotificationManager:
    """Notification management system for task events."""

    def __init__(self, storage: InMemoryStorage):
        """Initialize notification manager with storage reference."""
        self.storage = storage
        self.subscribers: List[Callable] = []
        logger.info("NotificationManager initialized")

    def subscribe(self, callback: Callable) -> None:
        """Subscribe to notification events."""
        self.subscribers.append(callback)
        logger.debug(f"New subscriber added. Total: {len(self.subscribers)}")

    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from notification events."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.debug(f"Subscriber removed. Total: {len(self.subscribers)}")

    async def create_notification(self, notification_type: NotificationType, 
                                  task: Task, message: str, 
                                  metadata: Optional[Dict[str, Any]] = None) -> Notification:
        """Create and store a new notification."""
        notification = Notification(
            id=str(uuid.uuid4()),
            notification_type=notification_type,
            task_id=task.id,
            message=message,
            created_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        await self.storage.save_notification(notification)
        logger.info(f"Notification created: {notification_type.value} for task {task.id}")
        
        # Notify subscribers
        await self._notify_subscribers(notification)
        
        return notification

    async def _notify_subscribers(self, notification: Notification) -> None:
        """Notify all subscribers of new notification."""
        for callback in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(notification)
                else:
                    callback(notification)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")

    async def get_unread_notifications(self, task_id: Optional[str] = None) -> List[Notification]:
        """Retrieve unread notifications."""
        return await self.storage.get_notifications(task_id=task_id, unread_only=True)

    async def mark_as_read(self, notification_id: str) -> bool:
        """Mark a notification as read."""
        result = await self.storage.mark_notification_read(notification_id)
        if result:
            logger.info(f"Notification marked as read: {notification_id}")
        return result


class PulseEngine:
    """
    Main PulseEngine class - orchestrates all task management operations.
    Provides CRUD operations, analytics, notifications, and background tasks.
    """

    def __init__(self, storage: Optional[InMemoryStorage] = None):
        """Initialize PulseEngine with all subsystems."""
        self.storage = storage or InMemoryStorage()
        self.analytics = AnalyticsEngine(self.storage)
        self.notifications = NotificationManager(self.storage)
        self._background_tasks: List[asyncio.Task] = []
        self._running = False
        logger.info("PulseEngine initialized successfully")

    # Task CRUD Operations

    async def create_task(self, title: str, description: str, 
                         priority: TaskPriority = TaskPriority.MEDIUM,
                         due_date: Optional[datetime] = None,
                         assigned_to: Optional[str] = None,
                         tags: Optional[List[str]] = None) -> Task:
        """Create a new task."""
        task = Task(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            status=TaskStatus.PENDING,
            priority=priority,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            due_date=due_date,
            assigned_to=assigned_to,
            tags=tags or []
        )
        
        await self.storage.save_task(task)
        logger.info(f"Task created: {task.id} - {task.title}")
        
        await self.notifications.create_notification(
            NotificationType.TASK_CREATED,
            task,
            f"New task created: {task.title}"
        )
        
        return task

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by ID."""
        return await self.storage.get_task(task_id)

    async def update_task(self, task_id: str, **updates) -> Optional[Task]:
        """Update an existing task."""
        task = await self.storage.get_task(task_id)
        if not task:
            logger.warning(f"Update failed: Task not found {task_id}")
            return None
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)
        
        task.updated_at = datetime.utcnow()
        
        # Handle completion
        if updates.get('status') == TaskStatus.COMPLETED and not task.completed_at:
            task.completed_at = datetime.utcnow()
            await self.notifications.create_notification(
                NotificationType.TASK_COMPLETED,
                task,
                f"Task completed: {task.title}"
            )
        
        await self.storage.save_task(task)
        logger.info(f"Task updated: {task_id}")
        
        await self.notifications.create_notification(
            NotificationType.TASK_UPDATED,
            task,
            f"Task updated: {task.title}"
        )
        
        return task

    async def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        task = await self.storage.get_task(task_id)
        if not task:
            return False
        
        success = await self.storage.delete_task(task_id)
        if success:
            logger.info(f"Task deleted: {task_id}")
            await self.notifications.create_notification(
                NotificationType.TASK_DELETED,
                task,
                f"Task deleted: {task.title}"
            )
        return success

    async def list_tasks(self, status: Optional[TaskStatus] = None,
                        priority: Optional[TaskPriority] = None,
                        assigned_to: Optional[str] = None) -> List[Task]:
        """List tasks with optional filters."""
        tasks = await self.storage.get_all_tasks()
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        if priority:
            tasks = [t for t in tasks if t.priority == priority]
        
        if assigned_to:
            tasks = [t for t in tasks if t.assigned_to == assigned_to]
        
        logger.debug(f"Listed {len(tasks)} tasks with filters")
        return tasks

    # Analytics Operations

    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        return await self.analytics.generate_task_summary()

    async def get_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get task trends analysis."""
        return await self.analytics.get_task_trends(days)

    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user-specific statistics."""
        return await self.analytics.get_user_statistics(user_id)

    # Background Tasks

    async def start_background_tasks(self) -> None:
        """Start background task monitoring."""
        if self._running:
            logger.warning("Background tasks already running")
            return
        
        self._running = True
        task = asyncio.create_task(self._check_overdue_tasks())
        self._background_tasks.append(task)
        logger.info("Background tasks started")

    async def stop_background_tasks(self) -> None:
        """Stop all background tasks."""
        self._running = False
        for task in self._background_tasks:
            task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        logger.info("Background tasks stopped")

    async def _check_overdue_tasks(self) -> None:
        """Background task to check for overdue tasks periodically."""
        logger.info("Overdue task checker started")
        
        while self._running:
            try:
                tasks = await self.storage.get_all_tasks()
                
                for task in tasks:
                    if task.is_overdue() and task.status != TaskStatus.OVERDUE:
                        task.status = TaskStatus.OVERDUE
                        task.updated_at = datetime.utcnow()
                        await self.storage.save_task(task)
                        
                        await self.notifications.create_notification(
                            NotificationType.TASK_OVERDUE,
                            task,
                            f"Task is overdue: {task.title}"
                        )
                        logger.warning(f"Task marked as overdue: {task.id}")
                
                # Check for tasks due soon (within 24 hours)
                for task in tasks:
                    if (task.due_date and task.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]):
                        time_until_due = task.due_date - datetime.utcnow()
                        if timedelta(0) < time_until_due < timedelta(hours=24):
                            await self.notifications.create_notification(
                                NotificationType.TASK_DUE_SOON,
                                task,
                                f"Task due soon: {task.title} (due in {time_until_due.seconds // 3600} hours)"
                            )
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                logger.info("Overdue task checker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in overdue task checker: {e}")
                await asyncio.sleep(60)  # Wait before retrying


# Module-level convenience functions

async def create_engine() -> PulseEngine:
    """Factory function to create a new PulseEngine instance."""
    engine = PulseEngine()
    logger.info("New PulseEngine instance created")
    return engine


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        engine = await create_engine()
        
        # Create sample tasks
        task1 = await engine.create_task(
            "Implement authentication",
            "Add JWT-based authentication to the API",
            priority=TaskPriority.HIGH,
            due_date=datetime.utcnow() + timedelta(days=3)
        )
        
        task2 = await engine.create_task(
            "Write documentation",
            "Complete API documentation",
            priority=TaskPriority.MEDIUM
        )
        
        # Start background monitoring
        await engine.start_background_tasks()
        
        # Get analytics
        summary = await engine.get_analytics_summary()
        print(f"Analytics: {json.dumps(summary, indent=2)}")
        
        # Wait a bit for background tasks
        await asyncio.sleep(2)
        
        # Cleanup
        await engine.stop_background_tasks()
        logger.info("Example execution completed")
    
    asyncio.run(main())
