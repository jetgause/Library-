#!/usr/bin/env python3
"""
Security Patch 20/23 - Incident Response Core Module
Part 1 of 3: Core data structures and incident models

Author: jetgause
Created: 2025-12-11
Version: 1.0.0
"""

import json
import logging
import hashlib
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

    def priority_score(self) -> int:
        """Get numeric priority for severity"""
        scores = {
            "CRITICAL": 1000,
            "HIGH": 750,
            "MEDIUM": 500,
            "LOW": 250,
            "INFO": 100
        }
        return scores.get(self.value, 0)


class IncidentStatus(Enum):
    """Incident status workflow"""
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    CONTAINED = "CONTAINED"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"


@dataclass
class TimelineEvent:
    """Single event in incident timeline"""
    timestamp: datetime
    event_type: str
    description: str
    actor: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "description": self.description,
            "actor": self.actor,
            "metadata": self.metadata
        }


@dataclass
class SecurityIncident:
    """Represents a security incident"""
    id: str
    severity: IncidentSeverity
    status: IncidentStatus
    title: str
    description: str
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    affected_systems: List[str] = field(default_factory=list)
    timeline: List[TimelineEvent] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_timeline_event(self, event_type: str, description: str, actor: str = "system"):
        """Add event to incident timeline"""
        event = TimelineEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            description=description,
            actor=actor
        )
        self.timeline.append(event)
        self.updated_at = datetime.utcnow()
        logger.info(f"Incident {self.id}: {event_type} - {description}")
    
    def update_status(self, new_status: IncidentStatus, actor: str = "system"):
        """Update incident status with timeline tracking"""
        old_status = self.status
        self.status = new_status
        self.updated_at = datetime.utcnow()
        
        if new_status == IncidentStatus.RESOLVED:
            self.resolved_at = datetime.utcnow()
        elif new_status == IncidentStatus.CLOSED:
            self.closed_at = datetime.utcnow()
        
        self.add_timeline_event(
            "status_change",
            f"Status changed from {old_status.value} to {new_status.value}",
            actor
        )
    
    def assign(self, assignee: str, actor: str = "system"):
        """Assign incident to team member"""
        self.assigned_to = assignee
        self.add_timeline_event(
            "assignment",
            f"Incident assigned to {assignee}",
            actor
        )
    
    def get_age_minutes(self) -> float:
        """Get incident age in minutes"""
        return (datetime.utcnow() - self.created_at).total_seconds() / 60
    
    def get_time_to_resolve_minutes(self) -> Optional[float]:
        """Get time to resolution in minutes"""
        if self.resolved_at:
            return (self.resolved_at - self.created_at).total_seconds() / 60
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert incident to dictionary"""
        return {
            "id": self.id,
            "severity": self.severity.value,
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "assigned_to": self.assigned_to,
            "affected_systems": self.affected_systems,
            "tags": self.tags,
            "timeline": [e.to_dict() for e in self.timeline],
            "metadata": self.metadata,
            "age_minutes": self.get_age_minutes(),
            "time_to_resolve": self.get_time_to_resolve_minutes()
        }


@dataclass
class ResponseAction:
    """Automated response action"""
    action_type: str
    description: str
    function: Callable
    params: Dict[str, Any] = field(default_factory=dict)
    
    def execute(self, incident: SecurityIncident) -> bool:
        """Execute the response action"""
        try:
            logger.info(f"Executing action: {self.action_type} for incident {incident.id}")
            result = self.function(incident, **self.params)
            incident.add_timeline_event(
                "automated_action",
                f"Executed: {self.description}",
                "automation"
            )
            return result
        except Exception as e:
            logger.error(f"Action {self.action_type} failed: {e}")
            incident.add_timeline_event(
                "action_failed",
                f"Failed to execute: {self.description} - {str(e)}",
                "automation"
            )
            return False


@dataclass
class ResponsePlaybook:
    """Automated incident response playbook"""
    name: str
    description: str
    trigger_conditions: Dict[str, Any]
    actions: List[ResponseAction] = field(default_factory=list)
    enabled: bool = True
    
    def matches_incident(self, incident: SecurityIncident) -> bool:
        """Check if playbook should be triggered for incident"""
        if not self.enabled:
            return False
        
        if "severity" in self.trigger_conditions:
            required_severities = self.trigger_conditions["severity"]
            if incident.severity.value not in required_severities:
                return False
        
        if "tags" in self.trigger_conditions:
            required_tags = self.trigger_conditions["tags"]
            if not any(tag in incident.tags for tag in required_tags):
                return False
        
        if "affected_systems" in self.trigger_conditions:
            required_systems = self.trigger_conditions["affected_systems"]
            if not any(sys in incident.affected_systems for sys in required_systems):
                return False
        
        return True
    
    def execute(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Execute all actions in playbook"""
        logger.info(f"Executing playbook '{self.name}' for incident {incident.id}")
        
        results = {
            "playbook": self.name,
            "incident_id": incident.id,
            "actions_executed": 0,
            "actions_succeeded": 0,
            "actions_failed": 0,
            "start_time": datetime.utcnow().isoformat()
        }
        
        for action in self.actions:
            results["actions_executed"] += 1
            success = action.execute(incident)
            
            if success:
                results["actions_succeeded"] += 1
            else:
                results["actions_failed"] += 1
        
        results["end_time"] = datetime.utcnow().isoformat()
        incident.add_timeline_event(
            "playbook_executed",
            f"Playbook '{self.name}' completed: {results['actions_succeeded']}/{results['actions_executed']} actions succeeded",
            "automation"
        )
        
        return results


@dataclass
class IncidentMetrics:
    """Incident response metrics and KPIs"""
    total_incidents: int = 0
    open_incidents: int = 0
    incidents_by_severity: Dict[str, int] = field(default_factory=dict)
    incidents_by_status: Dict[str, int] = field(default_factory=dict)
    mean_time_to_detect: float = 0.0
    mean_time_to_respond: float = 0.0
    mean_time_to_resolve: float = 0.0
    resolution_times: List[float] = field(default_factory=list)
    
    def update(self, incidents: List[SecurityIncident]):
        """Update metrics from incident list"""
        from collections import defaultdict
        
        self.total_incidents = len(incidents)
        self.open_incidents = sum(1 for i in incidents if i.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING])
        
        self.incidents_by_severity = defaultdict(int)
        self.incidents_by_status = defaultdict(int)
        self.resolution_times = []
        
        for incident in incidents:
            self.incidents_by_severity[incident.severity.value] += 1
            self.incidents_by_status[incident.status.value] += 1
            
            if incident.resolved_at:
                resolution_time = incident.get_time_to_resolve_minutes()
                if resolution_time:
                    self.resolution_times.append(resolution_time)
        
        if self.resolution_times:
            self.mean_time_to_resolve = sum(self.resolution_times) / len(self.resolution_times)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "total_incidents": self.total_incidents,
            "open_incidents": self.open_incidents,
            "incidents_by_severity": dict(self.incidents_by_severity),
            "incidents_by_status": dict(self.incidents_by_status),
            "mean_time_to_detect": self.mean_time_to_detect,
            "mean_time_to_respond": self.mean_time_to_respond,
            "mean_time_to_resolve": self.mean_time_to_resolve,
            "min_resolution_time": min(self.resolution_times) if self.resolution_times else None,
            "max_resolution_time": max(self.resolution_times) if self.resolution_times else None,
        }


if __name__ == "__main__":
    print("Incident Response Core Module loaded")
    print("Available classes: SecurityIncident, ResponsePlaybook, IncidentMetrics")
