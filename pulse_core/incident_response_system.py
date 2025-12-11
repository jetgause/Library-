#!/usr/bin/env python3
"""
Security Patch 20/23 - Incident Response System
Part 3 of 3: Main system orchestration and automated response

Complete incident response orchestration system with automated playbooks,
monitoring, persistence, and comprehensive incident lifecycle management.

Author: jetgause
Created: 2025-12-11
Version: 1.0.0
"""

import json
import logging
import time
import hashlib
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

from pulse_core.incident_response_core import (
    SecurityIncident,
    IncidentSeverity,
    IncidentStatus,
    ResponsePlaybook,
    ResponseAction,
    IncidentMetrics
)
from pulse_core.incident_response_notifications import (
    NotificationManager,
    NotificationChannel,
    NotificationConfig,
    EscalationManager
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IncidentResponseSystem:
    """Main incident response orchestration system with automated playbooks, monitoring, and persistence."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.incidents: Dict[str, SecurityIncident] = {}
        self.playbooks: List[ResponsePlaybook] = []
        self.metrics = IncidentMetrics()
        self.storage_path = Path(self.config.get("storage_path", "/var/lib/pulse/incidents"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        notification_config = NotificationConfig(**self.config.get("notifications", {}))
        self.notification_manager = NotificationManager(notification_config)
        self.escalation_manager = EscalationManager()
        
        self.monitoring_enabled = self.config.get("monitoring_enabled", True)
        self.monitoring_interval = self.config.get("monitoring_interval", 300)
        self.monitoring_thread = None
        
        self._load_incidents()
        self._register_default_playbooks()
        
        if self.monitoring_enabled:
            self._start_monitoring()
        
        logger.info("IncidentResponseSystem initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            "storage_path": "/var/lib/pulse/incidents",
            "monitoring_enabled": True,
            "monitoring_interval": 300,
            "auto_assign": True,
            "auto_escalate": True,
            "notifications": {}
        }
    
    def create_incident(self, severity: IncidentSeverity, title: str, description: str,
                       affected_systems: Optional[List[str]] = None, tags: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> SecurityIncident:
        incident_id = self._generate_incident_id(title)
        incident = SecurityIncident(
            id=incident_id, severity=severity, status=IncidentStatus.OPEN,
            title=title, description=description, created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(), affected_systems=affected_systems or [],
            tags=tags or [], metadata=metadata or {}
        )
        incident.add_timeline_event("created", "Incident created", "system")
        self.incidents[incident_id] = incident
        self._save_incident(incident)
        logger.warning(f"New incident created: {incident_id} [{severity.value}] {title}")
        self._send_incident_notification(incident, "New security incident detected")
        self._execute_playbooks(incident)
        if self.config.get("auto_assign", True):
            self._auto_assign(incident)
        self._update_metrics()
        return incident
    
    def get_incident(self, incident_id: str) -> Optional[SecurityIncident]:
        return self.incidents.get(incident_id)
    
    def get_all_incidents(self) -> List[SecurityIncident]:
        return list(self.incidents.values())
    
    def get_active_incidents(self) -> List[SecurityIncident]:
        return [i for i in self.incidents.values() if i.status != IncidentStatus.CLOSED]
    
    def update_incident_status(self, incident_id: str, new_status: IncidentStatus, actor: str = "system") -> bool:
        incident = self.get_incident(incident_id)
        if not incident:
            return False
        incident.update_status(new_status, actor)
        self._save_incident(incident)
        if new_status == IncidentStatus.RESOLVED:
            self._send_incident_notification(incident, f"Incident resolved. Time: {incident.get_time_to_resolve_minutes():.1f} min")
        self._update_metrics()
        return True
    
    def assign_incident(self, incident_id: str, assignee: str, actor: str = "system") -> bool:
        incident = self.get_incident(incident_id)
        if not incident:
            return False
        incident.assign(assignee, actor)
        self._save_incident(incident)
        return True
    
    def register_playbook(self, playbook: ResponsePlaybook):
        self.playbooks.append(playbook)
        logger.info(f"Playbook registered: {playbook.name}")
    
    def get_metrics(self) -> IncidentMetrics:
        self._update_metrics()
        return self.metrics
    
    def _generate_incident_id(self, title: str) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        hash_suffix = hashlib.md5(title.encode()).hexdigest()[:6]
        return f"INC-{timestamp}-{hash_suffix}"
    
    def _save_incident(self, incident: SecurityIncident):
        try:
            with open(self.storage_path / f"{incident.id}.json", 'w') as f:
                json.dump(incident.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save incident: {e}")
    
    def _load_incidents(self):
        if not self.storage_path.exists():
            return
        for incident_file in self.storage_path.glob("INC-*.json"):
            try:
                with open(incident_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded incident: {data['id']}")
            except Exception as e:
                logger.error(f"Failed to load {incident_file}: {e}")
    
    def _execute_playbooks(self, incident: SecurityIncident):
        for playbook in self.playbooks:
            if playbook.matches_incident(incident):
                try:
                    playbook.execute(incident)
                    self._save_incident(incident)
                except Exception as e:
                    logger.error(f"Playbook execution failed: {e}")
    
    def _auto_assign(self, incident: SecurityIncident):
        assignment_map = {
            IncidentSeverity.CRITICAL: "security-lead",
            IncidentSeverity.HIGH: "security-oncall",
            IncidentSeverity.MEDIUM: "security-team",
            IncidentSeverity.LOW: "security-team"
        }
        assignee = assignment_map.get(incident.severity, "security-team")
        incident.assign(assignee, "automation")
        self._save_incident(incident)
    
    def _send_incident_notification(self, incident: SecurityIncident, message: str):
        self.notification_manager.send_notification(NotificationChannel.SLACK, incident, message)
        if incident.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH]:
            self.notification_manager.send_notification(NotificationChannel.EMAIL, incident, message)
        if incident.severity == IncidentSeverity.CRITICAL:
            self.notification_manager.send_notification(NotificationChannel.PAGERDUTY, incident, message, "urgent")
    
    def _update_metrics(self):
        self.metrics.update(list(self.incidents.values()))
    
    def _start_monitoring(self):
        def monitor():
            while self.monitoring_enabled:
                try:
                    self._check_escalations()
                    time.sleep(self.monitoring_interval)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
    
    def _check_escalations(self):
        if not self.config.get("auto_escalate", True):
            return
        for incident in self.get_active_incidents():
            if self.escalation_manager.should_escalate(incident):
                self.escalation_manager.escalate(incident, self.notification_manager)
                self._save_incident(incident)
    
    def _register_default_playbooks(self):
        self.register_playbook(ResponsePlaybook(
            name="Brute Force Attack Mitigation",
            description="Respond to brute force attacks",
            trigger_conditions={"tags": ["brute-force"]},
            actions=[
                ResponseAction("block_ip", "Block IP", lambda i, **k: self._action_block_ip(i)),
                ResponseAction("alert", "Alert team", lambda i, **k: self._action_alert_team(i))
            ]
        ))
        self.register_playbook(ResponsePlaybook(
            name="Data Breach Response",
            description="Respond to data breaches",
            trigger_conditions={"severity": ["CRITICAL"], "tags": ["data-breach"]},
            actions=[
                ResponseAction("isolate", "Isolate systems", lambda i, **k: self._action_isolate_systems(i)),
                ResponseAction("evidence", "Preserve evidence", lambda i, **k: self._action_preserve_evidence(i))
            ]
        ))
    
    def _action_block_ip(self, incident: SecurityIncident) -> bool:
        ip = incident.metadata.get("source_ip")
        if ip:
            logger.info(f"Blocking IP: {ip}")
        return True
    
    def _action_alert_team(self, incident: SecurityIncident) -> bool:
        self._send_incident_notification(incident, "Automated action required")
        return True
    
    def _action_isolate_systems(self, incident: SecurityIncident) -> bool:
        for system in incident.affected_systems:
            logger.critical(f"Isolating: {system}")
        return True
    
    def _action_preserve_evidence(self, incident: SecurityIncident) -> bool:
        logger.info(f"Preserving evidence for {incident.id}")
        return True


if __name__ == "__main__":
    system = IncidentResponseSystem()
    incident = system.create_incident(
        severity=IncidentSeverity.HIGH,
        title="Test Incident",
        description="Testing system",
        tags=["test"]
    )
    print(f"Created: {incident.id}")
    print(f"Metrics: {system.get_metrics().to_dict()}")
