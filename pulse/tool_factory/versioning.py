"""
Version Manager - Handles semantic versioning and tool evolution
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import re


class VersionManager:
    """Manages tool versions and evolution tracking"""
    
    def __init__(self):
        self.version_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def parse_version(self, version_string: str) -> tuple[int, int, int]:
        """Parse semantic version string (e.g., '1.2.3')"""
        match = re.match(r'(\d+)\.(\d+)\.(\d+)', version_string)
        if not match:
            raise ValueError(f"Invalid version format: {version_string}")
        return tuple(map(int, match.groups()))
    
    def compare_versions(self, v1: str, v2: str) -> int:
        """
        Compare two version strings
        Returns: -1 if v1 < v2, 0 if equal, 1 if v1 > v2
        """
        major1, minor1, patch1 = self.parse_version(v1)
        major2, minor2, patch2 = self.parse_version(v2)
        
        if (major1, minor1, patch1) < (major2, minor2, patch2):
            return -1
        elif (major1, minor1, patch1) > (major2, minor2, patch2):
            return 1
        return 0
    
    def increment_version(self, current_version: str, level: str = "patch") -> str:
        """
        Increment version based on level: 'major', 'minor', or 'patch'
        """
        major, minor, patch = self.parse_version(current_version)
        
        if level == "major":
            return f"{major + 1}.0.0"
        elif level == "minor":
            return f"{major}.{minor + 1}.0"
        elif level == "patch":
            return f"{major}.{minor}.{patch + 1}"
        else:
            raise ValueError(f"Invalid level: {level}")
    
    def record_version(self, tool_id: str, version: str, changes: Dict[str, Any]):
        """Record a version in the history"""
        if tool_id not in self.version_history:
            self.version_history[tool_id] = []
        
        version_record = {
            "version": version,
            "timestamp": datetime.utcnow().isoformat(),
            "changes": changes
        }
        
        self.version_history[tool_id].append(version_record)
    
    def get_version_history(self, tool_id: str) -> List[Dict[str, Any]]:
        """Get complete version history for a tool"""
        return self.version_history.get(tool_id, [])
    
    def get_latest_version(self, tool_id: str) -> Optional[str]:
        """Get the latest version number for a tool"""
        history = self.version_history.get(tool_id, [])
        if not history:
            return None
        
        versions = [h["version"] for h in history]
        versions.sort(key=lambda v: self.parse_version(v), reverse=True)
        return versions[0]
    
    def create_changelog(self, tool_id: str) -> str:
        """Generate a markdown changelog for a tool"""
        history = self.get_version_history(tool_id)
        if not history:
            return f"# Changelog for {tool_id}\n\nNo version history available."
        
        changelog = f"# Changelog for {tool_id}\n\n"
        
        # Sort by version (newest first)
        sorted_history = sorted(
            history,
            key=lambda h: self.parse_version(h["version"]),
            reverse=True
        )
        
        for record in sorted_history:
            changelog += f"## Version {record['version']}\n"
            changelog += f"*Released: {record['timestamp']}*\n\n"
            
            changes = record.get("changes", {})
            for change_type, items in changes.items():
                changelog += f"### {change_type.title()}\n"
                if isinstance(items, list):
                    for item in items:
                        changelog += f"- {item}\n"
                else:
                    changelog += f"- {items}\n"
                changelog += "\n"
        
        return changelog
    
    def is_compatible(self, required_version: str, available_version: str) -> bool:
        """
        Check if available version is compatible with required version
        Uses semantic versioning rules: major version must match
        """
        req_major, req_minor, req_patch = self.parse_version(required_version)
        avail_major, avail_minor, avail_patch = self.parse_version(available_version)
        
        # Major version must match
        if req_major != avail_major:
            return False
        
        # Available version must be >= required version
        if (avail_minor, avail_patch) < (req_minor, req_patch):
            return False
        
        return True
