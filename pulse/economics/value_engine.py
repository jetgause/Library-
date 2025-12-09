"""
Value Engine - Calculates and tracks tool value metrics
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


class ValueEngine:
    """Calculates economic value of tools based on usage, impact, and efficiency"""
    
    def __init__(self):
        self.value_history: Dict[str, List[Dict[str, Any]]] = {}
        self.weights = {
            "usage_frequency": 0.3,
            "impact_score": 0.4,
            "efficiency": 0.2,
            "recency": 0.1
        }
    
    def calculate_value(self, tool: Dict[str, Any], metrics: Dict[str, Any]) -> float:
        """
        Calculate overall value score for a tool
        
        Metrics expected:
        - usage_count: Number of times used
        - success_rate: Percentage of successful executions
        - avg_execution_time: Average time to complete
        - user_feedback: Average user rating (0-1)
        - last_used: Timestamp of last usage
        """
        usage_score = self._calculate_usage_score(metrics.get("usage_count", 0))
        impact_score = self._calculate_impact_score(
            metrics.get("success_rate", 0.5),
            metrics.get("user_feedback", 0.5)
        )
        efficiency_score = self._calculate_efficiency_score(
            metrics.get("avg_execution_time", 100)
        )
        recency_score = self._calculate_recency_score(
            metrics.get("last_used")
        )
        
        total_value = (
            self.weights["usage_frequency"] * usage_score +
            self.weights["impact_score"] * impact_score +
            self.weights["efficiency"] * efficiency_score +
            self.weights["recency"] * recency_score
        )
        
        # Record in history
        self._record_value(tool["id"], total_value, metrics)
        
        return round(total_value, 3)
    
    def _calculate_usage_score(self, usage_count: int) -> float:
        """Normalize usage count to 0-1 scale using logarithmic scaling"""
        if usage_count <= 0:
            return 0.0
        # Log scale: 1 use = 0.1, 10 uses = 0.5, 100 uses = 0.8, 1000+ uses = 1.0
        import math
        score = math.log10(usage_count + 1) / 3.0  # log10(1000) ≈ 3
        return min(score, 1.0)
    
    def _calculate_impact_score(self, success_rate: float, user_feedback: float) -> float:
        """Combine success rate and user feedback"""
        return (success_rate * 0.6 + user_feedback * 0.4)
    
    def _calculate_efficiency_score(self, avg_time: float) -> float:
        """
        Score based on execution time (lower is better)
        < 1s = 1.0, 1-10s = 0.8, 10-60s = 0.6, 60-300s = 0.4, > 300s = 0.2
        """
        if avg_time < 1:
            return 1.0
        elif avg_time < 10:
            return 0.8
        elif avg_time < 60:
            return 0.6
        elif avg_time < 300:
            return 0.4
        else:
            return 0.2
    
    def _calculate_recency_score(self, last_used: Optional[str]) -> float:
        """Score based on how recently the tool was used"""
        if not last_used:
            return 0.0
        
        try:
            last_used_dt = datetime.fromisoformat(last_used.replace('Z', '+00:00'))
            age = datetime.utcnow() - last_used_dt.replace(tzinfo=None)
            
            # < 1 day = 1.0, 1-7 days = 0.8, 7-30 days = 0.5, 30-90 days = 0.3, > 90 days = 0.1
            if age < timedelta(days=1):
                return 1.0
            elif age < timedelta(days=7):
                return 0.8
            elif age < timedelta(days=30):
                return 0.5
            elif age < timedelta(days=90):
                return 0.3
            else:
                return 0.1
        except Exception:
            return 0.0
    
    def _record_value(self, tool_id: str, value: float, metrics: Dict[str, Any]):
        """Record value calculation in history"""
        if tool_id not in self.value_history:
            self.value_history[tool_id] = []
        
        self.value_history[tool_id].append({
            "timestamp": datetime.utcnow().isoformat(),
            "value": value,
            "metrics": metrics
        })
    
    def get_value_trend(self, tool_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get value trend over time"""
        history = self.value_history.get(tool_id, [])
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        return [
            h for h in history
            if datetime.fromisoformat(h["timestamp"]) > cutoff
        ]
    
    def get_top_tools(self, tools: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top N tools by value score"""
        sorted_tools = sorted(
            tools,
            key=lambda t: t.get("value_score", 0),
            reverse=True
        )
        return sorted_tools[:limit]
    
    def compare_tools(self, tool_id1: str, tool_id2: str) -> Dict[str, Any]:
        """Compare value metrics between two tools"""
        history1 = self.value_history.get(tool_id1, [])
        history2 = self.value_history.get(tool_id2, [])
        
        avg_value1 = sum(h["value"] for h in history1) / len(history1) if history1 else 0
        avg_value2 = sum(h["value"] for h in history2) / len(history2) if history2 else 0
        
        return {
            "tool1": {
                "id": tool_id1,
                "avg_value": round(avg_value1, 3),
                "data_points": len(history1)
            },
            "tool2": {
                "id": tool_id2,
                "avg_value": round(avg_value2, 3),
                "data_points": len(history2)
            },
            "difference": round(abs(avg_value1 - avg_value2), 3)
        }
