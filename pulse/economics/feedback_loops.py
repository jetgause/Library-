"""
Feedback Loops - Implements economic feedback mechanisms for self-optimization
"""
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime


class FeedbackLoop:
    """Manages feedback loops for continuous system improvement"""
    
    def __init__(self):
        self.loops: Dict[str, Dict[str, Any]] = {}
        self.feedback_history: Dict[str, List[Dict[str, Any]]] = {}
        self.triggers: Dict[str, Callable] = {}
    
    def create_loop(
        self,
        loop_id: str,
        metric: str,
        threshold: float,
        action: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a feedback loop
        
        Args:
            loop_id: Unique identifier
            metric: Metric to monitor (e.g., 'value_score', 'cpu_usage')
            threshold: Threshold value that triggers action
            action: Action to take (e.g., 'optimize', 'deprecate', 'scale_up')
            parameters: Additional parameters for the action
        """
        loop = {
            "id": loop_id,
            "metric": metric,
            "threshold": threshold,
            "action": action,
            "parameters": parameters or {},
            "created_at": datetime.utcnow().isoformat(),
            "status": "active",
            "trigger_count": 0
        }
        
        self.loops[loop_id] = loop
        return loop
    
    def evaluate(self, loop_id: str, current_value: float) -> Dict[str, Any]:
        """
        Evaluate if a feedback loop should trigger
        
        Returns:
            {
                "triggered": bool,
                "action": str,
                "details": dict
            }
        """
        if loop_id not in self.loops:
            return {"triggered": False, "error": "Loop not found"}
        
        loop = self.loops[loop_id]
        
        if loop["status"] != "active":
            return {"triggered": False, "reason": "Loop not active"}
        
        threshold = loop["threshold"]
        triggered = current_value >= threshold
        
        result = {
            "triggered": triggered,
            "loop_id": loop_id,
            "metric": loop["metric"],
            "current_value": current_value,
            "threshold": threshold,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if triggered:
            result["action"] = loop["action"]
            result["parameters"] = loop["parameters"]
            
            # Update trigger count
            loop["trigger_count"] += 1
            loop["last_triggered"] = result["timestamp"]
            
            # Record in history
            self._record_feedback(loop_id, result)
        
        return result
    
    def batch_evaluate(self, evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple loops at once
        
        Evaluations format:
        [
            {"loop_id": "loop1", "current_value": 0.85},
            {"loop_id": "loop2", "current_value": 45.2}
        ]
        """
        results = []
        for evaluation in evaluations:
            result = self.evaluate(
                evaluation["loop_id"],
                evaluation["current_value"]
            )
            results.append(result)
        
        return results
    
    def register_trigger(self, action: str, callback: Callable):
        """Register a callback function for an action"""
        self.triggers[action] = callback
    
    def execute_action(self, loop_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the action associated with a loop"""
        if loop_id not in self.loops:
            return {"success": False, "error": "Loop not found"}
        
        loop = self.loops[loop_id]
        action = loop["action"]
        
        if action not in self.triggers:
            return {
                "success": False,
                "error": f"No trigger registered for action: {action}"
            }
        
        try:
            callback = self.triggers[action]
            result = callback(loop, context or {})
            
            return {
                "success": True,
                "action": action,
                "result": result,
                "executed_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _record_feedback(self, loop_id: str, feedback: Dict[str, Any]):
        """Record feedback event in history"""
        if loop_id not in self.feedback_history:
            self.feedback_history[loop_id] = []
        
        self.feedback_history[loop_id].append(feedback)
    
    def get_history(self, loop_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get feedback history for a loop"""
        history = self.feedback_history.get(loop_id, [])
        
        if limit:
            return history[-limit:]
        return history
    
    def get_loop_stats(self, loop_id: str) -> Dict[str, Any]:
        """Get statistics for a feedback loop"""
        if loop_id not in self.loops:
            return {"error": "Loop not found"}
        
        loop = self.loops[loop_id]
        history = self.feedback_history.get(loop_id, [])
        
        return {
            "loop_id": loop_id,
            "status": loop["status"],
            "trigger_count": loop["trigger_count"],
            "last_triggered": loop.get("last_triggered"),
            "history_count": len(history),
            "created_at": loop["created_at"]
        }
    
    def update_threshold(self, loop_id: str, new_threshold: float) -> bool:
        """Update the threshold for a loop"""
        if loop_id not in self.loops:
            return False
        
        old_threshold = self.loops[loop_id]["threshold"]
        self.loops[loop_id]["threshold"] = new_threshold
        self.loops[loop_id]["threshold_updated_at"] = datetime.utcnow().isoformat()
        
        # Record the change
        self._record_feedback(loop_id, {
            "event": "threshold_updated",
            "old_threshold": old_threshold,
            "new_threshold": new_threshold,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return True
    
    def activate_loop(self, loop_id: str) -> bool:
        """Activate a feedback loop"""
        if loop_id in self.loops:
            self.loops[loop_id]["status"] = "active"
            return True
        return False
    
    def deactivate_loop(self, loop_id: str) -> bool:
        """Deactivate a feedback loop"""
        if loop_id in self.loops:
            self.loops[loop_id]["status"] = "inactive"
            return True
        return False
    
    def get_all_loops(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all loops, optionally filtered by status"""
        loops = list(self.loops.values())
        
        if status:
            loops = [l for l in loops if l["status"] == status]
        
        return loops
    
    def analyze_trends(self, loop_id: str) -> Dict[str, Any]:
        """Analyze trigger trends for a loop"""
        history = self.feedback_history.get(loop_id, [])
        
        if not history:
            return {"error": "No history available"}
        
        triggered_events = [h for h in history if h.get("triggered")]
        
        if not triggered_events:
            return {
                "loop_id": loop_id,
                "total_evaluations": len(history),
                "triggers": 0,
                "trigger_rate": 0.0
            }
        
        trigger_rate = len(triggered_events) / len(history)
        
        # Calculate average value when triggered
        avg_value = sum(e["current_value"] for e in triggered_events) / len(triggered_events)
        
        return {
            "loop_id": loop_id,
            "total_evaluations": len(history),
            "triggers": len(triggered_events),
            "trigger_rate": round(trigger_rate, 3),
            "avg_trigger_value": round(avg_value, 3)
        }
