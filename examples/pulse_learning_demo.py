#!/usr/bin/env python3
"""
PULSE Learning System Demo
===========================

This example demonstrates PULSE's learning and feedback mechanisms:
- Recording user feedback
- Learning from interactions
- Pattern recognition
- Adaptive behavior

Author: PULSE Development Team
Date: 2025-12-10
"""

import sys
from datetime import datetime
from typing import Dict, List

class Colors:
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class LearningSystem:
    """PULSE learning and feedback system."""
    
    def __init__(self):
        self.feedback_history = []
        self.learning_patterns = {}
        
    def record_feedback(self, tool_id: str, rating: int, comment: str = ""):
        """Record user feedback for a tool."""
        feedback = {
            'tool_id': tool_id,
            'rating': rating,
            'comment': comment,
            'timestamp': datetime.now().isoformat()
        }
        self.feedback_history.append(feedback)
        self._update_patterns(tool_id, rating)
        return feedback
    
    def _update_patterns(self, tool_id: str, rating: int):
        """Update learning patterns based on feedback."""
        if tool_id not in self.learning_patterns:
            self.learning_patterns[tool_id] = {
                'total_ratings': 0,
                'average_rating': 0.0,
                'feedback_count': 0
            }
        
        pattern = self.learning_patterns[tool_id]
        pattern['feedback_count'] += 1
        pattern['total_ratings'] += rating
        pattern['average_rating'] = pattern['total_ratings'] / pattern['feedback_count']
    
    def get_tool_rating(self, tool_id: str) -> float:
        """Get average rating for a tool."""
        pattern = self.learning_patterns.get(tool_id)
        return pattern['average_rating'] if pattern else 0.0
    
    def run_demo(self):
        """Run learning system demo."""
        print(f"\n{Colors.BOLD}{'='*60}")
        print("  PULSE Learning System Demo")
        print(f"{'='*60}{Colors.ENDC}\n")
        
        # Simulate feedback
        print(f"{Colors.BOLD}Recording User Feedback:{Colors.ENDC}\n")
        
        feedbacks = [
            ('data_cleaner', 5, 'Excellent tool!'),
            ('data_cleaner', 4, 'Very helpful'),
            ('api_gateway', 3, 'Needs improvement'),
        ]
        
        for tool_id, rating, comment in feedbacks:
            fb = self.record_feedback(tool_id, rating, comment)
            stars = '⭐' * rating
            print(f"{Colors.OKGREEN}✓{Colors.ENDC} {tool_id}: {stars} ({rating}/5)")
            print(f"  Comment: \"{comment}\"\n")
        
        # Show patterns
        print(f"{Colors.BOLD}Learning Patterns:{Colors.ENDC}\n")
        for tool_id, pattern in self.learning_patterns.items():
            avg = pattern['average_rating']
            color = Colors.OKGREEN if avg >= 4 else Colors.WARNING if avg >= 3 else Colors.FAIL
            print(f"  {tool_id}:")
            print(f"    Average Rating: {color}{avg:.1f}/5{Colors.ENDC}")
            print(f"    Feedback Count: {pattern['feedback_count']}\n")
        
        print(f"{Colors.OKGREEN}Learning demo complete!{Colors.ENDC}\n")


def main():
    try:
        system = LearningSystem()
        system.run_demo()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Interrupted.{Colors.ENDC}")
        sys.exit(0)


if __name__ == "__main__":
    main()
