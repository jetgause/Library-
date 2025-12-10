#!/usr/bin/env python3
"""
PULSE Context Awareness Demo
============================
Demonstrates context tracking and awareness features in the PULSE system.
Shows how PULSE maintains conversational context, tracks state changes,
and adapts responses based on contextual understanding.

Author: jetgause
Date: 2025-12-10
"""

import time
from datetime import datetime
from typing import Dict, List, Any


# ANSI Color codes for console output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ContextTracker:
    """Tracks conversational context and state changes."""
    
    def __init__(self):
        self.context_stack: List[Dict[str, Any]] = []
        self.user_preferences: Dict[str, Any] = {}
        self.session_history: List[str] = []
        
    def push_context(self, context_type: str, data: Dict[str, Any]) -> None:
        """Add new context to the stack."""
        context = {
            'type': context_type,
            'data': data,
            'timestamp': datetime.now().isoformat(),
        }
        self.context_stack.append(context)
        self.session_history.append(f"{context_type}: {data.get('summary', 'N/A')}")
        
    def get_current_context(self) -> Dict[str, Any]:
        """Retrieve the most recent context."""
        return self.context_stack[-1] if self.context_stack else {}
    
    def get_context_depth(self) -> int:
        """Return the depth of the context stack."""
        return len(self.context_stack)
    
    def update_preferences(self, key: str, value: Any) -> None:
        """Update user preferences based on interaction patterns."""
        self.user_preferences[key] = value


def print_header(text: str) -> None:
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")


def print_info(label: str, value: str, color: str = Colors.CYAN) -> None:
    """Print formatted information."""
    print(f"{color}{Colors.BOLD}{label}:{Colors.ENDC} {value}")


def print_success(message: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")


def print_activity(message: str) -> None:
    """Print activity message."""
    print(f"{Colors.YELLOW}⚡ {message}{Colors.ENDC}")


def demo_context_initialization() -> ContextTracker:
    """Demonstrate context tracker initialization."""
    print_header("Context Tracker Initialization")
    
    tracker = ContextTracker()
    print_success("Context tracker initialized")
    print_info("Context Depth", str(tracker.get_context_depth()), Colors.BLUE)
    
    return tracker


def demo_context_building(tracker: ContextTracker) -> None:
    """Demonstrate building context through interactions."""
    print_header("Building Conversational Context")
    
    interactions = [
        ("user_query", {"summary": "Request for weather info", "topic": "weather", "location": "NYC"}),
        ("system_response", {"summary": "Provided weather data", "confidence": 0.95}),
        ("user_query", {"summary": "Follow-up about temperature", "topic": "weather", "references_previous": True}),
        ("system_response", {"summary": "Clarified temperature details", "confidence": 0.98}),
    ]
    
    for idx, (ctx_type, data) in enumerate(interactions, 1):
        print_activity(f"Processing interaction {idx}: {ctx_type}")
        tracker.push_context(ctx_type, data)
        time.sleep(0.3)
        
        if data.get("references_previous"):
            print_info("  Context Awareness", "Detected reference to previous exchange", Colors.GREEN)
        
        print_info("  Context Depth", str(tracker.get_context_depth()), Colors.CYAN)
    
    print_success(f"Built context stack with {tracker.get_context_depth()} entries")


def demo_context_awareness(tracker: ContextTracker) -> None:
    """Demonstrate context-aware response generation."""
    print_header("Context-Aware Response Generation")
    
    current = tracker.get_current_context()
    print_info("Current Context Type", current.get('type', 'None'), Colors.BLUE)
    print_info("Context Data", str(current.get('data', {})), Colors.CYAN)
    
    # Simulate context-aware decision making
    if current.get('data', {}).get('references_previous'):
        print_success("Context awareness: Linking to previous conversation")
        print_info("  Strategy", "Use contextual information from history", Colors.YELLOW)
    
    print_activity("Analyzing context stack for patterns...")
    time.sleep(0.5)
    
    weather_count = sum(1 for ctx in tracker.context_stack 
                       if ctx.get('data', {}).get('topic') == 'weather')
    print_info("Topic Focus", f"Weather-related queries: {weather_count}", Colors.GREEN)


def demo_preference_learning(tracker: ContextTracker) -> None:
    """Demonstrate learning user preferences from context."""
    print_header("Adaptive Preference Learning")
    
    # Simulate preference extraction
    preferences = [
        ("preferred_detail_level", "high"),
        ("response_style", "technical"),
        ("topic_interest", "weather"),
    ]
    
    for key, value in preferences:
        print_activity(f"Learning preference: {key}")
        tracker.update_preferences(key, value)
        time.sleep(0.3)
        print_success(f"Updated: {key} = {value}")
    
    print_info("\nLearned Preferences", str(tracker.user_preferences), Colors.CYAN)


def main():
    """Run the PULSE context awareness demonstration."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}PULSE Context Awareness Demo{Colors.ENDC}")
    print(f"{Colors.CYAN}Demonstrating context tracking and awareness features{Colors.ENDC}\n")
    
    # Initialize context tracker
    tracker = demo_context_initialization()
    time.sleep(1)
    
    # Build conversational context
    demo_context_building(tracker)
    time.sleep(1)
    
    # Demonstrate context awareness
    demo_context_awareness(tracker)
    time.sleep(1)
    
    # Learn user preferences
    demo_preference_learning(tracker)
    
    # Summary
    print_header("Demo Summary")
    print_info("Total Interactions", str(len(tracker.session_history)), Colors.BLUE)
    print_info("Context Stack Depth", str(tracker.get_context_depth()), Colors.CYAN)
    print_info("Preferences Learned", str(len(tracker.user_preferences)), Colors.GREEN)
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Context awareness demo completed successfully!{Colors.ENDC}\n")


if __name__ == "__main__":
    main()
