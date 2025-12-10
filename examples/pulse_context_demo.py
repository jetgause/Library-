#!/usr/bin/env python3
"""
PULSE Context Awareness Demo
==============================

Demonstrates PULSE's context tracking and awareness features:
- Context creation and management
- State tracking across operations
- Context-aware decision making
- Historical context retrieval

Author: PULSE Development Team
Date: 2025-12-10
"""

from datetime import datetime
from typing import Dict, List, Optional

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class ContextManager:
    """Manages execution context and state tracking."""
    
    def __init__(self):
        self.contexts = {}
        self.active_context = None
        self.context_history = []
        
    def create_context(self, context_id: str, initial_state: Dict) -> Dict:
        """Create a new execution context."""
        context = {
            'id': context_id,
            'created_at': datetime.now().isoformat(),
            'state': initial_state.copy(),
            'history': [],
            'metadata': {
                'operations': 0,
                'last_update': datetime.now().isoformat()
            }
        }
        
        self.contexts[context_id] = context
        self.context_history.append({
            'action': 'created',
            'context_id': context_id,
            'timestamp': context['created_at']
        })
        
        return context
    
    def activate_context(self, context_id: str) -> bool:
        """Activate a specific context."""
        if context_id in self.contexts:
            self.active_context = context_id
            return True
        return False
    
    def update_state(self, key: str, value, context_id: Optional[str] = None):
        """Update context state."""
        ctx_id = context_id or self.active_context
        
        if ctx_id not in self.contexts:
            raise ValueError(f"Context {ctx_id} not found")
        
        context = self.contexts[ctx_id]
        old_value = context['state'].get(key)
        
        context['state'][key] = value
        context['metadata']['operations'] += 1
        context['metadata']['last_update'] = datetime.now().isoformat()
        
        context['history'].append({
            'timestamp': datetime.now().isoformat(),
            'key': key,
            'old_value': old_value,
            'new_value': value
        })
    
    def get_state(self, key: str, context_id: Optional[str] = None):
        """Retrieve state value from context."""
        ctx_id = context_id or self.active_context
        
        if ctx_id not in self.contexts:
            return None
        
        return self.contexts[ctx_id]['state'].get(key)
    
    def get_context_history(self, context_id: str) -> List[Dict]:
        """Get operation history for a context."""
        if context_id not in self.contexts:
            return []
        
        return self.contexts[context_id]['history']


def run_demo():
    """Run context awareness demonstration."""
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print("  PULSE Context Awareness Demo")
    print(f"{'='*70}{Colors.ENDC}\n")
    
    manager = ContextManager()
    
    # Demo 1: Create trading context
    print(f"{Colors.BOLD}1. Creating Trading Context{Colors.ENDC}\n")
    
    trading_ctx = manager.create_context('trading_session_1', {
        'account_balance': 100000.0,
        'positions': [],
        'risk_level': 'moderate',
        'market_regime': 'trending'
    })
    
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} Created context: {trading_ctx['id']}")
    print(f"  Initial State:")
    for key, value in trading_ctx['state'].items():
        print(f"    • {key}: {Colors.OKBLUE}{value}{Colors.ENDC}")
    
    # Demo 2: Activate and update context
    print(f"\n{Colors.BOLD}2. Activating and Updating Context{Colors.ENDC}\n")
    
    manager.activate_context('trading_session_1')
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} Activated context")
    
    # Simulate trading operations
    manager.update_state('account_balance', 102500.0)
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} Updated balance: $102,500")
    
    manager.update_state('positions', ['SPY_CALL_450'])
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} Added position: SPY_CALL_450")
    
    manager.update_state('market_regime', 'volatile')
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} Market regime changed to volatile")
    
    # Demo 3: Context history
    print(f"\n{Colors.BOLD}3. Context History{Colors.ENDC}\n")
    
    history = manager.get_context_history('trading_session_1')
    print(f"Operation History ({len(history)} operations):\n")
    
    for i, entry in enumerate(history, 1):
        print(f"  {i}. {entry['key']}: {Colors.WARNING}{entry['old_value']}{Colors.ENDC} → "
              f"{Colors.OKGREEN}{entry['new_value']}{Colors.ENDC}")
    
    # Demo 4: Multi-context management
    print(f"\n{Colors.BOLD}4. Multiple Contexts{Colors.ENDC}\n")
    
    backtest_ctx = manager.create_context('backtest_1', {
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'strategy': 'mean_reversion'
    })
    
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} Created backtest context")
    print(f"  Active Contexts: {len(manager.contexts)}")
    for ctx_id in manager.contexts:
        print(f"    • {ctx_id}")
    
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}Demo completed successfully!{Colors.ENDC}\n")


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Demo interrupted by user{Colors.ENDC}\n")
    except Exception as e:
        print(f"\n{Colors.WARNING}Error: {e}{Colors.ENDC}\n")
