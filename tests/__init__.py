"""
PULSE System Test Suite

Comprehensive testing framework for all PULSE modules including:
- Tool Factory tests
- Economics module tests
- Taxonomy and metadata tests
- Integration tests
- API tests

Author: jetgause
Created: 2025-12-09
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
test_dir = Path(__file__).parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))

__version__ = "1.0.0"
__all__ = []
