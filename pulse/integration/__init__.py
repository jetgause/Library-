"""
Integration Module
Handles external API integration and GitHub connectivity
"""

from pulse.integration.api import PulseAPI
from pulse.integration.github_integration import GitHubIntegration

__all__ = ["PulseAPI", "GitHubIntegration"]
