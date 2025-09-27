"""
Multi-Agent System for Space Biology Knowledge Engine

This package implements a sophisticated multi-agent architecture using LangGraph
for intelligent processing of space biology queries.
"""

from .multi_agent_system import SpaceBiologyMultiAgentSystem, create_multi_agent_system
from .core.architecture import AgentType, TaskType

__all__ = [
    "SpaceBiologyMultiAgentSystem",
    "create_multi_agent_system",
    "AgentType",
    "TaskType"
]