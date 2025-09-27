"""
Main Multi-Agent System Integration

This module provides the main interface for the multi-agent system,
integrating all components and providing a clean API for the rest of the application.
"""

from typing import Dict, List, Any, Optional
import asyncio
import logging
from datetime import datetime

from .core.architecture import MultiAgentOrchestrator, AgentType, TaskType, AgentState
from .tools.knowledge_tools import create_tool_registry, ToolRegistry
from .workflows.dynamic_orchestrator import AdaptiveWorkflowEngine, WorkflowStatus

from ..services.knowledge_graph import KnowledgeGraphService
from ..services.semantic_search import SemanticSearchService
from ..services.llm_service import LLMService
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class SpaceBiologyMultiAgentSystem:
    """
    Main multi-agent system for the Space Biology Knowledge Engine.

    This system orchestrates multiple specialized AI agents to process complex
    queries about space biology research, providing intelligent responses
    that combine literature search, data analysis, and expert reasoning.
    """

    def __init__(self, services: Dict[str, Any]):
        """
        Initialize the multi-agent system.

        Args:
            services: Dictionary containing initialized services
                     (kg_service, search_service, llm_service, etc.)
        """
        self.services = services
        self.settings = get_settings()

        # Initialize core components
        self.tool_registry = create_tool_registry(services)
        self.workflow_engine = AdaptiveWorkflowEngine(self.tool_registry)

        # Initialize the orchestrator
        llm_config = {
            "model": self.settings.default_llm_model,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "openai_api_key": self.settings.openai_api_key
        }
        self.orchestrator = MultiAgentOrchestrator(llm_config)

        # Performance tracking
        self.query_history = []
        self.system_metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "average_response_time": 0,
            "agent_utilization": {},
            "tool_usage": {}
        }

        logger.info("Multi-agent system initialized successfully")

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None,
                          user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query through the multi-agent system.

        Args:
            query: User query string
            context: Additional context for the query
            user_preferences: User preferences for response format, detail level, etc.

        Returns:
            Comprehensive response including answer, sources, metadata, and workflow info
        """
        start_time = datetime.utcnow()
        query_id = f"query_{start_time.isoformat()}"

        try:
            # Prepare context
            full_context = {
                "user_preferences": user_preferences or {},
                "query_id": query_id,
                "timestamp": start_time.isoformat(),
                **(context or {})
            }

            # Log query
            logger.info(f"Processing query {query_id}: {query[:100]}...")

            # Plan workflow
            workflow_plan = await self.workflow_engine.plan_workflow(query, full_context)

            # Create initial agent state
            initial_state = AgentState(
                messages=[],
                current_task=workflow_plan.task_type,
                query=query,
                context=full_context,
                intermediate_results={},
                final_response=None,
                confidence_scores={},
                sources=[],
                metadata={"query_id": query_id, "workflow_plan": workflow_plan.__dict__},
                next_agent=None,
                workflow_status="planned",
                error_messages=[]
            )

            # Execute workflow
            workflow_execution = await self.workflow_engine.execute_workflow(
                workflow_plan, initial_state
            )

            # Generate final response using orchestrator
            if workflow_execution.status == WorkflowStatus.COMPLETED:
                final_result = await self.orchestrator.process_query(query, full_context)
            else:
                # Fallback to simple processing if workflow failed
                final_result = await self._fallback_processing(query, full_context)

            # Calculate response time
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()

            # Prepare comprehensive response
            response = {
                "query_id": query_id,
                "query": query,
                "response": final_result.get("response", ""),
                "confidence": self._calculate_overall_confidence(
                    final_result.get("confidence_scores", {}),
                    workflow_execution.performance_metrics
                ),
                "sources": final_result.get("sources", []),
                "workflow_info": {
                    "workflow_id": workflow_execution.workflow_id,
                    "status": workflow_execution.status.value,
                    "agents_used": list(workflow_execution.completed_nodes),
                    "execution_time": response_time,
                    "performance_metrics": workflow_execution.performance_metrics
                },
                "metadata": {
                    "timestamp": end_time.isoformat(),
                    "processing_time_seconds": response_time,
                    "agents_involved": final_result.get("agents_used", []),
                    "tools_used": workflow_plan.required_tools,
                    "task_type": workflow_plan.task_type.value if workflow_plan.task_type else None
                }
            }

            # Update system metrics
            await self._update_system_metrics(response, workflow_execution)

            # Store query history
            self.query_history.append({
                "query_id": query_id,
                "query": query,
                "response_time": response_time,
                "success": workflow_execution.status == WorkflowStatus.COMPLETED,
                "agents_used": list(workflow_execution.completed_nodes)
            })

            logger.info(f"Query {query_id} processed successfully in {response_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Error processing query {query_id}: {str(e)}")

            # Return error response
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()

            return {
                "query_id": query_id,
                "query": query,
                "response": f"I apologize, but I encountered an error processing your query: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "workflow_info": {
                    "status": "failed",
                    "error": str(e),
                    "execution_time": response_time
                },
                "metadata": {
                    "timestamp": end_time.isoformat(),
                    "processing_time_seconds": response_time,
                    "error": True
                }
            }

    async def _fallback_processing(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback processing when main workflow fails."""
        try:
            # Simple direct LLM call with knowledge base context
            search_results = await self.services["search_service"].search(query, limit=5)

            context_text = "\n".join([
                f"- {result.get('title', '')}: {result.get('summary', '')}"
                for result in search_results
            ])

            prompt = f"""
            Based on the following space biology research context, please answer this query:

            Query: {query}

            Context:
            {context_text}

            Please provide a comprehensive answer based on the available research.
            """

            response = await self.services["llm_service"].generate_response(prompt, max_tokens=1500)

            return {
                "response": response,
                "confidence_scores": {"fallback": 0.6},
                "sources": search_results,
                "agents_used": ["fallback_llm"],
                "metadata": {"processing_mode": "fallback"}
            }

        except Exception as e:
            return {
                "response": "I apologize, but I'm unable to process your query at this time due to system issues.",
                "confidence_scores": {"error": 0.0},
                "sources": [],
                "agents_used": [],
                "metadata": {"error": str(e)}
            }

    def _calculate_overall_confidence(self, agent_confidences: Dict[str, float],
                                    performance_metrics: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the response."""
        if not agent_confidences:
            return 0.5

        # Weight by agent performance and workflow success
        base_confidence = sum(agent_confidences.values()) / len(agent_confidences)

        # Adjust based on workflow performance
        success_rate = performance_metrics.get("success_rate", 1.0)
        confidence_adjustment = success_rate * 0.2  # Max 20% adjustment

        final_confidence = min(base_confidence + confidence_adjustment, 1.0)
        return round(final_confidence, 3)

    async def _update_system_metrics(self, response: Dict[str, Any],
                                   workflow_execution) -> None:
        """Update system-wide performance metrics."""
        self.system_metrics["total_queries"] += 1

        if workflow_execution.status == WorkflowStatus.COMPLETED:
            self.system_metrics["successful_queries"] += 1

        # Update average response time
        current_avg = self.system_metrics["average_response_time"]
        total_queries = self.system_metrics["total_queries"]
        new_time = response["metadata"]["processing_time_seconds"]

        self.system_metrics["average_response_time"] = (
            (current_avg * (total_queries - 1) + new_time) / total_queries
        )

        # Update agent utilization
        for agent in workflow_execution.completed_nodes:
            if agent not in self.system_metrics["agent_utilization"]:
                self.system_metrics["agent_utilization"][agent] = 0
            self.system_metrics["agent_utilization"][agent] += 1

        # Update tool usage
        for tool in response["metadata"].get("tools_used", []):
            if tool not in self.system_metrics["tool_usage"]:
                self.system_metrics["tool_usage"][tool] = 0
            self.system_metrics["tool_usage"][tool] += 1

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and performance metrics."""
        workflow_analytics = self.workflow_engine.get_workflow_analytics()

        return {
            "system_health": "healthy",  # Could be computed based on error rates
            "components": {
                "orchestrator": "active",
                "workflow_engine": "active",
                "tool_registry": f"{len(self.tool_registry.tools)} tools available"
            },
            "performance_metrics": self.system_metrics,
            "workflow_analytics": workflow_analytics,
            "recent_queries": self.query_history[-10:] if self.query_history else [],
            "capabilities": {
                "supported_agent_types": [agent.value for agent in AgentType],
                "supported_task_types": [task.value for task in TaskType],
                "available_tools": self.tool_registry.list_available_tools()
            }
        }

    async def explain_reasoning(self, query_id: str) -> Dict[str, Any]:
        """Explain the reasoning process for a specific query."""
        # Find the query in history
        query_info = None
        for query in self.query_history:
            if query["query_id"] == query_id:
                query_info = query
                break

        if not query_info:
            return {"error": f"Query {query_id} not found in history"}

        # Get workflow execution details from engine
        workflow_analytics = self.workflow_engine.get_workflow_analytics()

        return {
            "query_id": query_id,
            "original_query": query_info["query"],
            "reasoning_process": {
                "workflow_selection": "Selected based on query complexity and domain",
                "agent_sequence": query_info["agents_used"],
                "decision_points": "Adaptive routing based on intermediate results",
                "confidence_calculation": "Weighted average of agent confidences"
            },
            "performance_analysis": {
                "execution_time": query_info["response_time"],
                "success_indicators": query_info["success"],
                "efficiency_score": "Good" if query_info["response_time"] < 5 else "Needs improvement"
            }
        }

    async def optimize_system(self) -> Dict[str, Any]:
        """Perform system optimization based on performance history."""
        optimizations_applied = []

        # Analyze agent performance
        workflow_analytics = self.workflow_engine.get_workflow_analytics()
        agent_performance = workflow_analytics.get("agent_performance", {})

        for agent_type, performance in agent_performance.items():
            if performance["success_rate"] < 0.7:
                optimizations_applied.append(f"Flagged {agent_type} for performance review")

        # Optimize workflow templates based on execution history
        if len(self.query_history) > 10:
            avg_response_time = sum(q["response_time"] for q in self.query_history[-10:]) / 10
            if avg_response_time > 10:  # If average response time > 10 seconds
                optimizations_applied.append("Recommended workflow simplification")

        # Tool usage optimization
        tool_usage = self.system_metrics.get("tool_usage", {})
        if tool_usage:
            most_used_tool = max(tool_usage, key=tool_usage.get)
            least_used_tool = min(tool_usage, key=tool_usage.get)
            optimizations_applied.append(f"Most used tool: {most_used_tool}, least used: {least_used_tool}")

        return {
            "optimization_timestamp": datetime.utcnow().isoformat(),
            "optimizations_applied": optimizations_applied,
            "recommendations": [
                "Consider caching frequently requested information",
                "Monitor agent performance and retrain if necessary",
                "Optimize tool selection based on usage patterns"
            ],
            "system_health_score": self._calculate_system_health_score()
        }

    def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score."""
        if self.system_metrics["total_queries"] == 0:
            return 1.0

        success_rate = self.system_metrics["successful_queries"] / self.system_metrics["total_queries"]
        response_time_score = 1.0 if self.system_metrics["average_response_time"] < 5 else 0.8

        return (success_rate * 0.7 + response_time_score * 0.3)


# Factory function for easy initialization
async def create_multi_agent_system(services: Dict[str, Any]) -> SpaceBiologyMultiAgentSystem:
    """
    Factory function to create and initialize the multi-agent system.

    Args:
        services: Dictionary of initialized services

    Returns:
        Initialized SpaceBiologyMultiAgentSystem instance
    """
    system = SpaceBiologyMultiAgentSystem(services)

    # Perform any additional initialization
    logger.info("Multi-agent system created and ready for operation")

    return system


# Utility functions for integration
async def process_query_with_multi_agent_system(
    query: str,
    services: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to process a query with the multi-agent system.

    Args:
        query: User query
        services: Initialized services
        context: Optional context

    Returns:
        Processed response
    """
    system = await create_multi_agent_system(services)
    return await system.process_query(query, context)


def get_available_capabilities() -> Dict[str, List[str]]:
    """Get list of available system capabilities."""
    return {
        "agent_types": [agent.value for agent in AgentType],
        "task_types": [task.value for task in TaskType],
        "processing_modes": ["simple", "complex", "mission_specific", "research_focused"],
        "output_formats": ["comprehensive", "summary", "technical", "executive"]
    }