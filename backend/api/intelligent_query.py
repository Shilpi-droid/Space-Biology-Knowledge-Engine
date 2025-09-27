"""
Intelligent Query API - Multi-Agent System Integration

This endpoint provides access to the advanced multi-agent system for processing
complex space biology queries with adaptive workflows and intelligent reasoning.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from ..agents.multi_agent_system import SpaceBiologyMultiAgentSystem, create_multi_agent_system, get_available_capabilities
from ..core.dependencies import (
    get_kg_service, get_search_service, get_llm_service,
    get_evidence_synthesis_service, get_temporal_analysis_service
)

router = APIRouter(prefix="/intelligent", tags=["intelligent_query"])


class QueryComplexity(str, Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class ResponseFormat(str, Enum):
    """Response format options."""
    COMPREHENSIVE = "comprehensive"
    SUMMARY = "summary"
    TECHNICAL = "technical"
    EXECUTIVE = "executive"


class UserPreferences(BaseModel):
    """User preferences for query processing."""
    response_format: ResponseFormat = Field(default=ResponseFormat.COMPREHENSIVE)
    detail_level: str = Field(default="medium", regex="^(low|medium|high)$")
    include_sources: bool = Field(default=True)
    include_workflow_info: bool = Field(default=False)
    max_response_time: int = Field(default=30, ge=5, le=300, description="Max response time in seconds")
    preferred_agents: Optional[List[str]] = Field(default=None, description="Preferred agent types")


class QueryContext(BaseModel):
    """Context information for the query."""
    mission_type: Optional[str] = Field(default=None, regex="^(mars|moon|iss|deep_space)$")
    domain_focus: Optional[List[str]] = Field(default=None, description="Specific domain areas to focus on")
    time_constraint: Optional[str] = Field(default=None, description="Time constraints for the analysis")
    priority_level: str = Field(default="normal", regex="^(low|normal|high|critical)$")
    previous_context: Optional[Dict[str, Any]] = Field(default=None, description="Previous conversation context")


class IntelligentQueryRequest(BaseModel):
    """Request model for intelligent query processing."""
    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    context: Optional[QueryContext] = Field(default=None, description="Query context")
    preferences: Optional[UserPreferences] = Field(default=None, description="User preferences")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for continuity")


class WorkflowInfo(BaseModel):
    """Information about the workflow execution."""
    workflow_id: str
    status: str
    agents_used: List[str]
    execution_time: float
    performance_metrics: Dict[str, Any]


class IntelligentQueryResponse(BaseModel):
    """Response model for intelligent query processing."""
    query_id: str
    query: str
    response: str
    confidence: float
    sources: List[Dict[str, Any]]
    workflow_info: WorkflowInfo
    metadata: Dict[str, Any]


class SystemStatusResponse(BaseModel):
    """System status response model."""
    system_health: str
    components: Dict[str, str]
    performance_metrics: Dict[str, Any]
    capabilities: Dict[str, List[str]]


# Global system instance (will be initialized on startup)
_multi_agent_system: Optional[SpaceBiologyMultiAgentSystem] = None


async def get_multi_agent_system() -> SpaceBiologyMultiAgentSystem:
    """Dependency to get the multi-agent system instance."""
    global _multi_agent_system

    if _multi_agent_system is None:
        # Initialize the system with all required services
        services = {
            "kg_service": get_kg_service(),
            "search_service": get_search_service(),
            "llm_service": get_llm_service(),
            "evidence_service": get_evidence_synthesis_service(),
            "temporal_service": get_temporal_analysis_service()
        }
        _multi_agent_system = await create_multi_agent_system(services)

    return _multi_agent_system


@router.post("/query", response_model=IntelligentQueryResponse)
async def process_intelligent_query(
    request: IntelligentQueryRequest,
    system: SpaceBiologyMultiAgentSystem = Depends(get_multi_agent_system)
):
    """
    Process a query using the intelligent multi-agent system.

    This endpoint leverages adaptive workflows, specialized agents, and intelligent
    reasoning to provide comprehensive responses to complex space biology queries.
    """
    try:
        # Prepare context
        context_dict = {}
        if request.context:
            context_dict = request.context.dict(exclude_none=True)

        # Prepare user preferences
        preferences_dict = {}
        if request.preferences:
            preferences_dict = request.preferences.dict(exclude_none=True)

        # Process the query
        result = await system.process_query(
            query=request.query,
            context=context_dict,
            user_preferences=preferences_dict
        )

        # Format the response
        return IntelligentQueryResponse(
            query_id=result["query_id"],
            query=result["query"],
            response=result["response"],
            confidence=result["confidence"],
            sources=result["sources"],
            workflow_info=WorkflowInfo(**result["workflow_info"]),
            metadata=result["metadata"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    system: SpaceBiologyMultiAgentSystem = Depends(get_multi_agent_system)
):
    """Get comprehensive system status and performance metrics."""
    try:
        status = await system.get_system_status()

        return SystemStatusResponse(
            system_health=status["system_health"],
            components=status["components"],
            performance_metrics=status["performance_metrics"],
            capabilities=status["capabilities"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@router.get("/capabilities")
async def get_system_capabilities():
    """Get available system capabilities and supported features."""
    try:
        capabilities = get_available_capabilities()

        return {
            "capabilities": capabilities,
            "description": "Multi-agent system for intelligent space biology query processing",
            "features": [
                "Adaptive workflow planning",
                "Specialized agent routing",
                "Evidence synthesis and contradiction detection",
                "Mission-specific intelligence",
                "Real-time performance optimization",
                "Comprehensive source attribution"
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get capabilities: {str(e)}")


@router.post("/explain/{query_id}")
async def explain_reasoning(
    query_id: str,
    system: SpaceBiologyMultiAgentSystem = Depends(get_multi_agent_system)
):
    """
    Explain the reasoning process used for a specific query.

    This endpoint provides transparency into how the multi-agent system
    processed a query, including workflow decisions and agent interactions.
    """
    try:
        explanation = await system.explain_reasoning(query_id)

        if "error" in explanation:
            raise HTTPException(status_code=404, detail=explanation["error"])

        return explanation

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to explain reasoning: {str(e)}")


@router.post("/optimize")
async def optimize_system(
    background_tasks: BackgroundTasks,
    system: SpaceBiologyMultiAgentSystem = Depends(get_multi_agent_system)
):
    """
    Trigger system optimization based on performance history.

    This endpoint analyzes system performance and applies optimizations
    to improve response quality and execution efficiency.
    """
    try:
        # Run optimization in background
        background_tasks.add_task(_run_optimization, system)

        return {
            "message": "System optimization started",
            "status": "in_progress",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start optimization: {str(e)}")


async def _run_optimization(system: SpaceBiologyMultiAgentSystem):
    """Run system optimization in background."""
    try:
        result = await system.optimize_system()
        # Could store optimization results in database for later retrieval
        print(f"Optimization completed: {result}")
    except Exception as e:
        print(f"Optimization failed: {str(e)}")


@router.get("/analytics")
async def get_system_analytics(
    time_range: str = "24h",
    system: SpaceBiologyMultiAgentSystem = Depends(get_multi_agent_system)
):
    """
    Get detailed analytics about system performance and usage patterns.
    """
    try:
        status = await system.get_system_status()

        # Extract relevant analytics
        analytics = {
            "time_range": time_range,
            "query_statistics": {
                "total_queries": status["performance_metrics"]["total_queries"],
                "successful_queries": status["performance_metrics"]["successful_queries"],
                "success_rate": (
                    status["performance_metrics"]["successful_queries"] /
                    max(status["performance_metrics"]["total_queries"], 1)
                ),
                "average_response_time": status["performance_metrics"]["average_response_time"]
            },
            "agent_utilization": status["performance_metrics"]["agent_utilization"],
            "tool_usage": status["performance_metrics"]["tool_usage"],
            "recent_activity": status["recent_queries"],
            "workflow_analytics": status.get("workflow_analytics", {})
        }

        return analytics

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


@router.post("/simulate")
async def simulate_query_processing(
    query: str,
    complexity: QueryComplexity = QueryComplexity.MODERATE,
    dry_run: bool = True
):
    """
    Simulate query processing to understand workflow planning without execution.

    This endpoint is useful for understanding how the system would process
    a query without actually executing the full workflow.
    """
    try:
        if not dry_run:
            raise HTTPException(status_code=400, detail="Only dry run simulation is currently supported")

        # Create a temporary system for simulation
        services = {
            "kg_service": get_kg_service(),
            "search_service": get_search_service(),
            "llm_service": get_llm_service(),
        }

        system = await create_multi_agent_system(services)

        # Simulate workflow planning
        context = {"complexity": complexity.value, "simulation": True}
        workflow_plan = await system.workflow_engine.plan_workflow(query, context)

        return {
            "query": query,
            "simulation_results": {
                "planned_workflow": {
                    "task_type": workflow_plan.task_type.value if workflow_plan.task_type else None,
                    "agent_sequence": [agent.value for agent in workflow_plan.agent_sequence],
                    "estimated_duration": workflow_plan.estimated_duration,
                    "required_tools": workflow_plan.required_tools,
                    "dependencies": workflow_plan.dependencies
                },
                "complexity_assessment": complexity.value,
                "estimated_performance": {
                    "confidence_prediction": "high" if complexity != QueryComplexity.EXPERT else "medium",
                    "resource_requirements": "moderate",
                    "expected_agents": len(workflow_plan.agent_sequence)
                }
            },
            "metadata": {
                "simulation_timestamp": datetime.utcnow().isoformat(),
                "dry_run": dry_run
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@router.websocket("/stream")
async def stream_query_processing(websocket):
    """
    WebSocket endpoint for streaming query processing updates.

    This endpoint provides real-time updates during query processing,
    allowing clients to see the progress of multi-agent workflows.
    """
    await websocket.accept()

    try:
        while True:
            # Receive query from client
            data = await websocket.receive_json()
            query = data.get("query", "")

            if not query:
                await websocket.send_json({"error": "No query provided"})
                continue

            # Process query with streaming updates
            await _stream_query_processing(websocket, query, data.get("context", {}))

    except Exception as e:
        await websocket.send_json({"error": f"Streaming failed: {str(e)}"})
    finally:
        await websocket.close()


async def _stream_query_processing(websocket, query: str, context: Dict[str, Any]):
    """Stream query processing with real-time updates."""
    try:
        # Send initial status
        await websocket.send_json({
            "status": "started",
            "message": "Query processing initiated",
            "timestamp": datetime.utcnow().isoformat()
        })

        # Get system instance
        system = await get_multi_agent_system()

        # Send workflow planning update
        await websocket.send_json({
            "status": "planning",
            "message": "Planning optimal workflow",
            "stage": "workflow_planning"
        })

        # Process query (simplified for streaming example)
        result = await system.process_query(query, context)

        # Send completion update
        await websocket.send_json({
            "status": "completed",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        await websocket.send_json({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })