"""
AI Agents API endpoints for Space Biology Knowledge Engine.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from uuid import uuid4
from datetime import datetime

from ..services.agents.research_agent import ResearchAgent
from ..services.agents.mission_agent import MissionAgent
from ..services.agents.risk_agent import RiskAgent
from ..core.dependencies import get_research_agent, get_mission_agent, get_risk_agent

router = APIRouter(prefix="/agents", tags=["ai_agents"])


class AgentRequest(BaseModel):
    """Request model for agent interactions."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message/query")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for continuity")
    session_id: Optional[str] = Field(default=None, description="Session ID")


class AgentResponse(BaseModel):
    """Response model for agent interactions."""
    response: str
    agent_type: str
    conversation_id: str
    confidence: float
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: datetime


class ConversationHistory(BaseModel):
    """Conversation history model."""
    conversation_id: str
    agent_type: str
    messages: List[Dict[str, Any]]
    created_at: datetime
    last_updated: datetime


class MissionAssessmentRequest(BaseModel):
    """Request model for mission assessments."""
    mission_type: str = Field(..., regex="^(mars|moon|iss|deep_space)$")
    duration_months: Optional[int] = Field(default=None, ge=1, le=36)
    crew_size: int = Field(default=4, ge=1, le=10)
    specific_concerns: Optional[List[str]] = Field(default=None)
    mission_objectives: Optional[List[str]] = Field(default=None)


class RiskAssessmentRequest(BaseModel):
    """Request model for risk assessments."""
    risk_factors: List[str] = Field(..., min_items=1)
    mission_context: Optional[Dict[str, Any]] = Field(default=None)
    assessment_type: str = Field(default="comprehensive", regex="^(quick|comprehensive|detailed)$")
    time_horizon: Optional[str] = Field(default=None, regex="^(short_term|medium_term|long_term)$")


# Research Agent Endpoints
@router.post("/research/query", response_model=AgentResponse)
async def research_agent_query(
    request: AgentRequest,
    research_agent: ResearchAgent = Depends(get_research_agent)
):
    """Query the research agent for scientific information."""
    try:
        conversation_id = request.conversation_id or str(uuid4())

        response_data = await research_agent.process_query(
            query=request.message,
            context=request.context,
            conversation_id=conversation_id
        )

        return AgentResponse(
            response=response_data["response"],
            agent_type="research",
            conversation_id=conversation_id,
            confidence=response_data.get("confidence", 0.8),
            sources=response_data.get("sources", []),
            metadata=response_data.get("metadata", {}),
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research agent query failed: {str(e)}")


@router.post("/research/compare", response_model=AgentResponse)
async def research_compare_studies(
    study_ids: List[str] = Field(..., min_items=2, max_items=5),
    comparison_aspects: Optional[List[str]] = Field(default=None),
    research_agent: ResearchAgent = Depends(get_research_agent)
):
    """Compare multiple studies using the research agent."""
    try:
        response_data = await research_agent.compare_studies(
            study_ids=study_ids,
            aspects=comparison_aspects
        )

        return AgentResponse(
            response=response_data["response"],
            agent_type="research",
            conversation_id=str(uuid4()),
            confidence=response_data.get("confidence", 0.8),
            sources=response_data.get("sources", []),
            metadata=response_data.get("metadata", {}),
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Study comparison failed: {str(e)}")


# Mission Agent Endpoints
@router.post("/mission/assess", response_model=AgentResponse)
async def mission_assessment(
    request: MissionAssessmentRequest,
    mission_agent: MissionAgent = Depends(get_mission_agent)
):
    """Get mission-specific biological assessment."""
    try:
        assessment_data = await mission_agent.assess_mission(
            mission_type=request.mission_type,
            duration_months=request.duration_months,
            crew_size=request.crew_size,
            specific_concerns=request.specific_concerns,
            mission_objectives=request.mission_objectives
        )

        return AgentResponse(
            response=assessment_data["assessment"],
            agent_type="mission",
            conversation_id=str(uuid4()),
            confidence=assessment_data.get("confidence", 0.8),
            sources=assessment_data.get("sources", []),
            metadata=assessment_data.get("metadata", {}),
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mission assessment failed: {str(e)}")


@router.post("/mission/query", response_model=AgentResponse)
async def mission_agent_query(
    request: AgentRequest,
    mission_agent: MissionAgent = Depends(get_mission_agent)
):
    """Query the mission agent for mission-specific guidance."""
    try:
        conversation_id = request.conversation_id or str(uuid4())

        response_data = await mission_agent.process_query(
            query=request.message,
            context=request.context,
            conversation_id=conversation_id
        )

        return AgentResponse(
            response=response_data["response"],
            agent_type="mission",
            conversation_id=conversation_id,
            confidence=response_data.get("confidence", 0.8),
            sources=response_data.get("sources", []),
            metadata=response_data.get("metadata", {}),
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mission agent query failed: {str(e)}")


@router.get("/mission/countermeasures")
async def get_mission_countermeasures(
    mission_type: str = Field(..., regex="^(mars|moon|iss|deep_space)$"),
    biological_system: Optional[str] = Field(default=None),
    mission_agent: MissionAgent = Depends(get_mission_agent)
):
    """Get countermeasures for specific mission type."""
    try:
        countermeasures = await mission_agent.get_countermeasures(
            mission_type=mission_type,
            biological_system=biological_system
        )

        return {
            "mission_type": mission_type,
            "biological_system": biological_system,
            "countermeasures": countermeasures
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get countermeasures: {str(e)}")


# Risk Agent Endpoints
@router.post("/risk/assess", response_model=AgentResponse)
async def risk_assessment(
    request: RiskAssessmentRequest,
    risk_agent: RiskAgent = Depends(get_risk_agent)
):
    """Perform comprehensive risk assessment."""
    try:
        assessment_data = await risk_agent.assess_risks(
            risk_factors=request.risk_factors,
            mission_context=request.mission_context,
            assessment_type=request.assessment_type,
            time_horizon=request.time_horizon
        )

        return AgentResponse(
            response=assessment_data["assessment"],
            agent_type="risk",
            conversation_id=str(uuid4()),
            confidence=assessment_data.get("confidence", 0.8),
            sources=assessment_data.get("sources", []),
            metadata=assessment_data.get("metadata", {}),
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")


@router.post("/risk/query", response_model=AgentResponse)
async def risk_agent_query(
    request: AgentRequest,
    risk_agent: RiskAgent = Depends(get_risk_agent)
):
    """Query the risk agent for safety and health guidance."""
    try:
        conversation_id = request.conversation_id or str(uuid4())

        response_data = await risk_agent.process_query(
            query=request.message,
            context=request.context,
            conversation_id=conversation_id
        )

        return AgentResponse(
            response=response_data["response"],
            agent_type="risk",
            conversation_id=conversation_id,
            confidence=response_data.get("confidence", 0.8),
            sources=response_data.get("sources", []),
            metadata=response_data.get("metadata", {}),
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk agent query failed: {str(e)}")


@router.get("/risk/factors")
async def get_known_risk_factors(
    category: Optional[str] = Field(default=None),
    mission_type: Optional[str] = Field(default=None),
    risk_agent: RiskAgent = Depends(get_risk_agent)
):
    """Get known risk factors, optionally filtered by category or mission type."""
    try:
        risk_factors = await risk_agent.get_risk_factors(
            category=category,
            mission_type=mission_type
        )

        return {
            "category": category,
            "mission_type": mission_type,
            "risk_factors": risk_factors
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get risk factors: {str(e)}")


# Conversation Management
@router.get("/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(
    conversation_id: str
):
    """Get conversation history for a specific conversation."""
    try:
        # This would typically retrieve from a database
        # For now, return a placeholder response
        return ConversationHistory(
            conversation_id=conversation_id,
            agent_type="unknown",
            messages=[],
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversation: {str(e)}")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str
):
    """Delete a conversation and its history."""
    try:
        # Implementation would delete from database
        return {"message": f"Conversation {conversation_id} deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")


@router.get("/status")
async def get_agents_status(
    research_agent: ResearchAgent = Depends(get_research_agent),
    mission_agent: MissionAgent = Depends(get_mission_agent),
    risk_agent: RiskAgent = Depends(get_risk_agent)
):
    """Get status of all AI agents."""
    try:
        status = {
            "research_agent": {
                "status": "active",
                "capabilities": [
                    "literature_search",
                    "study_comparison",
                    "methodology_explanation",
                    "evidence_synthesis"
                ]
            },
            "mission_agent": {
                "status": "active",
                "capabilities": [
                    "mission_assessment",
                    "countermeasure_recommendation",
                    "risk_evaluation",
                    "protocol_generation"
                ]
            },
            "risk_agent": {
                "status": "active",
                "capabilities": [
                    "risk_assessment",
                    "health_monitoring",
                    "safety_protocols",
                    "emergency_procedures"
                ]
            }
        }

        return status

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agents status: {str(e)}")


# Multi-agent collaboration
@router.post("/collaborate")
async def multi_agent_collaboration(
    query: str = Field(..., description="Query requiring multiple agent types"),
    agents: List[str] = Field(..., description="Agent types to involve"),
    research_agent: ResearchAgent = Depends(get_research_agent),
    mission_agent: MissionAgent = Depends(get_mission_agent),
    risk_agent: RiskAgent = Depends(get_risk_agent)
):
    """Coordinate multiple agents for complex queries."""
    try:
        results = {}
        conversation_id = str(uuid4())

        # Route to appropriate agents
        if "research" in agents:
            research_result = await research_agent.process_query(
                query=query,
                conversation_id=conversation_id
            )
            results["research"] = research_result

        if "mission" in agents:
            mission_result = await mission_agent.process_query(
                query=query,
                conversation_id=conversation_id
            )
            results["mission"] = mission_result

        if "risk" in agents:
            risk_result = await risk_agent.process_query(
                query=query,
                conversation_id=conversation_id
            )
            results["risk"] = risk_result

        # Synthesize responses
        synthesized_response = await _synthesize_multi_agent_response(results, query)

        return {
            "query": query,
            "agents_involved": agents,
            "individual_responses": results,
            "synthesized_response": synthesized_response,
            "conversation_id": conversation_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-agent collaboration failed: {str(e)}")


async def _synthesize_multi_agent_response(results: Dict[str, Any], query: str) -> str:
    """Synthesize responses from multiple agents."""
    # Simple synthesis - in practice this would use LLM
    sections = []

    if "research" in results:
        sections.append(f"Research Perspective: {results['research'].get('response', '')}")

    if "mission" in results:
        sections.append(f"Mission Planning Perspective: {results['mission'].get('response', '')}")

    if "risk" in results:
        sections.append(f"Risk Assessment Perspective: {results['risk'].get('response', '')}")

    return "\n\n".join(sections)