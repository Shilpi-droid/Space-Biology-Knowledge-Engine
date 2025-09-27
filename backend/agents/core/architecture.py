"""
Multi-Agent Architecture for Space Biology Knowledge Engine

This module defines the core architecture for a multi-agent system using LangGraph.
The system consists of specialized agents that collaborate to process complex queries
and generate comprehensive responses about space biology research.
"""

from typing import Dict, List, Any, Optional, TypedDict, Callable
from enum import Enum
import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain.tools import BaseTool
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Types of specialized agents in the system."""
    COORDINATOR = "coordinator"
    RESEARCH_SPECIALIST = "research_specialist"
    DATA_ANALYST = "data_analyst"
    MISSION_PLANNER = "mission_planner"
    RISK_ASSESSOR = "risk_assessor"
    EVIDENCE_SYNTHESIZER = "evidence_synthesizer"
    QUERY_ROUTER = "query_router"
    QUALITY_CHECKER = "quality_checker"


class TaskType(str, Enum):
    """Types of tasks that can be processed."""
    LITERATURE_SEARCH = "literature_search"
    DATA_ANALYSIS = "data_analysis"
    MISSION_PLANNING = "mission_planning"
    RISK_ASSESSMENT = "risk_assessment"
    EVIDENCE_SYNTHESIS = "evidence_synthesis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"


class AgentState(TypedDict):
    """State shared between agents during execution."""
    messages: List[BaseMessage]
    current_task: Optional[TaskType]
    query: str
    context: Dict[str, Any]
    intermediate_results: Dict[str, Any]
    final_response: Optional[str]
    confidence_scores: Dict[str, float]
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    next_agent: Optional[AgentType]
    workflow_status: str
    error_messages: List[str]


@dataclass
class AgentCapability:
    """Defines the capabilities of an agent."""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    tools: List[str]
    confidence_threshold: float = 0.7


@dataclass
class WorkflowPlan:
    """Represents a planned workflow for task execution."""
    task_type: TaskType
    agent_sequence: List[AgentType]
    dependencies: Dict[str, List[str]]
    estimated_duration: int
    required_tools: List[str]


class MultiAgentOrchestrator:
    """
    Orchestrates multiple specialized agents using LangGraph.

    This class manages the creation, coordination, and execution of specialized
    agents that work together to process complex space biology queries.
    """

    def __init__(self, llm_config: Dict[str, Any]):
        self.llm = ChatOpenAI(**llm_config)
        self.agents: Dict[AgentType, Any] = {}
        self.tools: Dict[str, BaseTool] = {}
        self.workflows: Dict[TaskType, WorkflowPlan] = {}
        self.graph: Optional[StateGraph] = None

        # Agent capabilities registry
        self.agent_capabilities = self._define_agent_capabilities()

        # Initialize the system
        self._initialize_agents()
        self._create_workflow_graph()

    def _define_agent_capabilities(self) -> Dict[AgentType, AgentCapability]:
        """Define the capabilities of each agent type."""
        return {
            AgentType.COORDINATOR: AgentCapability(
                name="System Coordinator",
                description="Manages overall task execution and agent coordination",
                input_types=["user_query", "system_request"],
                output_types=["workflow_plan", "final_response"],
                tools=["workflow_planner", "response_synthesizer"]
            ),

            AgentType.QUERY_ROUTER: AgentCapability(
                name="Query Router",
                description="Analyzes queries and determines optimal processing strategy",
                input_types=["user_query"],
                output_types=["task_classification", "routing_decision"],
                tools=["query_analyzer", "intent_classifier", "complexity_assessor"]
            ),

            AgentType.RESEARCH_SPECIALIST: AgentCapability(
                name="Research Specialist",
                description="Conducts literature searches and extracts research insights",
                input_types=["research_query", "topic_analysis"],
                output_types=["literature_summary", "research_insights"],
                tools=["semantic_search", "citation_analyzer", "methodology_extractor"]
            ),

            AgentType.DATA_ANALYST: AgentCapability(
                name="Data Analyst",
                description="Analyzes datasets and extracts statistical insights",
                input_types=["dataset_query", "analysis_request"],
                output_types=["statistical_analysis", "data_visualization"],
                tools=["graph_query", "statistical_analyzer", "trend_detector"]
            ),

            AgentType.MISSION_PLANNER: AgentCapability(
                name="Mission Planner",
                description="Provides mission-specific guidance and recommendations",
                input_types=["mission_context", "planning_request"],
                output_types=["mission_recommendations", "protocol_suggestions"],
                tools=["mission_database", "countermeasure_lookup", "risk_calculator"]
            ),

            AgentType.RISK_ASSESSOR: AgentCapability(
                name="Risk Assessor",
                description="Evaluates biological and operational risks",
                input_types=["risk_factors", "assessment_request"],
                output_types=["risk_analysis", "mitigation_strategies"],
                tools=["risk_calculator", "evidence_evaluator", "severity_assessor"]
            ),

            AgentType.EVIDENCE_SYNTHESIZER: AgentCapability(
                name="Evidence Synthesizer",
                description="Combines and synthesizes evidence from multiple sources",
                input_types=["evidence_collection", "synthesis_request"],
                output_types=["evidence_summary", "contradiction_analysis"],
                tools=["evidence_combiner", "contradiction_detector", "confidence_calculator"]
            ),

            AgentType.QUALITY_CHECKER: AgentCapability(
                name="Quality Checker",
                description="Validates outputs and ensures quality standards",
                input_types=["response_draft", "validation_request"],
                output_types=["quality_assessment", "improvement_suggestions"],
                tools=["fact_checker", "completeness_validator", "accuracy_assessor"]
            )
        }

    def _initialize_agents(self):
        """Initialize all agents with their specific configurations."""
        for agent_type, capability in self.agent_capabilities.items():
            agent_config = self._create_agent_config(agent_type, capability)
            self.agents[agent_type] = self._create_agent(agent_config)

    def _create_agent_config(self, agent_type: AgentType, capability: AgentCapability) -> Dict[str, Any]:
        """Create configuration for a specific agent."""
        return {
            "type": agent_type,
            "capability": capability,
            "llm": self.llm,
            "tools": [self.tools.get(tool_name) for tool_name in capability.tools if tool_name in self.tools],
            "system_prompt": self._get_agent_system_prompt(agent_type),
            "temperature": self._get_agent_temperature(agent_type),
            "max_tokens": 2000
        }

    def _create_agent(self, config: Dict[str, Any]) -> Callable:
        """Create an agent function with the given configuration."""

        async def agent_function(state: AgentState) -> AgentState:
            """Agent execution function."""
            try:
                agent_type = config["type"]
                capability = config["capability"]

                # Get the latest message
                current_message = state["messages"][-1] if state["messages"] else None

                # Create context for the agent
                agent_context = {
                    "query": state["query"],
                    "task": state["current_task"],
                    "previous_results": state["intermediate_results"],
                    "context": state["context"]
                }

                # Prepare the system prompt
                system_prompt = config["system_prompt"].format(
                    capability_description=capability.description,
                    current_context=str(agent_context)
                )

                # Create messages for the LLM
                messages = [SystemMessage(content=system_prompt)]
                if current_message:
                    messages.append(current_message)

                # Execute the agent
                response = await config["llm"].ainvoke(messages)

                # Process the response based on agent type
                processed_result = await self._process_agent_response(
                    agent_type, response.content, agent_context
                )

                # Update state
                state["messages"].append(response)
                state["intermediate_results"][agent_type.value] = processed_result
                state["confidence_scores"][agent_type.value] = processed_result.get("confidence", 0.8)

                # Determine next agent
                state["next_agent"] = await self._determine_next_agent(agent_type, state)

                logger.info(f"Agent {agent_type.value} completed processing")

            except Exception as e:
                logger.error(f"Error in agent {config['type'].value}: {str(e)}")
                state["error_messages"].append(f"Agent {config['type'].value}: {str(e)}")

            return state

        return agent_function

    def _create_workflow_graph(self):
        """Create the LangGraph workflow for agent coordination."""
        workflow = StateGraph(AgentState)

        # Add agent nodes
        for agent_type in self.agents.keys():
            workflow.add_node(agent_type.value, self.agents[agent_type])

        # Add coordination nodes
        workflow.add_node("start", self._start_node)
        workflow.add_node("router", self._router_node)
        workflow.add_node("coordinator", self._coordinator_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        workflow.add_node("quality_check", self._quality_check_node)

        # Define the workflow edges
        workflow.set_entry_point("start")

        # Conditional routing based on query analysis
        workflow.add_conditional_edges(
            "start",
            self._route_initial_query,
            {
                "simple_query": AgentType.RESEARCH_SPECIALIST.value,
                "complex_query": "router",
                "mission_query": AgentType.MISSION_PLANNER.value,
                "risk_query": AgentType.RISK_ASSESSOR.value,
                "data_query": AgentType.DATA_ANALYST.value
            }
        )

        # Router determines the workflow
        workflow.add_conditional_edges(
            "router",
            self._route_to_specialists,
            {
                "research_workflow": AgentType.RESEARCH_SPECIALIST.value,
                "analysis_workflow": AgentType.DATA_ANALYST.value,
                "mission_workflow": AgentType.MISSION_PLANNER.value,
                "synthesis_workflow": AgentType.EVIDENCE_SYNTHESIZER.value,
                "multi_agent_workflow": "coordinator"
            }
        )

        # Specialist agents route to synthesizer or coordinator
        for agent_type in [AgentType.RESEARCH_SPECIALIST, AgentType.DATA_ANALYST,
                          AgentType.MISSION_PLANNER, AgentType.RISK_ASSESSOR]:
            workflow.add_conditional_edges(
                agent_type.value,
                self._route_from_specialist,
                {
                    "needs_synthesis": AgentType.EVIDENCE_SYNTHESIZER.value,
                    "needs_coordination": "coordinator",
                    "complete": "quality_check",
                    "needs_more_data": AgentType.DATA_ANALYST.value
                }
            )

        # Evidence synthesizer routes to quality check
        workflow.add_edge(AgentType.EVIDENCE_SYNTHESIZER.value, "quality_check")

        # Coordinator manages complex workflows
        workflow.add_conditional_edges(
            "coordinator",
            self._coordinate_next_step,
            {
                "continue_research": AgentType.RESEARCH_SPECIALIST.value,
                "analyze_data": AgentType.DATA_ANALYST.value,
                "assess_risks": AgentType.RISK_ASSESSOR.value,
                "synthesize": AgentType.EVIDENCE_SYNTHESIZER.value,
                "finalize": "quality_check"
            }
        )

        # Quality check determines if we're done or need more work
        workflow.add_conditional_edges(
            "quality_check",
            self._quality_check_decision,
            {
                "approved": "synthesizer",
                "needs_improvement": "coordinator",
                "needs_more_research": AgentType.RESEARCH_SPECIALIST.value,
                "needs_risk_review": AgentType.RISK_ASSESSOR.value
            }
        )

        # Final synthesizer creates the response
        workflow.add_edge("synthesizer", END)

        self.graph = workflow.compile()

    async def _start_node(self, state: AgentState) -> AgentState:
        """Initialize the workflow state."""
        state["workflow_status"] = "started"
        state["metadata"]["start_time"] = datetime.utcnow().isoformat()
        return state

    async def _router_node(self, state: AgentState) -> AgentState:
        """Route the query to appropriate workflow."""
        query_analysis = await self._analyze_query_complexity(state["query"])
        state["metadata"]["query_analysis"] = query_analysis
        state["current_task"] = query_analysis.get("primary_task")
        return state

    async def _coordinator_node(self, state: AgentState) -> AgentState:
        """Coordinate multi-agent workflows."""
        # Analyze what's been done and what's needed
        completed_agents = list(state["intermediate_results"].keys())

        # Determine next steps based on task requirements
        workflow_plan = await self._create_dynamic_workflow_plan(
            state["current_task"], completed_agents, state["context"]
        )

        state["metadata"]["workflow_plan"] = workflow_plan
        return state

    async def _synthesizer_node(self, state: AgentState) -> AgentState:
        """Synthesize final response from all agent outputs."""
        synthesis_prompt = f"""
        Synthesize a comprehensive response based on the following agent outputs:

        Query: {state['query']}

        Agent Results:
        {self._format_agent_results(state['intermediate_results'])}

        Create a cohesive, well-structured response that addresses the original query.
        Include confidence scores and cite relevant sources.
        """

        messages = [SystemMessage(content=synthesis_prompt)]
        response = await self.llm.ainvoke(messages)

        state["final_response"] = response.content
        state["workflow_status"] = "completed"
        state["metadata"]["end_time"] = datetime.utcnow().isoformat()

        return state

    async def _quality_check_node(self, state: AgentState) -> AgentState:
        """Perform quality checks on the response."""
        quality_metrics = await self._assess_response_quality(state)
        state["metadata"]["quality_metrics"] = quality_metrics
        return state

    # Routing functions
    async def _route_initial_query(self, state: AgentState) -> str:
        """Determine initial routing based on query analysis."""
        query = state["query"].lower()

        # Simple keyword-based routing (can be enhanced with ML)
        if any(word in query for word in ["mission", "mars", "moon", "iss"]):
            return "mission_query"
        elif any(word in query for word in ["risk", "danger", "safety", "hazard"]):
            return "risk_query"
        elif any(word in query for word in ["data", "statistics", "analysis", "trends"]):
            return "data_query"
        elif len(query.split()) < 10:
            return "simple_query"
        else:
            return "complex_query"

    async def _route_to_specialists(self, state: AgentState) -> str:
        """Route from router to appropriate specialist workflow."""
        task = state.get("current_task")
        complexity = state["metadata"].get("query_analysis", {}).get("complexity", "medium")

        if complexity == "high":
            return "multi_agent_workflow"
        elif task == TaskType.LITERATURE_SEARCH:
            return "research_workflow"
        elif task == TaskType.DATA_ANALYSIS:
            return "analysis_workflow"
        elif task == TaskType.MISSION_PLANNING:
            return "mission_workflow"
        else:
            return "synthesis_workflow"

    async def _route_from_specialist(self, state: AgentState) -> str:
        """Route from specialist agents to next step."""
        confidence = state["confidence_scores"].get(state.get("next_agent", ""), 0.5)

        if confidence < 0.6:
            return "needs_more_data"
        elif len(state["intermediate_results"]) > 1:
            return "needs_synthesis"
        elif state["metadata"].get("complexity") == "high":
            return "needs_coordination"
        else:
            return "complete"

    async def _coordinate_next_step(self, state: AgentState) -> str:
        """Coordinate the next step in complex workflows."""
        completed = set(state["intermediate_results"].keys())

        if "research_specialist" not in completed:
            return "continue_research"
        elif "data_analyst" not in completed:
            return "analyze_data"
        elif "risk_assessor" not in completed and "risk" in state["query"]:
            return "assess_risks"
        elif len(completed) >= 2:
            return "synthesize"
        else:
            return "finalize"

    async def _quality_check_decision(self, state: AgentState) -> str:
        """Decide on quality check results."""
        quality_score = state["metadata"].get("quality_metrics", {}).get("overall_score", 0.8)

        if quality_score >= 0.8:
            return "approved"
        elif quality_score >= 0.6:
            return "needs_improvement"
        else:
            return "needs_more_research"

    # Helper methods
    async def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity and determine processing requirements."""
        analysis_prompt = f"""
        Analyze this space biology query and provide:
        1. Complexity level (low/medium/high)
        2. Primary task type
        3. Required capabilities
        4. Estimated processing time

        Query: {query}

        Return analysis as JSON.
        """

        messages = [SystemMessage(content=analysis_prompt)]
        response = await self.llm.ainvoke(messages)

        # Parse the response (simplified - would use structured output in practice)
        return {
            "complexity": "medium",
            "primary_task": TaskType.LITERATURE_SEARCH,
            "required_capabilities": ["research", "analysis"],
            "estimated_time": 60
        }

    async def _create_dynamic_workflow_plan(self, task: TaskType, completed: List[str], context: Dict[str, Any]) -> WorkflowPlan:
        """Create a dynamic workflow plan based on current state."""
        # This would be more sophisticated in practice
        return WorkflowPlan(
            task_type=task,
            agent_sequence=[AgentType.RESEARCH_SPECIALIST, AgentType.EVIDENCE_SYNTHESIZER],
            dependencies={},
            estimated_duration=120,
            required_tools=["semantic_search", "evidence_combiner"]
        )

    def _format_agent_results(self, results: Dict[str, Any]) -> str:
        """Format agent results for synthesis."""
        formatted = []
        for agent, result in results.items():
            formatted.append(f"{agent.upper()}:\n{result}\n")
        return "\n".join(formatted)

    async def _assess_response_quality(self, state: AgentState) -> Dict[str, Any]:
        """Assess the quality of the generated response."""
        return {
            "overall_score": 0.85,
            "completeness": 0.9,
            "accuracy": 0.8,
            "relevance": 0.9
        }

    def _get_agent_system_prompt(self, agent_type: AgentType) -> str:
        """Get the system prompt for a specific agent type."""
        prompts = {
            AgentType.RESEARCH_SPECIALIST: """
            You are a Research Specialist for space biology. Your role is to:
            1. Search and analyze scientific literature
            2. Extract key findings and methodologies
            3. Identify relevant studies and papers
            4. Summarize research insights

            Current context: {current_context}
            Your capabilities: {capability_description}

            Provide detailed, accurate, and well-sourced responses.
            """,

            AgentType.DATA_ANALYST: """
            You are a Data Analyst specializing in space biology data. Your role is to:
            1. Analyze datasets and extract patterns
            2. Perform statistical analysis
            3. Identify trends and correlations
            4. Create data visualizations

            Current context: {current_context}
            Your capabilities: {capability_description}

            Focus on quantitative insights and data-driven conclusions.
            """,

            AgentType.MISSION_PLANNER: """
            You are a Mission Planning Specialist. Your role is to:
            1. Assess mission-specific biological factors
            2. Recommend countermeasures and protocols
            3. Evaluate mission risks and requirements
            4. Provide operational guidance

            Current context: {current_context}
            Your capabilities: {capability_description}

            Prioritize crew safety and mission success.
            """
        }

        return prompts.get(agent_type, "You are a specialized AI agent. Follow your capabilities and provide helpful responses.")

    def _get_agent_temperature(self, agent_type: AgentType) -> float:
        """Get the temperature setting for a specific agent type."""
        temperatures = {
            AgentType.DATA_ANALYST: 0.1,  # Low creativity for data analysis
            AgentType.RESEARCH_SPECIALIST: 0.3,  # Medium-low for research
            AgentType.MISSION_PLANNER: 0.5,  # Medium for planning
            AgentType.EVIDENCE_SYNTHESIZER: 0.4,  # Medium-low for synthesis
        }
        return temperatures.get(agent_type, 0.3)

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a query through the multi-agent system."""
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            current_task=None,
            query=query,
            context=context or {},
            intermediate_results={},
            final_response=None,
            confidence_scores={},
            sources=[],
            metadata={"query_received": datetime.utcnow().isoformat()},
            next_agent=None,
            workflow_status="initialized",
            error_messages=[]
        )

        # Execute the workflow
        final_state = await self.graph.ainvoke(initial_state)

        return {
            "response": final_state["final_response"],
            "confidence_scores": final_state["confidence_scores"],
            "sources": final_state["sources"],
            "metadata": final_state["metadata"],
            "workflow_status": final_state["workflow_status"],
            "agents_used": list(final_state["intermediate_results"].keys())
        }


async def _determine_next_agent(current_agent: AgentType, state: AgentState) -> Optional[AgentType]:
    """Determine the next agent based on current state and results."""
    # This would be more sophisticated in practice
    return None

async def _process_agent_response(agent_type: AgentType, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Process and structure the agent response."""
    return {
        "response": response,
        "confidence": 0.8,
        "timestamp": datetime.utcnow().isoformat(),
        "agent_type": agent_type.value
    }