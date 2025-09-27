"""
Dynamic Workflow Orchestrator for Multi-Agent System

This module provides intelligent workflow orchestration that adapts based on query complexity,
agent capabilities, and real-time performance metrics.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from ..core.architecture import AgentType, TaskType, AgentState, WorkflowPlan
from ..tools.knowledge_tools import ToolRegistry

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PLANNED = "planned"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class PriorityLevel(str, Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class WorkflowNode:
    """Represents a node in the workflow graph."""
    agent_type: AgentType
    dependencies: Set[str] = field(default_factory=set)
    outputs: Set[str] = field(default_factory=set)
    estimated_duration: int = 60  # seconds
    required_tools: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    fallback_agents: List[AgentType] = field(default_factory=list)


@dataclass
class WorkflowExecution:
    """Tracks the execution of a workflow."""
    workflow_id: str
    plan: WorkflowPlan
    status: WorkflowStatus
    current_node: Optional[str]
    completed_nodes: Set[str] = field(default_factory=set)
    failed_nodes: Set[str] = field(default_factory=set)
    node_results: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_messages: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class AdaptiveWorkflowEngine:
    """
    Intelligent workflow engine that adapts execution based on:
    - Query complexity analysis
    - Agent performance history
    - Resource availability
    - Real-time feedback
    """

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.workflow_templates = {}
        self.execution_history = []
        self.agent_performance = defaultdict(lambda: {
            "success_rate": 1.0,
            "avg_duration": 60,
            "quality_score": 0.8,
            "total_executions": 0
        })
        self.workflow_cache = {}

        # Initialize workflow templates
        self._initialize_workflow_templates()

    def _initialize_workflow_templates(self):
        """Initialize predefined workflow templates."""

        # Simple research workflow
        self.workflow_templates["simple_research"] = {
            "nodes": {
                "research": WorkflowNode(
                    agent_type=AgentType.RESEARCH_SPECIALIST,
                    required_tools=["semantic_search", "methodology_extractor"],
                    success_criteria={"min_sources": 3, "confidence": 0.7}
                ),
                "quality_check": WorkflowNode(
                    agent_type=AgentType.QUALITY_CHECKER,
                    dependencies={"research"},
                    required_tools=["fact_checker"],
                    success_criteria={"quality_score": 0.8}
                )
            },
            "entry_point": "research",
            "exit_points": ["quality_check"]
        }

        # Complex analysis workflow
        self.workflow_templates["complex_analysis"] = {
            "nodes": {
                "router": WorkflowNode(
                    agent_type=AgentType.QUERY_ROUTER,
                    required_tools=["query_analyzer"],
                    outputs={"routing_decision", "task_breakdown"}
                ),
                "research": WorkflowNode(
                    agent_type=AgentType.RESEARCH_SPECIALIST,
                    dependencies={"router"},
                    required_tools=["semantic_search", "citation_analyzer"],
                    fallback_agents=[AgentType.DATA_ANALYST]
                ),
                "data_analysis": WorkflowNode(
                    agent_type=AgentType.DATA_ANALYST,
                    dependencies={"router"},
                    required_tools=["graph_query", "statistical_analyzer"]
                ),
                "synthesis": WorkflowNode(
                    agent_type=AgentType.EVIDENCE_SYNTHESIZER,
                    dependencies={"research", "data_analysis"},
                    required_tools=["evidence_combiner", "contradiction_detector"]
                ),
                "quality_check": WorkflowNode(
                    agent_type=AgentType.QUALITY_CHECKER,
                    dependencies={"synthesis"},
                    success_criteria={"quality_score": 0.8, "completeness": 0.9}
                )
            },
            "entry_point": "router",
            "exit_points": ["quality_check"]
        }

        # Mission-specific workflow
        self.workflow_templates["mission_planning"] = {
            "nodes": {
                "mission_analysis": WorkflowNode(
                    agent_type=AgentType.MISSION_PLANNER,
                    required_tools=["mission_database"],
                    outputs={"mission_context", "requirements"}
                ),
                "risk_assessment": WorkflowNode(
                    agent_type=AgentType.RISK_ASSESSOR,
                    dependencies={"mission_analysis"},
                    required_tools=["risk_calculator", "evidence_evaluator"]
                ),
                "research_support": WorkflowNode(
                    agent_type=AgentType.RESEARCH_SPECIALIST,
                    dependencies={"mission_analysis"},
                    required_tools=["semantic_search"]
                ),
                "synthesis": WorkflowNode(
                    agent_type=AgentType.EVIDENCE_SYNTHESIZER,
                    dependencies={"risk_assessment", "research_support"},
                    required_tools=["evidence_combiner"]
                ),
                "final_recommendations": WorkflowNode(
                    agent_type=AgentType.MISSION_PLANNER,
                    dependencies={"synthesis"},
                    required_tools=["mission_database"]
                )
            },
            "entry_point": "mission_analysis",
            "exit_points": ["final_recommendations"]
        }

    async def plan_workflow(self, query: str, context: Dict[str, Any]) -> WorkflowPlan:
        """
        Intelligently plan a workflow based on query analysis and context.
        """
        # Analyze query complexity and requirements
        query_analysis = await self._analyze_query(query, context)

        # Select appropriate workflow template
        template_name = await self._select_workflow_template(query_analysis)

        # Customize workflow based on context and performance history
        customized_workflow = await self._customize_workflow(
            template_name, query_analysis, context
        )

        # Optimize workflow based on resource availability
        optimized_workflow = await self._optimize_workflow(customized_workflow)

        return WorkflowPlan(
            task_type=query_analysis["primary_task"],
            agent_sequence=optimized_workflow["execution_order"],
            dependencies=optimized_workflow["dependencies"],
            estimated_duration=optimized_workflow["estimated_duration"],
            required_tools=optimized_workflow["required_tools"]
        )

    async def execute_workflow(self, plan: WorkflowPlan, initial_state: AgentState) -> WorkflowExecution:
        """
        Execute a workflow plan with adaptive monitoring and error recovery.
        """
        workflow_id = f"workflow_{datetime.utcnow().isoformat()}"
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            plan=plan,
            status=WorkflowStatus.EXECUTING,
            current_node=None,
            start_time=datetime.utcnow()
        )

        try:
            # Execute workflow nodes in planned order
            current_state = initial_state.copy()

            for agent_type in plan.agent_sequence:
                execution.current_node = agent_type.value

                # Check if dependencies are satisfied
                if not await self._check_dependencies(agent_type, execution):
                    await self._handle_dependency_failure(agent_type, execution)
                    continue

                # Execute agent with monitoring
                result = await self._execute_agent_with_monitoring(
                    agent_type, current_state, execution
                )

                if result["success"]:
                    execution.completed_nodes.add(agent_type.value)
                    execution.node_results[agent_type.value] = result
                    current_state["intermediate_results"][agent_type.value] = result["data"]

                    # Update performance metrics
                    await self._update_agent_performance(agent_type, result)
                else:
                    execution.failed_nodes.add(agent_type.value)
                    execution.error_messages.append(result.get("error", "Unknown error"))

                    # Attempt recovery
                    recovery_successful = await self._attempt_recovery(
                        agent_type, execution, current_state
                    )

                    if not recovery_successful:
                        execution.status = WorkflowStatus.FAILED
                        break

                # Adaptive re-planning if needed
                if await self._should_replan(execution, current_state):
                    new_plan = await self._replan_workflow(execution, current_state)
                    if new_plan:
                        plan = new_plan
                        execution.plan = new_plan

            # Finalize execution
            execution.status = WorkflowStatus.COMPLETED if execution.status != WorkflowStatus.FAILED else WorkflowStatus.FAILED
            execution.end_time = datetime.utcnow()
            execution.performance_metrics = await self._calculate_execution_metrics(execution)

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            execution.status = WorkflowStatus.FAILED
            execution.error_messages.append(str(e))
            execution.end_time = datetime.utcnow()

        # Store execution history
        self.execution_history.append(execution)

        return execution

    async def _analyze_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze query to understand complexity and requirements."""
        analysis = {
            "complexity": "medium",
            "primary_task": TaskType.LITERATURE_SEARCH,
            "secondary_tasks": [],
            "domain_focus": [],
            "required_capabilities": [],
            "estimated_scope": "moderate",
            "urgency": PriorityLevel.NORMAL
        }

        # Keyword-based analysis (could be enhanced with ML)
        query_lower = query.lower()

        # Determine complexity
        complexity_indicators = {
            "simple": ["what is", "define", "explain"],
            "medium": ["compare", "analyze", "relationship"],
            "high": ["synthesize", "integrate", "comprehensive", "multi-factor"]
        }

        for level, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                analysis["complexity"] = level
                break

        # Identify domain focus
        domain_keywords = {
            "mission": ["mission", "mars", "moon", "iss", "spacecraft"],
            "risk": ["risk", "safety", "hazard", "danger"],
            "physiology": ["bone", "muscle", "cardiovascular", "immune"],
            "research": ["study", "research", "literature", "papers"]
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                analysis["domain_focus"].append(domain)

        # Determine primary task
        task_indicators = {
            TaskType.MISSION_PLANNING: ["mission", "planning", "protocol"],
            TaskType.RISK_ASSESSMENT: ["risk", "assess", "safety"],
            TaskType.DATA_ANALYSIS: ["data", "statistics", "trends"],
            TaskType.EVIDENCE_SYNTHESIS: ["synthesize", "combine", "integrate"]
        }

        for task, indicators in task_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                analysis["primary_task"] = task
                break

        # Context-based adjustments
        if context.get("mission_type"):
            analysis["domain_focus"].append("mission")
            if analysis["primary_task"] == TaskType.LITERATURE_SEARCH:
                analysis["primary_task"] = TaskType.MISSION_PLANNING

        return analysis

    async def _select_workflow_template(self, query_analysis: Dict[str, Any]) -> str:
        """Select the most appropriate workflow template."""

        # Mission-focused queries
        if "mission" in query_analysis["domain_focus"]:
            return "mission_planning"

        # Complex multi-domain queries
        if (query_analysis["complexity"] == "high" or
            len(query_analysis["domain_focus"]) > 2):
            return "complex_analysis"

        # Simple research queries
        return "simple_research"

    async def _customize_workflow(self, template_name: str, query_analysis: Dict[str, Any],
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Customize workflow template based on specific requirements."""

        template = self.workflow_templates[template_name].copy()

        # Adjust based on complexity
        if query_analysis["complexity"] == "high":
            # Add additional verification steps
            if "verification" not in template["nodes"]:
                template["nodes"]["verification"] = WorkflowNode(
                    agent_type=AgentType.QUALITY_CHECKER,
                    dependencies=set(template["exit_points"]),
                    success_criteria={"quality_score": 0.9}
                )
                template["exit_points"] = ["verification"]

        # Adjust based on domain focus
        if "risk" in query_analysis["domain_focus"]:
            if template_name != "mission_planning":
                # Add risk assessment node
                template["nodes"]["risk_check"] = WorkflowNode(
                    agent_type=AgentType.RISK_ASSESSOR,
                    dependencies={"research"} if "research" in template["nodes"] else set(),
                    required_tools=["risk_calculator"]
                )

        # Performance-based agent selection
        for node_name, node in template["nodes"].items():
            agent_performance = self.agent_performance[node.agent_type]

            # If agent has poor performance, consider fallbacks
            if agent_performance["success_rate"] < 0.7:
                if node.fallback_agents:
                    # Select best performing fallback
                    best_fallback = max(
                        node.fallback_agents,
                        key=lambda a: self.agent_performance[a]["success_rate"]
                    )
                    node.agent_type = best_fallback

        return template

    async def _optimize_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize workflow for performance and resource usage."""

        nodes = workflow["nodes"]

        # Calculate execution order considering dependencies
        execution_order = await self._calculate_execution_order(nodes)

        # Estimate total duration
        estimated_duration = sum(
            node.estimated_duration for node in nodes.values()
        )

        # Identify parallel execution opportunities
        parallel_groups = await self._identify_parallel_groups(nodes)

        # Collect all required tools
        required_tools = list(set(
            tool for node in nodes.values()
            for tool in node.required_tools
        ))

        # Build dependency map
        dependencies = {
            node_name: list(node.dependencies)
            for node_name, node in nodes.items()
        }

        return {
            "execution_order": execution_order,
            "estimated_duration": estimated_duration,
            "parallel_groups": parallel_groups,
            "required_tools": required_tools,
            "dependencies": dependencies,
            "nodes": nodes
        }

    async def _calculate_execution_order(self, nodes: Dict[str, WorkflowNode]) -> List[AgentType]:
        """Calculate optimal execution order using topological sort."""

        # Build dependency graph
        in_degree = defaultdict(int)
        adj_list = defaultdict(list)

        for node_name, node in nodes.items():
            for dep in node.dependencies:
                adj_list[dep].append(node_name)
                in_degree[node_name] += 1

        # Topological sort
        queue = [name for name in nodes.keys() if in_degree[name] == 0]
        execution_order = []

        while queue:
            current = queue.pop(0)
            execution_order.append(nodes[current].agent_type)

            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return execution_order

    async def _identify_parallel_groups(self, nodes: Dict[str, WorkflowNode]) -> List[List[str]]:
        """Identify nodes that can be executed in parallel."""
        parallel_groups = []

        # Group nodes by dependency level
        levels = defaultdict(list)

        def get_max_dep_level(node_name: str, memo: Dict[str, int] = {}) -> int:
            if node_name in memo:
                return memo[node_name]

            if not nodes[node_name].dependencies:
                memo[node_name] = 0
                return 0

            max_level = max(
                get_max_dep_level(dep, memo)
                for dep in nodes[node_name].dependencies
            )
            memo[node_name] = max_level + 1
            return max_level + 1

        for node_name in nodes:
            level = get_max_dep_level(node_name)
            levels[level].append(node_name)

        return [group for group in levels.values() if len(group) > 1]

    async def _execute_agent_with_monitoring(self, agent_type: AgentType,
                                           state: AgentState,
                                           execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute agent with comprehensive monitoring."""

        start_time = datetime.utcnow()

        try:
            # Create agent-specific context
            agent_context = {
                "workflow_id": execution.workflow_id,
                "execution_history": execution.node_results,
                "available_tools": self.tool_registry.list_available_tools()
            }

            # Execute agent (this would call the actual agent)
            # For now, simulate execution
            result = await self._simulate_agent_execution(agent_type, state, agent_context)

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            return {
                "success": True,
                "data": result,
                "duration": duration,
                "agent_type": agent_type.value,
                "timestamp": end_time.isoformat()
            }

        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            return {
                "success": False,
                "error": str(e),
                "duration": duration,
                "agent_type": agent_type.value,
                "timestamp": end_time.isoformat()
            }

    async def _simulate_agent_execution(self, agent_type: AgentType,
                                      state: AgentState,
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate agent execution (replace with actual agent calls)."""

        # Simulate processing time
        await asyncio.sleep(0.1)

        # Return mock results based on agent type
        if agent_type == AgentType.RESEARCH_SPECIALIST:
            return {
                "findings": ["Finding 1", "Finding 2"],
                "sources": ["Source 1", "Source 2"],
                "confidence": 0.85
            }
        elif agent_type == AgentType.DATA_ANALYST:
            return {
                "analysis": {"trend": "increasing", "correlation": 0.7},
                "visualizations": ["chart1.png"],
                "confidence": 0.9
            }
        elif agent_type == AgentType.MISSION_PLANNER:
            return {
                "recommendations": ["Recommendation 1", "Recommendation 2"],
                "risks": ["Risk 1", "Risk 2"],
                "confidence": 0.8
            }
        else:
            return {"result": "Generic agent result", "confidence": 0.75}

    async def _check_dependencies(self, agent_type: AgentType,
                                 execution: WorkflowExecution) -> bool:
        """Check if all dependencies for an agent are satisfied."""

        # Find the node for this agent type
        template = self.workflow_templates.get("complex_analysis", {})
        nodes = template.get("nodes", {})

        for node_name, node in nodes.items():
            if node.agent_type == agent_type:
                # Check if all dependencies are completed
                return all(
                    dep in execution.completed_nodes
                    for dep in node.dependencies
                )

        return True  # No dependencies found

    async def _handle_dependency_failure(self, agent_type: AgentType,
                                        execution: WorkflowExecution):
        """Handle dependency failures."""
        error_msg = f"Dependencies not satisfied for agent {agent_type.value}"
        execution.error_messages.append(error_msg)
        logger.warning(error_msg)

    async def _update_agent_performance(self, agent_type: AgentType,
                                       result: Dict[str, Any]):
        """Update agent performance metrics."""

        performance = self.agent_performance[agent_type]
        performance["total_executions"] += 1

        # Update success rate
        if result["success"]:
            success_count = performance["success_rate"] * (performance["total_executions"] - 1) + 1
            performance["success_rate"] = success_count / performance["total_executions"]
        else:
            success_count = performance["success_rate"] * (performance["total_executions"] - 1)
            performance["success_rate"] = success_count / performance["total_executions"]

        # Update average duration
        total_duration = performance["avg_duration"] * (performance["total_executions"] - 1) + result["duration"]
        performance["avg_duration"] = total_duration / performance["total_executions"]

        # Update quality score if available
        if "confidence" in result.get("data", {}):
            confidence = result["data"]["confidence"]
            total_quality = performance["quality_score"] * (performance["total_executions"] - 1) + confidence
            performance["quality_score"] = total_quality / performance["total_executions"]

    async def _attempt_recovery(self, agent_type: AgentType,
                               execution: WorkflowExecution,
                               state: AgentState) -> bool:
        """Attempt to recover from agent failure."""

        # Try fallback agents if available
        template = self.workflow_templates.get("complex_analysis", {})
        nodes = template.get("nodes", {})

        for node_name, node in nodes.items():
            if node.agent_type == agent_type and node.fallback_agents:
                for fallback_agent in node.fallback_agents:
                    logger.info(f"Attempting recovery with fallback agent: {fallback_agent.value}")

                    result = await self._execute_agent_with_monitoring(
                        fallback_agent, state, execution
                    )

                    if result["success"]:
                        execution.completed_nodes.add(agent_type.value)
                        execution.node_results[agent_type.value] = result
                        return True

        return False

    async def _should_replan(self, execution: WorkflowExecution,
                            state: AgentState) -> bool:
        """Determine if workflow should be replanned."""

        # Replan if too many failures
        if len(execution.failed_nodes) > 2:
            return True

        # Replan if execution is taking too long
        if execution.start_time:
            elapsed = datetime.utcnow() - execution.start_time
            if elapsed > timedelta(minutes=10):
                return True

        # Replan if quality scores are consistently low
        avg_confidence = 0
        if execution.node_results:
            confidences = [
                result.get("data", {}).get("confidence", 0.5)
                for result in execution.node_results.values()
                if result.get("success", False)
            ]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                if avg_confidence < 0.6:
                    return True

        return False

    async def _replan_workflow(self, execution: WorkflowExecution,
                              state: AgentState) -> Optional[WorkflowPlan]:
        """Create a new workflow plan based on current state."""

        # Analyze what went wrong
        failed_capabilities = [
            self.agent_capabilities[AgentType(node.split("_")[0])]
            for node in execution.failed_nodes
        ]

        # Create simpler workflow avoiding failed agents
        simplified_plan = await self.plan_workflow(
            state["query"],
            {**state["context"], "avoid_agents": list(execution.failed_nodes)}
        )

        return simplified_plan

    async def _calculate_execution_metrics(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """Calculate comprehensive execution metrics."""

        total_duration = 0
        if execution.start_time and execution.end_time:
            total_duration = (execution.end_time - execution.start_time).total_seconds()

        success_rate = len(execution.completed_nodes) / max(
            len(execution.completed_nodes) + len(execution.failed_nodes), 1
        )

        avg_confidence = 0
        if execution.node_results:
            confidences = [
                result.get("data", {}).get("confidence", 0.5)
                for result in execution.node_results.values()
                if result.get("success", False)
            ]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)

        return {
            "total_duration": total_duration,
            "success_rate": success_rate,
            "completed_nodes": len(execution.completed_nodes),
            "failed_nodes": len(execution.failed_nodes),
            "average_confidence": avg_confidence,
            "error_count": len(execution.error_messages)
        }

    def get_workflow_analytics(self) -> Dict[str, Any]:
        """Get analytics about workflow performance."""

        if not self.execution_history:
            return {"message": "No workflow executions recorded"}

        total_executions = len(self.execution_history)
        successful_executions = sum(
            1 for exec in self.execution_history
            if exec.status == WorkflowStatus.COMPLETED
        )

        avg_duration = sum(
            exec.performance_metrics.get("total_duration", 0)
            for exec in self.execution_history
        ) / total_executions

        avg_success_rate = sum(
            exec.performance_metrics.get("success_rate", 0)
            for exec in self.execution_history
        ) / total_executions

        return {
            "total_executions": total_executions,
            "success_rate": successful_executions / total_executions,
            "average_duration": avg_duration,
            "average_node_success_rate": avg_success_rate,
            "agent_performance": dict(self.agent_performance)
        }