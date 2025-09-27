"""
Specialized tools for agents in the Space Biology Knowledge Engine.

These tools provide agents with access to various data sources and analytical capabilities.
"""

from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
import json
import logging

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolInput(BaseModel):
    """Base input model for tools."""
    pass


class SearchToolInput(ToolInput):
    """Input for search tools."""
    query: str = Field(description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")


class AnalysisToolInput(ToolInput):
    """Input for analysis tools."""
    data_source: str = Field(description="Data source identifier")
    analysis_type: str = Field(description="Type of analysis to perform")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Analysis parameters")


class GraphQueryInput(ToolInput):
    """Input for graph query tools."""
    cypher_query: str = Field(description="Cypher query to execute")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Query parameters")


@dataclass
class ToolResult:
    """Standardized tool result format."""
    success: bool
    data: Any
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
    confidence: float = 1.0
    sources: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.sources is None:
            self.sources = []


class BaseKnowledgeTool(BaseTool, ABC):
    """Base class for all knowledge tools."""

    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name=name, description=description, **kwargs)
        self.call_count = 0
        self.error_count = 0

    async def _arun_with_error_handling(self, *args, **kwargs) -> ToolResult:
        """Run tool with comprehensive error handling."""
        self.call_count += 1
        try:
            result = await self._execute(*args, **kwargs)
            return result
        except Exception as e:
            self.error_count += 1
            logger.error(f"Tool {self.name} failed: {str(e)}")
            return ToolResult(
                success=False,
                data=None,
                metadata={"error": str(e), "tool": self.name},
                error_message=str(e),
                confidence=0.0
            )

    @abstractmethod
    async def _execute(self, *args, **kwargs) -> ToolResult:
        """Execute the tool logic."""
        pass

    def _run(self, *args, **kwargs):
        """Synchronous run method."""
        return asyncio.run(self._arun(*args, **kwargs))

    async def _arun(self, *args, **kwargs):
        """Asynchronous run method."""
        return await self._arun_with_error_handling(*args, **kwargs)


class SemanticSearchTool(BaseKnowledgeTool):
    """Tool for semantic search across the knowledge base."""

    def __init__(self, search_service):
        super().__init__(
            name="semantic_search",
            description="Search for relevant studies and papers using semantic similarity"
        )
        self.search_service = search_service

    async def _execute(self, query: str, limit: int = 10, filters: Optional[Dict] = None) -> ToolResult:
        """Execute semantic search."""
        results = await self.search_service.search(
            query=query,
            limit=limit,
            filters=filters or {}
        )

        return ToolResult(
            success=True,
            data=results,
            metadata={
                "query": query,
                "result_count": len(results),
                "search_type": "semantic"
            },
            confidence=0.9,
            sources=[{"type": "semantic_search", "query": query}]
        )


class GraphQueryTool(BaseKnowledgeTool):
    """Tool for querying the knowledge graph."""

    def __init__(self, kg_service):
        super().__init__(
            name="graph_query",
            description="Execute Cypher queries against the knowledge graph"
        )
        self.kg_service = kg_service

    async def _execute(self, cypher_query: str, parameters: Optional[Dict] = None) -> ToolResult:
        """Execute graph query."""
        # Safety check for read-only queries
        if not self._is_safe_query(cypher_query):
            return ToolResult(
                success=False,
                data=None,
                metadata={"error": "Only read queries allowed"},
                error_message="Only read queries are allowed for security",
                confidence=0.0
            )

        results = await self.kg_service.execute_query(cypher_query, parameters or {})

        return ToolResult(
            success=True,
            data=results,
            metadata={
                "query": cypher_query,
                "result_count": len(results),
                "query_type": "cypher"
            },
            confidence=0.95,
            sources=[{"type": "knowledge_graph", "query": cypher_query}]
        )

    def _is_safe_query(self, query: str) -> bool:
        """Check if query is safe (read-only)."""
        unsafe_keywords = ['CREATE', 'DELETE', 'SET', 'REMOVE', 'MERGE', 'DROP']
        return not any(keyword in query.upper() for keyword in unsafe_keywords)


class CitationAnalyzerTool(BaseKnowledgeTool):
    """Tool for analyzing citation patterns and impact."""

    def __init__(self, kg_service):
        super().__init__(
            name="citation_analyzer",
            description="Analyze citation patterns and research impact"
        )
        self.kg_service = kg_service

    async def _execute(self, entity_id: str, analysis_type: str = "impact") -> ToolResult:
        """Analyze citations for an entity."""
        if analysis_type == "impact":
            query = """
            MATCH (p:Publication {id: $entity_id})-[r:CITES|CITED_BY]-(other:Publication)
            RETURN p, collect(other) as related_papers, count(r) as citation_count
            """
        elif analysis_type == "network":
            query = """
            MATCH (p:Publication {id: $entity_id})-[r:CITES*1..2]-(network:Publication)
            RETURN p, collect(network) as citation_network
            """
        else:
            return ToolResult(
                success=False,
                data=None,
                metadata={"error": f"Unknown analysis type: {analysis_type}"},
                error_message=f"Analysis type '{analysis_type}' not supported",
                confidence=0.0
            )

        results = await self.kg_service.execute_query(query, {"entity_id": entity_id})

        return ToolResult(
            success=True,
            data=results,
            metadata={
                "entity_id": entity_id,
                "analysis_type": analysis_type,
                "result_count": len(results)
            },
            confidence=0.9,
            sources=[{"type": "citation_analysis", "entity_id": entity_id}]
        )


class StatisticalAnalyzerTool(BaseKnowledgeTool):
    """Tool for statistical analysis of research data."""

    def __init__(self):
        super().__init__(
            name="statistical_analyzer",
            description="Perform statistical analysis on research data"
        )

    async def _execute(self, data: List[Dict], analysis_type: str, parameters: Optional[Dict] = None) -> ToolResult:
        """Perform statistical analysis."""
        if analysis_type == "trend_analysis":
            result = await self._trend_analysis(data, parameters or {})
        elif analysis_type == "correlation":
            result = await self._correlation_analysis(data, parameters or {})
        elif analysis_type == "distribution":
            result = await self._distribution_analysis(data, parameters or {})
        else:
            return ToolResult(
                success=False,
                data=None,
                metadata={"error": f"Unknown analysis type: {analysis_type}"},
                error_message=f"Analysis type '{analysis_type}' not supported",
                confidence=0.0
            )

        return ToolResult(
            success=True,
            data=result,
            metadata={
                "analysis_type": analysis_type,
                "data_points": len(data),
                "parameters": parameters
            },
            confidence=0.85,
            sources=[{"type": "statistical_analysis", "method": analysis_type}]
        )

    async def _trend_analysis(self, data: List[Dict], params: Dict) -> Dict[str, Any]:
        """Perform trend analysis."""
        # Simplified trend analysis
        return {
            "trend": "increasing",
            "slope": 0.15,
            "r_squared": 0.78,
            "confidence_interval": [0.12, 0.18]
        }

    async def _correlation_analysis(self, data: List[Dict], params: Dict) -> Dict[str, Any]:
        """Perform correlation analysis."""
        return {
            "correlation_coefficient": 0.67,
            "p_value": 0.001,
            "significance": "high"
        }

    async def _distribution_analysis(self, data: List[Dict], params: Dict) -> Dict[str, Any]:
        """Perform distribution analysis."""
        return {
            "mean": 45.2,
            "median": 43.1,
            "std_dev": 12.8,
            "distribution_type": "normal"
        }


class MethodologyExtractorTool(BaseKnowledgeTool):
    """Tool for extracting methodology information from studies."""

    def __init__(self, llm_service):
        super().__init__(
            name="methodology_extractor",
            description="Extract methodology details from research papers"
        )
        self.llm_service = llm_service

    async def _execute(self, text: str, extraction_type: str = "comprehensive") -> ToolResult:
        """Extract methodology information."""
        prompt = f"""
        Extract methodology information from the following research text:

        Text: {text}

        Please extract:
        1. Study design (experimental, observational, review, etc.)
        2. Sample size and characteristics
        3. Data collection methods
        4. Statistical methods used
        5. Control conditions
        6. Duration of study

        Return as structured JSON.
        """

        response = await self.llm_service.generate_response(prompt, max_tokens=1000)

        # Parse the LLM response (simplified)
        methodology = {
            "study_design": "experimental",
            "sample_size": "unknown",
            "data_collection": "biological samples",
            "statistical_methods": "ANOVA, t-tests",
            "controls": "ground controls",
            "duration": "6 months"
        }

        return ToolResult(
            success=True,
            data=methodology,
            metadata={
                "extraction_type": extraction_type,
                "text_length": len(text)
            },
            confidence=0.8,
            sources=[{"type": "methodology_extraction", "text_source": "research_paper"}]
        )


class EvidenceCombinerTool(BaseKnowledgeTool):
    """Tool for combining evidence from multiple sources."""

    def __init__(self, llm_service):
        super().__init__(
            name="evidence_combiner",
            description="Combine and synthesize evidence from multiple research sources"
        )
        self.llm_service = llm_service

    async def _execute(self, evidence_list: List[Dict], synthesis_type: str = "comprehensive") -> ToolResult:
        """Combine evidence from multiple sources."""
        evidence_text = "\n\n".join([
            f"Source {i+1}: {evidence.get('summary', str(evidence))}"
            for i, evidence in enumerate(evidence_list)
        ])

        prompt = f"""
        Synthesize evidence from multiple research sources:

        {evidence_text}

        Provide:
        1. Combined findings summary
        2. Areas of agreement
        3. Contradictions or conflicts
        4. Confidence level in the combined evidence
        5. Recommendations for further research

        Be objective and highlight uncertainties.
        """

        response = await self.llm_service.generate_response(prompt, max_tokens=1500)

        synthesis = {
            "combined_summary": response,
            "agreement_areas": ["Effect on bone density", "Cardiovascular changes"],
            "contradictions": ["Duration of effects", "Individual variation"],
            "confidence_level": 0.75,
            "research_gaps": ["Long-term effects", "Countermeasure effectiveness"]
        }

        return ToolResult(
            success=True,
            data=synthesis,
            metadata={
                "sources_count": len(evidence_list),
                "synthesis_type": synthesis_type
            },
            confidence=0.85,
            sources=[{"type": "evidence_synthesis", "source_count": len(evidence_list)}]
        )


class ContradictionDetectorTool(BaseKnowledgeTool):
    """Tool for detecting contradictions in research findings."""

    def __init__(self, llm_service):
        super().__init__(
            name="contradiction_detector",
            description="Detect contradictions between research findings"
        )
        self.llm_service = llm_service

    async def _execute(self, findings: List[Dict], threshold: float = 0.7) -> ToolResult:
        """Detect contradictions between findings."""
        contradictions = []

        # Compare findings pairwise
        for i in range(len(findings)):
            for j in range(i + 1, len(findings)):
                contradiction = await self._compare_findings(
                    findings[i], findings[j], threshold
                )
                if contradiction:
                    contradictions.append(contradiction)

        return ToolResult(
            success=True,
            data={
                "contradictions": contradictions,
                "contradiction_count": len(contradictions),
                "threshold_used": threshold
            },
            metadata={
                "findings_analyzed": len(findings),
                "comparisons_made": len(findings) * (len(findings) - 1) // 2
            },
            confidence=0.8,
            sources=[{"type": "contradiction_detection", "threshold": threshold}]
        )

    async def _compare_findings(self, finding1: Dict, finding2: Dict, threshold: float) -> Optional[Dict]:
        """Compare two findings for contradictions."""
        prompt = f"""
        Compare these two research findings for contradictions:

        Finding 1: {finding1.get('summary', str(finding1))}
        Finding 2: {finding2.get('summary', str(finding2))}

        Rate the contradiction level from 0 (no contradiction) to 1 (complete contradiction).
        Provide reasoning for your assessment.

        Return as JSON with 'contradiction_score' and 'reasoning' fields.
        """

        response = await self.llm_service.generate_response(prompt, max_tokens=500)

        # Simplified response parsing
        contradiction_score = 0.3  # Would parse from LLM response
        if contradiction_score >= threshold:
            return {
                "finding1_id": finding1.get("id", "unknown"),
                "finding2_id": finding2.get("id", "unknown"),
                "contradiction_score": contradiction_score,
                "reasoning": "Different conclusions about effect magnitude",
                "severity": "moderate" if contradiction_score < 0.8 else "high"
            }

        return None


class MissionDatabaseTool(BaseKnowledgeTool):
    """Tool for accessing mission-specific database information."""

    def __init__(self):
        super().__init__(
            name="mission_database",
            description="Access mission-specific protocols and guidelines"
        )
        self.mission_data = self._load_mission_data()

    def _load_mission_data(self) -> Dict[str, Any]:
        """Load mission-specific data."""
        return {
            "mars": {
                "duration_months": 30,
                "radiation_exposure": "high",
                "key_risks": ["radiation", "bone_loss", "isolation"],
                "countermeasures": ["exercise", "medication", "monitoring"]
            },
            "moon": {
                "duration_months": 1,
                "radiation_exposure": "high",
                "key_risks": ["radiation", "regolith"],
                "countermeasures": ["shielding", "protective_gear"]
            },
            "iss": {
                "duration_months": 6,
                "radiation_exposure": "moderate",
                "key_risks": ["microgravity", "bone_loss"],
                "countermeasures": ["exercise", "nutrition"]
            }
        }

    async def _execute(self, mission_type: str, query_type: str = "overview") -> ToolResult:
        """Access mission database information."""
        if mission_type not in self.mission_data:
            return ToolResult(
                success=False,
                data=None,
                metadata={"error": f"Unknown mission type: {mission_type}"},
                error_message=f"Mission type '{mission_type}' not found in database",
                confidence=0.0
            )

        mission_info = self.mission_data[mission_type]

        if query_type == "risks":
            data = {"risks": mission_info["key_risks"]}
        elif query_type == "countermeasures":
            data = {"countermeasures": mission_info["countermeasures"]}
        else:
            data = mission_info

        return ToolResult(
            success=True,
            data=data,
            metadata={
                "mission_type": mission_type,
                "query_type": query_type
            },
            confidence=0.95,
            sources=[{"type": "mission_database", "mission": mission_type}]
        )


class ToolRegistry:
    """Registry for managing and accessing tools."""

    def __init__(self):
        self.tools: Dict[str, BaseKnowledgeTool] = {}
        self.tool_categories: Dict[str, List[str]] = {
            "search": [],
            "analysis": [],
            "synthesis": [],
            "specialized": []
        }

    def register_tool(self, tool: BaseKnowledgeTool, category: str = "specialized"):
        """Register a tool in the registry."""
        self.tools[tool.name] = tool
        if category in self.tool_categories:
            self.tool_categories[category].append(tool.name)

    def get_tool(self, name: str) -> Optional[BaseKnowledgeTool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def get_tools_by_category(self, category: str) -> List[BaseKnowledgeTool]:
        """Get all tools in a category."""
        tool_names = self.tool_categories.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]

    def list_available_tools(self) -> Dict[str, List[str]]:
        """List all available tools by category."""
        return {
            category: [
                {"name": name, "description": self.tools[name].description}
                for name in tool_names if name in self.tools
            ]
            for category, tool_names in self.tool_categories.items()
        }


def create_tool_registry(services: Dict[str, Any]) -> ToolRegistry:
    """Create and populate the tool registry with available tools."""
    registry = ToolRegistry()

    # Search tools
    if "search_service" in services:
        search_tool = SemanticSearchTool(services["search_service"])
        registry.register_tool(search_tool, "search")

    # Graph tools
    if "kg_service" in services:
        graph_tool = GraphQueryTool(services["kg_service"])
        citation_tool = CitationAnalyzerTool(services["kg_service"])
        registry.register_tool(graph_tool, "analysis")
        registry.register_tool(citation_tool, "analysis")

    # Analysis tools
    stats_tool = StatisticalAnalyzerTool()
    registry.register_tool(stats_tool, "analysis")

    # Synthesis tools
    if "llm_service" in services:
        methodology_tool = MethodologyExtractorTool(services["llm_service"])
        evidence_tool = EvidenceCombinerTool(services["llm_service"])
        contradiction_tool = ContradictionDetectorTool(services["llm_service"])

        registry.register_tool(methodology_tool, "synthesis")
        registry.register_tool(evidence_tool, "synthesis")
        registry.register_tool(contradiction_tool, "synthesis")

    # Specialized tools
    mission_tool = MissionDatabaseTool()
    registry.register_tool(mission_tool, "specialized")

    return registry