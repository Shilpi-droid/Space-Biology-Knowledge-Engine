
"""
Dependency injection for FastAPI endpoints.
"""

from functools import lru_cache
from typing import AsyncGenerator

from ..config.settings import get_settings
from ..services.knowledge_graph import KnowledgeGraphService
from ..services.semantic_search import SemanticSearchService
from ..services.data_ingestion import DataIngestionService
from ..services.llm_service import LLMService
from ..services.evidence_synthesis import EvidenceSynthesisService
from ..services.temporal_analysis import TemporalAnalysisService
from ..services.mission_intelligence import MissionIntelligenceService
from ..services.agents.research_agent import ResearchAgent
from ..services.agents.mission_agent import MissionAgent
from ..services.agents.risk_agent import RiskAgent


# Service instances cache
_service_cache = {}


@lru_cache()
def get_kg_service() -> KnowledgeGraphService:
    """Get knowledge graph service instance."""
    if "kg_service" not in _service_cache:
        settings = get_settings()
        _service_cache["kg_service"] = KnowledgeGraphService(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
    return _service_cache["kg_service"]


@lru_cache()
def get_search_service() -> SemanticSearchService:
    """Get semantic search service instance."""
    if "search_service" not in _service_cache:
        settings = get_settings()
        _service_cache["search_service"] = SemanticSearchService(
            embedding_model_name=settings.embedding_model_name,
            embedding_dimension=settings.embedding_dimension
        )
    return _service_cache["search_service"]


@lru_cache()
def get_llm_service() -> LLMService:
    """Get LLM service instance."""
    if "llm_service" not in _service_cache:
        settings = get_settings()
        _service_cache["llm_service"] = LLMService()
    return _service_cache["llm_service"]


@lru_cache()
def get_data_ingestion_service() -> DataIngestionService:
    """Get data ingestion service instance."""
    if "data_ingestion_service" not in _service_cache:
        settings = get_settings()
        _service_cache["data_ingestion_service"] = DataIngestionService(
            osdr_base_url=settings.osdr_base_url,
            pmc_base_url=settings.pmc_base_url,
            ntrs_base_url=settings.ntrs_base_url,
            api_delay=settings.api_delay,
            max_requests_per_minute=settings.max_requests_per_minute
        )
    return _service_cache["data_ingestion_service"]


@lru_cache()
def get_evidence_synthesis_service() -> EvidenceSynthesisService:
    """Get evidence synthesis service instance."""
    if "evidence_synthesis_service" not in _service_cache:
        kg_service = get_kg_service()
        llm_service = get_llm_service()
        _service_cache["evidence_synthesis_service"] = EvidenceSynthesisService(
            kg_service=kg_service,
            llm_service=llm_service
        )
    return _service_cache["evidence_synthesis_service"]


@lru_cache()
def get_temporal_analysis_service() -> TemporalAnalysisService:
    """Get temporal analysis service instance."""
    if "temporal_analysis_service" not in _service_cache:
        kg_service = get_kg_service()
        _service_cache["temporal_analysis_service"] = TemporalAnalysisService(
            kg_service=kg_service
        )
    return _service_cache["temporal_analysis_service"]


@lru_cache()
def get_mission_intelligence_service() -> MissionIntelligenceService:
    """Get mission intelligence service instance."""
    if "mission_intelligence_service" not in _service_cache:
        kg_service = get_kg_service()
        search_service = get_search_service()
        llm_service = get_llm_service()
        temporal_service = get_temporal_analysis_service()
        evidence_service = get_evidence_synthesis_service()

        _service_cache["mission_intelligence_service"] = MissionIntelligenceService(
            kg_service=kg_service,
            search_service=search_service,
            llm_service=llm_service,
            temporal_service=temporal_service,
            evidence_service=evidence_service
        )
    return _service_cache["mission_intelligence_service"]


@lru_cache()
def get_research_agent() -> ResearchAgent:
    """Get research agent instance."""
    if "research_agent" not in _service_cache:
        kg_service = get_kg_service()
        search_service = get_search_service()
        llm_service = get_llm_service()
        evidence_service = get_evidence_synthesis_service()

        _service_cache["research_agent"] = ResearchAgent(
            kg_service=kg_service,
            search_service=search_service,
            llm_service=llm_service,
            evidence_service=evidence_service
        )
    return _service_cache["research_agent"]


@lru_cache()
def get_mission_agent() -> MissionAgent:
    """Get mission agent instance."""
    if "mission_agent" not in _service_cache:
        kg_service = get_kg_service()
        search_service = get_search_service()
        llm_service = get_llm_service()
        mission_intelligence_service = get_mission_intelligence_service()

        _service_cache["mission_agent"] = MissionAgent(
            kg_service=kg_service,
            search_service=search_service,
            llm_service=llm_service,
            mission_intelligence_service=mission_intelligence_service
        )
    return _service_cache["mission_agent"]


@lru_cache()
def get_risk_agent() -> RiskAgent:
    """Get risk agent instance."""
    if "risk_agent" not in _service_cache:
        kg_service = get_kg_service()
        search_service = get_search_service()
        llm_service = get_llm_service()
        evidence_service = get_evidence_synthesis_service()

        _service_cache["risk_agent"] = RiskAgent(
            kg_service=kg_service,
            search_service=search_service,
            llm_service=llm_service,
            evidence_service=evidence_service
        )
    return _service_cache["risk_agent"]


async def cleanup_services():
    """Cleanup service instances on shutdown."""
    for service_name, service in _service_cache.items():
        if hasattr(service, 'close'):
            await service.close()

    _service_cache.clear()


# Dependency providers for testing
def override_kg_service(service: KnowledgeGraphService):
    """Override knowledge graph service for testing."""
    _service_cache["kg_service"] = service


def override_search_service(service: SemanticSearchService):
    """Override search service for testing."""
    _service_cache["search_service"] = service


def override_llm_service(service: LLMService):
    """Override LLM service for testing."""
    _service_cache["llm_service"] = service