"""
Pydantic models for API request/response validation in the Space Biology Knowledge Engine.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


class SourceType(str, Enum):
    """Valid data source types."""
    OSDR = "osdr"
    PMC = "pmc"
    NTRS = "ntrs"
    ALL = "all"


class ResearchType(str, Enum):
    """Valid research type categories."""
    GENOMICS = "genomics"
    PROTEOMICS = "proteomics"
    PHYSIOLOGY = "physiology"
    BIOCHEMISTRY = "biochemistry"
    CELL_BIOLOGY = "cell_biology"
    MICROBIOLOGY = "microbiology"
    PSYCHOLOGY = "psychology"
    BIOMEDICAL = "biomedical"
    RADIATION = "radiation"
    OTHER = "other"


class MissionType(str, Enum):
    """Valid mission types."""
    MARS = "mars"
    MOON = "moon"
    ISS = "iss"
    DEEP_SPACE = "deep_space"
    OTHER = "other"


class OrganismType(str, Enum):
    """Valid organism types."""
    HUMAN = "human"
    MOUSE = "mouse"
    RAT = "rat"
    PLANT = "plant"
    BACTERIA = "bacteria"
    YEAST = "yeast"
    OTHER = "other"


# Request Models
class SearchRequest(BaseModel):
    """Request model for search endpoints."""
    query: str = Field(..., min_length=2, max_length=500, description="Search query text")
    filters: Optional[Dict[str, Any]] = Field(default={}, description="Search filters")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    offset: int = Field(default=0, ge=0, description="Result offset for pagination")
    source: Optional[SourceType] = Field(default=None, description="Filter by data source")
    research_type: Optional[ResearchType] = Field(default=None, description="Filter by research type")
    mission_type: Optional[MissionType] = Field(default=None, description="Filter by mission relevance")
    organism: Optional[str] = Field(default=None, description="Filter by organism name")
    date_from: Optional[datetime] = Field(default=None, description="Filter by publication date from")
    date_to: Optional[datetime] = Field(default=None, description="Filter by publication date to")

    @validator('query')
    def validate_query(cls, v):
        """Validate and sanitize query string."""
        if not v.strip():
            raise ValueError("Query cannot be empty")

        # Basic sanitization
        sanitized = v.strip()
        if len(sanitized) < 2:
            raise ValueError("Query must be at least 2 characters")

        return sanitized

    @root_validator
    def validate_date_range(cls, values):
        """Validate date range."""
        date_from = values.get('date_from')
        date_to = values.get('date_to')

        if date_from and date_to and date_from > date_to:
            raise ValueError("date_from must be before date_to")

        return values


class AgentRequest(BaseModel):
    """Request model for AI agent interactions."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message to agent")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for context")
    agent_type: str = Field(..., description="Type of agent (research, mission, risk)")
    mission_context: Optional[Dict[str, str]] = Field(default=None, description="Mission-specific parameters")

    @validator('agent_type')
    def validate_agent_type(cls, v):
        """Validate agent type."""
        valid_types = ['research', 'mission', 'risk']
        if v not in valid_types:
            raise ValueError(f"agent_type must be one of: {valid_types}")
        return v


class GraphQuery(BaseModel):
    """Request model for knowledge graph queries."""
    node_id: Optional[str] = Field(default=None, description="Center node ID")
    node_type: Optional[str] = Field(default=None, description="Node type filter")
    relationship_type: Optional[str] = Field(default=None, description="Relationship type filter")
    depth: int = Field(default=2, ge=1, le=3, description="Maximum traversal depth")
    limit: int = Field(default=50, ge=1, le=200, description="Maximum number of nodes/relationships")


class DataIngestionRequest(BaseModel):
    """Request model for triggering data ingestion."""
    sources: List[SourceType] = Field(default=[SourceType.ALL], description="Data sources to ingest")
    limit: Optional[int] = Field(default=None, ge=1, le=1000, description="Limit number of items to ingest")
    force_refresh: bool = Field(default=False, description="Force refresh of existing data")


# Response Models
class StudyResponse(BaseModel):
    """Response model for study data."""
    id: str
    title: str
    description: str
    source: str
    organisms: List[str]
    factors: List[str]
    methodology: str
    research_type: str
    mission_relevance: str
    duration: Optional[int]
    sample_size: Optional[int]
    publication_date: Optional[datetime]
    relevance_score: Optional[float] = Field(default=None, description="Search relevance score")
    created_at: datetime
    updated_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OrganismResponse(BaseModel):
    """Response model for organism data."""
    name: str
    scientific_name: str
    organism_type: str
    taxonomy_id: Optional[str]
    description: str
    space_relevance: str
    study_count: Optional[int] = Field(default=None, description="Number of associated studies")


class PublicationResponse(BaseModel):
    """Response model for publication data."""
    title: str
    doi: Optional[str]
    pmid: Optional[str]
    abstract: str
    authors: List[str]
    journal: str
    publication_date: Optional[datetime]
    source: str
    url: str
    keywords: List[str]
    research_areas: List[str]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchResponse(BaseModel):
    """Response model for search results."""
    query: str
    total_results: int
    results: List[StudyResponse]
    facets: Optional[Dict[str, Any]] = Field(default=None, description="Search facets/filters")
    pagination: Dict[str, Any]
    execution_time: float = Field(description="Query execution time in seconds")


class AgentResponse(BaseModel):
    """Response model for AI agent interactions."""
    message: str
    agent_type: str
    conversation_id: str
    references: List[str] = Field(default=[], description="Referenced study/publication IDs")
    confidence: float = Field(ge=0.0, le=1.0, description="Response confidence score")
    timestamp: datetime = Field(default_factory=datetime.now)
    context_used: Optional[Dict[str, Any]] = Field(default=None, description="Context used in response")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class GraphNode(BaseModel):
    """Response model for graph nodes."""
    id: str
    label: str
    properties: Dict[str, Any]
    type: str


class GraphEdge(BaseModel):
    """Response model for graph edges."""
    id: str
    source: str
    target: str
    type: str
    properties: Dict[str, Any]


class GraphResponse(BaseModel):
    """Response model for graph data."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    total_nodes: int
    total_edges: int
    query_depth: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    uptime: float
    components: Dict[str, str]
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StatsResponse(BaseModel):
    """Response model for system statistics."""
    studies: Dict[str, Any]
    organisms: Dict[str, Any]
    publications: Dict[str, Any]
    relationships: Dict[str, Any]
    data_files: Dict[str, Any]
    embeddings: Dict[str, Any]
    last_updated: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DataIngestionResponse(BaseModel):
    """Response model for data ingestion status."""
    status: str
    sources_processed: List[str]
    items_ingested: Dict[str, int]
    errors: List[str] = Field(default=[])
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    error_code: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Filter Models
class SearchFilters(BaseModel):
    """Available search filters."""
    sources: List[str]
    research_types: List[str]
    mission_types: List[str]
    organisms: List[str]
    date_range: Dict[str, Optional[datetime]]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RiskAssessmentRequest(BaseModel):
    """Request model for risk assessment agent."""
    mission_type: MissionType
    mission_duration: int = Field(ge=1, le=1000, description="Mission duration in days")
    crew_size: int = Field(ge=1, le=20, description="Number of crew members")
    environmental_factors: List[str] = Field(default=[], description="Specific environmental concerns")
    health_baseline: Optional[Dict[str, Any]] = Field(default=None, description="Crew health baseline data")


class MissionPlanningRequest(BaseModel):
    """Request model for mission planning agent."""
    mission_type: MissionType
    mission_duration: int = Field(ge=1, le=1000, description="Mission duration in days")
    launch_date: Optional[datetime] = Field(default=None, description="Planned launch date")
    objectives: List[str] = Field(default=[], description="Mission objectives")
    constraints: Dict[str, Any] = Field(default={}, description="Mission constraints")
    crew_profile: Optional[Dict[str, Any]] = Field(default=None, description="Crew characteristics")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }