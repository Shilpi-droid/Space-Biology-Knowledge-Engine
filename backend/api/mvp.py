"""
Core MVP API Endpoints

This module provides the main API endpoints for the guaranteed working demo.
Clean, simple interface over the hybrid storage GraphRAG system.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from pydantic import BaseModel, Field
from datetime import datetime

from ..core.ingestion_service import get_mvp_service, SpaceBiologyMVPService, demo_query_examples

router = APIRouter(prefix="/mvp", tags=["mvp_core"])


class QueryRequest(BaseModel):
    """Request model for GraphRAG queries."""
    query: str = Field(..., min_length=1, max_length=1000, description="Research query")
    strategy: Optional[str] = Field(default=None, description="Processing strategy (auto-detected if None)")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results to return")


class SearchRequest(BaseModel):
    """Request model for simple semantic search."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results")


class MVPQueryResponse(BaseModel):
    """Response model for MVP queries."""
    query: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response model for search results."""
    query: str
    results: List[Dict[str, Any]]
    total_found: int
    search_type: str
    timestamp: str


class SystemStatus(BaseModel):
    """System status response."""
    service_status: str
    dataset_loaded: bool
    last_ingestion: Optional[str]
    storage: Dict[str, Any]
    timestamp: str


class IngestionRequest(BaseModel):
    """Request model for data ingestion."""
    max_papers: int = Field(default=1000, ge=100, le=5000, description="Maximum papers to collect")
    force_refresh: bool = Field(default=False, description="Force fresh data collection")


@router.get("/status", response_model=SystemStatus)
async def get_system_status(
    service: SpaceBiologyMVPService = Depends(get_mvp_service)
):
    """
    Get comprehensive system status.

    Returns current status of all system components including:
    - Service initialization status
    - Dataset loading status
    - Storage system health
    - Component readiness
    """
    try:
        status = await service.get_system_status()
        return SystemStatus(
            service_status=status["service_status"],
            dataset_loaded=status["dataset_loaded"],
            last_ingestion=status.get("last_ingestion"),
            storage=status.get("storage", {}),
            timestamp=status["timestamp"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@router.post("/ingest")
async def ingest_dataset(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    service: SpaceBiologyMVPService = Depends(get_mvp_service)
):
    """
    Ingest 5-year space biology dataset from NASA ADS.

    This endpoint triggers the complete data pipeline:
    1. Collect papers from NASA ADS API
    2. Store in hybrid architecture (SQLite + Neo4j + FAISS)
    3. Build semantic indexes
    4. Enable GraphRAG queries

    **Note**: Requires NASA_ADS_TOKEN environment variable.
    """
    try:
        # Start ingestion in background for large datasets
        if request.max_papers > 500:
            background_tasks.add_task(
                _background_ingestion,
                service,
                request.max_papers,
                request.force_refresh
            )
            return {
                "status": "started",
                "message": f"Dataset ingestion started in background for {request.max_papers} papers",
                "estimated_time_minutes": request.max_papers // 100 * 2,  # Rough estimate
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            # Process immediately for smaller datasets
            result = await service.ingest_five_year_dataset(
                max_papers=request.max_papers,
                force_refresh=request.force_refresh
            )
            return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


async def _background_ingestion(service: SpaceBiologyMVPService,
                               max_papers: int,
                               force_refresh: bool):
    """Background task for large dataset ingestion."""
    try:
        await service.ingest_five_year_dataset(
            max_papers=max_papers,
            force_refresh=force_refresh
        )
    except Exception as e:
        # Log error - in production would notify via webhook/email
        print(f"Background ingestion failed: {str(e)}")


@router.post("/query", response_model=MVPQueryResponse)
async def graphrag_query(
    request: QueryRequest,
    service: SpaceBiologyMVPService = Depends(get_mvp_service)
):
    """
    Process research query using GraphRAG methodology.

    This is the main intelligence endpoint that:
    1. Analyzes query complexity and selects optimal strategy
    2. Performs semantic search across paper abstracts
    3. Traverses knowledge graph for related entities
    4. Enriches context with metadata from relational store
    5. Synthesizes comprehensive answer using LLM

    **Available Strategies:**
    - `literature_search`: General research questions
    - `author_analysis`: Questions about researchers
    - `citation_analysis`: Questions about research impact
    - `topic_exploration`: Questions about research areas
    - `temporal_analysis`: Questions about research evolution
    - `comparative_analysis`: Comparison questions
    """
    try:
        result = await service.query(
            user_query=request.query,
            strategy=request.strategy,
            max_results=request.max_results
        )

        return MVPQueryResponse(
            query=result["query"],
            answer=result["answer"],
            confidence=result["confidence"],
            sources=result["sources"],
            metadata=result["metadata"]
        )

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.post("/search", response_model=SearchResponse)
async def semantic_search(
    request: SearchRequest,
    service: SpaceBiologyMVPService = Depends(get_mvp_service)
):
    """
    Simple semantic search across research papers.

    Lightweight alternative to full GraphRAG for when you just need
    to find relevant papers quickly. Uses FAISS vector similarity
    search on paper titles and abstracts.
    """
    try:
        result = await service.search_papers(
            query=request.query,
            limit=request.limit
        )

        return SearchResponse(
            query=result["query"],
            results=result["results"],
            total_found=result["total_found"],
            search_type=result["search_type"],
            timestamp=result["timestamp"]
        )

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/author/{author_name}")
async def get_author_insights(
    author_name: str,
    service: SpaceBiologyMVPService = Depends(get_mvp_service)
):
    """
    Get comprehensive insights about a researcher.

    Combines data from all storage systems to provide:
    - Publication list and statistics
    - Collaboration network analysis
    - Research impact metrics
    - Recent work and trends
    """
    try:
        result = await service.get_author_insights(author_name)
        return result

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Author analysis failed: {str(e)}")


@router.get("/topic/{topic_name}")
async def get_topic_analysis(
    topic_name: str,
    service: SpaceBiologyMVPService = Depends(get_mvp_service)
):
    """
    Get analysis for a specific research topic.

    Provides comprehensive overview including:
    - Related papers and key publications
    - Research statistics and trends
    - Temporal evolution of the field
    - Key researchers and institutions
    """
    try:
        result = await service.get_topic_analysis(topic_name)
        return result

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Topic analysis failed: {str(e)}")


@router.get("/examples")
async def get_demo_examples():
    """
    Get example queries for testing the system.

    Returns a curated list of example queries that demonstrate
    different capabilities and processing strategies.
    """
    try:
        examples = await demo_query_examples()
        return {
            "examples": examples,
            "usage_tips": [
                "Try different strategies to see how processing changes",
                "Ask about specific authors to see collaboration networks",
                "Query temporal trends to understand research evolution",
                "Compare different research areas or approaches"
            ],
            "sample_authors": [
                "Smith, J",  # Common space biology authors would be here
                "Johnson, M",
                "Williams, K"
            ],
            "sample_topics": [
                "bone physiology",
                "muscle atrophy",
                "radiation biology",
                "microgravity",
                "space medicine"
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get examples: {str(e)}")


@router.get("/analytics")
async def get_system_analytics(
    service: SpaceBiologyMVPService = Depends(get_mvp_service)
):
    """
    Get system analytics and usage statistics.

    Provides insights into:
    - Dataset composition and statistics
    - Query processing performance
    - Storage system utilization
    - Research landscape overview
    """
    try:
        status = await service.get_system_status()

        # Build analytics from system status
        analytics = {
            "dataset_overview": {
                "papers_loaded": status.get("storage", {}).get("sqlite", {}).get("papers", 0),
                "authors_tracked": status.get("storage", {}).get("sqlite", {}).get("authors", 0),
                "graph_nodes": status.get("storage", {}).get("neo4j", {}).get("total_nodes", 0),
                "semantic_vectors": status.get("storage", {}).get("faiss", {}).get("vectors", 0)
            },
            "system_health": {
                "service_status": status["service_status"],
                "dataset_loaded": status["dataset_loaded"],
                "storage_systems": {
                    "sqlite": status.get("storage", {}).get("sqlite", {}).get("status", "unknown"),
                    "neo4j": status.get("storage", {}).get("neo4j", {}).get("status", "unknown"),
                    "faiss": status.get("storage", {}).get("faiss", {}).get("status", "unknown")
                }
            },
            "capabilities": {
                "graphrag_strategies": status.get("query_engine", {}).get("available_strategies", []),
                "search_types": ["semantic", "hybrid", "graph_traversal"],
                "analysis_types": ["author", "topic", "citation", "temporal"]
            },
            "last_updated": status["timestamp"]
        }

        return analytics

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


@router.post("/demo")
async def run_demo_query(
    service: SpaceBiologyMVPService = Depends(get_mvp_service)
):
    """
    Run a demonstration query to showcase system capabilities.

    Executes a pre-selected query that demonstrates the GraphRAG
    methodology and returns both the result and explanation of
    the processing steps.
    """
    try:
        # Demo query that showcases multiple capabilities
        demo_query = "What are the main effects of microgravity on human physiology and what countermeasures are being researched?"

        # Process the query
        result = await service.query(
            user_query=demo_query,
            strategy=None,  # Let it auto-detect
            max_results=10
        )

        # Add explanation of what happened
        explanation = {
            "what_happened": [
                "1. Query analyzed and 'literature_search' strategy selected",
                "2. Semantic search found relevant papers using FAISS vector similarity",
                "3. Neo4j graph traversal identified related authors and citations",
                "4. SQLite provided metadata enrichment (journals, dates, citations)",
                "5. LLM synthesized comprehensive answer from assembled context",
                "6. Confidence calculated based on source quality and relevance"
            ],
            "technologies_used": [
                "NASA ADS API (data source)",
                "FAISS (semantic search)",
                "Neo4j (graph relationships)",
                "SQLite (metadata storage)",
                "Sentence Transformers (embeddings)",
                "OpenAI GPT (synthesis)"
            ],
            "demo_highlights": [
                f"Analyzed {result['metadata']['papers_analyzed']} papers",
                f"Explored {result['metadata']['graph_entities_explored']} graph entities",
                f"Achieved {result['confidence']:.1%} confidence",
                f"Processing time: {result['metadata']['processing_time']:.2f} seconds"
            ]
        }

        return {
            "demo_query": demo_query,
            "result": result,
            "explanation": explanation,
            "next_steps": [
                "Try your own queries using /mvp/query",
                "Explore specific authors with /mvp/author/{name}",
                "Search papers with /mvp/search",
                "Check system status with /mvp/status"
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Space Biology MVP"
    }