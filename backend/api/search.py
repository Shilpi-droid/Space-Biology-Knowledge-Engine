"""
Search API endpoints for Space Biology Knowledge Engine.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ..services.semantic_search import SemanticSearchService
from ..services.knowledge_graph import KnowledgeGraphService
from ..core.dependencies import get_search_service, get_kg_service

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    """Request model for search operations."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")
    search_type: str = Field(default="hybrid", regex="^(semantic|keyword|hybrid)$")


class SearchResponse(BaseModel):
    """Response model for search results."""
    results: List[Dict[str, Any]]
    total_count: int
    query: str
    filters_applied: Optional[Dict[str, Any]]
    search_metadata: Dict[str, Any]


class FilterOptions(BaseModel):
    """Available filter options."""
    organisms: List[str]
    factors: List[str]
    research_types: List[str]
    date_ranges: List[str]
    data_sources: List[str]


@router.post("/semantic", response_model=SearchResponse)
async def semantic_search(
    request: SearchRequest,
    search_service: SemanticSearchService = Depends(get_search_service)
):
    """Perform semantic search across knowledge base."""
    try:
        results = await search_service.search(
            query=request.query,
            limit=request.limit,
            filters=request.filters
        )

        # Calculate relevance scores and metadata
        search_metadata = {
            "search_type": "semantic",
            "embedding_model": search_service.embedding_model_name,
            "similarity_threshold": search_service.similarity_threshold,
            "processing_time_ms": 0  # TODO: Add timing
        }

        return SearchResponse(
            results=results,
            total_count=len(results),
            query=request.query,
            filters_applied=request.filters,
            search_metadata=search_metadata
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/advanced", response_model=SearchResponse)
async def advanced_search(
    request: SearchRequest,
    kg_service: KnowledgeGraphService = Depends(get_kg_service),
    search_service: SemanticSearchService = Depends(get_search_service)
):
    """Perform advanced search with graph-based filtering."""
    try:
        # Combine semantic search with graph queries
        semantic_results = await search_service.search(
            query=request.query,
            limit=request.limit * 2,  # Get more for filtering
            filters=request.filters
        )

        # Apply graph-based filters if specified
        if request.filters:
            filtered_results = await _apply_graph_filters(
                semantic_results, request.filters, kg_service
            )
        else:
            filtered_results = semantic_results

        # Limit final results
        final_results = filtered_results[:request.limit]

        search_metadata = {
            "search_type": "advanced",
            "graph_filters_applied": bool(request.filters),
            "semantic_candidates": len(semantic_results),
            "filtered_results": len(filtered_results),
            "final_results": len(final_results)
        }

        return SearchResponse(
            results=final_results,
            total_count=len(final_results),
            query=request.query,
            filters_applied=request.filters,
            search_metadata=search_metadata
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced search failed: {str(e)}")


@router.get("/filters", response_model=FilterOptions)
async def get_filter_options(
    kg_service: KnowledgeGraphService = Depends(get_kg_service)
):
    """Get available filter options for search."""
    try:
        # Query graph for available filter values
        organisms = await kg_service.get_distinct_organisms()
        factors = await kg_service.get_distinct_factors()
        research_types = await kg_service.get_research_types()

        # Predefined options
        date_ranges = [
            "last_year",
            "last_3_years",
            "last_5_years",
            "last_10_years",
            "all_time"
        ]

        data_sources = [
            "osdr",
            "pmc",
            "ntrs",
            "other"
        ]

        return FilterOptions(
            organisms=organisms,
            factors=factors,
            research_types=research_types,
            date_ranges=date_ranges,
            data_sources=data_sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get filter options: {str(e)}")


@router.get("/suggestions")
async def get_search_suggestions(
    q: str = Query(..., min_length=1, description="Partial query for suggestions"),
    limit: int = Query(default=5, ge=1, le=20),
    search_service: SemanticSearchService = Depends(get_search_service)
):
    """Get search suggestions based on partial query."""
    try:
        suggestions = await search_service.get_suggestions(q, limit)
        return {"suggestions": suggestions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")


@router.get("/autocomplete")
async def autocomplete_entities(
    term: str = Query(..., min_length=1, description="Term to autocomplete"),
    entity_type: str = Query(default="all", regex="^(organism|factor|researcher|all)$"),
    kg_service: KnowledgeGraphService = Depends(get_kg_service)
):
    """Autocomplete entities from knowledge graph."""
    try:
        results = await kg_service.autocomplete_entities(term, entity_type)
        return {"completions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Autocomplete failed: {str(e)}")


async def _apply_graph_filters(results: List[Dict[str, Any]],
                              filters: Dict[str, Any],
                              kg_service: KnowledgeGraphService) -> List[Dict[str, Any]]:
    """Apply graph-based filters to search results."""
    filtered_results = []

    for result in results:
        if await _matches_filters(result, filters, kg_service):
            filtered_results.append(result)

    return filtered_results


async def _matches_filters(result: Dict[str, Any],
                          filters: Dict[str, Any],
                          kg_service: KnowledgeGraphService) -> bool:
    """Check if a result matches the specified filters."""
    # Check organism filter
    if "organisms" in filters:
        result_organisms = result.get("organisms", [])
        if not any(org in filters["organisms"] for org in result_organisms):
            return False

    # Check factor filter
    if "factors" in filters:
        result_factors = result.get("factors", [])
        if not any(factor in filters["factors"] for factor in result_factors):
            return False

    # Check date range filter
    if "date_range" in filters:
        if not _matches_date_range(result, filters["date_range"]):
            return False

    # Check data source filter
    if "data_sources" in filters:
        result_source = result.get("source", "")
        if result_source not in filters["data_sources"]:
            return False

    return True


def _matches_date_range(result: Dict[str, Any], date_range: str) -> bool:
    """Check if result matches date range filter."""
    # Implementation for date range matching
    # This would need actual date parsing and comparison
    return True  # Placeholder