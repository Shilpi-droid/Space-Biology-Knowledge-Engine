"""
Knowledge Graph API endpoints for Space Biology Knowledge Engine.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ..services.knowledge_graph import KnowledgeGraphService
from ..core.dependencies import get_kg_service

router = APIRouter(prefix="/graph", tags=["knowledge_graph"])


class NodeRequest(BaseModel):
    """Request model for node operations."""
    node_id: str = Field(..., description="Node ID")
    include_relationships: bool = Field(default=True, description="Include node relationships")
    max_depth: int = Field(default=1, ge=0, le=3, description="Maximum relationship depth")


class SubgraphRequest(BaseModel):
    """Request model for subgraph extraction."""
    center_nodes: List[str] = Field(..., description="Central node IDs")
    max_depth: int = Field(default=2, ge=1, le=3, description="Maximum traversal depth")
    relationship_types: Optional[List[str]] = Field(default=None, description="Filter by relationship types")
    node_types: Optional[List[str]] = Field(default=None, description="Filter by node types")
    max_nodes: int = Field(default=100, ge=1, le=500, description="Maximum nodes in result")


class PathRequest(BaseModel):
    """Request model for path finding."""
    start_node: str = Field(..., description="Starting node ID")
    end_node: str = Field(..., description="Ending node ID")
    max_path_length: int = Field(default=5, ge=1, le=10, description="Maximum path length")
    relationship_types: Optional[List[str]] = Field(default=None, description="Allowed relationship types")


class GraphNode(BaseModel):
    """Graph node representation."""
    id: str
    labels: List[str]
    properties: Dict[str, Any]
    relationships: Optional[List[Dict[str, Any]]] = None


class GraphRelationship(BaseModel):
    """Graph relationship representation."""
    id: str
    type: str
    start_node: str
    end_node: str
    properties: Dict[str, Any]


class SubgraphResponse(BaseModel):
    """Subgraph response model."""
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    metadata: Dict[str, Any]


class PathResponse(BaseModel):
    """Path finding response model."""
    paths: List[Dict[str, Any]]
    shortest_path_length: Optional[int]
    total_paths_found: int


@router.get("/nodes/{node_id}", response_model=GraphNode)
async def get_node(
    node_id: str,
    include_relationships: bool = Query(default=True),
    max_depth: int = Query(default=1, ge=0, le=3),
    kg_service: KnowledgeGraphService = Depends(get_kg_service)
):
    """Get detailed information about a specific node."""
    try:
        node = await kg_service.get_node_by_id(
            node_id=node_id,
            include_relationships=include_relationships,
            max_depth=max_depth
        )

        if not node:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

        return GraphNode(
            id=node["id"],
            labels=node.get("labels", []),
            properties=node.get("properties", {}),
            relationships=node.get("relationships") if include_relationships else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get node: {str(e)}")


@router.post("/subgraph", response_model=SubgraphResponse)
async def extract_subgraph(
    request: SubgraphRequest,
    kg_service: KnowledgeGraphService = Depends(get_kg_service)
):
    """Extract a subgraph around specified center nodes."""
    try:
        subgraph_data = await kg_service.extract_subgraph(
            center_nodes=request.center_nodes,
            max_depth=request.max_depth,
            relationship_types=request.relationship_types,
            node_types=request.node_types,
            max_nodes=request.max_nodes
        )

        # Convert to API format
        nodes = [
            GraphNode(
                id=node["id"],
                labels=node.get("labels", []),
                properties=node.get("properties", {})
            )
            for node in subgraph_data["nodes"]
        ]

        relationships = [
            GraphRelationship(
                id=rel["id"],
                type=rel["type"],
                start_node=rel["start_node"],
                end_node=rel["end_node"],
                properties=rel.get("properties", {})
            )
            for rel in subgraph_data["relationships"]
        ]

        metadata = {
            "center_nodes": request.center_nodes,
            "max_depth": request.max_depth,
            "total_nodes": len(nodes),
            "total_relationships": len(relationships),
            "node_types": list(set(label for node in nodes for label in node.labels)),
            "relationship_types": list(set(rel.type for rel in relationships))
        }

        return SubgraphResponse(
            nodes=nodes,
            relationships=relationships,
            metadata=metadata
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract subgraph: {str(e)}")


@router.post("/paths", response_model=PathResponse)
async def find_paths(
    request: PathRequest,
    kg_service: KnowledgeGraphService = Depends(get_kg_service)
):
    """Find paths between two nodes in the knowledge graph."""
    try:
        paths_data = await kg_service.find_paths(
            start_node=request.start_node,
            end_node=request.end_node,
            max_path_length=request.max_path_length,
            relationship_types=request.relationship_types
        )

        return PathResponse(
            paths=paths_data["paths"],
            shortest_path_length=paths_data.get("shortest_path_length"),
            total_paths_found=len(paths_data["paths"])
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find paths: {str(e)}")


@router.get("/relationships")
async def get_relationship_types(
    kg_service: KnowledgeGraphService = Depends(get_kg_service)
):
    """Get all relationship types in the knowledge graph."""
    try:
        relationship_types = await kg_service.get_relationship_types()
        return {
            "relationship_types": relationship_types,
            "total_count": len(relationship_types)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get relationship types: {str(e)}")


@router.get("/node-types")
async def get_node_types(
    kg_service: KnowledgeGraphService = Depends(get_kg_service)
):
    """Get all node types (labels) in the knowledge graph."""
    try:
        node_types = await kg_service.get_node_types()
        return {
            "node_types": node_types,
            "total_count": len(node_types)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get node types: {str(e)}")


@router.get("/statistics")
async def get_graph_statistics(
    kg_service: KnowledgeGraphService = Depends(get_kg_service)
):
    """Get knowledge graph statistics."""
    try:
        stats = await kg_service.get_graph_statistics()
        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get graph statistics: {str(e)}")


@router.get("/search/neighbors")
async def search_neighbors(
    node_id: str = Query(..., description="Node ID to find neighbors for"),
    relationship_type: Optional[str] = Query(default=None, description="Filter by relationship type"),
    node_type: Optional[str] = Query(default=None, description="Filter by neighbor node type"),
    limit: int = Query(default=20, ge=1, le=100),
    kg_service: KnowledgeGraphService = Depends(get_kg_service)
):
    """Search for neighbors of a specific node."""
    try:
        neighbors = await kg_service.get_node_neighbors(
            node_id=node_id,
            relationship_type=relationship_type,
            node_type=node_type,
            limit=limit
        )

        return {
            "node_id": node_id,
            "neighbors": neighbors,
            "total_count": len(neighbors)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search neighbors: {str(e)}")


@router.get("/centrality/{node_id}")
async def get_node_centrality(
    node_id: str,
    centrality_type: str = Query(default="degree", regex="^(degree|betweenness|closeness|pagerank)$"),
    kg_service: KnowledgeGraphService = Depends(get_kg_service)
):
    """Get centrality measures for a specific node."""
    try:
        centrality = await kg_service.calculate_centrality(node_id, centrality_type)
        return {
            "node_id": node_id,
            "centrality_type": centrality_type,
            "centrality_score": centrality
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate centrality: {str(e)}")


@router.get("/communities")
async def detect_communities(
    algorithm: str = Query(default="louvain", regex="^(louvain|leiden|label_propagation)$"),
    min_community_size: int = Query(default=3, ge=2),
    kg_service: KnowledgeGraphService = Depends(get_kg_service)
):
    """Detect communities in the knowledge graph."""
    try:
        communities = await kg_service.detect_communities(
            algorithm=algorithm,
            min_community_size=min_community_size
        )

        return {
            "algorithm": algorithm,
            "communities": communities,
            "total_communities": len(communities)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect communities: {str(e)}")


@router.post("/query/cypher")
async def execute_cypher_query(
    query: str = Field(..., description="Cypher query to execute"),
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Query parameters"),
    kg_service: KnowledgeGraphService = Depends(get_kg_service)
):
    """Execute a custom Cypher query (admin only)."""
    try:
        # Add safety checks for read-only queries
        if not _is_safe_query(query):
            raise HTTPException(status_code=400, detail="Only read queries are allowed")

        results = await kg_service.execute_query(query, parameters or {})
        return {
            "query": query,
            "results": results,
            "result_count": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


def _is_safe_query(query: str) -> bool:
    """Check if a Cypher query is safe (read-only)."""
    dangerous_keywords = [
        'CREATE', 'DELETE', 'SET', 'REMOVE', 'MERGE',
        'DROP', 'ALTER', 'DETACH DELETE'
    ]

    query_upper = query.upper()
    return not any(keyword in query_upper for keyword in dangerous_keywords)