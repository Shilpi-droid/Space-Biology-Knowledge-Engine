"""
Data Management API endpoints for Space Biology Knowledge Engine.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field
from datetime import datetime

from ..services.data_ingestion import DataIngestionService
from ..services.knowledge_graph import KnowledgeGraphService
from ..services.semantic_search import SemanticSearchService
from ..core.dependencies import get_data_ingestion_service, get_kg_service, get_search_service

router = APIRouter(prefix="/data", tags=["data_management"])


class IngestionRequest(BaseModel):
    """Request model for data ingestion."""
    sources: List[str] = Field(..., description="Data sources to ingest from")
    priority: str = Field(default="normal", regex="^(low|normal|high)$")
    incremental: bool = Field(default=True, description="Perform incremental update")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Source-specific filters")


class IngestionStatus(BaseModel):
    """Status model for data ingestion."""
    task_id: str
    status: str
    progress: float
    current_source: Optional[str]
    items_processed: int
    items_total: Optional[int]
    started_at: datetime
    estimated_completion: Optional[datetime]
    errors: List[str]


class DataSourceInfo(BaseModel):
    """Information about a data source."""
    name: str
    description: str
    last_updated: Optional[datetime]
    total_records: int
    status: str
    supported_filters: List[str]


class SystemStats(BaseModel):
    """System-wide statistics."""
    total_studies: int
    total_publications: int
    total_organisms: int
    total_factors: int
    last_ingestion: Optional[datetime]
    database_size: Dict[str, Any]
    index_sizes: Dict[str, Any]


@router.get("/stats", response_model=SystemStats)
async def get_system_statistics(
    kg_service: KnowledgeGraphService = Depends(get_kg_service),
    search_service: SemanticSearchService = Depends(get_search_service)
):
    """Get comprehensive system statistics."""
    try:
        # Get graph statistics
        graph_stats = await kg_service.get_graph_statistics()

        # Get search index statistics
        search_stats = await search_service.get_index_statistics()

        return SystemStats(
            total_studies=graph_stats.get("study_count", 0),
            total_publications=graph_stats.get("publication_count", 0),
            total_organisms=graph_stats.get("organism_count", 0),
            total_factors=graph_stats.get("factor_count", 0),
            last_ingestion=graph_stats.get("last_updated"),
            database_size={
                "nodes": graph_stats.get("total_nodes", 0),
                "relationships": graph_stats.get("total_relationships", 0),
                "size_mb": graph_stats.get("database_size_mb", 0)
            },
            index_sizes={
                "embedding_vectors": search_stats.get("vector_count", 0),
                "index_size_mb": search_stats.get("index_size_mb", 0)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.get("/sources", response_model=List[DataSourceInfo])
async def get_data_sources(
    data_service: DataIngestionService = Depends(get_data_ingestion_service)
):
    """Get information about available data sources."""
    try:
        sources = await data_service.get_available_sources()

        return [
            DataSourceInfo(
                name=source["name"],
                description=source["description"],
                last_updated=source.get("last_updated"),
                total_records=source.get("total_records", 0),
                status=source.get("status", "unknown"),
                supported_filters=source.get("supported_filters", [])
            )
            for source in sources
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get data sources: {str(e)}")


@router.post("/ingest")
async def trigger_data_ingestion(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    data_service: DataIngestionService = Depends(get_data_ingestion_service)
):
    """Trigger data ingestion from specified sources."""
    try:
        # Validate sources
        available_sources = await data_service.get_available_sources()
        available_source_names = [s["name"] for s in available_sources]

        invalid_sources = [s for s in request.sources if s not in available_source_names]
        if invalid_sources:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sources: {invalid_sources}. Available: {available_source_names}"
            )

        # Start ingestion as background task
        task_id = await data_service.start_ingestion(
            sources=request.sources,
            priority=request.priority,
            incremental=request.incremental,
            filters=request.filters
        )

        return {
            "task_id": task_id,
            "status": "started",
            "message": f"Data ingestion started for sources: {', '.join(request.sources)}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger ingestion: {str(e)}")


@router.get("/ingest/status/{task_id}", response_model=IngestionStatus)
async def get_ingestion_status(
    task_id: str,
    data_service: DataIngestionService = Depends(get_data_ingestion_service)
):
    """Get status of a specific ingestion task."""
    try:
        status = await data_service.get_ingestion_status(task_id)

        if not status:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        return IngestionStatus(
            task_id=task_id,
            status=status["status"],
            progress=status.get("progress", 0.0),
            current_source=status.get("current_source"),
            items_processed=status.get("items_processed", 0),
            items_total=status.get("items_total"),
            started_at=status["started_at"],
            estimated_completion=status.get("estimated_completion"),
            errors=status.get("errors", [])
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get ingestion status: {str(e)}")


@router.get("/ingest/active")
async def get_active_ingestions(
    data_service: DataIngestionService = Depends(get_data_ingestion_service)
):
    """Get all currently active ingestion tasks."""
    try:
        active_tasks = await data_service.get_active_ingestions()
        return {"active_tasks": active_tasks}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active ingestions: {str(e)}")


@router.delete("/ingest/{task_id}")
async def cancel_ingestion(
    task_id: str,
    data_service: DataIngestionService = Depends(get_data_ingestion_service)
):
    """Cancel a running ingestion task."""
    try:
        success = await data_service.cancel_ingestion(task_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found or cannot be cancelled")

        return {"message": f"Ingestion task {task_id} cancelled successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel ingestion: {str(e)}")


@router.get("/health")
async def check_data_health(
    kg_service: KnowledgeGraphService = Depends(get_kg_service),
    search_service: SemanticSearchService = Depends(get_search_service)
):
    """Check the health of data systems."""
    try:
        health_status = {
            "knowledge_graph": await kg_service.health_check(),
            "search_index": await search_service.health_check(),
            "overall_status": "healthy"
        }

        # Determine overall status
        if not all(health_status[key].get("healthy", False)
                  for key in ["knowledge_graph", "search_index"]):
            health_status["overall_status"] = "unhealthy"

        return health_status

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.post("/rebuild-index")
async def rebuild_search_index(
    background_tasks: BackgroundTasks,
    search_service: SemanticSearchService = Depends(get_search_service)
):
    """Rebuild the semantic search index from knowledge graph."""
    try:
        task_id = await search_service.rebuild_index_async()

        return {
            "task_id": task_id,
            "message": "Search index rebuild started",
            "status": "in_progress"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {str(e)}")


@router.get("/export")
async def export_data(
    format: str = Query(default="json", regex="^(json|csv|rdf|cypher)$"),
    entity_types: Optional[List[str]] = Query(default=None),
    filters: Optional[Dict[str, Any]] = Query(default=None),
    kg_service: KnowledgeGraphService = Depends(get_kg_service)
):
    """Export data from the knowledge graph."""
    try:
        export_data = await kg_service.export_data(
            format=format,
            entity_types=entity_types,
            filters=filters
        )

        return {
            "format": format,
            "export_size": len(export_data),
            "data": export_data,
            "exported_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data export failed: {str(e)}")


@router.post("/import")
async def import_data(
    data: Dict[str, Any],
    format: str = Field(default="json", regex="^(json|rdf|cypher)$"),
    validate: bool = Field(default=True, description="Validate data before import"),
    kg_service: KnowledgeGraphService = Depends(get_kg_service)
):
    """Import data into the knowledge graph."""
    try:
        result = await kg_service.import_data(
            data=data,
            format=format,
            validate=validate
        )

        return {
            "imported_nodes": result.get("nodes_created", 0),
            "imported_relationships": result.get("relationships_created", 0),
            "import_time": result.get("processing_time_ms", 0),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data import failed: {str(e)}")


@router.get("/validation/integrity")
async def check_data_integrity(
    kg_service: KnowledgeGraphService = Depends(get_kg_service)
):
    """Check data integrity and consistency."""
    try:
        integrity_report = await kg_service.check_data_integrity()

        return {
            "integrity_status": integrity_report.get("status", "unknown"),
            "issues_found": integrity_report.get("issues", []),
            "statistics": integrity_report.get("statistics", {}),
            "recommendations": integrity_report.get("recommendations", [])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Integrity check failed: {str(e)}")


@router.post("/maintenance/cleanup")
async def cleanup_data(
    remove_orphans: bool = Field(default=True, description="Remove orphaned nodes"),
    remove_duplicates: bool = Field(default=True, description="Remove duplicate entities"),
    dry_run: bool = Field(default=True, description="Perform dry run without changes"),
    kg_service: KnowledgeGraphService = Depends(get_kg_service)
):
    """Perform data cleanup operations."""
    try:
        cleanup_result = await kg_service.cleanup_data(
            remove_orphans=remove_orphans,
            remove_duplicates=remove_duplicates,
            dry_run=dry_run
        )

        return {
            "dry_run": dry_run,
            "orphans_found": cleanup_result.get("orphans_found", 0),
            "duplicates_found": cleanup_result.get("duplicates_found", 0),
            "cleanup_actions": cleanup_result.get("actions", []),
            "estimated_space_saved": cleanup_result.get("space_saved_mb", 0)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data cleanup failed: {str(e)}")


@router.get("/backup")
async def create_backup(
    include_indexes: bool = Query(default=True, description="Include search indexes"),
    kg_service: KnowledgeGraphService = Depends(get_kg_service),
    search_service: SemanticSearchService = Depends(get_search_service)
):
    """Create a backup of the knowledge base."""
    try:
        backup_info = await kg_service.create_backup(include_indexes=include_indexes)

        if include_indexes:
            index_backup = await search_service.backup_indexes()
            backup_info["index_backup"] = index_backup

        return {
            "backup_id": backup_info.get("backup_id"),
            "backup_size_mb": backup_info.get("size_mb", 0),
            "backup_path": backup_info.get("backup_path"),
            "created_at": backup_info.get("created_at"),
            "includes_indexes": include_indexes
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup creation failed: {str(e)}")


@router.post("/restore/{backup_id}")
async def restore_backup(
    backup_id: str,
    confirm: bool = Field(default=False, description="Confirm destructive operation"),
    kg_service: KnowledgeGraphService = Depends(get_kg_service)
):
    """Restore from a backup (destructive operation)."""
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="This is a destructive operation. Set confirm=true to proceed."
            )

        restore_result = await kg_service.restore_backup(backup_id)

        return {
            "backup_id": backup_id,
            "restore_status": restore_result.get("status"),
            "nodes_restored": restore_result.get("nodes_restored", 0),
            "relationships_restored": restore_result.get("relationships_restored", 0),
            "restore_time_ms": restore_result.get("processing_time_ms", 0)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup restore failed: {str(e)}")


@router.get("/metrics")
async def get_data_metrics(
    time_range: str = Query(default="24h", regex="^(1h|24h|7d|30d)$"),
    kg_service: KnowledgeGraphService = Depends(get_kg_service)
):
    """Get data ingestion and usage metrics."""
    try:
        metrics = await kg_service.get_metrics(time_range=time_range)

        return {
            "time_range": time_range,
            "ingestion_metrics": metrics.get("ingestion", {}),
            "query_metrics": metrics.get("queries", {}),
            "performance_metrics": metrics.get("performance", {}),
            "growth_metrics": metrics.get("growth", {})
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")