"""
Central API routing and request handling for the Space Biology Knowledge Engine.
"""

import time
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pathlib import Path

from config.database import get_neo4j_driver
from config.settings import get_settings

router = APIRouter()

# Store startup time for uptime calculation
startup_time = time.time()


@router.get("/health")
async def get_health() -> Dict[str, Any]:
    """
    System health check endpoint.

    Returns:
        Dict containing system health status, uptime, and component status
    """
    health_status = {
        "status": "healthy",
        "uptime": time.time() - startup_time,
        "components": {}
    }

    try:
        # Check Neo4j connectivity
        driver = get_neo4j_driver()
        if driver:
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                if result.single():
                    health_status["components"]["neo4j"] = "healthy"
                else:
                    health_status["components"]["neo4j"] = "unhealthy"
        else:
            health_status["components"]["neo4j"] = "unavailable"
    except Exception as e:
        health_status["components"]["neo4j"] = f"error: {str(e)}"

    # Check data directories
    data_dirs = ["data/raw", "data/processed", "data/embeddings"]
    dirs_healthy = all(Path(d).exists() and Path(d).is_dir() for d in data_dirs)
    health_status["components"]["data_directories"] = "healthy" if dirs_healthy else "unhealthy"

    # Check embedding model availability
    try:
        from utils.embeddings import get_embedding_model
        model = get_embedding_model()
        health_status["components"]["embedding_model"] = "healthy" if model else "unavailable"
    except Exception:
        health_status["components"]["embedding_model"] = "unavailable"

    # Overall status
    unhealthy_components = [k for k, v in health_status["components"].items()
                          if v not in ["healthy", "unavailable"]]
    if unhealthy_components:
        health_status["status"] = "degraded"

    return health_status


@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """
    Get overall system statistics including study counts, organisms, etc.

    Returns:
        Dict containing system-wide statistics
    """
    stats = {
        "studies": {"total": 0, "by_source": {}},
        "organisms": {"total": 0, "list": []},
        "publications": {"total": 0},
        "relationships": {"total": 0},
        "data_files": {"processed": 0, "raw": 0},
        "embeddings": {"total": 0, "index_size": 0}
    }

    try:
        driver = get_neo4j_driver()
        if driver:
            with driver.session() as session:
                # Count studies
                result = session.run("MATCH (s:Study) RETURN count(s) as total")
                stats["studies"]["total"] = result.single()["total"]

                # Count studies by source
                result = session.run(
                    "MATCH (s:Study) WHERE s.source IS NOT NULL "
                    "RETURN s.source as source, count(s) as count"
                )
                for record in result:
                    stats["studies"]["by_source"][record["source"]] = record["count"]

                # Count organisms
                result = session.run("MATCH (o:Organism) RETURN count(o) as total")
                stats["organisms"]["total"] = result.single()["total"]

                # Get organism list (limited to 20)
                result = session.run(
                    "MATCH (o:Organism) RETURN DISTINCT o.name as name LIMIT 20"
                )
                stats["organisms"]["list"] = [record["name"] for record in result]

                # Count publications
                result = session.run("MATCH (p:Publication) RETURN count(p) as total")
                stats["publications"]["total"] = result.single()["total"]

                # Count relationships
                result = session.run("MATCH ()-[r]->() RETURN count(r) as total")
                stats["relationships"]["total"] = result.single()["total"]

    except Exception as e:
        stats["error"] = f"Database query failed: {str(e)}"

    # Count data files
    try:
        processed_dir = Path("data/processed")
        if processed_dir.exists():
            stats["data_files"]["processed"] = len(list(processed_dir.rglob("*.json")))

        raw_dir = Path("data/raw")
        if raw_dir.exists():
            stats["data_files"]["raw"] = len(list(raw_dir.rglob("*")))

        # Check embeddings
        embeddings_dir = Path("data/embeddings")
        if embeddings_dir.exists():
            embedding_files = list(embeddings_dir.glob("*.npy"))
            stats["embeddings"]["total"] = len(embedding_files)
            if embedding_files:
                import os
                total_size = sum(os.path.getsize(f) for f in embedding_files)
                stats["embeddings"]["index_size"] = total_size

    except Exception as e:
        stats["file_error"] = f"File system error: {str(e)}"

    return stats


@router.get("/")
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "Space Biology Knowledge Engine API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "stats": "/stats"
    }