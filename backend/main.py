"""
Space Biology Knowledge Engine - Main Application Entry Point

This module serves as the FastAPI application bootstrap and configuration entry point.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import get_settings
from config.database import test_neo4j_connection, create_indexes
from api.search import router as search_router
from api.graph import router as graph_router
from api.agents import router as agents_router
from api.data import router as data_router
from api.intelligent_query import router as intelligent_router
from app.main import router as app_router


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    # Startup
    logger.info("Starting Space Biology Knowledge Engine...")

    try:
        # Test database connectivity
        settings = get_settings()
        if await test_neo4j_connection():
            logger.info("Neo4j connection successful")
            await create_indexes()
        else:
            logger.error("Failed to connect to Neo4j database")

        # Create data directories
        data_dirs = ["data/raw", "data/processed", "data/embeddings", "data/cache", "data/logs"]
        for dir_path in data_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Load embedding model
        try:
            from utils.embeddings import load_embedding_model
            await load_embedding_model()
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")

        logger.info("Application startup completed successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Space Biology Knowledge Engine...")
    try:
        # Close database connections
        from config.database import close_connections
        await close_connections()

        # Save any in-memory data
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application instance.

    Returns:
        FastAPI: Configured application instance
    """

    app = FastAPI(
        title="Space Biology Knowledge Engine",
        description="AI-powered knowledge engine for space biology research",
        version="1.0.0",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(app_router, prefix="", tags=["main"])
    app.include_router(intelligent_router, tags=["intelligent_query"])  # Primary AI endpoint
    app.include_router(search_router, tags=["search"])
    app.include_router(graph_router, tags=["graph"])
    app.include_router(agents_router, tags=["agents"])
    app.include_router(data_router, tags=["data"])

    # Serve static files for documentation
    static_path = Path("static")
    if static_path.exists():
        app.mount("/static", StaticFiles(directory="static"), name="static")

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )