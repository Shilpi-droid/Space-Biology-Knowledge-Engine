"""
Core MVP Data Ingestion Service

This service orchestrates the complete data pipeline:
1. NASA ADS API data collection
2. Hybrid storage (SQLite + Neo4j + FAISS)
3. Partitioning strategy for scaling
4. Deduplication and data quality

Designed for guaranteed working demo with clear scaling path.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from pathlib import Path

from .data_sources.nasa_ads import collect_space_biology_dataset, Paper
from .storage.hybrid_storage import HybridStorageManager
from .graphrag.query_engine import GraphRAGQueryEngine, GraphRAGResult
from ..services.llm_service import LLMService
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class SpaceBiologyMVPService:
    """
    Core MVP service for Space Biology Knowledge Engine.

    Provides guaranteed working demo functionality:
    - NASA ADS data collection (5 years of papers)
    - Hybrid storage architecture
    - GraphRAG query processing
    - Automatic partitioning and deduplication
    """

    def __init__(self):
        self.settings = get_settings()
        self.storage_manager = None
        self.query_engine = None
        self.llm_service = None

        # Initialize data directories
        self._ensure_data_directories()

        # System status
        self.is_initialized = False
        self.dataset_loaded = False
        self.last_ingestion = None

    def _ensure_data_directories(self):
        """Ensure required data directories exist."""
        directories = [
            "data",
            "data/sqlite",
            "data/neo4j",
            "data/faiss",
            "data/cache",
            "data/logs"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize the MVP service.

        Returns:
            Initialization status and configuration
        """
        try:
            logger.info("Initializing Space Biology MVP Service...")

            # Initialize LLM service
            self.llm_service = LLMService()

            # Initialize hybrid storage
            self.storage_manager = HybridStorageManager(
                sqlite_path="data/sqlite/space_biology.db",
                neo4j_uri=self.settings.neo4j_uri,
                neo4j_user=self.settings.neo4j_user,
                neo4j_password=self.settings.neo4j_password,
                faiss_index_path="data/faiss/space_biology_index"
            )

            await self.storage_manager.initialize()

            # Initialize GraphRAG query engine
            self.query_engine = GraphRAGQueryEngine(
                storage_manager=self.storage_manager,
                llm_service=self.llm_service
            )

            self.is_initialized = True

            logger.info("MVP Service initialized successfully")

            return {
                "status": "initialized",
                "timestamp": datetime.utcnow().isoformat(),
                "components": {
                    "storage_manager": "ready",
                    "query_engine": "ready",
                    "llm_service": "ready"
                },
                "dataset_loaded": self.dataset_loaded
            }

        except Exception as e:
            logger.error(f"Failed to initialize MVP service: {str(e)}")
            raise

    async def ingest_five_year_dataset(self, max_papers: int = 1000,
                                     force_refresh: bool = False) -> Dict[str, Any]:
        """
        Ingest 5-year dataset from NASA ADS API.

        Args:
            max_papers: Maximum number of papers to collect
            force_refresh: Force fresh data collection even if data exists

        Returns:
            Ingestion results and statistics
        """
        if not self.is_initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")

        try:
            logger.info(f"Starting dataset ingestion (max_papers: {max_papers})")

            # Check if we need to collect new data
            if self.dataset_loaded and not force_refresh:
                logger.info("Dataset already loaded. Use force_refresh=True to reload.")
                return {"status": "already_loaded", "message": "Dataset already loaded"}

            # Validate NASA ADS API token
            ads_token = os.getenv("NASA_ADS_TOKEN")
            if not ads_token:
                raise ValueError("NASA_ADS_TOKEN environment variable not set")

            # Step 1: Collect data from NASA ADS
            logger.info("Collecting data from NASA ADS API...")
            dataset = await collect_space_biology_dataset(
                ads_token=ads_token,
                max_papers=max_papers
            )

            papers = dataset["papers"]
            logger.info(f"Collected {len(papers)} papers")

            # Step 2: Store in hybrid storage with partitioning
            logger.info("Storing data in hybrid storage...")
            partition_id = await self.storage_manager.store_dataset(
                papers=papers,
                partition_strategy="year"
            )

            # Step 3: Update service status
            self.dataset_loaded = True
            self.last_ingestion = datetime.utcnow()

            # Step 4: Generate ingestion report
            report = {
                "status": "completed",
                "timestamp": self.last_ingestion.isoformat(),
                "partition_id": partition_id,
                "statistics": {
                    "total_papers": len(papers),
                    "total_authors": len(dataset["unique_authors"]),
                    "date_range": f"{dataset['date_range']}",
                    "metrics": dataset["metrics"]
                },
                "storage_distribution": {
                    "sqlite_records": len(papers),
                    "neo4j_nodes": len(papers) + len(dataset["unique_authors"]),
                    "faiss_vectors": len([p for p in papers if p.abstract])
                }
            }

            logger.info(f"Dataset ingestion completed successfully")
            return report

        except Exception as e:
            logger.error(f"Dataset ingestion failed: {str(e)}")
            raise

    async def query(self, user_query: str,
                   strategy: Optional[str] = None,
                   max_results: int = 10) -> Dict[str, Any]:
        """
        Process a user query using GraphRAG.

        Args:
            user_query: Natural language query
            strategy: Processing strategy (auto-detected if None)
            max_results: Maximum number of results

        Returns:
            GraphRAG result with answer and sources
        """
        if not self.is_initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")

        if not self.dataset_loaded:
            raise RuntimeError("No dataset loaded. Call ingest_five_year_dataset() first.")

        try:
            logger.info(f"Processing query: {user_query[:100]}...")

            # Process query using GraphRAG
            result = await self.query_engine.query(
                user_query=user_query,
                strategy=strategy,
                max_papers=max_results
            )

            # Format response for API
            response = {
                "query": result.query,
                "answer": result.answer,
                "confidence": result.confidence,
                "sources": result.sources,
                "metadata": {
                    "processing_time": result.processing_metadata["processing_time_seconds"],
                    "strategy_used": result.processing_metadata["strategy_used"],
                    "papers_analyzed": result.processing_metadata["papers_retrieved"],
                    "graph_entities_explored": result.processing_metadata["graph_entities_explored"],
                    "timestamp": result.processing_metadata["timestamp"]
                }
            }

            logger.info(f"Query processed successfully in {response['metadata']['processing_time']:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Basic status
            status = {
                "service_status": "ready" if self.is_initialized else "not_initialized",
                "dataset_loaded": self.dataset_loaded,
                "last_ingestion": self.last_ingestion.isoformat() if self.last_ingestion else None,
                "timestamp": datetime.utcnow().isoformat()
            }

            if not self.is_initialized:
                return status

            # Storage status
            storage_status = {}

            # SQLite status
            if self.storage_manager.sqlite.connection:
                cursor = self.storage_manager.sqlite.connection.cursor()
                cursor.execute("SELECT COUNT(*) FROM papers")
                paper_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM authors")
                author_count = cursor.fetchone()[0]

                storage_status["sqlite"] = {
                    "status": "connected",
                    "papers": paper_count,
                    "authors": author_count
                }

            # Neo4j status
            try:
                async with self.storage_manager.neo4j.driver.session() as session:
                    result = await session.run("MATCH (n) RETURN COUNT(n) as node_count")
                    record = await result.single()
                    node_count = record["node_count"] if record else 0

                storage_status["neo4j"] = {
                    "status": "connected",
                    "total_nodes": node_count
                }
            except Exception as e:
                storage_status["neo4j"] = {"status": "error", "error": str(e)}

            # FAISS status
            faiss_status = {
                "status": "ready" if self.storage_manager.faiss.index else "not_loaded",
                "vectors": len(self.storage_manager.faiss.id_to_bibcode) if self.storage_manager.faiss.index else 0
            }
            storage_status["faiss"] = faiss_status

            status["storage"] = storage_status

            # Query engine status
            status["query_engine"] = {
                "status": "ready" if self.query_engine else "not_initialized",
                "available_strategies": list(self.query_engine.strategies.keys()) if self.query_engine else []
            }

            return status

        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                "service_status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def search_papers(self, query: str, limit: int = 20) -> Dict[str, Any]:
        """
        Simple semantic search for papers (lightweight alternative to full GraphRAG).

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            Search results with metadata
        """
        if not self.is_initialized or not self.dataset_loaded:
            raise RuntimeError("Service not ready")

        try:
            # Semantic search using FAISS
            similar_papers = await self.storage_manager.faiss.search_similar(query, k=limit)

            # Enrich with metadata
            results = []
            for bibcode, similarity in similar_papers:
                paper_metadata = await self.storage_manager.sqlite.connection.execute(
                    "SELECT * FROM papers WHERE bibcode = ?", (bibcode,)
                ).fetchone()

                if paper_metadata:
                    paper_dict = dict(paper_metadata)
                    paper_dict["similarity_score"] = similarity
                    results.append(paper_dict)

            return {
                "query": query,
                "results": results,
                "total_found": len(results),
                "search_type": "semantic",
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Paper search failed: {str(e)}")
            raise

    async def get_author_insights(self, author_name: str) -> Dict[str, Any]:
        """Get comprehensive insights about an author."""
        if not self.is_initialized or not self.dataset_loaded:
            raise RuntimeError("Service not ready")

        try:
            insights = await self.storage_manager.get_author_insights(author_name)
            return {
                "author": author_name,
                "insights": insights,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Author insights failed: {str(e)}")
            raise

    async def get_topic_analysis(self, topic: str) -> Dict[str, Any]:
        """Get analysis for a specific research topic."""
        if not self.is_initialized or not self.dataset_loaded:
            raise RuntimeError("Service not ready")

        try:
            # Get papers related to the topic
            cursor = self.storage_manager.sqlite.connection.cursor()
            cursor.execute("""
                SELECT p.*, t.topic_name FROM papers p
                JOIN paper_topics pt ON p.bibcode = pt.paper_bibcode
                JOIN topics t ON pt.topic_id = t.topic_id
                WHERE t.topic_name LIKE ?
                ORDER BY p.citation_count DESC
                LIMIT 50
            """, (f"%{topic}%",))

            papers = [dict(row) for row in cursor.fetchall()]

            # Calculate topic statistics
            total_papers = len(papers)
            total_citations = sum(p.get("citation_count", 0) for p in papers)

            # Get temporal distribution
            year_counts = {}
            for paper in papers:
                if paper.get("pub_date"):
                    year = paper["pub_date"][:4]
                    year_counts[year] = year_counts.get(year, 0) + 1

            return {
                "topic": topic,
                "analysis": {
                    "total_papers": total_papers,
                    "total_citations": total_citations,
                    "temporal_distribution": year_counts,
                    "top_papers": papers[:10]
                },
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Topic analysis failed: {str(e)}")
            raise

    async def cleanup(self):
        """Clean up resources."""
        if self.storage_manager:
            await self.storage_manager.close()


# Global service instance
_mvp_service: Optional[SpaceBiologyMVPService] = None


async def get_mvp_service() -> SpaceBiologyMVPService:
    """Get or create the global MVP service instance."""
    global _mvp_service

    if _mvp_service is None:
        _mvp_service = SpaceBiologyMVPService()
        await _mvp_service.initialize()

    return _mvp_service


async def initialize_mvp_system() -> Dict[str, Any]:
    """Initialize the complete MVP system."""
    service = await get_mvp_service()
    return await service.get_system_status()


async def demo_query_examples() -> List[Dict[str, Any]]:
    """Generate demo query examples for testing."""
    examples = [
        {
            "query": "What are the effects of microgravity on bone density?",
            "description": "Literature search about bone physiology in space",
            "expected_strategy": "literature_search"
        },
        {
            "query": "Who are the leading researchers in space radiation biology?",
            "description": "Author analysis for radiation research experts",
            "expected_strategy": "author_analysis"
        },
        {
            "query": "Which papers have the highest impact in space medicine?",
            "description": "Citation analysis for influential research",
            "expected_strategy": "citation_analysis"
        },
        {
            "query": "How has research on muscle atrophy in space evolved over time?",
            "description": "Temporal analysis of muscle physiology research",
            "expected_strategy": "temporal_analysis"
        },
        {
            "query": "Compare countermeasures for bone loss versus muscle atrophy",
            "description": "Comparative analysis of different interventions",
            "expected_strategy": "comparative_analysis"
        }
    ]

    return examples