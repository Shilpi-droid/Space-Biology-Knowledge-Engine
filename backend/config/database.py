"""
Database connection management for the Space Biology Knowledge Engine.
"""

import logging
from typing import Optional
import asyncio
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

from .settings import get_settings

logger = logging.getLogger(__name__)

# Global driver instance
_driver: Optional[Driver] = None


def get_neo4j_driver() -> Optional[Driver]:
    """
    Get or create Neo4j driver instance.

    Returns:
        Neo4j driver instance or None if connection fails
    """
    global _driver

    if _driver is None:
        settings = get_settings()
        try:
            _driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                max_connection_lifetime=3600,  # 1 hour
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,
                resolver=None,
                connection_timeout=30
            )
            logger.info("Neo4j driver created successfully")
        except Exception as e:
            logger.error(f"Failed to create Neo4j driver: {e}")
            _driver = None

    return _driver


async def test_neo4j_connection() -> bool:
    """
    Test Neo4j database connectivity.

    Returns:
        True if connection is successful, False otherwise
    """
    driver = get_neo4j_driver()
    if not driver:
        logger.error("No Neo4j driver available")
        return False

    try:
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()["test"]
            if test_value == 1:
                logger.info("Neo4j connection test successful")
                return True
            else:
                logger.error("Neo4j connection test failed: unexpected result")
                return False

    except ServiceUnavailable as e:
        logger.error(f"Neo4j service unavailable: {e}")
        return False
    except AuthError as e:
        logger.error(f"Neo4j authentication failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Neo4j connection test failed: {e}")
        return False


async def create_indexes() -> bool:
    """
    Create database indexes for performance optimization.

    Returns:
        True if indexes were created successfully, False otherwise
    """
    driver = get_neo4j_driver()
    if not driver:
        logger.error("No Neo4j driver available for index creation")
        return False

    indexes = [
        # Study indexes
        "CREATE INDEX study_id IF NOT EXISTS FOR (s:Study) ON (s.id)",
        "CREATE INDEX study_title IF NOT EXISTS FOR (s:Study) ON (s.title)",
        "CREATE INDEX study_source IF NOT EXISTS FOR (s:Study) ON (s.source)",

        # Organism indexes
        "CREATE INDEX organism_name IF NOT EXISTS FOR (o:Organism) ON (o.name)",
        "CREATE INDEX organism_type IF NOT EXISTS FOR (o:Organism) ON (o.type)",

        # Publication indexes
        "CREATE INDEX publication_doi IF NOT EXISTS FOR (p:Publication) ON (p.doi)",
        "CREATE INDEX publication_pmid IF NOT EXISTS FOR (p:Publication) ON (p.pmid)",
        "CREATE INDEX publication_title IF NOT EXISTS FOR (p:Publication) ON (p.title)",

        # Factor indexes
        "CREATE INDEX factor_name IF NOT EXISTS FOR (f:Factor) ON (f.name)",
        "CREATE INDEX factor_type IF NOT EXISTS FOR (f:Factor) ON (f.type)",

        # Constraint for unique study IDs
        "CREATE CONSTRAINT study_id_unique IF NOT EXISTS FOR (s:Study) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT publication_doi_unique IF NOT EXISTS FOR (p:Publication) REQUIRE p.doi IS UNIQUE"
    ]

    try:
        with driver.session() as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                    logger.debug(f"Created index/constraint: {index_query}")
                except Exception as e:
                    logger.warning(f"Failed to create index/constraint: {index_query}, Error: {e}")

            # Wait for indexes to come online
            session.run("CALL db.awaitIndexes()")
            logger.info("All database indexes created and online")
            return True

    except Exception as e:
        logger.error(f"Failed to create database indexes: {e}")
        return False


async def migrate_schema() -> bool:
    """
    Handle database schema migrations.

    Returns:
        True if migration was successful, False otherwise
    """
    driver = get_neo4j_driver()
    if not driver:
        logger.error("No Neo4j driver available for schema migration")
        return False

    try:
        with driver.session() as session:
            # Check current schema version
            result = session.run(
                "MATCH (v:SchemaVersion) RETURN v.version as version ORDER BY v.created DESC LIMIT 1"
            )
            current_version = result.single()

            if not current_version:
                # First time setup
                session.run(
                    "CREATE (v:SchemaVersion {version: '1.0.0', created: datetime()})"
                )
                logger.info("Schema version initialized to 1.0.0")

            logger.info("Schema migration completed successfully")
            return True

    except Exception as e:
        logger.error(f"Schema migration failed: {e}")
        return False


async def close_connections():
    """Close all database connections."""
    global _driver

    if _driver:
        try:
            _driver.close()
            _driver = None
            logger.info("Neo4j driver closed successfully")
        except Exception as e:
            logger.error(f"Error closing Neo4j driver: {e}")


def execute_query(query: str, parameters: dict = None, database: str = None):
    """
    Execute a Cypher query with error handling.

    Args:
        query: Cypher query string
        parameters: Query parameters
        database: Target database name (optional)

    Returns:
        Query results as list of dictionaries
    """
    driver = get_neo4j_driver()
    if not driver:
        raise RuntimeError("No Neo4j driver available")

    try:
        with driver.session(database=database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        logger.error(f"Query: {query}")
        logger.error(f"Parameters: {parameters}")
        raise


def execute_write_query(query: str, parameters: dict = None, database: str = None):
    """
    Execute a write query in a transaction.

    Args:
        query: Cypher query string
        parameters: Query parameters
        database: Target database name (optional)

    Returns:
        Query results as list of dictionaries
    """
    driver = get_neo4j_driver()
    if not driver:
        raise RuntimeError("No Neo4j driver available")

    try:
        with driver.session(database=database) as session:
            with session.begin_transaction() as tx:
                result = tx.run(query, parameters or {})
                records = [record.data() for record in result]
                tx.commit()
                return records

    except Exception as e:
        logger.error(f"Write query execution failed: {e}")
        logger.error(f"Query: {query}")
        logger.error(f"Parameters: {parameters}")
        raise


def batch_execute(queries_and_params: list, database: str = None, batch_size: int = 1000):
    """
    Execute multiple queries in batches for better performance.

    Args:
        queries_and_params: List of (query, parameters) tuples
        database: Target database name (optional)
        batch_size: Number of queries per batch

    Returns:
        Number of successfully executed queries
    """
    driver = get_neo4j_driver()
    if not driver:
        raise RuntimeError("No Neo4j driver available")

    successful_count = 0
    total_queries = len(queries_and_params)

    try:
        with driver.session(database=database) as session:
            for i in range(0, total_queries, batch_size):
                batch = queries_and_params[i:i + batch_size]

                with session.begin_transaction() as tx:
                    for query, params in batch:
                        try:
                            tx.run(query, params or {})
                            successful_count += 1
                        except Exception as e:
                            logger.warning(f"Query failed in batch: {e}")
                            logger.warning(f"Query: {query}")

                    tx.commit()

                logger.info(f"Processed batch {i//batch_size + 1}/{(total_queries-1)//batch_size + 1}")

        logger.info(f"Batch execution completed: {successful_count}/{total_queries} queries successful")
        return successful_count

    except Exception as e:
        logger.error(f"Batch execution failed: {e}")
        raise