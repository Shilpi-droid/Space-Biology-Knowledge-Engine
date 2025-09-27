"""
Hybrid Storage Architecture for Space Biology Knowledge Engine

This module implements a three-tier storage system:
1. SQLite - Tabular data (papers, authors, citations)
2. Neo4j - Relationships and graph traversal
3. FAISS - Vector embeddings for semantic search

Designed for efficient querying and easy scaling through partitioning.
"""

import sqlite3
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import numpy as np
import hashlib

# Neo4j
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError

# Vector storage
import faiss
from sentence_transformers import SentenceTransformer

from ..data_sources.nasa_ads import Paper

logger = logging.getLogger(__name__)


@dataclass
class StoragePartition:
    """Represents a data partition for scaling."""
    partition_id: str
    year_start: int
    year_end: int
    topic_cluster: Optional[str]
    paper_count: int
    created_at: datetime


class SQLiteManager:
    """Manages SQLite database for tabular data storage."""

    def __init__(self, db_path: str = "data/space_biology.db"):
        self.db_path = db_path
        self.connection = None

    async def initialize(self):
        """Initialize SQLite database with required tables."""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row  # Enable dict-like access

        # Create tables
        await self._create_tables()
        logger.info(f"SQLite database initialized at {self.db_path}")

    async def _create_tables(self):
        """Create database tables."""
        cursor = self.connection.cursor()

        # Papers table - core metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                bibcode TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                journal TEXT,
                pub_date TEXT,
                citation_count INTEGER DEFAULT 0,
                doi TEXT,
                arxiv_id TEXT,
                partition_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX(partition_id),
                INDEX(pub_date),
                INDEX(citation_count)
            )
        """)

        # Authors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS authors (
                author_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                paper_count INTEGER DEFAULT 0,
                total_citations INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX(name),
                INDEX(paper_count)
            )
        """)

        # Paper-Author relationships
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_authors (
                paper_bibcode TEXT,
                author_id INTEGER,
                author_position INTEGER,
                PRIMARY KEY (paper_bibcode, author_id),
                FOREIGN KEY (paper_bibcode) REFERENCES papers(bibcode),
                FOREIGN KEY (author_id) REFERENCES authors(author_id),
                INDEX(paper_bibcode),
                INDEX(author_id)
            )
        """)

        # Citations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS citations (
                citing_paper TEXT,
                cited_paper TEXT,
                PRIMARY KEY (citing_paper, cited_paper),
                FOREIGN KEY (citing_paper) REFERENCES papers(bibcode),
                FOREIGN KEY (cited_paper) REFERENCES papers(bibcode),
                INDEX(citing_paper),
                INDEX(cited_paper)
            )
        """)

        # Topics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_name TEXT UNIQUE NOT NULL,
                paper_count INTEGER DEFAULT 0,
                INDEX(topic_name)
            )
        """)

        # Paper-Topic relationships
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_topics (
                paper_bibcode TEXT,
                topic_id INTEGER,
                PRIMARY KEY (paper_bibcode, topic_id),
                FOREIGN KEY (paper_bibcode) REFERENCES papers(bibcode),
                FOREIGN KEY (topic_id) REFERENCES topics(topic_id),
                INDEX(paper_bibcode),
                INDEX(topic_id)
            )
        """)

        # Partitions table for scaling
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS partitions (
                partition_id TEXT PRIMARY KEY,
                year_start INTEGER,
                year_end INTEGER,
                topic_cluster TEXT,
                paper_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX(year_start, year_end),
                INDEX(topic_cluster)
            )
        """)

        self.connection.commit()

    async def store_papers(self, papers: List[Paper], partition_id: str):
        """Store papers in SQLite with deduplication."""
        cursor = self.connection.cursor()

        for paper in papers:
            try:
                # Insert or update paper
                cursor.execute("""
                    INSERT OR REPLACE INTO papers
                    (bibcode, title, journal, pub_date, citation_count, doi, arxiv_id, partition_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    paper.bibcode, paper.title, paper.journal, paper.pub_date,
                    paper.citation_count, paper.doi, paper.arxiv_id, partition_id
                ))

                # Store authors
                for i, author_name in enumerate(paper.authors):
                    # Insert or ignore author
                    cursor.execute("""
                        INSERT OR IGNORE INTO authors (name, paper_count, total_citations)
                        VALUES (?, 0, 0)
                    """, (author_name,))

                    # Get author ID
                    cursor.execute("SELECT author_id FROM authors WHERE name = ?", (author_name,))
                    author_id = cursor.fetchone()[0]

                    # Link paper to author
                    cursor.execute("""
                        INSERT OR REPLACE INTO paper_authors
                        (paper_bibcode, author_id, author_position)
                        VALUES (?, ?, ?)
                    """, (paper.bibcode, author_id, i))

                # Store topics
                for topic_name in paper.topics:
                    # Insert or ignore topic
                    cursor.execute("""
                        INSERT OR IGNORE INTO topics (topic_name, paper_count)
                        VALUES (?, 0)
                    """, (topic_name,))

                    # Get topic ID
                    cursor.execute("SELECT topic_id FROM topics WHERE topic_name = ?", (topic_name,))
                    topic_result = cursor.fetchone()
                    if topic_result:
                        topic_id = topic_result[0]

                        # Link paper to topic
                        cursor.execute("""
                            INSERT OR REPLACE INTO paper_topics (paper_bibcode, topic_id)
                            VALUES (?, ?)
                        """, (paper.bibcode, topic_id))

                # Store citations
                for cited_bibcode in paper.citations:
                    cursor.execute("""
                        INSERT OR IGNORE INTO citations (citing_paper, cited_paper)
                        VALUES (?, ?)
                    """, (paper.bibcode, cited_bibcode))

            except Exception as e:
                logger.error(f"Error storing paper {paper.bibcode}: {str(e)}")

        self.connection.commit()

    async def get_papers_by_partition(self, partition_id: str) -> List[Dict[str, Any]]:
        """Get all papers in a partition."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM papers WHERE partition_id = ?", (partition_id,))
        return [dict(row) for row in cursor.fetchall()]

    async def get_author_papers(self, author_name: str) -> List[Dict[str, Any]]:
        """Get all papers by an author."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT p.* FROM papers p
            JOIN paper_authors pa ON p.bibcode = pa.paper_bibcode
            JOIN authors a ON pa.author_id = a.author_id
            WHERE a.name = ?
        """, (author_name,))
        return [dict(row) for row in cursor.fetchall()]

    async def get_paper_citations(self, bibcode: str) -> List[str]:
        """Get papers that cite the given paper."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT citing_paper FROM citations WHERE cited_paper = ?
        """, (bibcode,))
        return [row[0] for row in cursor.fetchall()]

    async def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()


class Neo4jManager:
    """Manages Neo4j graph database for relationships."""

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    async def initialize(self):
        """Initialize Neo4j connection and create constraints."""
        self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))

        # Test connection
        async with self.driver.session() as session:
            result = await session.run("RETURN 1")
            await result.consume()

        # Create constraints and indexes
        await self._create_constraints()
        logger.info("Neo4j database initialized")

    async def _create_constraints(self):
        """Create Neo4j constraints and indexes."""
        constraints = [
            "CREATE CONSTRAINT paper_bibcode IF NOT EXISTS FOR (p:Paper) REQUIRE p.bibcode IS UNIQUE",
            "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE INDEX paper_partition IF NOT EXISTS FOR (p:Paper) ON (p.partition_id)",
            "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)",
            "CREATE INDEX author_citations IF NOT EXISTS FOR (a:Author) ON (a.total_citations)"
        ]

        async with self.driver.session() as session:
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Neo4jError as e:
                    # Constraint might already exist
                    logger.debug(f"Constraint creation note: {str(e)}")

    async def store_papers_graph(self, papers: List[Paper], partition_id: str):
        """Store papers and relationships in Neo4j."""
        async with self.driver.session() as session:
            for paper in papers:
                await self._store_single_paper(session, paper, partition_id)

    async def _store_single_paper(self, session, paper: Paper, partition_id: str):
        """Store a single paper and its relationships."""
        try:
            # Extract year from pub_date
            year = None
            if paper.pub_date:
                try:
                    year = int(paper.pub_date[:4])
                except:
                    pass

            # Create or merge paper node
            await session.run("""
                MERGE (p:Paper {bibcode: $bibcode})
                SET p.title = $title,
                    p.journal = $journal,
                    p.year = $year,
                    p.citation_count = $citation_count,
                    p.partition_id = $partition_id
            """, {
                "bibcode": paper.bibcode,
                "title": paper.title,
                "journal": paper.journal,
                "year": year,
                "citation_count": paper.citation_count,
                "partition_id": partition_id
            })

            # Create author nodes and relationships
            for i, author_name in enumerate(paper.authors):
                await session.run("""
                    MERGE (a:Author {name: $author_name})
                    WITH a
                    MATCH (p:Paper {bibcode: $bibcode})
                    MERGE (a)-[r:AUTHORED]->(p)
                    SET r.position = $position
                """, {
                    "author_name": author_name,
                    "bibcode": paper.bibcode,
                    "position": i
                })

            # Create topic nodes and relationships
            for topic_name in paper.topics:
                await session.run("""
                    MERGE (t:Topic {name: $topic_name})
                    WITH t
                    MATCH (p:Paper {bibcode: $bibcode})
                    MERGE (p)-[:HAS_TOPIC]->(t)
                """, {
                    "topic_name": topic_name,
                    "bibcode": paper.bibcode
                })

            # Create citation relationships
            for cited_bibcode in paper.citations:
                await session.run("""
                    MATCH (citing:Paper {bibcode: $citing_bibcode})
                    MERGE (cited:Paper {bibcode: $cited_bibcode})
                    MERGE (citing)-[:CITES]->(cited)
                """, {
                    "citing_bibcode": paper.bibcode,
                    "cited_bibcode": cited_bibcode
                })

        except Exception as e:
            logger.error(f"Error storing paper {paper.bibcode} in Neo4j: {str(e)}")

    async def get_author_collaboration_network(self, author_name: str, depth: int = 2) -> Dict[str, Any]:
        """Get collaboration network for an author."""
        query = """
            MATCH (center:Author {name: $author_name})
            MATCH path = (center)-[:AUTHORED*1..%d]-(collaborator:Author)
            WHERE collaborator <> center
            RETURN collaborator.name as collaborator, length(path) as distance
            ORDER BY distance, collaborator.name
        """ % depth

        async with self.driver.session() as session:
            result = await session.run(query, {"author_name": author_name})
            collaborators = []
            async for record in result:
                collaborators.append({
                    "name": record["collaborator"],
                    "distance": record["distance"]
                })
            return {"center_author": author_name, "collaborators": collaborators}

    async def get_citation_network(self, bibcode: str, depth: int = 2) -> Dict[str, Any]:
        """Get citation network for a paper."""
        query = """
            MATCH (center:Paper {bibcode: $bibcode})
            OPTIONAL MATCH citing_path = (citing:Paper)-[:CITES*1..%d]->(center)
            OPTIONAL MATCH cited_path = (center)-[:CITES*1..%d]->(cited:Paper)
            RETURN
                collect(DISTINCT citing.bibcode) as citing_papers,
                collect(DISTINCT cited.bibcode) as cited_papers
        """ % (depth, depth)

        async with self.driver.session() as session:
            result = await session.run(query, {"bibcode": bibcode})
            record = await result.single()
            return {
                "center_paper": bibcode,
                "citing_papers": record["citing_papers"] if record else [],
                "cited_papers": record["cited_papers"] if record else []
            }

    async def get_papers_by_topic_and_partition(self, topic: str, partition_id: str) -> List[str]:
        """Get papers by topic within a partition."""
        query = """
            MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic {name: $topic})
            WHERE p.partition_id = $partition_id
            RETURN p.bibcode
        """

        async with self.driver.session() as session:
            result = await session.run(query, {"topic": topic, "partition_id": partition_id})
            return [record["p.bibcode"] async for record in result]

    async def close(self):
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()


class FAISSVectorStore:
    """FAISS-based vector storage for semantic search."""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", index_path: str = "data/faiss_index"):
        self.embedding_model_name = embedding_model_name
        self.index_path = index_path
        self.model = None
        self.index = None
        self.bibcode_to_id = {}
        self.id_to_bibcode = {}
        self.embedding_cache = {}

    async def initialize(self):
        """Initialize FAISS index and embedding model."""
        # Load sentence transformer model
        self.model = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity

        logger.info(f"FAISS vector store initialized with {self.embedding_model_name}")

    async def add_papers(self, papers: List[Paper], partition_id: str):
        """Add papers to vector index."""
        texts = []
        bibcodes = []

        for paper in papers:
            # Combine title and abstract for embedding
            text = f"{paper.title} {paper.abstract}".strip()
            if text:
                texts.append(text)
                bibcodes.append(paper.bibcode)

        if not texts:
            return

        # Generate embeddings
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

        # Add to index
        start_id = len(self.id_to_bibcode)
        for i, (embedding, bibcode) in enumerate(zip(embeddings, bibcodes)):
            vector_id = start_id + i
            self.bibcode_to_id[bibcode] = vector_id
            self.id_to_bibcode[vector_id] = bibcode

        # Add embeddings to FAISS index
        self.index.add(embeddings.astype(np.float32))

        logger.info(f"Added {len(texts)} papers to vector index")

    async def search_similar(self, query: str, k: int = 10, partition_filter: Optional[str] = None) -> List[Tuple[str, float]]:
        """Search for similar papers using semantic similarity."""
        if not self.model or not self.index:
            return []

        # Generate query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        # Search FAISS index
        similarities, indices = self.index.search(query_embedding.astype(np.float32), k * 2)  # Get more for filtering

        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx in self.id_to_bibcode:
                bibcode = self.id_to_bibcode[idx]

                # Apply partition filter if specified
                if partition_filter:
                    # This would require checking partition membership
                    # For now, we'll skip this filtering step
                    pass

                results.append((bibcode, float(similarity)))

        return results[:k]

    async def get_paper_embedding(self, bibcode: str) -> Optional[np.ndarray]:
        """Get embedding for a specific paper."""
        if bibcode in self.bibcode_to_id:
            vector_id = self.bibcode_to_id[bibcode]
            # Get embedding from index
            embedding = self.index.reconstruct(vector_id)
            return embedding
        return None

    async def save_index(self):
        """Save FAISS index to disk."""
        if self.index:
            faiss.write_index(self.index, f"{self.index_path}.idx")

            # Save mapping
            mapping = {
                "bibcode_to_id": self.bibcode_to_id,
                "id_to_bibcode": self.id_to_bibcode
            }
            with open(f"{self.index_path}_mapping.json", "w") as f:
                json.dump(mapping, f)

            logger.info(f"FAISS index saved to {self.index_path}")

    async def load_index(self):
        """Load FAISS index from disk."""
        try:
            self.index = faiss.read_index(f"{self.index_path}.idx")

            # Load mapping
            with open(f"{self.index_path}_mapping.json", "r") as f:
                mapping = json.load(f)
                self.bibcode_to_id = mapping["bibcode_to_id"]
                self.id_to_bibcode = {int(k): v for k, v in mapping["id_to_bibcode"].items()}

            logger.info(f"FAISS index loaded from {self.index_path}")
            return True
        except Exception as e:
            logger.warning(f"Could not load FAISS index: {str(e)}")
            return False


class HybridStorageManager:
    """Unified manager for hybrid storage architecture."""

    def __init__(self, sqlite_path: str = "data/space_biology.db",
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 faiss_index_path: str = "data/faiss_index"):

        self.sqlite = SQLiteManager(sqlite_path)
        self.neo4j = Neo4jManager(neo4j_uri, neo4j_user, neo4j_password)
        self.faiss = FAISSVectorStore(index_path=faiss_index_path)

        self.partitions = {}

    async def initialize(self):
        """Initialize all storage components."""
        await self.sqlite.initialize()
        await self.neo4j.initialize()
        await self.faiss.initialize()

        # Try to load existing FAISS index
        await self.faiss.load_index()

        logger.info("Hybrid storage system initialized")

    async def store_dataset(self, papers: List[Paper], partition_strategy: str = "year") -> str:
        """Store complete dataset across all storage systems."""
        # Create partition
        partition_id = self._create_partition_id(papers, partition_strategy)

        # Store in parallel across all systems
        await asyncio.gather(
            self.sqlite.store_papers(papers, partition_id),
            self.neo4j.store_papers_graph(papers, partition_id),
            self.faiss.add_papers(papers, partition_id)
        )

        # Update partition tracking
        self.partitions[partition_id] = StoragePartition(
            partition_id=partition_id,
            year_start=self._get_min_year(papers),
            year_end=self._get_max_year(papers),
            topic_cluster=None,  # Could implement topic clustering
            paper_count=len(papers),
            created_at=datetime.now()
        )

        # Save FAISS index
        await self.faiss.save_index()

        logger.info(f"Stored {len(papers)} papers in partition {partition_id}")
        return partition_id

    def _create_partition_id(self, papers: List[Paper], strategy: str) -> str:
        """Create partition ID based on strategy."""
        if strategy == "year":
            min_year = self._get_min_year(papers)
            max_year = self._get_max_year(papers)
            return f"year_{min_year}_{max_year}"
        else:
            # Hash-based partitioning
            content = "".join(sorted([p.bibcode for p in papers]))
            hash_obj = hashlib.md5(content.encode())
            return f"hash_{hash_obj.hexdigest()[:8]}"

    def _get_min_year(self, papers: List[Paper]) -> int:
        """Get minimum year from papers."""
        years = []
        for paper in papers:
            if paper.pub_date:
                try:
                    years.append(int(paper.pub_date[:4]))
                except:
                    pass
        return min(years) if years else datetime.now().year

    def _get_max_year(self, papers: List[Paper]) -> int:
        """Get maximum year from papers."""
        years = []
        for paper in papers:
            if paper.pub_date:
                try:
                    years.append(int(paper.pub_date[:4]))
                except:
                    pass
        return max(years) if years else datetime.now().year

    async def hybrid_search(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search across all storage systems.

        This implements the GraphRAG pattern:
        1. Vector search for semantic similarity
        2. Graph traversal for related entities
        3. SQL for metadata enrichment
        """
        # Step 1: Semantic search with FAISS
        similar_papers = await self.faiss.search_similar(query, k * 2)  # Get more for filtering

        if not similar_papers:
            return []

        # Step 2: Get paper metadata from SQLite
        results = []
        for bibcode, similarity in similar_papers:
            # Get paper details from SQLite
            cursor = self.sqlite.connection.cursor()
            cursor.execute("SELECT * FROM papers WHERE bibcode = ?", (bibcode,))
            paper_row = cursor.fetchone()

            if paper_row:
                paper_data = dict(paper_row)
                paper_data["similarity_score"] = similarity

                # Step 3: Enhance with graph data from Neo4j
                citation_network = await self.neo4j.get_citation_network(bibcode, depth=1)
                paper_data["citing_papers_count"] = len(citation_network["citing_papers"])
                paper_data["cited_papers_count"] = len(citation_network["cited_papers"])

                results.append(paper_data)

        # Sort by similarity and limit
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:k]

    async def get_author_insights(self, author_name: str) -> Dict[str, Any]:
        """Get comprehensive author insights using all storage systems."""
        # SQLite: Get basic author stats
        papers = await self.sqlite.get_author_papers(author_name)

        # Neo4j: Get collaboration network
        collaboration_network = await self.neo4j.get_author_collaboration_network(author_name)

        # FAISS: Find semantically similar authors (based on paper content)
        # This would require author embeddings - simplified for now

        return {
            "author_name": author_name,
            "paper_count": len(papers),
            "papers": papers,
            "collaboration_network": collaboration_network,
            "recent_papers": [p for p in papers if p.get("pub_date", "").startswith("202")]
        }

    async def close(self):
        """Close all storage connections."""
        await asyncio.gather(
            self.sqlite.close(),
            self.neo4j.close(),
            # FAISS doesn't need explicit closing
        )