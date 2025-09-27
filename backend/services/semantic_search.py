"""
Vector embeddings and similarity search service for the Space Biology Knowledge Engine.
"""

import logging
import numpy as np
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
import json
from datetime import datetime

import faiss
from sentence_transformers import SentenceTransformer

from config.settings import get_settings, get_data_paths
from config.database import execute_query

logger = logging.getLogger(__name__)


class SemanticSearchService:
    """
    Service for vector embeddings and semantic similarity search.
    """

    def __init__(self):
        self.settings = get_settings()
        self.data_paths = get_data_paths()
        self.model = None
        self.index = None
        self.document_map = {}
        self.embeddings_cache = {}
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            model_name = self.settings.embedding_model_name
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")

            # Load existing index if available
            asyncio.create_task(self._load_existing_index())

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.model = None

    async def _load_existing_index(self):
        """Load existing FAISS index and document mappings."""
        try:
            index_file = self.data_paths['embeddings'] / 'faiss_index.bin'
            mapping_file = self.data_paths['embeddings'] / 'document_mapping.json'

            if index_file.exists() and mapping_file.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(index_file))

                # Load document mapping
                with open(mapping_file, 'r') as f:
                    self.document_map = json.load(f)

                logger.info(f"Loaded existing index with {self.index.ntotal} embeddings")
            else:
                logger.info("No existing index found, will create new one")

        except Exception as e:
            logger.error(f"Failed to load existing index: {e}")
            self.index = None
            self.document_map = {}

    async def generate_embeddings(self, texts: List[str],
                                batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            Array of embeddings
        """
        if not self.model:
            raise ValueError("Embedding model not initialized")

        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")

            # Process in batches to manage memory
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Generate embeddings for batch
                batch_embeddings = await asyncio.to_thread(
                    self.model.encode,
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

                all_embeddings.append(batch_embeddings)

                # Log progress
                if i % (batch_size * 5) == 0:
                    logger.debug(f"Processed {i + len(batch_texts)}/{len(texts)} texts")

            # Concatenate all embeddings
            embeddings = np.vstack(all_embeddings)
            logger.info(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    async def build_faiss_index(self, embeddings: np.ndarray,
                              document_ids: List[str],
                              index_type: str = 'flat') -> bool:
        """
        Build FAISS index from embeddings.

        Args:
            embeddings: Array of embeddings
            document_ids: List of document IDs corresponding to embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')

        Returns:
            True if successful, False otherwise
        """
        try:
            if len(embeddings) != len(document_ids):
                raise ValueError("Number of embeddings must match number of document IDs")

            dimension = embeddings.shape[1]
            logger.info(f"Building FAISS index for {len(embeddings)} embeddings, dimension {dimension}")

            # Create appropriate index type
            if index_type == 'flat':
                # Exact search, good for smaller datasets
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            elif index_type == 'ivf':
                # Inverted file index, good for larger datasets
                nlist = min(100, len(embeddings) // 10)  # Number of clusters
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                # Train the index
                self.index.train(embeddings)
            elif index_type == 'hnsw':
                # Hierarchical Navigable Small World, very fast
                M = 16  # Number of connections per element
                self.index = faiss.IndexHNSWFlat(dimension, M)
                self.index.hnsw.efConstruction = 200
            else:
                raise ValueError(f"Unknown index type: {index_type}")

            # Add embeddings to index
            self.index.add(embeddings)

            # Update document mapping
            for i, doc_id in enumerate(document_ids):
                self.document_map[str(i)] = doc_id

            # Save index and mapping
            await self._save_index()

            logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors")
            return True

        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            return False

    async def _save_index(self):
        """Save FAISS index and document mapping to disk."""
        try:
            if self.index is None:
                return

            index_file = self.data_paths['embeddings'] / 'faiss_index.bin'
            mapping_file = self.data_paths['embeddings'] / 'document_mapping.json'

            # Save FAISS index
            faiss.write_index(self.index, str(index_file))

            # Save document mapping
            with open(mapping_file, 'w') as f:
                json.dump(self.document_map, f, indent=2)

            logger.info("Index and mapping saved successfully")

        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    async def semantic_search(self, query: str, k: int = 10,
                            threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform semantic search using the query text.

        Args:
            query: Search query text
            k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of search results with similarity scores
        """
        if not self.model or not self.index:
            logger.error("Model or index not available for search")
            return []

        try:
            # Generate embedding for query
            query_embedding = await asyncio.to_thread(
                self.model.encode,
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            # Search the index
            scores, indices = self.index.search(query_embedding, k)

            # Process results
            results = []
            for i in range(len(indices[0])):
                if scores[0][i] >= threshold:
                    idx = str(indices[0][i])
                    if idx in self.document_map:
                        document_id = self.document_map[idx]
                        results.append({
                            'document_id': document_id,
                            'similarity_score': float(scores[0][i]),
                            'rank': i + 1
                        })

            logger.info(f"Semantic search returned {len(results)} results for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    async def hybrid_search(self, query: str, filters: Optional[Dict[str, Any]] = None,
                          k: int = 10, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search.

        Args:
            query: Search query
            filters: Additional filters for Neo4j query
            k: Number of results to return
            alpha: Weight for semantic search (1-alpha for keyword search)

        Returns:
            List of ranked search results
        """
        try:
            # Perform semantic search
            semantic_results = await self.semantic_search(query, k * 2)

            # Perform keyword search via Neo4j
            keyword_results = await self._keyword_search(query, filters, k * 2)

            # Combine and rank results
            combined_results = await self._combine_search_results(
                semantic_results, keyword_results, alpha
            )

            # Get additional metadata for results
            enriched_results = await self._enrich_search_results(combined_results[:k])

            return enriched_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    async def _keyword_search(self, query: str, filters: Optional[Dict[str, Any]] = None,
                            k: int = 10) -> List[Dict[str, Any]]:
        """Perform keyword-based search in Neo4j."""
        try:
            # Build keyword search query
            where_clauses = []
            params = {'query': query, 'limit': k}

            # Text search in title and description
            where_clauses.append(
                "(s.title CONTAINS $query OR s.description CONTAINS $query)"
            )

            # Add filters
            if filters:
                if 'research_type' in filters:
                    where_clauses.append("s.research_type = $research_type")
                    params['research_type'] = filters['research_type']

                if 'mission_type' in filters:
                    where_clauses.append("s.mission_relevance = $mission_type")
                    params['mission_type'] = filters['mission_type']

                if 'organism' in filters:
                    where_clauses.append(
                        "EXISTS { (s)-[:STUDIES]->(o:Organism) WHERE o.name CONTAINS $organism }"
                    )
                    params['organism'] = filters['organism']

            where_clause = " AND ".join(where_clauses) if where_clauses else "true"

            query_cypher = f"""
            MATCH (s:Study)
            WHERE {where_clause}
            RETURN s.id as document_id, s.title as title, s.description as description
            ORDER BY size(s.title) + size(s.description) DESC
            LIMIT $limit
            """

            results = execute_query(query_cypher, params)

            # Calculate keyword relevance scores
            keyword_results = []
            query_words = set(query.lower().split())

            for record in results:
                text = f"{record['title']} {record['description']}".lower()
                text_words = set(text.split())

                # Simple relevance scoring based on word overlap
                overlap = len(query_words.intersection(text_words))
                relevance_score = overlap / len(query_words) if query_words else 0

                keyword_results.append({
                    'document_id': record['document_id'],
                    'relevance_score': relevance_score,
                    'title': record['title']
                })

            return keyword_results

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    async def _combine_search_results(self, semantic_results: List[Dict[str, Any]],
                                    keyword_results: List[Dict[str, Any]],
                                    alpha: float) -> List[Dict[str, Any]]:
        """Combine semantic and keyword search results."""
        try:
            # Create unified result set
            all_results = {}

            # Add semantic results
            for result in semantic_results:
                doc_id = result['document_id']
                all_results[doc_id] = {
                    'document_id': doc_id,
                    'semantic_score': result['similarity_score'],
                    'keyword_score': 0.0,
                    'combined_score': 0.0
                }

            # Add keyword results
            for result in keyword_results:
                doc_id = result['document_id']
                if doc_id in all_results:
                    all_results[doc_id]['keyword_score'] = result['relevance_score']
                else:
                    all_results[doc_id] = {
                        'document_id': doc_id,
                        'semantic_score': 0.0,
                        'keyword_score': result['relevance_score'],
                        'combined_score': 0.0
                    }

            # Calculate combined scores
            for result in all_results.values():
                result['combined_score'] = (
                    alpha * result['semantic_score'] +
                    (1 - alpha) * result['keyword_score']
                )

            # Sort by combined score
            sorted_results = sorted(
                all_results.values(),
                key=lambda x: x['combined_score'],
                reverse=True
            )

            return sorted_results

        except Exception as e:
            logger.error(f"Failed to combine search results: {e}")
            return []

    async def _enrich_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich search results with additional metadata from Neo4j."""
        if not results:
            return []

        try:
            # Get document IDs
            doc_ids = [r['document_id'] for r in results]

            # Query Neo4j for additional details
            query = """
            MATCH (s:Study)
            WHERE s.id IN $doc_ids
            OPTIONAL MATCH (s)-[:STUDIES]->(o:Organism)
            OPTIONAL MATCH (s)-[:HAS_FACTOR]->(f:Factor)
            RETURN s.id as id, s.title as title, s.description as description,
                   s.research_type as research_type, s.mission_relevance as mission_relevance,
                   s.publication_date as publication_date, s.source as source,
                   collect(DISTINCT o.name) as organisms,
                   collect(DISTINCT f.name) as factors
            """

            metadata_results = execute_query(query, {'doc_ids': doc_ids})

            # Create metadata lookup
            metadata_map = {r['id']: r for r in metadata_results}

            # Enrich results
            enriched_results = []
            for result in results:
                doc_id = result['document_id']
                if doc_id in metadata_map:
                    metadata = metadata_map[doc_id]
                    enriched_result = {
                        **result,
                        'title': metadata['title'],
                        'description': metadata['description'],
                        'research_type': metadata['research_type'],
                        'mission_relevance': metadata['mission_relevance'],
                        'publication_date': metadata['publication_date'],
                        'source': metadata['source'],
                        'organisms': [o for o in metadata['organisms'] if o],
                        'factors': [f for f in metadata['factors'] if f]
                    }
                    enriched_results.append(enriched_result)

            return enriched_results

        except Exception as e:
            logger.error(f"Failed to enrich search results: {e}")
            return results

    async def find_similar_documents(self, document_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Find documents similar to a given document.

        Args:
            document_id: ID of the reference document
            k: Number of similar documents to return

        Returns:
            List of similar documents with similarity scores
        """
        try:
            # Get the document's embedding index
            doc_index = None
            for idx, doc_id in self.document_map.items():
                if doc_id == document_id:
                    doc_index = int(idx)
                    break

            if doc_index is None:
                logger.warning(f"Document {document_id} not found in index")
                return []

            # Get the document's embedding
            if doc_index >= self.index.ntotal:
                logger.error(f"Document index {doc_index} out of range")
                return []

            # Reconstruct the embedding (this is approximate for some index types)
            doc_embedding = self.index.reconstruct(doc_index).reshape(1, -1)

            # Search for similar documents
            scores, indices = self.index.search(doc_embedding, k + 1)  # +1 to exclude self

            # Process results (skip the first result if it's the same document)
            results = []
            for i in range(len(indices[0])):
                idx = str(indices[0][i])
                if idx in self.document_map and self.document_map[idx] != document_id:
                    similar_doc_id = self.document_map[idx]
                    results.append({
                        'document_id': similar_doc_id,
                        'similarity_score': float(scores[0][i]),
                        'rank': len(results) + 1
                    })

                if len(results) >= k:
                    break

            return results

        except Exception as e:
            logger.error(f"Failed to find similar documents: {e}")
            return []

    async def update_index_with_new_documents(self, new_texts: List[str],
                                            new_doc_ids: List[str]) -> bool:
        """
        Update the existing index with new documents.

        Args:
            new_texts: List of new document texts
            new_doc_ids: List of new document IDs

        Returns:
            True if successful, False otherwise
        """
        try:
            if len(new_texts) != len(new_doc_ids):
                raise ValueError("Number of texts must match number of document IDs")

            # Generate embeddings for new documents
            new_embeddings = await self.generate_embeddings(new_texts)

            # Add to existing index
            if self.index is None:
                # Create new index if none exists
                await self.build_faiss_index(new_embeddings, new_doc_ids)
            else:
                # Add to existing index
                start_idx = self.index.ntotal
                self.index.add(new_embeddings)

                # Update document mapping
                for i, doc_id in enumerate(new_doc_ids):
                    self.document_map[str(start_idx + i)] = doc_id

                # Save updated index
                await self._save_index()

            logger.info(f"Added {len(new_texts)} new documents to index")
            return True

        except Exception as e:
            logger.error(f"Failed to update index: {e}")
            return False

    async def rebuild_index_from_database(self) -> bool:
        """
        Rebuild the entire search index from the database.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Rebuilding search index from database...")

            # Get all studies from database
            query = """
            MATCH (s:Study)
            RETURN s.id as id, s.title as title, s.description as description
            ORDER BY s.created_at DESC
            """

            results = execute_query(query)

            if not results:
                logger.warning("No studies found in database")
                return False

            # Prepare texts and IDs
            texts = []
            doc_ids = []

            for record in results:
                # Combine title and description for embedding
                text = f"{record['title']} {record['description']}".strip()
                if text:
                    texts.append(text)
                    doc_ids.append(record['id'])

            logger.info(f"Found {len(texts)} documents to index")

            # Generate embeddings and build index
            embeddings = await self.generate_embeddings(texts)
            success = await self.build_faiss_index(embeddings, doc_ids)

            if success:
                logger.info("Index rebuilt successfully")
            else:
                logger.error("Failed to rebuild index")

            return success

        except Exception as e:
            logger.error(f"Failed to rebuild index from database: {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        stats = {
            'model_loaded': self.model is not None,
            'index_loaded': self.index is not None,
            'total_documents': len(self.document_map),
            'model_name': self.settings.embedding_model_name,
            'embedding_dimension': self.settings.embedding_dimension
        }

        if self.index:
            stats.update({
                'index_size': self.index.ntotal,
                'index_type': type(self.index).__name__
            })

        return stats