"""
GraphRAG Query Engine for Space Biology Knowledge Engine

This module implements the GraphRAG (Graph + Retrieval Augmented Generation) pattern:
1. Semantic retrieval from vector store (FAISS)
2. Graph traversal for related entities (Neo4j)
3. Metadata enrichment from relational store (SQLite)
4. LLM synthesis with structured context

Optimized for space biology research queries with intelligent context assembly.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from ..storage.hybrid_storage import HybridStorageManager
from ...services.llm_service import LLMService

logger = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """Structured context for GraphRAG queries."""
    semantic_matches: List[Dict[str, Any]]
    graph_entities: Dict[str, List[Dict[str, Any]]]
    metadata_enrichments: Dict[str, Any]
    confidence_scores: Dict[str, float]


@dataclass
class GraphRAGResult:
    """Result from GraphRAG query processing."""
    query: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    context_used: QueryContext
    processing_metadata: Dict[str, Any]


class GraphRAGQueryEngine:
    """
    Main query engine implementing GraphRAG pattern for space biology research.
    """

    def __init__(self, storage_manager: HybridStorageManager, llm_service: LLMService):
        self.storage = storage_manager
        self.llm = llm_service

        # Query processing strategies
        self.strategies = {
            "literature_search": self._literature_search_strategy,
            "author_analysis": self._author_analysis_strategy,
            "citation_analysis": self._citation_analysis_strategy,
            "topic_exploration": self._topic_exploration_strategy,
            "temporal_analysis": self._temporal_analysis_strategy,
            "comparative_analysis": self._comparative_analysis_strategy
        }

        # Context assembly weights
        self.context_weights = {
            "semantic_similarity": 0.4,
            "citation_relevance": 0.3,
            "temporal_relevance": 0.2,
            "author_authority": 0.1
        }

    async def query(self, user_query: str,
                   strategy: Optional[str] = None,
                   max_papers: int = 10,
                   context_depth: int = 2) -> GraphRAGResult:
        """
        Process a query using GraphRAG methodology.

        Args:
            user_query: Natural language query from user
            strategy: Specific processing strategy (auto-detected if None)
            max_papers: Maximum papers to retrieve
            context_depth: Depth of graph traversal

        Returns:
            Comprehensive GraphRAG result with answer and context
        """
        start_time = datetime.utcnow()

        try:
            # Step 1: Query analysis and strategy selection
            if not strategy:
                strategy = await self._detect_query_strategy(user_query)

            logger.info(f"Processing query with strategy: {strategy}")

            # Step 2: Multi-stage retrieval and context assembly
            context = await self._assemble_query_context(
                user_query, strategy, max_papers, context_depth
            )

            # Step 3: LLM synthesis with structured context
            answer = await self._synthesize_answer(user_query, context, strategy)

            # Step 4: Calculate confidence and prepare sources
            confidence = self._calculate_answer_confidence(context)
            sources = self._extract_sources(context)

            # Processing metadata
            end_time = datetime.utcnow()
            processing_metadata = {
                "strategy_used": strategy,
                "processing_time_seconds": (end_time - start_time).total_seconds(),
                "papers_retrieved": len(context.semantic_matches),
                "graph_entities_explored": sum(len(entities) for entities in context.graph_entities.values()),
                "timestamp": end_time.isoformat()
            }

            result = GraphRAGResult(
                query=user_query,
                answer=answer,
                confidence=confidence,
                sources=sources,
                context_used=context,
                processing_metadata=processing_metadata
            )

            logger.info(f"Query processed successfully in {processing_metadata['processing_time_seconds']:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    async def _detect_query_strategy(self, query: str) -> str:
        """Detect the best processing strategy for the query."""
        query_lower = query.lower()

        # Strategy detection patterns
        if any(word in query_lower for word in ["author", "researcher", "scientist", "who"]):
            return "author_analysis"
        elif any(word in query_lower for word in ["cite", "citation", "reference", "impact"]):
            return "citation_analysis"
        elif any(word in query_lower for word in ["topic", "field", "area", "domain"]):
            return "topic_exploration"
        elif any(word in query_lower for word in ["trend", "over time", "recent", "latest", "evolution"]):
            return "temporal_analysis"
        elif any(word in query_lower for word in ["compare", "versus", "vs", "difference", "similar"]):
            return "comparative_analysis"
        else:
            return "literature_search"

    async def _assemble_query_context(self, query: str, strategy: str,
                                    max_papers: int, context_depth: int) -> QueryContext:
        """
        Assemble comprehensive context using all storage systems.
        This is the core of the GraphRAG implementation.
        """

        # Step 1: Semantic retrieval from FAISS
        semantic_matches = await self.storage.faiss.search_similar(
            query, k=max_papers * 2  # Get more for filtering
        )

        # Enrich semantic matches with metadata from SQLite
        enriched_papers = []
        for bibcode, similarity in semantic_matches[:max_papers]:
            paper_metadata = await self._get_paper_metadata(bibcode)
            if paper_metadata:
                paper_metadata["semantic_similarity"] = similarity
                enriched_papers.append(paper_metadata)

        # Step 2: Graph traversal for related entities
        graph_entities = {}

        if strategy in ["author_analysis", "citation_analysis"]:
            # Explore author networks and citation patterns
            for paper in enriched_papers[:5]:  # Limit graph exploration
                bibcode = paper["bibcode"]

                # Citation network
                citation_network = await self.storage.neo4j.get_citation_network(
                    bibcode, depth=context_depth
                )
                graph_entities[f"citations_{bibcode}"] = [
                    {"bibcode": bc, "type": "citation"}
                    for bc in citation_network["citing_papers"]
                ]

                # Author collaboration network (if author-focused)
                if strategy == "author_analysis" and paper.get("authors"):
                    first_author = paper["authors"][0] if isinstance(paper["authors"], list) else paper["authors"]
                    collab_network = await self.storage.neo4j.get_author_collaboration_network(
                        first_author, depth=context_depth
                    )
                    graph_entities[f"collaborations_{first_author}"] = [
                        {"name": collab["name"], "type": "collaborator", "distance": collab["distance"]}
                        for collab in collab_network["collaborators"]
                    ]

        elif strategy == "topic_exploration":
            # Explore topic relationships
            topics_explored = set()
            for paper in enriched_papers[:3]:
                paper_topics = await self._get_paper_topics(paper["bibcode"])
                for topic in paper_topics:
                    if topic not in topics_explored:
                        topic_papers = await self.storage.neo4j.get_papers_by_topic_and_partition(
                            topic, paper.get("partition_id", "")
                        )
                        graph_entities[f"topic_{topic}"] = [
                            {"bibcode": bc, "type": "topic_paper", "topic": topic}
                            for bc in topic_papers[:10]
                        ]
                        topics_explored.add(topic)

        # Step 3: Metadata enrichment
        metadata_enrichments = await self._calculate_metadata_enrichments(
            enriched_papers, graph_entities, strategy
        )

        # Step 4: Confidence scoring
        confidence_scores = self._calculate_context_confidence(
            enriched_papers, graph_entities, metadata_enrichments
        )

        return QueryContext(
            semantic_matches=enriched_papers,
            graph_entities=graph_entities,
            metadata_enrichments=metadata_enrichments,
            confidence_scores=confidence_scores
        )

    async def _get_paper_metadata(self, bibcode: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive paper metadata from SQLite."""
        cursor = self.storage.sqlite.connection.cursor()

        # Get paper info with authors and topics
        cursor.execute("""
            SELECT p.*,
                   GROUP_CONCAT(DISTINCT a.name) as authors,
                   GROUP_CONCAT(DISTINCT t.topic_name) as topics
            FROM papers p
            LEFT JOIN paper_authors pa ON p.bibcode = pa.paper_bibcode
            LEFT JOIN authors a ON pa.author_id = a.author_id
            LEFT JOIN paper_topics pt ON p.bibcode = pt.paper_bibcode
            LEFT JOIN topics t ON pt.topic_id = t.topic_id
            WHERE p.bibcode = ?
            GROUP BY p.bibcode
        """, (bibcode,))

        row = cursor.fetchone()
        if row:
            paper_data = dict(row)
            # Parse comma-separated strings back to lists
            if paper_data["authors"]:
                paper_data["authors"] = paper_data["authors"].split(",")
            if paper_data["topics"]:
                paper_data["topics"] = paper_data["topics"].split(",")
            return paper_data

        return None

    async def _get_paper_topics(self, bibcode: str) -> List[str]:
        """Get topics for a specific paper."""
        cursor = self.storage.sqlite.connection.cursor()
        cursor.execute("""
            SELECT t.topic_name FROM topics t
            JOIN paper_topics pt ON t.topic_id = pt.topic_id
            WHERE pt.paper_bibcode = ?
        """, (bibcode,))
        return [row[0] for row in cursor.fetchall()]

    async def _calculate_metadata_enrichments(self, papers: List[Dict[str, Any]],
                                            graph_entities: Dict[str, List[Dict[str, Any]]],
                                            strategy: str) -> Dict[str, Any]:
        """Calculate metadata enrichments for context."""
        enrichments = {
            "total_papers": len(papers),
            "total_citations": sum(p.get("citation_count", 0) for p in papers),
            "date_range": self._calculate_date_range(papers),
            "top_authors": self._get_top_authors(papers),
            "topic_distribution": self._calculate_topic_distribution(papers),
            "journal_distribution": self._calculate_journal_distribution(papers)
        }

        # Strategy-specific enrichments
        if strategy == "author_analysis":
            enrichments["collaboration_stats"] = self._calculate_collaboration_stats(graph_entities)
        elif strategy == "citation_analysis":
            enrichments["citation_patterns"] = self._analyze_citation_patterns(graph_entities)
        elif strategy == "temporal_analysis":
            enrichments["temporal_trends"] = self._calculate_temporal_trends(papers)

        return enrichments

    def _calculate_date_range(self, papers: List[Dict[str, Any]]) -> Dict[str, str]:
        """Calculate date range of papers."""
        dates = [p.get("pub_date", "") for p in papers if p.get("pub_date")]
        if dates:
            return {"earliest": min(dates), "latest": max(dates)}
        return {"earliest": "", "latest": ""}

    def _get_top_authors(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get top authors by paper count."""
        author_counts = {}
        for paper in papers:
            authors = paper.get("authors", [])
            if isinstance(authors, list):
                for author in authors:
                    author_counts[author] = author_counts.get(author, 0) + 1

        top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"name": name, "paper_count": count} for name, count in top_authors[:10]]

    def _calculate_topic_distribution(self, papers: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of topics."""
        topic_counts = {}
        for paper in papers:
            topics = paper.get("topics", [])
            if isinstance(topics, list):
                for topic in topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
        return topic_counts

    def _calculate_journal_distribution(self, papers: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of journals."""
        journal_counts = {}
        for paper in papers:
            journal = paper.get("journal", "")
            if journal:
                journal_counts[journal] = journal_counts.get(journal, 0) + 1
        return journal_counts

    def _calculate_collaboration_stats(self, graph_entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Calculate collaboration statistics from graph entities."""
        all_collaborators = []
        for key, entities in graph_entities.items():
            if "collaborations_" in key:
                all_collaborators.extend(entities)

        return {
            "total_collaborators": len(all_collaborators),
            "average_distance": sum(c.get("distance", 0) for c in all_collaborators) / max(len(all_collaborators), 1),
            "collaboration_network_size": len(set(c.get("name", "") for c in all_collaborators))
        }

    def _analyze_citation_patterns(self, graph_entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze citation patterns from graph entities."""
        all_citations = []
        for key, entities in graph_entities.items():
            if "citations_" in key:
                all_citations.extend(entities)

        return {
            "total_citations_found": len(all_citations),
            "citation_network_density": len(all_citations) / max(len(graph_entities), 1)
        }

    def _calculate_temporal_trends(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate temporal trends in the papers."""
        year_counts = {}
        for paper in papers:
            pub_date = paper.get("pub_date", "")
            if pub_date and len(pub_date) >= 4:
                year = pub_date[:4]
                year_counts[year] = year_counts.get(year, 0) + 1

        sorted_years = sorted(year_counts.items())
        return {
            "papers_by_year": dict(sorted_years),
            "trend_direction": "increasing" if len(sorted_years) > 1 and sorted_years[-1][1] > sorted_years[0][1] else "stable"
        }

    def _calculate_context_confidence(self, papers: List[Dict[str, Any]],
                                    graph_entities: Dict[str, List[Dict[str, Any]]],
                                    metadata: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for different aspects of the context."""
        confidence_scores = {}

        # Semantic confidence based on similarity scores
        if papers:
            similarities = [p.get("semantic_similarity", 0) for p in papers]
            confidence_scores["semantic"] = sum(similarities) / len(similarities)

        # Graph completeness confidence
        graph_size = sum(len(entities) for entities in graph_entities.values())
        confidence_scores["graph_completeness"] = min(graph_size / 50, 1.0)  # Normalize to 1.0

        # Citation confidence
        total_citations = metadata.get("total_citations", 0)
        confidence_scores["citation_authority"] = min(total_citations / 100, 1.0)  # Normalize

        # Temporal confidence based on recency
        date_range = metadata.get("date_range", {})
        if date_range.get("latest"):
            try:
                latest_year = int(date_range["latest"][:4])
                current_year = datetime.now().year
                recency_score = max(0, 1 - (current_year - latest_year) / 10)  # Decay over 10 years
                confidence_scores["temporal_relevance"] = recency_score
            except:
                confidence_scores["temporal_relevance"] = 0.5

        return confidence_scores

    async def _synthesize_answer(self, query: str, context: QueryContext, strategy: str) -> str:
        """Synthesize answer using LLM with structured context."""

        # Build context for LLM
        context_text = self._build_llm_context(context, strategy)

        # Strategy-specific prompts
        strategy_prompts = {
            "literature_search": self._create_literature_search_prompt(query, context_text),
            "author_analysis": self._create_author_analysis_prompt(query, context_text),
            "citation_analysis": self._create_citation_analysis_prompt(query, context_text),
            "topic_exploration": self._create_topic_exploration_prompt(query, context_text),
            "temporal_analysis": self._create_temporal_analysis_prompt(query, context_text),
            "comparative_analysis": self._create_comparative_analysis_prompt(query, context_text)
        }

        prompt = strategy_prompts.get(strategy, strategy_prompts["literature_search"])

        # Generate answer
        answer = await self.llm.generate_response(prompt, max_tokens=1500)

        return answer

    def _build_llm_context(self, context: QueryContext, strategy: str) -> str:
        """Build structured context for LLM synthesis."""
        context_parts = []

        # Add semantic matches
        if context.semantic_matches:
            context_parts.append("## Relevant Research Papers")
            for i, paper in enumerate(context.semantic_matches[:8], 1):
                authors_str = ", ".join(paper.get("authors", [])[:3]) if paper.get("authors") else "Unknown"
                context_parts.append(
                    f"{i}. **{paper.get('title', 'Unknown Title')}** "
                    f"({authors_str}, {paper.get('pub_date', 'Unknown Date')[:4]})\n"
                    f"   Journal: {paper.get('journal', 'Unknown')}\n"
                    f"   Citations: {paper.get('citation_count', 0)}\n"
                    f"   Similarity: {paper.get('semantic_similarity', 0):.3f}\n"
                )

        # Add metadata enrichments
        if context.metadata_enrichments:
            context_parts.append("\n## Research Landscape Summary")
            meta = context.metadata_enrichments
            context_parts.append(f"- Total papers analyzed: {meta.get('total_papers', 0)}")
            context_parts.append(f"- Total citations: {meta.get('total_citations', 0)}")
            context_parts.append(f"- Date range: {meta.get('date_range', {}).get('earliest', '')} to {meta.get('date_range', {}).get('latest', '')}")

            if meta.get("top_authors"):
                authors_list = ", ".join([f"{a['name']} ({a['paper_count']})" for a in meta["top_authors"][:5]])
                context_parts.append(f"- Top authors: {authors_list}")

        # Add graph insights
        if context.graph_entities:
            context_parts.append("\n## Network Analysis")
            for entity_type, entities in context.graph_entities.items():
                if entities:
                    context_parts.append(f"- {entity_type}: {len(entities)} connections found")

        # Add confidence indicators
        if context.confidence_scores:
            conf_items = [f"{k}: {v:.2f}" for k, v in context.confidence_scores.items()]
            context_parts.append(f"\n## Analysis Confidence: {', '.join(conf_items)}")

        return "\n".join(context_parts)

    def _create_literature_search_prompt(self, query: str, context: str) -> str:
        """Create prompt for literature search strategy."""
        return f"""
You are an expert space biology researcher. Based on the comprehensive research context provided,
answer the following query with scientific accuracy and depth.

Query: {query}

Research Context:
{context}

Please provide a comprehensive answer that:
1. Directly addresses the query
2. Synthesizes findings from multiple papers
3. Highlights key discoveries and consensus
4. Notes any contradictions or uncertainties
5. Provides specific citations to support claims
6. Suggests areas for future research if relevant

Focus on being scientifically accurate while making the information accessible.
"""

    def _create_author_analysis_prompt(self, query: str, context: str) -> str:
        """Create prompt for author analysis strategy."""
        return f"""
You are analyzing researchers and their contributions to space biology. Based on the research context,
provide insights about authors, their work, and collaboration patterns.

Query: {query}

Research Context:
{context}

Please provide an analysis that includes:
1. Author contributions and research focus areas
2. Collaboration patterns and networks
3. Impact and influence in the field
4. Evolution of their research over time
5. Key publications and their significance

Be specific about research contributions and provide evidence from the data.
"""

    def _create_citation_analysis_prompt(self, query: str, context: str) -> str:
        """Create prompt for citation analysis strategy."""
        return f"""
You are analyzing citation patterns and research impact in space biology.

Query: {query}

Research Context:
{context}

Please provide an analysis that covers:
1. Citation patterns and research impact
2. Influential papers and their contributions
3. Knowledge flow and research evolution
4. Emerging vs established research areas
5. Research network structure and key connections

Focus on how ideas have spread and evolved in the field.
"""

    def _create_topic_exploration_prompt(self, query: str, context: str) -> str:
        """Create prompt for topic exploration strategy."""
        return f"""
You are exploring research topics and themes in space biology.

Query: {query}

Research Context:
{context}

Please provide a comprehensive exploration that includes:
1. Main research themes and sub-topics
2. Relationships between different research areas
3. Current state of knowledge in each area
4. Research gaps and opportunities
5. Methodological approaches being used

Organize your response by topic areas and show connections between them.
"""

    def _create_temporal_analysis_prompt(self, query: str, context: str) -> str:
        """Create prompt for temporal analysis strategy."""
        return f"""
You are analyzing how space biology research has evolved over time.

Query: {query}

Research Context:
{context}

Please provide a temporal analysis that covers:
1. Evolution of research questions and approaches
2. Emerging trends and new directions
3. Shifts in focus or methodology
4. Historical context and milestones
5. Future research trajectories

Show how the field has developed and where it might be heading.
"""

    def _create_comparative_analysis_prompt(self, query: str, context: str) -> str:
        """Create prompt for comparative analysis strategy."""
        return f"""
You are comparing different aspects of space biology research.

Query: {query}

Research Context:
{context}

Please provide a comparative analysis that includes:
1. Systematic comparison of the requested topics/approaches/findings
2. Similarities and differences
3. Strengths and limitations of each
4. Evidence supporting different perspectives
5. Implications for future research

Be objective and evidence-based in your comparisons.
"""

    def _calculate_answer_confidence(self, context: QueryContext) -> float:
        """Calculate overall confidence in the answer."""
        if not context.confidence_scores:
            return 0.5

        # Weighted average of confidence scores
        weighted_sum = 0
        total_weight = 0

        for metric, score in context.confidence_scores.items():
            weight = self.context_weights.get(metric, 0.1)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight > 0:
            return min(weighted_sum / total_weight, 1.0)
        else:
            return 0.5

    def _extract_sources(self, context: QueryContext) -> List[Dict[str, Any]]:
        """Extract source information for citations."""
        sources = []

        for paper in context.semantic_matches:
            source = {
                "type": "research_paper",
                "bibcode": paper.get("bibcode", ""),
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "journal": paper.get("journal", ""),
                "pub_date": paper.get("pub_date", ""),
                "citation_count": paper.get("citation_count", 0),
                "similarity_score": paper.get("semantic_similarity", 0),
                "doi": paper.get("doi", "")
            }
            sources.append(source)

        return sources

    # Strategy-specific processing methods
    async def _literature_search_strategy(self, query: str, context: QueryContext) -> Dict[str, Any]:
        """Process literature search queries."""
        # This could include additional processing specific to literature searches
        return {"strategy": "literature_search", "processed": True}

    async def _author_analysis_strategy(self, query: str, context: QueryContext) -> Dict[str, Any]:
        """Process author analysis queries."""
        # Could include author-specific processing
        return {"strategy": "author_analysis", "processed": True}

    async def _citation_analysis_strategy(self, query: str, context: QueryContext) -> Dict[str, Any]:
        """Process citation analysis queries."""
        # Could include citation-specific processing
        return {"strategy": "citation_analysis", "processed": True}

    async def _topic_exploration_strategy(self, query: str, context: QueryContext) -> Dict[str, Any]:
        """Process topic exploration queries."""
        # Could include topic-specific processing
        return {"strategy": "topic_exploration", "processed": True}

    async def _temporal_analysis_strategy(self, query: str, context: QueryContext) -> Dict[str, Any]:
        """Process temporal analysis queries."""
        # Could include temporal-specific processing
        return {"strategy": "temporal_analysis", "processed": True}

    async def _comparative_analysis_strategy(self, query: str, context: QueryContext) -> Dict[str, Any]:
        """Process comparative analysis queries."""
        # Could include comparison-specific processing
        return {"strategy": "comparative_analysis", "processed": True}