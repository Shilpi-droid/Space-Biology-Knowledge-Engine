"""
Research agent for scientific research queries in the Space Biology Knowledge Engine.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_agent import BaseAgent
from config.database import execute_query

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    AI agent specialized in scientific research queries and analysis.
    """

    def __init__(self):
        specialized_knowledge = {
            'research_methodologies': [
                'controlled experiments', 'observational studies', 'meta-analysis',
                'systematic reviews', 'case studies', 'longitudinal studies'
            ],
            'research_domains': [
                'genomics', 'proteomics', 'physiology', 'cell biology',
                'microbiology', 'psychology', 'radiation biology'
            ],
            'space_environments': [
                'microgravity', 'radiation exposure', 'isolation',
                'confined spaces', 'altered atmosphere', 'circadian disruption'
            ]
        }
        super().__init__('research', specialized_knowledge)

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None,
                          conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process research-related queries.

        Args:
            query: User research query
            context: Additional context
            conversation_id: Conversation tracking ID

        Returns:
            Research agent response
        """
        try:
            # Validate input
            validation = await self.validate_input(query, context)
            if not validation['valid']:
                return self.format_response(
                    f"Invalid query: {', '.join(validation['errors'])}",
                    confidence=0.0
                )

            # Extract intent
            intent = await self.extract_intent(query)
            logger.info(f"Research agent processing query with intent: {intent['primary_intent']}")

            # Route to specific handler based on intent
            if intent['primary_intent'] == 'search':
                response_data = await self._handle_search_query(query, context)
            elif intent['primary_intent'] == 'compare':
                response_data = await self._handle_comparison_query(query, context)
            elif intent['primary_intent'] == 'explain':
                response_data = await self._handle_explanation_query(query, context)
            elif intent['primary_intent'] == 'summarize':
                response_data = await self._handle_summary_query(query, context)
            else:
                response_data = await self._handle_general_query(query, context)

            # Track conversation
            if conversation_id:
                await self.track_conversation(
                    conversation_id, query, response_data['message'],
                    {'intent': intent, 'context': context}
                )

            # Generate follow-up questions
            follow_ups = await self.generate_follow_up_questions(query, response_data['message'])
            response_data['metadata']['follow_up_questions'] = follow_ups

            return response_data

        except Exception as e:
            logger.error(f"Research agent query processing failed: {e}")
            return self.format_response(
                "I encountered an error while processing your research query. Please try rephrasing your question.",
                confidence=0.0,
                metadata={'error': str(e)}
            )

    async def _handle_search_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle search-type queries."""
        try:
            # Get relevant studies
            studies = await self.find_related_studies(query)

            if not studies:
                return self.format_response(
                    "I couldn't find any research studies directly related to your query. "
                    "Try rephrasing your question or using more general terms.",
                    confidence=0.3
                )

            # Generate response based on found studies
            response = await self._generate_search_response(query, studies)
            references = [study['id'] for study in studies]

            return self.format_response(
                response,
                references=references,
                confidence=0.8,
                metadata={'studies_found': len(studies)}
            )

        except Exception as e:
            logger.error(f"Search query handling failed: {e}")
            return self.format_response(
                "I encountered an issue while searching for relevant research. Please try again.",
                confidence=0.0
            )

    async def _handle_comparison_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle comparison queries between studies or findings."""
        try:
            # Extract entities to compare
            entities = await self._extract_comparison_entities(query)

            if len(entities) < 2:
                return self.format_response(
                    "For comparison queries, please specify at least two items to compare. "
                    "For example: 'Compare the effects of microgravity on mice versus humans'",
                    confidence=0.4
                )

            # Find studies related to each entity
            comparison_data = {}
            all_references = []

            for entity in entities:
                entity_studies = await self.find_related_studies(entity)
                comparison_data[entity] = entity_studies
                all_references.extend([s['id'] for s in entity_studies])

            # Generate comparison response
            response = await self._generate_comparison_response(query, comparison_data)

            return self.format_response(
                response,
                references=list(set(all_references)),
                confidence=0.8,
                metadata={'entities_compared': entities}
            )

        except Exception as e:
            logger.error(f"Comparison query handling failed: {e}")
            return self.format_response(
                "I encountered an issue while processing your comparison. Please try again.",
                confidence=0.0
            )

    async def _handle_explanation_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle explanation queries about mechanisms, processes, etc."""
        try:
            # Find relevant studies for explanation
            studies = await self.find_related_studies(query)

            # Get related concepts for comprehensive explanation
            key_concepts = await self.get_related_concepts(query)

            # Generate detailed explanation
            response = await self._generate_explanation_response(query, studies, key_concepts)
            references = [study['id'] for study in studies]

            return self.format_response(
                response,
                references=references,
                confidence=0.8,
                metadata={'key_concepts': key_concepts}
            )

        except Exception as e:
            logger.error(f"Explanation query handling failed: {e}")
            return self.format_response(
                "I encountered an issue while generating the explanation. Please try again.",
                confidence=0.0
            )

    async def _handle_summary_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle summary requests for research areas or topics."""
        try:
            # Extract the topic to summarize
            topic = await self._extract_summary_topic(query)

            # Find comprehensive set of studies
            studies = await self.find_related_studies(topic, limit=10)

            if not studies:
                return self.format_response(
                    f"I couldn't find enough research on '{topic}' to provide a comprehensive summary.",
                    confidence=0.3
                )

            # Generate summary
            response = await self._generate_research_summary(topic, studies)
            references = [study['id'] for study in studies]

            return self.format_response(
                response,
                references=references,
                confidence=0.8,
                metadata={'topic': topic, 'studies_analyzed': len(studies)}
            )

        except Exception as e:
            logger.error(f"Summary query handling failed: {e}")
            return self.format_response(
                "I encountered an issue while generating the summary. Please try again.",
                confidence=0.0
            )

    async def _handle_general_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle general research queries that don't fit specific patterns."""
        try:
            # Get relevant studies
            studies = await self.find_related_studies(query)

            # Generate general research response
            if studies:
                response = await self._generate_general_response(query, studies)
                references = [study['id'] for study in studies]
                confidence = 0.7
            else:
                response = await self._generate_fallback_response(query)
                references = []
                confidence = 0.5

            return self.format_response(
                response,
                references=references,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"General query handling failed: {e}")
            return self.format_response(
                "I'm having trouble processing your query. Could you please rephrase it?",
                confidence=0.0
            )

    async def find_related_studies(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find studies related to the query with detailed metadata.

        Args:
            query: Search query
            limit: Maximum number of studies to return

        Returns:
            List of related studies with metadata
        """
        try:
            # Get basic search results
            search_results = await self.get_relevant_studies(query, limit)

            # Enrich with study details from database
            if not search_results:
                return []

            study_ids = [result['id'] for result in search_results]

            # Query database for detailed study information
            detailed_studies = await self._get_study_details(study_ids)

            # Combine search scores with study details
            enriched_studies = []
            for result in search_results:
                study_detail = detailed_studies.get(result['id'])
                if study_detail:
                    enriched_study = {
                        **study_detail,
                        'similarity_score': result['similarity_score'],
                        'rank': result['rank']
                    }
                    enriched_studies.append(enriched_study)

            return enriched_studies

        except Exception as e:
            logger.error(f"Failed to find related studies: {e}")
            return []

    async def _get_study_details(self, study_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get detailed study information from the database."""
        try:
            query = """
            MATCH (s:Study)
            WHERE s.id IN $study_ids
            OPTIONAL MATCH (s)-[:STUDIES]->(o:Organism)
            OPTIONAL MATCH (s)-[:HAS_FACTOR]->(f:Factor)
            RETURN s.id as id, s.title as title, s.description as description,
                   s.research_type as research_type, s.methodology as methodology,
                   s.mission_relevance as mission_relevance, s.source as source,
                   s.publication_date as publication_date,
                   collect(DISTINCT o.name) as organisms,
                   collect(DISTINCT f.name) as factors
            """

            results = execute_query(query, {'study_ids': study_ids})

            study_details = {}
            for record in results:
                study_details[record['id']] = {
                    'id': record['id'],
                    'title': record['title'],
                    'description': record['description'],
                    'research_type': record['research_type'],
                    'methodology': record['methodology'],
                    'mission_relevance': record['mission_relevance'],
                    'source': record['source'],
                    'publication_date': record['publication_date'],
                    'organisms': [o for o in record['organisms'] if o],
                    'factors': [f for f in record['factors'] if f]
                }

            return study_details

        except Exception as e:
            logger.error(f"Failed to get study details: {e}")
            return {}

    async def _generate_search_response(self, query: str, studies: List[Dict[str, Any]]) -> str:
        """Generate response for search queries."""
        try:
            # Prepare context for LLM
            studies_context = "\n".join([
                f"Study: {study['title']}\n"
                f"Type: {study.get('research_type', 'Unknown')}\n"
                f"Organisms: {', '.join(study.get('organisms', []))}\n"
                f"Summary: {study.get('description', '')[:200]}...\n"
                for study in studies[:3]  # Limit to top 3 for context
            ])

            prompt = f"""
            Based on the following space biology research studies, answer the user's query: "{query}"

            Relevant Studies:
            {studies_context}

            Please provide a comprehensive answer that:
            1. Directly addresses the user's question
            2. Synthesizes findings from the relevant studies
            3. Highlights key insights for space missions
            4. Mentions any important limitations or gaps

            Response:
            """

            response = await self.llm_service.generate_text(prompt, max_tokens=400)
            return response

        except Exception as e:
            logger.error(f"Failed to generate search response: {e}")
            return f"Found {len(studies)} relevant studies, but I encountered an issue generating a detailed response."

    async def _generate_comparison_response(self, query: str, comparison_data: Dict[str, List[Dict]]) -> str:
        """Generate response for comparison queries."""
        try:
            entities = list(comparison_data.keys())
            comparison_context = ""

            for entity in entities:
                studies = comparison_data[entity]
                if studies:
                    comparison_context += f"\n{entity.upper()}:\n"
                    for study in studies[:2]:  # Top 2 studies per entity
                        comparison_context += f"- {study['title']}: {study.get('description', '')[:150]}...\n"

            prompt = f"""
            Compare the following entities in space biology research: {', '.join(entities)}

            Research context:
            {comparison_context}

            User query: "{query}"

            Please provide a detailed comparison that:
            1. Highlights similarities and differences
            2. Discusses implications for space missions
            3. Notes any gaps in research
            4. Provides practical insights

            Response:
            """

            response = await self.llm_service.generate_text(prompt, max_tokens=500)
            return response

        except Exception as e:
            logger.error(f"Failed to generate comparison response: {e}")
            return f"I found research on {', '.join(comparison_data.keys())}, but encountered an issue generating the comparison."

    async def _generate_explanation_response(self, query: str, studies: List[Dict[str, Any]],
                                           concepts: List[str]) -> str:
        """Generate detailed explanation response."""
        try:
            studies_context = "\n".join([
                f"- {study['title']}: {study.get('methodology', '')}"
                for study in studies[:3]
            ])

            concepts_context = ", ".join(concepts[:5])

            prompt = f"""
            Explain the space biology topic in this query: "{query}"

            Relevant research studies:
            {studies_context}

            Related concepts: {concepts_context}

            Please provide a clear, scientific explanation that:
            1. Explains the underlying mechanisms or processes
            2. References relevant research findings
            3. Discusses practical implications for space missions
            4. Uses accessible language while maintaining scientific accuracy

            Explanation:
            """

            response = await self.llm_service.generate_text(prompt, max_tokens=500)
            return response

        except Exception as e:
            logger.error(f"Failed to generate explanation response: {e}")
            return "I found relevant research but encountered an issue generating a detailed explanation."

    async def _generate_research_summary(self, topic: str, studies: List[Dict[str, Any]]) -> str:
        """Generate comprehensive research summary."""
        try:
            # Group studies by research type
            research_types = {}
            for study in studies:
                r_type = study.get('research_type', 'Unknown')
                if r_type not in research_types:
                    research_types[r_type] = []
                research_types[r_type].append(study)

            summary_context = f"Topic: {topic}\n\n"
            for r_type, type_studies in research_types.items():
                summary_context += f"{r_type.upper()} ({len(type_studies)} studies):\n"
                for study in type_studies[:2]:
                    summary_context += f"- {study['title']}\n"
                summary_context += "\n"

            prompt = f"""
            Provide a comprehensive research summary for: {topic}

            Research overview:
            {summary_context}

            Please create a summary that includes:
            1. Current state of research in this area
            2. Key findings and consensus
            3. Important gaps or contradictions
            4. Implications for space missions
            5. Future research directions

            Summary:
            """

            response = await self.llm_service.generate_text(prompt, max_tokens=600)
            return response

        except Exception as e:
            logger.error(f"Failed to generate research summary: {e}")
            return f"I found {len(studies)} studies on {topic}, but encountered an issue generating the summary."

    async def _generate_general_response(self, query: str, studies: List[Dict[str, Any]]) -> str:
        """Generate general research response."""
        try:
            prompt = f"""
            Answer this space biology research question: "{query}"

            Based on available research, provide a helpful response that:
            1. Addresses the question directly
            2. Provides scientific context
            3. Mentions relevant research findings
            4. Discusses space mission implications

            Response:
            """

            response = await self.llm_service.generate_text(prompt, max_tokens=400)
            return response

        except Exception as e:
            logger.error(f"Failed to generate general response: {e}")
            return "I understand your question but encountered an issue generating a detailed response."

    async def _generate_fallback_response(self, query: str) -> str:
        """Generate fallback response when no studies are found."""
        try:
            prompt = f"""
            A user asked about space biology research: "{query}"

            No specific studies were found in our database. Provide a helpful response that:
            1. Acknowledges the topic is important
            2. Provides general scientific context about the topic
            3. Suggests what kind of research would be valuable
            4. Encourages the user to try different search terms

            Response:
            """

            response = await self.llm_service.generate_text(prompt, max_tokens=300)
            return response

        except Exception as e:
            logger.error(f"Failed to generate fallback response: {e}")
            return "I don't have specific research on this topic in our database. Try rephrasing your question or searching for related terms."

    async def _extract_comparison_entities(self, query: str) -> List[str]:
        """Extract entities to compare from the query."""
        # Simple pattern matching for comparison entities
        import re

        # Look for patterns like "X vs Y", "X and Y", "between X and Y"
        patterns = [
            r'(\w+(?:\s+\w+)*)\s+(?:vs|versus|and)\s+(\w+(?:\s+\w+)*)',
            r'between\s+(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)',
            r'compare\s+(\w+(?:\s+\w+)*)\s+(?:to|with)\s+(\w+(?:\s+\w+)*)'
        ]

        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, query.lower())
            for match in matches:
                entities.extend(match)

        # Clean up entities
        entities = [entity.strip() for entity in entities if entity.strip()]

        # If no patterns found, try to extract key nouns
        if not entities:
            # This is a simplified approach - in practice, you'd use NLP
            words = query.lower().split()
            potential_entities = [word for word in words
                                if word in ['human', 'mouse', 'rat', 'plant', 'bacteria',
                                           'microgravity', 'radiation', 'mars', 'moon', 'iss']]
            entities = potential_entities[:2]

        return entities[:4]  # Limit to 4 entities max

    async def _extract_summary_topic(self, query: str) -> str:
        """Extract the main topic to summarize from the query."""
        # Remove common summary words
        import re

        clean_query = re.sub(r'\b(summarize|summary|overview|brief|review)\b', '', query.lower())
        clean_query = re.sub(r'\b(of|on|about|the|a|an)\b', '', clean_query)
        clean_query = clean_query.strip()

        return clean_query if clean_query else query