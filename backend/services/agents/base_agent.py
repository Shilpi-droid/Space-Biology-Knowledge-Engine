"""
Base agent interface for AI agents in the Space Biology Knowledge Engine.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from services.llm_service import LLMService
from services.semantic_search import SemanticSearchService
from services.knowledge_graph import KnowledgeGraphService

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Manages conversation context and memory for agents."""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversations = {}

    def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation history."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        self.conversations[conversation_id].append(message)

        # Maintain max history
        if len(self.conversations[conversation_id]) > self.max_history:
            self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_history:]

    def get_conversation_context(self, conversation_id: str, last_n: Optional[int] = None) -> List[Dict]:
        """Get conversation context for the given conversation."""
        if conversation_id not in self.conversations:
            return []

        messages = self.conversations[conversation_id]
        if last_n:
            messages = messages[-last_n:]

        return messages

    def get_conversation_summary(self, conversation_id: str) -> str:
        """Generate a summary of the conversation."""
        messages = self.get_conversation_context(conversation_id)
        if not messages:
            return "No conversation history."

        # Create a simple summary
        user_messages = [m['content'] for m in messages if m['role'] == 'user']
        agent_messages = [m['content'] for m in messages if m['role'] == 'agent']

        summary = f"Conversation with {len(user_messages)} user messages and {len(agent_messages)} agent responses."
        if user_messages:
            summary += f" Recent topics: {', '.join(user_messages[-3:])}"

        return summary


class BaseAgent(ABC):
    """
    Abstract base class for AI agents in the Space Biology Knowledge Engine.
    """

    def __init__(self, agent_type: str, specialized_knowledge: Optional[Dict[str, Any]] = None):
        self.agent_type = agent_type
        self.specialized_knowledge = specialized_knowledge or {}
        self.llm_service = LLMService()
        self.search_service = SemanticSearchService()
        self.knowledge_graph = KnowledgeGraphService()
        self.memory = ConversationMemory()

    @abstractmethod
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None,
                          conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query and generate a response.

        Args:
            query: User query string
            context: Additional context for the query
            conversation_id: ID for conversation tracking

        Returns:
            Dictionary with response and metadata
        """
        pass

    async def validate_input(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate input query and context.

        Args:
            query: User query
            context: Query context

        Returns:
            Validation result with status and any errors
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Check query length
        if not query or len(query.strip()) < 2:
            validation_result['valid'] = False
            validation_result['errors'].append("Query is too short or empty")

        if len(query) > 2000:
            validation_result['valid'] = False
            validation_result['errors'].append("Query is too long (max 2000 characters)")

        # Check for inappropriate content
        inappropriate_terms = ['hack', 'exploit', 'malicious', 'attack']
        if any(term in query.lower() for term in inappropriate_terms):
            validation_result['valid'] = False
            validation_result['errors'].append("Query contains inappropriate content")

        # Check context if provided
        if context:
            if not isinstance(context, dict):
                validation_result['warnings'].append("Context should be a dictionary")

        return validation_result

    def format_response(self, content: str, references: Optional[List[str]] = None,
                       confidence: float = 0.8, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format agent response with consistent structure.

        Args:
            content: Response content
            references: List of reference IDs
            confidence: Confidence score for the response
            metadata: Additional metadata

        Returns:
            Formatted response dictionary
        """
        return {
            'message': content,
            'agent_type': self.agent_type,
            'references': references or [],
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

    async def track_conversation(self, conversation_id: str, user_query: str,
                               agent_response: str, metadata: Optional[Dict] = None):
        """
        Track conversation for context and learning.

        Args:
            conversation_id: Conversation identifier
            user_query: User's query
            agent_response: Agent's response
            metadata: Additional metadata
        """
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        # Add user message
        self.memory.add_message(conversation_id, 'user', user_query, metadata)

        # Add agent response
        self.memory.add_message(conversation_id, 'agent', agent_response, metadata)

        logger.info(f"Tracked conversation {conversation_id} for agent {self.agent_type}")

        return conversation_id

    async def get_relevant_studies(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant studies for the query using semantic search.

        Args:
            query: Search query
            limit: Maximum number of studies to return

        Returns:
            List of relevant studies
        """
        try:
            # Perform semantic search
            search_results = await self.search_service.semantic_search(query, k=limit)

            # Get detailed study information
            relevant_studies = []
            for result in search_results:
                # This would be enriched with actual study data from the knowledge graph
                study_info = {
                    'id': result['document_id'],
                    'similarity_score': result['similarity_score'],
                    'rank': result['rank']
                }
                relevant_studies.append(study_info)

            return relevant_studies

        except Exception as e:
            logger.error(f"Failed to get relevant studies: {e}")
            return []

    async def get_related_concepts(self, concept: str) -> List[str]:
        """
        Get related concepts from the knowledge graph.

        Args:
            concept: Main concept to find relations for

        Returns:
            List of related concepts
        """
        try:
            # Use knowledge graph to find related concepts
            # This is a simplified implementation
            related_concepts = []

            # Search for organisms related to the concept
            if 'organism' in concept.lower() or 'species' in concept.lower():
                # Query for related organisms
                related_concepts.extend(['microgravity effects', 'space adaptation', 'physiological changes'])

            # Search for factors related to the concept
            if any(term in concept.lower() for term in ['radiation', 'gravity', 'environment']):
                related_concepts.extend(['health risks', 'countermeasures', 'protection systems'])

            return related_concepts[:10]  # Limit to top 10

        except Exception as e:
            logger.error(f"Failed to get related concepts: {e}")
            return []

    async def extract_intent(self, query: str) -> Dict[str, Any]:
        """
        Extract user intent from the query.

        Args:
            query: User query

        Returns:
            Dictionary with intent classification
        """
        intent_patterns = {
            'search': ['find', 'search', 'look for', 'show me', 'what is'],
            'compare': ['compare', 'difference', 'versus', 'vs', 'contrast'],
            'explain': ['explain', 'why', 'how does', 'what causes', 'mechanism'],
            'recommend': ['recommend', 'suggest', 'advice', 'should', 'what would'],
            'analyze': ['analyze', 'assessment', 'evaluate', 'risk', 'impact'],
            'summarize': ['summarize', 'summary', 'overview', 'brief', 'key points']
        }

        query_lower = query.lower()
        detected_intents = []

        for intent, patterns in intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                detected_intents.append(intent)

        # Default to search if no specific intent detected
        if not detected_intents:
            detected_intents = ['search']

        return {
            'primary_intent': detected_intents[0],
            'all_intents': detected_intents,
            'confidence': 0.8 if len(detected_intents) == 1 else 0.6
        }

    async def generate_follow_up_questions(self, query: str, response: str) -> List[str]:
        """
        Generate relevant follow-up questions based on the query and response.

        Args:
            query: Original user query
            response: Agent response

        Returns:
            List of follow-up questions
        """
        try:
            prompt = f"""
            Based on this space biology query and response, suggest 3 relevant follow-up questions:

            Query: {query}
            Response: {response}

            Generate questions that would help the user explore related topics or get more specific information.
            Focus on practical applications for space missions.
            """

            follow_up_text = await self.llm_service.generate_text(prompt, max_tokens=200)

            # Parse questions from the response
            questions = []
            lines = follow_up_text.split('\n')

            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith(('"', '-', '*'))):
                    question = line.lstrip('0123456789."-* ').strip()
                    if question and question.endswith('?'):
                        questions.append(question)

            return questions[:3]  # Return top 3 questions

        except Exception as e:
            logger.error(f"Failed to generate follow-up questions: {e}")
            return []

    async def assess_response_quality(self, query: str, response: str) -> Dict[str, Any]:
        """
        Assess the quality of the generated response.

        Args:
            query: Original query
            response: Generated response

        Returns:
            Quality assessment
        """
        assessment = {
            'completeness': 0.0,
            'relevance': 0.0,
            'accuracy': 0.0,
            'clarity': 0.0,
            'overall_score': 0.0
        }

        try:
            # Simple heuristic-based assessment
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())

            # Relevance: check overlap between query and response
            word_overlap = len(query_words.intersection(response_words))
            assessment['relevance'] = min(1.0, word_overlap / len(query_words))

            # Completeness: check response length relative to query complexity
            response_length = len(response.split())
            if response_length >= 50:
                assessment['completeness'] = 0.9
            elif response_length >= 20:
                assessment['completeness'] = 0.7
            else:
                assessment['completeness'] = 0.5

            # Clarity: check for clear structure and readability
            if '.' in response and len(response.split('.')) >= 2:
                assessment['clarity'] = 0.8
            else:
                assessment['clarity'] = 0.6

            # Accuracy: placeholder (would need domain-specific validation)
            assessment['accuracy'] = 0.8

            # Calculate overall score
            assessment['overall_score'] = sum(assessment.values()) / 4

            return assessment

        except Exception as e:
            logger.error(f"Failed to assess response quality: {e}")
            return assessment

    def get_agent_capabilities(self) -> Dict[str, Any]:
        """
        Get information about agent capabilities.

        Returns:
            Dictionary with agent capabilities and limitations
        """
        return {
            'agent_type': self.agent_type,
            'capabilities': [
                'Natural language query processing',
                'Semantic search across research literature',
                'Knowledge graph traversal',
                'Context-aware responses',
                'Conversation memory'
            ],
            'limitations': [
                'Responses based on available data only',
                'Cannot access real-time information',
                'May require clarification for ambiguous queries'
            ],
            'specialized_knowledge': list(self.specialized_knowledge.keys()) if self.specialized_knowledge else []
        }