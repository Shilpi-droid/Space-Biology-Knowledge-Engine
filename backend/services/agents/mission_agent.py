"""
Mission planning agent for space mission support in the Space Biology Knowledge Engine.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from .base_agent import BaseAgent
from config.database import execute_query

logger = logging.getLogger(__name__)


class MissionAgent(BaseAgent):
    """
    AI agent specialized in mission planning and space mission support.
    """

    def __init__(self):
        specialized_knowledge = {
            'mission_types': {
                'mars': {
                    'duration': '6-9 months transit + 18 months surface',
                    'key_risks': ['radiation', 'isolation', 'bone_loss', 'muscle_atrophy', 'psychological'],
                    'critical_systems': ['life_support', 'food_production', 'medical', 'exercise']
                },
                'moon': {
                    'duration': '3-30 days surface',
                    'key_risks': ['radiation', 'regolith_exposure', 'reduced_gravity'],
                    'critical_systems': ['EVA_suits', 'habitat', 'power_systems']
                },
                'iss': {
                    'duration': '6 months average',
                    'key_risks': ['microgravity_effects', 'radiation', 'air_quality'],
                    'critical_systems': ['life_support', 'exercise_equipment', 'research_facilities']
                }
            },
            'countermeasures': [
                'exercise_protocols', 'nutrition_supplements', 'medication',
                'psychological_support', 'radiation_shielding', 'artificial_gravity'
            ],
            'research_priorities': [
                'crew_health', 'life_support', 'food_systems', 'emergency_procedures',
                'equipment_reliability', 'human_factors'
            ]
        }
        super().__init__('mission', specialized_knowledge)

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None,
                          conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process mission planning related queries.

        Args:
            query: Mission planning query
            context: Mission context (mission_type, duration, etc.)
            conversation_id: Conversation tracking ID

        Returns:
            Mission agent response
        """
        try:
            # Validate input
            validation = await self.validate_input(query, context)
            if not validation['valid']:
                return self.format_response(
                    f"Invalid query: {', '.join(validation['errors'])}",
                    confidence=0.0
                )

            # Extract mission context
            mission_context = await self._extract_mission_context(query, context)
            logger.info(f"Mission agent processing query for {mission_context.get('mission_type', 'unknown')} mission")

            # Extract intent
            intent = await self.extract_intent(query)

            # Route to specific handler based on intent and mission context
            if 'risk' in query.lower() or 'assess' in intent['primary_intent']:
                response_data = await self._handle_risk_assessment(query, mission_context)
            elif 'countermeasure' in query.lower() or 'mitigation' in query.lower():
                response_data = await self._handle_countermeasure_query(query, mission_context)
            elif 'protocol' in query.lower() or 'procedure' in query.lower():
                response_data = await self._handle_protocol_query(query, mission_context)
            elif 'recommend' in intent['primary_intent']:
                response_data = await self._handle_recommendation_query(query, mission_context)
            else:
                response_data = await self._handle_general_mission_query(query, mission_context)

            # Track conversation
            if conversation_id:
                await self.track_conversation(
                    conversation_id, query, response_data['message'],
                    {'intent': intent, 'mission_context': mission_context}
                )

            # Generate follow-up questions
            follow_ups = await self.generate_follow_up_questions(query, response_data['message'])
            response_data['metadata']['follow_up_questions'] = follow_ups
            response_data['metadata']['mission_context'] = mission_context

            return response_data

        except Exception as e:
            logger.error(f"Mission agent query processing failed: {e}")
            return self.format_response(
                "I encountered an error while processing your mission planning query. Please try rephrasing your question.",
                confidence=0.0,
                metadata={'error': str(e)}
            )

    async def _extract_mission_context(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract mission context from query and provided context."""
        mission_context = {
            'mission_type': 'unknown',
            'duration': None,
            'crew_size': None,
            'launch_timeframe': None,
            'specific_objectives': []
        }

        # Extract from provided context
        if context:
            mission_context.update({
                'mission_type': context.get('mission_type', 'unknown'),
                'duration': context.get('duration'),
                'crew_size': context.get('crew_size'),
                'launch_timeframe': context.get('launch_date'),
                'specific_objectives': context.get('objectives', [])
            })

        # Extract from query text
        query_lower = query.lower()

        # Mission type detection
        if any(term in query_lower for term in ['mars', 'martian', 'red planet']):
            mission_context['mission_type'] = 'mars'
        elif any(term in query_lower for term in ['moon', 'lunar', 'artemis']):
            mission_context['mission_type'] = 'moon'
        elif any(term in query_lower for term in ['iss', 'space station', 'orbit']):
            mission_context['mission_type'] = 'iss'

        # Duration extraction
        import re
        duration_patterns = [
            r'(\d+)\s*(?:days?|weeks?|months?|years?)',
            r'(\d+)-(\d+)\s*(?:days?|weeks?|months?)',
        ]

        for pattern in duration_patterns:
            match = re.search(pattern, query_lower)
            if match:
                mission_context['duration'] = match.group(0)
                break

        return mission_context

    async def _handle_risk_assessment(self, query: str, mission_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle risk assessment queries."""
        try:
            mission_type = mission_context.get('mission_type', 'unknown')

            # Get mission-specific risks from knowledge
            mission_info = self.specialized_knowledge['mission_types'].get(mission_type, {})
            key_risks = mission_info.get('key_risks', [])

            # Find relevant studies for risk assessment
            risk_studies = await self._get_risk_related_studies(query, mission_type, key_risks)

            # Generate risk assessment
            assessment = await self._generate_risk_assessment(query, mission_context, risk_studies, key_risks)

            references = [study['id'] for study in risk_studies] if risk_studies else []

            return self.format_response(
                assessment,
                references=references,
                confidence=0.85,
                metadata={
                    'risk_categories': key_risks,
                    'studies_analyzed': len(risk_studies),
                    'mission_type': mission_type
                }
            )

        except Exception as e:
            logger.error(f"Risk assessment handling failed: {e}")
            return self.format_response(
                "I encountered an issue while assessing mission risks. Please provide more specific details about your mission parameters.",
                confidence=0.0
            )

    async def _handle_countermeasure_query(self, query: str, mission_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle countermeasure and mitigation queries."""
        try:
            mission_type = mission_context.get('mission_type', 'unknown')

            # Find countermeasure studies
            countermeasure_studies = await self._get_countermeasure_studies(query, mission_type)

            # Generate countermeasure recommendations
            recommendations = await self._generate_countermeasure_recommendations(
                query, mission_context, countermeasure_studies
            )

            references = [study['id'] for study in countermeasure_studies] if countermeasure_studies else []

            return self.format_response(
                recommendations,
                references=references,
                confidence=0.8,
                metadata={
                    'countermeasure_types': self.specialized_knowledge['countermeasures'],
                    'studies_found': len(countermeasure_studies)
                }
            )

        except Exception as e:
            logger.error(f"Countermeasure query handling failed: {e}")
            return self.format_response(
                "I encountered an issue while researching countermeasures. Please specify the particular risks or health concerns you want to address.",
                confidence=0.0
            )

    async def _handle_protocol_query(self, query: str, mission_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle protocol and procedure queries."""
        try:
            mission_type = mission_context.get('mission_type', 'unknown')

            # Find protocol-related studies
            protocol_studies = await self._get_protocol_studies(query, mission_type)

            # Generate protocol recommendations
            protocols = await self._generate_protocol_recommendations(
                query, mission_context, protocol_studies
            )

            references = [study['id'] for study in protocol_studies] if protocol_studies else []

            return self.format_response(
                protocols,
                references=references,
                confidence=0.8,
                metadata={'protocol_type': 'mission_specific', 'studies_referenced': len(protocol_studies)}
            )

        except Exception as e:
            logger.error(f"Protocol query handling failed: {e}")
            return self.format_response(
                "I encountered an issue while researching protocols. Please specify the type of procedure or protocol you need.",
                confidence=0.0
            )

    async def _handle_recommendation_query(self, query: str, mission_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general recommendation queries."""
        try:
            mission_type = mission_context.get('mission_type', 'unknown')

            # Get comprehensive studies for recommendations
            related_studies = await self.find_related_studies(query, limit=8)

            # Generate mission-specific recommendations
            recommendations = await self._generate_mission_recommendations(
                query, mission_context, related_studies
            )

            references = [study['id'] for study in related_studies] if related_studies else []

            return self.format_response(
                recommendations,
                references=references,
                confidence=0.8,
                metadata={'recommendation_type': 'mission_planning', 'mission_type': mission_type}
            )

        except Exception as e:
            logger.error(f"Recommendation query handling failed: {e}")
            return self.format_response(
                "I encountered an issue while generating recommendations. Please provide more specific mission requirements.",
                confidence=0.0
            )

    async def _handle_general_mission_query(self, query: str, mission_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general mission planning queries."""
        try:
            # Get relevant studies
            studies = await self.find_related_studies(query, limit=5)

            if studies:
                response = await self._generate_general_mission_response(query, mission_context, studies)
                references = [study['id'] for study in studies]
                confidence = 0.7
            else:
                response = await self._generate_mission_fallback_response(query, mission_context)
                references = []
                confidence = 0.5

            return self.format_response(
                response,
                references=references,
                confidence=confidence,
                metadata={'query_type': 'general_mission'}
            )

        except Exception as e:
            logger.error(f"General mission query handling failed: {e}")
            return self.format_response(
                "I'm having trouble processing your mission query. Could you please provide more specific details about your mission requirements?",
                confidence=0.0
            )

    async def find_related_studies(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find studies related to mission planning query."""
        try:
            # Get basic search results
            search_results = await self.get_relevant_studies(query, limit)

            # Enrich with study details from database
            if not search_results:
                return []

            study_ids = [result['id'] for result in search_results]

            # Query database for detailed study information with mission relevance
            detailed_studies = await self._get_mission_relevant_studies(study_ids)

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

    async def _get_mission_relevant_studies(self, study_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get detailed study information with mission relevance scoring."""
        try:
            query = """
            MATCH (s:Study)
            WHERE s.id IN $study_ids
            OPTIONAL MATCH (s)-[:STUDIES]->(o:Organism)
            OPTIONAL MATCH (s)-[:HAS_FACTOR]->(f:Factor)
            RETURN s.id as id, s.title as title, s.description as description,
                   s.research_type as research_type, s.methodology as methodology,
                   s.mission_relevance as mission_relevance, s.source as source,
                   s.publication_date as publication_date, s.sample_size as sample_size,
                   collect(DISTINCT o.name) as organisms,
                   collect(DISTINCT f.name) as factors
            """

            results = execute_query(query, {'study_ids': study_ids})

            study_details = {}
            for record in results:
                # Calculate mission relevance score
                relevance_score = self._calculate_mission_relevance(record)

                study_details[record['id']] = {
                    'id': record['id'],
                    'title': record['title'],
                    'description': record['description'],
                    'research_type': record['research_type'],
                    'methodology': record['methodology'],
                    'mission_relevance': record['mission_relevance'],
                    'source': record['source'],
                    'publication_date': record['publication_date'],
                    'sample_size': record['sample_size'],
                    'organisms': [o for o in record['organisms'] if o],
                    'factors': [f for f in record['factors'] if f],
                    'calculated_relevance': relevance_score
                }

            return study_details

        except Exception as e:
            logger.error(f"Failed to get mission relevant studies: {e}")
            return {}

    def _calculate_mission_relevance(self, study_record: Dict[str, Any]) -> float:
        """Calculate mission relevance score for a study."""
        score = 0.0

        # Base score from existing mission_relevance field
        mission_rel = study_record.get('mission_relevance', '').lower()
        if mission_rel in ['mars', 'moon', 'iss']:
            score += 0.4
        elif mission_rel in ['space', 'microgravity', 'spaceflight']:
            score += 0.3

        # Score based on organisms
        organisms = study_record.get('organisms', [])
        human_terms = ['human', 'astronaut', 'crew', 'homo sapiens']
        if any(any(term in org.lower() for term in human_terms) for org in organisms):
            score += 0.3
        elif organisms:  # Any model organism
            score += 0.2

        # Score based on factors
        factors = study_record.get('factors', [])
        mission_factors = ['microgravity', 'radiation', 'isolation', 'confinement', 'stress']
        factor_matches = sum(1 for factor in factors
                           for mission_factor in mission_factors
                           if mission_factor in factor.lower())
        score += min(0.3, factor_matches * 0.1)

        return min(1.0, score)

    async def _get_risk_related_studies(self, query: str, mission_type: str,
                                      key_risks: List[str]) -> List[Dict[str, Any]]:
        """Get studies related to specific mission risks."""
        try:
            # Expand query with risk-related terms
            risk_query = f"{query} {' '.join(key_risks)} {mission_type} health effects"
            return await self.find_related_studies(risk_query, limit=8)
        except Exception as e:
            logger.error(f"Failed to get risk-related studies: {e}")
            return []

    async def _get_countermeasure_studies(self, query: str, mission_type: str) -> List[Dict[str, Any]]:
        """Get studies related to countermeasures and mitigation strategies."""
        try:
            countermeasure_query = f"{query} countermeasure mitigation exercise nutrition {mission_type}"
            return await self.find_related_studies(countermeasure_query, limit=6)
        except Exception as e:
            logger.error(f"Failed to get countermeasure studies: {e}")
            return []

    async def _get_protocol_studies(self, query: str, mission_type: str) -> List[Dict[str, Any]]:
        """Get studies related to protocols and procedures."""
        try:
            protocol_query = f"{query} protocol procedure guidelines {mission_type} operations"
            return await self.find_related_studies(protocol_query, limit=6)
        except Exception as e:
            logger.error(f"Failed to get protocol studies: {e}")
            return []

    async def _generate_risk_assessment(self, query: str, mission_context: Dict[str, Any],
                                      studies: List[Dict[str, Any]], key_risks: List[str]) -> str:
        """Generate comprehensive risk assessment."""
        try:
            mission_type = mission_context.get('mission_type', 'unknown')
            duration = mission_context.get('duration', 'unspecified')

            studies_context = "\n".join([
                f"Study: {study['title']}\n"
                f"Relevance: {study.get('calculated_relevance', 0):.2f}\n"
                f"Findings: {study.get('description', '')[:200]}...\n"
                for study in studies[:4]
            ])

            prompt = f"""
            Conduct a comprehensive risk assessment for a {mission_type} mission.

            Mission Parameters:
            - Mission Type: {mission_type}
            - Duration: {duration}
            - Key Risk Areas: {', '.join(key_risks)}

            User Query: "{query}"

            Supporting Research:
            {studies_context}

            Provide a detailed risk assessment that includes:
            1. Primary health risks specific to this mission type
            2. Risk severity levels (High/Medium/Low) with justification
            3. Timeline of when risks emerge during the mission
            4. Crew population most at risk
            5. Evidence-based risk mitigation priorities
            6. Uncertainties and knowledge gaps

            Focus on actionable insights for mission planners and medical officers.
            """

            return await self.llm_service.generate_text(prompt, max_tokens=600)

        except Exception as e:
            logger.error(f"Failed to generate risk assessment: {e}")
            return f"Based on available research, this {mission_context.get('mission_type', '')} mission presents several health risks that require careful consideration and mitigation planning."

    async def _generate_countermeasure_recommendations(self, query: str, mission_context: Dict[str, Any],
                                                     studies: List[Dict[str, Any]]) -> str:
        """Generate countermeasure recommendations."""
        try:
            mission_type = mission_context.get('mission_type', 'unknown')

            studies_context = "\n".join([
                f"Study: {study['title']}\n"
                f"Methodology: {study.get('methodology', 'Not specified')}\n"
                f"Key Points: {study.get('description', '')[:200]}...\n"
                for study in studies[:3]
            ])

            prompt = f"""
            Generate evidence-based countermeasure recommendations for a {mission_type} mission.

            User Query: "{query}"

            Research Evidence:
            {studies_context}

            Provide recommendations that include:
            1. Primary countermeasures with evidence levels
            2. Implementation timeline and protocols
            3. Resource requirements (equipment, time, crew training)
            4. Effectiveness metrics and success criteria
            5. Alternative approaches if primary countermeasures fail
            6. Integration with existing mission operations

            Prioritize practical, mission-ready solutions.
            """

            return await self.llm_service.generate_text(prompt, max_tokens=500)

        except Exception as e:
            logger.error(f"Failed to generate countermeasure recommendations: {e}")
            return f"Based on available research, several countermeasures should be considered for {mission_context.get('mission_type', '')} missions."

    async def _generate_protocol_recommendations(self, query: str, mission_context: Dict[str, Any],
                                               studies: List[Dict[str, Any]]) -> str:
        """Generate protocol and procedure recommendations."""
        try:
            mission_type = mission_context.get('mission_type', 'unknown')

            prompt = f"""
            Generate specific protocols and procedures for a {mission_type} mission based on the query: "{query}"

            Mission Context: {mission_context}

            Create detailed protocols that include:
            1. Step-by-step procedures
            2. Crew roles and responsibilities
            3. Equipment and resource requirements
            4. Timeline and frequency
            5. Success criteria and monitoring
            6. Emergency procedures and contingencies

            Base recommendations on current best practices and research evidence.
            """

            return await self.llm_service.generate_text(prompt, max_tokens=500)

        except Exception as e:
            logger.error(f"Failed to generate protocol recommendations: {e}")
            return f"Protocol recommendations for {mission_context.get('mission_type', '')} missions should be developed based on current research and operational requirements."

    async def _generate_mission_recommendations(self, query: str, mission_context: Dict[str, Any],
                                              studies: List[Dict[str, Any]]) -> str:
        """Generate general mission planning recommendations."""
        try:
            mission_type = mission_context.get('mission_type', 'unknown')

            prompt = f"""
            Provide comprehensive mission planning recommendations for: "{query}"

            Mission Type: {mission_type}
            Mission Context: {mission_context}

            Generate recommendations covering:
            1. Critical mission planning considerations
            2. Crew health and safety priorities
            3. Resource allocation suggestions
            4. Timeline and operational considerations
            5. Risk mitigation strategies
            6. Success factors and metrics

            Focus on actionable insights for mission planners.
            """

            return await self.llm_service.generate_text(prompt, max_tokens=500)

        except Exception as e:
            logger.error(f"Failed to generate mission recommendations: {e}")
            return f"Mission planning for {mission_context.get('mission_type', '')} missions requires careful consideration of multiple factors including crew health, resource management, and operational efficiency."

    async def _generate_general_mission_response(self, query: str, mission_context: Dict[str, Any],
                                               studies: List[Dict[str, Any]]) -> str:
        """Generate general mission planning response."""
        try:
            prompt = f"""
            Answer this mission planning question: "{query}"

            Mission Context: {mission_context}

            Provide a comprehensive response that:
            1. Directly addresses the question
            2. Incorporates relevant research findings
            3. Considers mission-specific constraints
            4. Offers practical implementation guidance
            5. Identifies key decision points

            Base your response on current space biology research and mission planning best practices.
            """

            return await self.llm_service.generate_text(prompt, max_tokens=400)

        except Exception as e:
            logger.error(f"Failed to generate general mission response: {e}")
            return "I understand your mission planning question. Based on current research and best practices, there are several important factors to consider for successful mission execution."

    async def _generate_mission_fallback_response(self, query: str, mission_context: Dict[str, Any]) -> str:
        """Generate fallback response when no specific studies are found."""
        try:
            mission_type = mission_context.get('mission_type', 'space')

            prompt = f"""
            A mission planner asked: "{query}"

            Mission context: {mission_type} mission

            No specific research studies were found in our database. Provide a helpful response that:
            1. Acknowledges the importance of the question
            2. Provides general guidance based on mission planning principles
            3. Suggests what type of research would be valuable
            4. Offers alternative approaches or considerations

            Focus on practical mission planning insights.
            """

            return await self.llm_service.generate_text(prompt, max_tokens=300)

        except Exception as e:
            logger.error(f"Failed to generate mission fallback response: {e}")
            return f"Your question about {mission_context.get('mission_type', 'space')} mission planning is important. While I don't have specific research studies on this topic in our database, I recommend consulting current mission planning guidelines and considering the unique requirements of your mission profile."