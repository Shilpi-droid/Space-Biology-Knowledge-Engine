"""
Risk assessment agent for health and safety evaluation in the Space Biology Knowledge Engine.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .base_agent import BaseAgent
from config.database import execute_query

logger = logging.getLogger(__name__)


class RiskAgent(BaseAgent):
    """
    AI agent specialized in health and safety risk assessment for space missions.
    """

    def __init__(self):
        specialized_knowledge = {
            'risk_categories': {
                'physiological': {
                    'bone_loss': {'severity': 'high', 'timeline': 'weeks', 'reversible': 'partial'},
                    'muscle_atrophy': {'severity': 'high', 'timeline': 'days', 'reversible': 'mostly'},
                    'cardiovascular_deconditioning': {'severity': 'high', 'timeline': 'days', 'reversible': 'mostly'},
                    'vision_changes': {'severity': 'medium', 'timeline': 'months', 'reversible': 'unknown'},
                    'kidney_stones': {'severity': 'medium', 'timeline': 'weeks', 'reversible': 'yes'}
                },
                'radiation': {
                    'acute_exposure': {'severity': 'critical', 'timeline': 'immediate', 'reversible': 'no'},
                    'cancer_risk': {'severity': 'high', 'timeline': 'years', 'reversible': 'no'},
                    'cns_effects': {'severity': 'medium', 'timeline': 'months', 'reversible': 'partial'},
                    'cataracts': {'severity': 'medium', 'timeline': 'years', 'reversible': 'no'}
                },
                'psychological': {
                    'isolation_stress': {'severity': 'high', 'timeline': 'weeks', 'reversible': 'yes'},
                    'sleep_disorders': {'severity': 'medium', 'timeline': 'days', 'reversible': 'yes'},
                    'cognitive_decline': {'severity': 'medium', 'timeline': 'months', 'reversible': 'partial'},
                    'depression': {'severity': 'high', 'timeline': 'weeks', 'reversible': 'yes'}
                },
                'operational': {
                    'equipment_failure': {'severity': 'critical', 'timeline': 'immediate', 'reversible': 'depends'},
                    'communication_loss': {'severity': 'high', 'timeline': 'immediate', 'reversible': 'yes'},
                    'supply_shortage': {'severity': 'high', 'timeline': 'immediate', 'reversible': 'depends'},
                    'medical_emergency': {'severity': 'critical', 'timeline': 'immediate', 'reversible': 'depends'}
                }
            },
            'risk_factors': {
                'mission_duration': ['acute', 'short_term', 'long_term', 'chronic'],
                'crew_demographics': ['age', 'gender', 'health_history', 'experience'],
                'environmental': ['radiation_level', 'gravity_level', 'atmosphere', 'temperature'],
                'operational': ['workload', 'stress_level', 'sleep_schedule', 'nutrition']
            },
            'mitigation_strategies': {
                'exercise': ['COLPA', 'resistance_training', 'cardiovascular', 'bone_loading'],
                'pharmacological': ['bisphosphonates', 'antioxidants', 'sleep_aids', 'anti_nausea'],
                'nutritional': ['vitamin_d', 'calcium', 'protein', 'antioxidants'],
                'operational': ['workload_management', 'sleep_hygiene', 'communication_protocols']
            }
        }
        super().__init__('risk', specialized_knowledge)

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None,
                          conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process risk assessment queries.

        Args:
            query: Risk assessment query
            context: Risk assessment context
            conversation_id: Conversation tracking ID

        Returns:
            Risk agent response
        """
        try:
            # Validate input
            validation = await self.validate_input(query, context)
            if not validation['valid']:
                return self.format_response(
                    f"Invalid query: {', '.join(validation['errors'])}",
                    confidence=0.0
                )

            # Extract risk context
            risk_context = await self._extract_risk_context(query, context)
            logger.info(f"Risk agent processing query for {risk_context.get('risk_type', 'general')} risk assessment")

            # Extract intent
            intent = await self.extract_intent(query)

            # Route to specific handler based on intent and risk context
            if 'assess' in intent['primary_intent'] or 'evaluate' in query.lower():
                response_data = await self._handle_risk_assessment(query, risk_context)
            elif 'compare' in intent['primary_intent']:
                response_data = await self._handle_risk_comparison(query, risk_context)
            elif 'monitor' in query.lower() or 'track' in query.lower():
                response_data = await self._handle_monitoring_query(query, risk_context)
            elif 'emergency' in query.lower() or 'protocol' in query.lower():
                response_data = await self._handle_emergency_protocol_query(query, risk_context)
            elif 'timeline' in query.lower() or 'when' in query.lower():
                response_data = await self._handle_temporal_risk_query(query, risk_context)
            else:
                response_data = await self._handle_general_risk_query(query, risk_context)

            # Track conversation
            if conversation_id:
                await self.track_conversation(
                    conversation_id, query, response_data['message'],
                    {'intent': intent, 'risk_context': risk_context}
                )

            # Generate follow-up questions
            follow_ups = await self.generate_follow_up_questions(query, response_data['message'])
            response_data['metadata']['follow_up_questions'] = follow_ups
            response_data['metadata']['risk_context'] = risk_context

            return response_data

        except Exception as e:
            logger.error(f"Risk agent query processing failed: {e}")
            return self.format_response(
                "I encountered an error while processing your risk assessment query. Please try rephrasing your question.",
                confidence=0.0,
                metadata={'error': str(e)}
            )

    async def _extract_risk_context(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract risk assessment context from query and provided context."""
        risk_context = {
            'risk_type': 'general',
            'mission_type': 'unknown',
            'mission_duration': None,
            'crew_size': None,
            'specific_risks': [],
            'risk_category': 'unknown'
        }

        # Extract from provided context
        if context:
            risk_context.update({
                'mission_type': context.get('mission_type', 'unknown'),
                'mission_duration': context.get('mission_duration'),
                'crew_size': context.get('crew_size'),
                'environmental_factors': context.get('environmental_factors', []),
                'health_baseline': context.get('health_baseline')
            })

        # Extract from query text
        query_lower = query.lower()

        # Risk category detection
        for category, risks in self.specialized_knowledge['risk_categories'].items():
            if category in query_lower or any(risk in query_lower for risk in risks.keys()):
                risk_context['risk_category'] = category
                break

        # Specific risk detection
        for category, risks in self.specialized_knowledge['risk_categories'].items():
            for risk_name in risks.keys():
                if risk_name.replace('_', ' ') in query_lower:
                    risk_context['specific_risks'].append(risk_name)

        # Mission type detection
        if any(term in query_lower for term in ['mars', 'martian']):
            risk_context['mission_type'] = 'mars'
        elif any(term in query_lower for term in ['moon', 'lunar']):
            risk_context['mission_type'] = 'moon'
        elif any(term in query_lower for term in ['iss', 'station']):
            risk_context['mission_type'] = 'iss'

        return risk_context

    async def _handle_risk_assessment(self, query: str, risk_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle comprehensive risk assessment queries."""
        try:
            # Get relevant risk studies
            risk_studies = await self._get_risk_studies(query, risk_context)

            # Perform quantitative risk assessment
            risk_analysis = await self._perform_risk_analysis(risk_context, risk_studies)

            # Generate comprehensive assessment
            assessment = await self._generate_comprehensive_risk_assessment(
                query, risk_context, risk_studies, risk_analysis
            )

            references = [study['id'] for study in risk_studies] if risk_studies else []

            return self.format_response(
                assessment,
                references=references,
                confidence=0.85,
                metadata={
                    'risk_analysis': risk_analysis,
                    'studies_analyzed': len(risk_studies),
                    'risk_category': risk_context.get('risk_category')
                }
            )

        except Exception as e:
            logger.error(f"Risk assessment handling failed: {e}")
            return self.format_response(
                "I encountered an issue while conducting the risk assessment. Please provide more specific details about the risks you want to evaluate.",
                confidence=0.0
            )

    async def _handle_risk_comparison(self, query: str, risk_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle risk comparison queries."""
        try:
            # Extract risks to compare
            risks_to_compare = await self._extract_comparison_risks(query)

            if len(risks_to_compare) < 2:
                return self.format_response(
                    "For risk comparisons, please specify at least two risks to compare. For example: 'Compare bone loss risk versus radiation exposure risk for Mars missions'",
                    confidence=0.4
                )

            # Get studies for each risk
            comparison_data = {}
            all_references = []

            for risk in risks_to_compare:
                risk_query = f"{risk} {risk_context.get('mission_type', '')} space"
                risk_studies = await self._get_risk_studies(risk_query, risk_context)
                comparison_data[risk] = risk_studies
                all_references.extend([s['id'] for s in risk_studies])

            # Generate comparison analysis
            comparison = await self._generate_risk_comparison(query, risks_to_compare, comparison_data, risk_context)

            return self.format_response(
                comparison,
                references=list(set(all_references)),
                confidence=0.8,
                metadata={'risks_compared': risks_to_compare}
            )

        except Exception as e:
            logger.error(f"Risk comparison handling failed: {e}")
            return self.format_response(
                "I encountered an issue while comparing risks. Please specify which risks you want to compare.",
                confidence=0.0
            )

    async def _handle_monitoring_query(self, query: str, risk_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle risk monitoring and tracking queries."""
        try:
            # Get monitoring-related studies
            monitoring_studies = await self._get_monitoring_studies(query, risk_context)

            # Generate monitoring recommendations
            monitoring_plan = await self._generate_monitoring_recommendations(
                query, risk_context, monitoring_studies
            )

            references = [study['id'] for study in monitoring_studies] if monitoring_studies else []

            return self.format_response(
                monitoring_plan,
                references=references,
                confidence=0.8,
                metadata={'monitoring_type': 'health_tracking', 'studies_found': len(monitoring_studies)}
            )

        except Exception as e:
            logger.error(f"Monitoring query handling failed: {e}")
            return self.format_response(
                "I encountered an issue while developing monitoring recommendations. Please specify which health parameters you want to monitor.",
                confidence=0.0
            )

    async def _handle_emergency_protocol_query(self, query: str, risk_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emergency protocol and response queries."""
        try:
            # Get emergency protocol studies
            protocol_studies = await self._get_emergency_protocol_studies(query, risk_context)

            # Generate emergency protocols
            protocols = await self._generate_emergency_protocols(query, risk_context, protocol_studies)

            references = [study['id'] for study in protocol_studies] if protocol_studies else []

            return self.format_response(
                protocols,
                references=references,
                confidence=0.8,
                metadata={'protocol_type': 'emergency_response', 'criticality': 'high'}
            )

        except Exception as e:
            logger.error(f"Emergency protocol query handling failed: {e}")
            return self.format_response(
                "I encountered an issue while developing emergency protocols. Please specify the type of emergency scenario you need protocols for.",
                confidence=0.0
            )

    async def _handle_temporal_risk_query(self, query: str, risk_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle temporal risk evolution queries."""
        try:
            # Get temporal risk studies
            temporal_studies = await self._get_temporal_risk_studies(query, risk_context)

            # Generate temporal risk analysis
            temporal_analysis = await self._generate_temporal_risk_analysis(
                query, risk_context, temporal_studies
            )

            references = [study['id'] for study in temporal_studies] if temporal_studies else []

            return self.format_response(
                temporal_analysis,
                references=references,
                confidence=0.8,
                metadata={'analysis_type': 'temporal_risk', 'timeline_focus': True}
            )

        except Exception as e:
            logger.error(f"Temporal risk query handling failed: {e}")
            return self.format_response(
                "I encountered an issue while analyzing temporal risk patterns. Please specify the timeframe you're interested in.",
                confidence=0.0
            )

    async def _handle_general_risk_query(self, query: str, risk_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general risk queries."""
        try:
            # Get relevant studies
            studies = await self._get_risk_studies(query, risk_context)

            if studies:
                response = await self._generate_general_risk_response(query, risk_context, studies)
                references = [study['id'] for study in studies]
                confidence = 0.7
            else:
                response = await self._generate_risk_fallback_response(query, risk_context)
                references = []
                confidence = 0.5

            return self.format_response(
                response,
                references=references,
                confidence=confidence,
                metadata={'query_type': 'general_risk'}
            )

        except Exception as e:
            logger.error(f"General risk query handling failed: {e}")
            return self.format_response(
                "I'm having trouble processing your risk query. Could you please provide more specific details about the health risks you're concerned about?",
                confidence=0.0
            )

    async def _get_risk_studies(self, query: str, risk_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get studies related to specific risks."""
        try:
            # Enhance query with risk-related terms
            risk_terms = []

            # Add specific risks if identified
            if risk_context.get('specific_risks'):
                risk_terms.extend(risk_context['specific_risks'])

            # Add category-related terms
            risk_category = risk_context.get('risk_category', '')
            if risk_category in self.specialized_knowledge['risk_categories']:
                risk_terms.extend(self.specialized_knowledge['risk_categories'][risk_category].keys())

            enhanced_query = f"{query} {' '.join(risk_terms)} health effects safety"

            return await self.get_relevant_studies(enhanced_query, limit=8)

        except Exception as e:
            logger.error(f"Failed to get risk studies: {e}")
            return []

    async def _perform_risk_analysis(self, risk_context: Dict[str, Any],
                                   studies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform quantitative risk analysis."""
        try:
            analysis = {
                'identified_risks': [],
                'risk_scores': {},
                'confidence_levels': {},
                'temporal_patterns': {},
                'mitigation_effectiveness': {}
            }

            # Analyze identified risks
            for risk_name in risk_context.get('specific_risks', []):
                risk_category = risk_context.get('risk_category', 'unknown')

                if risk_category in self.specialized_knowledge['risk_categories']:
                    risk_info = self.specialized_knowledge['risk_categories'][risk_category].get(risk_name, {})

                    # Calculate risk score based on severity, timeline, and reversibility
                    severity_score = self._get_severity_score(risk_info.get('severity', 'medium'))
                    timeline_score = self._get_timeline_score(risk_info.get('timeline', 'unknown'))
                    reversibility_score = self._get_reversibility_score(risk_info.get('reversible', 'unknown'))

                    # Mission-specific modifiers
                    mission_modifier = self._get_mission_risk_modifier(risk_context.get('mission_type', 'unknown'))

                    overall_score = (severity_score + timeline_score + reversibility_score) * mission_modifier

                    analysis['risk_scores'][risk_name] = min(1.0, overall_score)
                    analysis['confidence_levels'][risk_name] = self._calculate_confidence_from_studies(studies, risk_name)
                    analysis['identified_risks'].append({
                        'name': risk_name,
                        'category': risk_category,
                        'score': overall_score,
                        'severity': risk_info.get('severity'),
                        'timeline': risk_info.get('timeline'),
                        'reversible': risk_info.get('reversible')
                    })

            return analysis

        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return {'error': str(e)}

    def _get_severity_score(self, severity: str) -> float:
        """Convert severity level to numeric score."""
        severity_map = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.5,
            'low': 0.3,
            'minimal': 0.1
        }
        return severity_map.get(severity.lower(), 0.5)

    def _get_timeline_score(self, timeline: str) -> float:
        """Convert timeline to risk urgency score."""
        timeline_map = {
            'immediate': 1.0,
            'days': 0.8,
            'weeks': 0.6,
            'months': 0.4,
            'years': 0.2
        }
        return timeline_map.get(timeline.lower(), 0.5)

    def _get_reversibility_score(self, reversible: str) -> float:
        """Convert reversibility to risk permanence score."""
        reversibility_map = {
            'no': 1.0,
            'partial': 0.7,
            'mostly': 0.4,
            'yes': 0.2,
            'unknown': 0.6
        }
        return reversibility_map.get(reversible.lower(), 0.6)

    def _get_mission_risk_modifier(self, mission_type: str) -> float:
        """Get mission-specific risk modifier."""
        mission_modifiers = {
            'mars': 1.2,  # Higher risk due to duration and distance
            'moon': 1.0,  # Baseline risk
            'iss': 0.8,   # Lower risk due to proximity to Earth
            'unknown': 1.0
        }
        return mission_modifiers.get(mission_type.lower(), 1.0)

    def _calculate_confidence_from_studies(self, studies: List[Dict[str, Any]], risk_name: str) -> float:
        """Calculate confidence level based on available studies."""
        if not studies:
            return 0.3

        # Simple confidence calculation based on number of studies and their quality
        study_count = len(studies)

        # More studies = higher confidence, but with diminishing returns
        study_confidence = min(0.9, 0.4 + (study_count * 0.1))

        # Check for human studies (higher confidence)
        human_studies = sum(1 for study in studies
                          if any('human' in org.lower() for org in study.get('organisms', [])))

        if human_studies > 0:
            study_confidence += 0.1

        return min(0.95, study_confidence)

    async def _extract_comparison_risks(self, query: str) -> List[str]:
        """Extract risks to compare from the query."""
        import re

        risks = []

        # Look for specific risk mentions in the query
        for category, risk_dict in self.specialized_knowledge['risk_categories'].items():
            for risk_name in risk_dict.keys():
                risk_phrase = risk_name.replace('_', ' ')
                if risk_phrase in query.lower():
                    risks.append(risk_name)

        # Look for patterns like "X vs Y", "X and Y", "between X and Y"
        patterns = [
            r'(\w+(?:\s+\w+)*)\s+(?:vs|versus|and)\s+(\w+(?:\s+\w+)*)',
            r'between\s+(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)',
            r'compare\s+(\w+(?:\s+\w+)*)\s+(?:to|with)\s+(\w+(?:\s+\w+)*)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, query.lower())
            for match in matches:
                risks.extend([risk.strip() for risk in match if risk.strip()])

        return list(set(risks))[:4]  # Limit to 4 risks max

    async def _get_monitoring_studies(self, query: str, risk_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get studies related to health monitoring and biomarkers."""
        try:
            monitoring_query = f"{query} biomarker monitoring health tracking assessment"
            return await self.get_relevant_studies(monitoring_query, limit=6)
        except Exception as e:
            logger.error(f"Failed to get monitoring studies: {e}")
            return []

    async def _get_emergency_protocol_studies(self, query: str, risk_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get studies related to emergency protocols and responses."""
        try:
            emergency_query = f"{query} emergency protocol response procedure medical crisis"
            return await self.get_relevant_studies(emergency_query, limit=5)
        except Exception as e:
            logger.error(f"Failed to get emergency protocol studies: {e}")
            return []

    async def _get_temporal_risk_studies(self, query: str, risk_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get studies related to temporal risk evolution."""
        try:
            temporal_query = f"{query} timeline progression development onset duration"
            return await self.get_relevant_studies(temporal_query, limit=6)
        except Exception as e:
            logger.error(f"Failed to get temporal risk studies: {e}")
            return []

    # LLM Response Generation Methods
    async def _generate_comprehensive_risk_assessment(self, query: str, risk_context: Dict[str, Any],
                                                    studies: List[Dict[str, Any]], risk_analysis: Dict[str, Any]) -> str:
        """Generate comprehensive risk assessment response."""
        try:
            mission_type = risk_context.get('mission_type', 'unknown')
            risk_category = risk_context.get('risk_category', 'general')

            studies_context = "\n".join([
                f"Study: {study['title']}\n"
                f"Risk Relevance: {study.get('calculated_relevance', 0):.2f}\n"
                f"Findings: {study.get('description', '')[:200]}...\n"
                for study in studies[:4]
            ])

            prompt = f"""
            Conduct a comprehensive health risk assessment based on this query: "{query}"

            Mission Context:
            - Mission Type: {mission_type}
            - Risk Category: {risk_category}
            - Specific Risks: {', '.join(risk_context.get('specific_risks', []))}

            Risk Analysis Results:
            {risk_analysis}

            Supporting Research:
            {studies_context}

            Provide a detailed assessment that includes:
            1. Risk identification and categorization
            2. Quantitative risk levels with confidence intervals
            3. Timeline and progression patterns
            4. Population-specific vulnerabilities
            5. Evidence quality and certainty levels
            6. Critical knowledge gaps and uncertainties

            Focus on actionable medical insights for mission planners and flight surgeons.
            """

            return await self.llm_service.generate_text(prompt, max_tokens=700)

        except Exception as e:
            logger.error(f"Failed to generate comprehensive risk assessment: {e}")
            return f"Based on available research, several health risks have been identified for {mission_type} missions that require careful evaluation and monitoring."

    async def _generate_risk_comparison(self, query: str, risks: List[str],
                                      comparison_data: Dict[str, List[Dict]], risk_context: Dict[str, Any]) -> str:
        """Generate risk comparison analysis."""
        try:
            comparison_context = ""
            for risk in risks:
                studies = comparison_data.get(risk, [])
                comparison_context += f"\n{risk.upper().replace('_', ' ')}:\n"
                if studies:
                    for study in studies[:2]:
                        comparison_context += f"- {study['title']}: {study.get('description', '')[:150]}...\n"
                else:
                    comparison_context += "- Limited research data available\n"

            prompt = f"""
            Compare these health risks for space missions: {', '.join(risks)}

            Research Context:
            {comparison_context}

            User Query: "{query}"

            Provide a comprehensive comparison that includes:
            1. Relative risk severity and likelihood
            2. Onset timeline and progression patterns
            3. Reversibility and long-term consequences
            4. Prevention and mitigation options
            5. Monitoring and detection requirements
            6. Mission-specific risk variations

            Present the comparison with clear risk rankings and evidence-based recommendations.
            """

            return await self.llm_service.generate_text(prompt, max_tokens=600)

        except Exception as e:
            logger.error(f"Failed to generate risk comparison: {e}")
            return f"Based on available research, these risks ({', '.join(risks)}) present different challenges and require specific mitigation approaches."

    async def _generate_monitoring_recommendations(self, query: str, risk_context: Dict[str, Any],
                                                 studies: List[Dict[str, Any]]) -> str:
        """Generate health monitoring recommendations."""
        try:
            prompt = f"""
            Develop health monitoring recommendations for: "{query}"

            Risk Context: {risk_context}

            Create monitoring protocols that include:
            1. Key biomarkers and health indicators to track
            2. Monitoring frequency and methods
            3. Early warning thresholds and alert criteria
            4. Equipment and resource requirements
            5. Data collection and analysis procedures
            6. Response protocols for abnormal findings

            Base recommendations on current research and operational constraints.
            """

            return await self.llm_service.generate_text(prompt, max_tokens=500)

        except Exception as e:
            logger.error(f"Failed to generate monitoring recommendations: {e}")
            return "Health monitoring protocols should be established based on mission-specific risks and available resources."

    async def _generate_emergency_protocols(self, query: str, risk_context: Dict[str, Any],
                                          studies: List[Dict[str, Any]]) -> str:
        """Generate emergency response protocols."""
        try:
            prompt = f"""
            Develop emergency response protocols for: "{query}"

            Risk Context: {risk_context}

            Create protocols that include:
            1. Emergency recognition and triage procedures
            2. Immediate response actions and timelines
            3. Communication and coordination protocols
            4. Medical intervention options and limitations
            5. Resource allocation and prioritization
            6. Evacuation and return-to-Earth criteria

            Focus on practical, mission-ready emergency procedures.
            """

            return await self.llm_service.generate_text(prompt, max_tokens=500)

        except Exception as e:
            logger.error(f"Failed to generate emergency protocols: {e}")
            return "Emergency response protocols should be developed based on mission-specific risks and operational capabilities."

    async def _generate_temporal_risk_analysis(self, query: str, risk_context: Dict[str, Any],
                                             studies: List[Dict[str, Any]]) -> str:
        """Generate temporal risk evolution analysis."""
        try:
            prompt = f"""
            Analyze the temporal evolution of health risks for: "{query}"

            Risk Context: {risk_context}

            Provide analysis that includes:
            1. Risk onset timeline and progression patterns
            2. Critical time windows and inflection points
            3. Cumulative vs. acute risk development
            4. Recovery and reversal timelines
            5. Mission phase-specific risk variations
            6. Long-term health consequences and monitoring needs

            Focus on actionable timeline-based insights for mission planning.
            """

            return await self.llm_service.generate_text(prompt, max_tokens=500)

        except Exception as e:
            logger.error(f"Failed to generate temporal risk analysis: {e}")
            return "Temporal risk analysis should consider the progression of health effects throughout different mission phases."

    async def _generate_general_risk_response(self, query: str, risk_context: Dict[str, Any],
                                            studies: List[Dict[str, Any]]) -> str:
        """Generate general risk assessment response."""
        try:
            prompt = f"""
            Answer this health risk question: "{query}"

            Risk Context: {risk_context}

            Provide a comprehensive response that:
            1. Addresses the specific risk concerns
            2. Incorporates relevant research evidence
            3. Considers mission-specific factors
            4. Offers practical risk management guidance
            5. Identifies critical decision points

            Base your response on current space medicine research and best practices.
            """

            return await self.llm_service.generate_text(prompt, max_tokens=400)

        except Exception as e:
            logger.error(f"Failed to generate general risk response: {e}")
            return "Health risk assessment requires careful consideration of multiple factors including mission parameters, crew characteristics, and available countermeasures."

    async def _generate_risk_fallback_response(self, query: str, risk_context: Dict[str, Any]) -> str:
        """Generate fallback response when no specific studies are found."""
        try:
            prompt = f"""
            A flight surgeon asked about health risks: "{query}"

            Risk context: {risk_context}

            No specific studies were found. Provide a helpful response that:
            1. Acknowledges the importance of the health concern
            2. Provides general risk assessment principles
            3. Suggests what research would be valuable
            4. Offers standard risk management approaches

            Focus on practical medical guidance for space missions.
            """

            return await self.llm_service.generate_text(prompt, max_tokens=300)

        except Exception as e:
            logger.error(f"Failed to generate risk fallback response: {e}")
            return "Health risk assessment is critical for mission success. While specific research may be limited, standard risk management principles and medical monitoring protocols should be applied."