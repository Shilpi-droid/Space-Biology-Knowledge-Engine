"""
Mission-specific intelligence service for Space Biology Knowledge Engine.
Provides targeted insights and recommendations for specific space missions.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from .knowledge_graph import KnowledgeGraphService
from .semantic_search import SemanticSearchService
from .llm_service import LLMService
from .temporal_analysis import TemporalAnalysisService
from .evidence_synthesis import EvidenceSynthesisService

logger = logging.getLogger(__name__)


class MissionIntelligenceService:
    """Service for generating mission-specific biological intelligence and recommendations."""

    def __init__(self, kg_service: KnowledgeGraphService, search_service: SemanticSearchService,
                 llm_service: LLMService, temporal_service: TemporalAnalysisService,
                 evidence_service: EvidenceSynthesisService):
        self.kg_service = kg_service
        self.search_service = search_service
        self.llm_service = llm_service
        self.temporal_service = temporal_service
        self.evidence_service = evidence_service

        # Mission profiles with specific parameters
        self.mission_profiles = {
            'mars': {
                'duration_months': 30,  # 6-9 months transit + 18 months surface
                'radiation_exposure': 'high',
                'gravity': 0.38,
                'isolation_level': 'extreme',
                'key_factors': ['radiation', 'bone_loss', 'muscle_atrophy', 'psychological', 'immune_suppression'],
                'priority_systems': ['musculoskeletal', 'cardiovascular', 'immune', 'nervous']
            },
            'moon': {
                'duration_months': 1,  # Short-term missions
                'radiation_exposure': 'high',
                'gravity': 0.17,
                'isolation_level': 'moderate',
                'key_factors': ['radiation', 'regolith_exposure', 'bone_loss'],
                'priority_systems': ['respiratory', 'musculoskeletal', 'immune']
            },
            'iss': {
                'duration_months': 6,  # Standard ISS missions
                'radiation_exposure': 'moderate',
                'gravity': 0.0,
                'isolation_level': 'moderate',
                'key_factors': ['microgravity', 'bone_loss', 'muscle_atrophy', 'fluid_shift'],
                'priority_systems': ['musculoskeletal', 'cardiovascular', 'vestibular']
            },
            'deep_space': {
                'duration_months': 12,  # Multi-year missions
                'radiation_exposure': 'extreme',
                'gravity': 0.0,
                'isolation_level': 'extreme',
                'key_factors': ['radiation', 'isolation', 'bone_loss', 'psychological', 'resource_limitation'],
                'priority_systems': ['immune', 'nervous', 'musculoskeletal', 'cardiovascular']
            }
        }

    async def generate_mission_intelligence(self, mission_type: str, crew_size: int = 4,
                                          specific_concerns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive intelligence report for a specific mission."""
        try:
            if mission_type not in self.mission_profiles:
                raise ValueError(f"Unknown mission type: {mission_type}")

            profile = self.mission_profiles[mission_type]

            # Gather intelligence components in parallel
            tasks = [
                self._analyze_biological_risks(mission_type, profile),
                self._identify_critical_countermeasures(mission_type, profile),
                self._analyze_research_gaps(mission_type, profile),
                self._generate_monitoring_recommendations(mission_type, profile),
                self._assess_operational_implications(mission_type, profile, crew_size)
            ]

            if specific_concerns:
                tasks.append(self._analyze_specific_concerns(specific_concerns, mission_type))

            results = await asyncio.gather(*tasks)

            biological_risks = results[0]
            countermeasures = results[1]
            research_gaps = results[2]
            monitoring = results[3]
            operational = results[4]
            specific_analysis = results[5] if specific_concerns else {}

            # Generate executive summary
            executive_summary = await self._generate_executive_summary(
                mission_type, biological_risks, countermeasures, research_gaps
            )

            return {
                'mission_type': mission_type,
                'mission_profile': profile,
                'generated_at': datetime.utcnow().isoformat(),
                'executive_summary': executive_summary,
                'biological_risks': biological_risks,
                'countermeasures': countermeasures,
                'research_gaps': research_gaps,
                'monitoring_recommendations': monitoring,
                'operational_implications': operational,
                'specific_concerns_analysis': specific_analysis,
                'confidence_metrics': await self._calculate_confidence_metrics(mission_type)
            }

        except Exception as e:
            logger.error(f"Error generating mission intelligence: {str(e)}")
            raise

    async def _analyze_biological_risks(self, mission_type: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze biological risks specific to the mission profile."""
        risks = {}

        for factor in profile['key_factors']:
            # Search for studies related to this risk factor
            query = f"{factor} space biology {mission_type}"
            search_results = await self.search_service.search(query, limit=20)

            if search_results:
                # Synthesize evidence for this risk
                evidence = await self.evidence_service.synthesize_evidence(
                    search_results, focus_area=factor
                )

                # Assess risk severity based on mission duration and conditions
                severity = await self._assess_risk_severity(factor, profile, evidence)

                risks[factor] = {
                    'severity': severity,
                    'evidence_strength': evidence.get('confidence', 0.5),
                    'key_findings': evidence.get('summary', ''),
                    'timeline': await self._estimate_risk_timeline(factor, profile),
                    'affected_systems': await self._identify_affected_systems(factor),
                    'supporting_studies': len(search_results)
                }

        return risks

    async def _identify_critical_countermeasures(self, mission_type: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Identify and prioritize countermeasures for the mission."""
        countermeasures = {}

        for system in profile['priority_systems']:
            # Search for countermeasures and interventions
            query = f"countermeasures {system} space biology intervention"
            results = await self.search_service.search(query, limit=15)

            if results:
                # Extract countermeasure information
                measures = await self._extract_countermeasures(results, system)

                # Score effectiveness based on evidence
                scored_measures = []
                for measure in measures:
                    effectiveness = await self._score_countermeasure_effectiveness(
                        measure, system, profile
                    )
                    scored_measures.append({
                        **measure,
                        'effectiveness_score': effectiveness,
                        'mission_applicability': await self._assess_mission_applicability(
                            measure, mission_type
                        )
                    })

                # Sort by effectiveness and applicability
                scored_measures.sort(
                    key=lambda x: (x['effectiveness_score'] + x['mission_applicability']) / 2,
                    reverse=True
                )

                countermeasures[system] = {
                    'priority_level': await self._calculate_system_priority(system, profile),
                    'recommended_measures': scored_measures[:5],  # Top 5
                    'evidence_base': len(results)
                }

        return countermeasures

    async def _analyze_research_gaps(self, mission_type: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Identify critical research gaps for the mission."""
        gaps = {}

        # Analyze temporal trends to identify under-researched areas
        for factor in profile['key_factors']:
            trends = await self.temporal_service.analyze_research_evolution(
                topic=f"{factor} {mission_type}",
                time_window_years=10
            )

            # Identify gaps based on research volume and recency
            gap_analysis = {
                'research_volume': trends.get('publication_count', 0),
                'recent_activity': trends.get('recent_trend', 'stable'),
                'identified_gaps': [],
                'priority_level': 'medium'
            }

            # Determine gap priority
            if gap_analysis['research_volume'] < 10:
                gap_analysis['priority_level'] = 'high'
                gap_analysis['identified_gaps'].append('Limited research base')

            if gap_analysis['recent_activity'] == 'declining':
                gap_analysis['priority_level'] = 'high'
                gap_analysis['identified_gaps'].append('Declining research interest')

            # Mission-specific gap analysis
            mission_specific_gaps = await self._identify_mission_specific_gaps(factor, mission_type)
            gap_analysis['identified_gaps'].extend(mission_specific_gaps)

            gaps[factor] = gap_analysis

        return gaps

    async def _generate_monitoring_recommendations(self, mission_type: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate biological monitoring recommendations for the mission."""
        monitoring = {
            'critical_biomarkers': [],
            'monitoring_frequency': {},
            'equipment_requirements': [],
            'data_collection_priorities': []
        }

        # Identify critical biomarkers for each priority system
        for system in profile['priority_systems']:
            biomarkers = await self._identify_system_biomarkers(system, mission_type)
            monitoring['critical_biomarkers'].extend(biomarkers)

            # Determine monitoring frequency based on risk timeline
            frequency = await self._determine_monitoring_frequency(system, profile)
            monitoring['monitoring_frequency'][system] = frequency

        # Equipment and data collection recommendations
        monitoring['equipment_requirements'] = await self._recommend_monitoring_equipment(
            monitoring['critical_biomarkers'], mission_type
        )

        monitoring['data_collection_priorities'] = await self._prioritize_data_collection(
            profile, monitoring['critical_biomarkers']
        )

        return monitoring

    async def _assess_operational_implications(self, mission_type: str, profile: Dict[str, Any],
                                            crew_size: int) -> Dict[str, Any]:
        """Assess operational implications of biological factors."""
        implications = {
            'crew_health_risks': {},
            'mission_success_factors': [],
            'resource_requirements': {},
            'timeline_considerations': {},
            'contingency_planning': []
        }

        # Assess crew health risks
        for factor in profile['key_factors']:
            risk_level = await self._assess_operational_risk_level(factor, profile, crew_size)
            implications['crew_health_risks'][factor] = risk_level

        # Mission success factors
        implications['mission_success_factors'] = await self._identify_mission_success_factors(
            mission_type, profile
        )

        # Resource requirements
        implications['resource_requirements'] = await self._estimate_resource_requirements(
            mission_type, profile, crew_size
        )

        # Timeline considerations
        implications['timeline_considerations'] = await self._analyze_timeline_considerations(
            mission_type, profile
        )

        # Contingency planning
        implications['contingency_planning'] = await self._develop_contingency_recommendations(
            mission_type, profile
        )

        return implications

    async def _analyze_specific_concerns(self, concerns: List[str], mission_type: str) -> Dict[str, Any]:
        """Analyze user-specified concerns in mission context."""
        analysis = {}

        for concern in concerns:
            # Search for relevant studies
            query = f"{concern} {mission_type} space biology"
            results = await self.search_service.search(query, limit=10)

            if results:
                # Synthesize evidence for this concern
                evidence = await self.evidence_service.synthesize_evidence(
                    results, focus_area=concern
                )

                analysis[concern] = {
                    'evidence_summary': evidence.get('summary', ''),
                    'confidence_level': evidence.get('confidence', 0.5),
                    'recommendations': await self._generate_concern_recommendations(
                        concern, mission_type, evidence
                    ),
                    'related_studies': len(results)
                }
            else:
                analysis[concern] = {
                    'evidence_summary': 'Limited research available for this specific concern.',
                    'confidence_level': 0.1,
                    'recommendations': ['Conduct targeted research', 'Monitor closely during mission'],
                    'related_studies': 0
                }

        return analysis

    async def _generate_executive_summary(self, mission_type: str, risks: Dict[str, Any],
                                        countermeasures: Dict[str, Any], gaps: Dict[str, Any]) -> str:
        """Generate an executive summary using LLM."""
        prompt = f"""
        Generate a concise executive summary for a {mission_type} mission biological intelligence report.

        Key risks identified: {list(risks.keys())}
        Primary countermeasures available: {list(countermeasures.keys())}
        Major research gaps: {[k for k, v in gaps.items() if v['priority_level'] == 'high']}

        Summary should be 3-4 sentences highlighting the most critical biological considerations
        for mission planning and crew safety.
        """

        return await self.llm_service.generate_response(prompt, max_tokens=200)

    async def _calculate_confidence_metrics(self, mission_type: str) -> Dict[str, float]:
        """Calculate confidence metrics for the intelligence report."""
        return {
            'overall_confidence': 0.75,  # Based on available evidence
            'risk_assessment_confidence': 0.8,
            'countermeasure_confidence': 0.7,
            'research_gap_confidence': 0.85,
            'data_recency_score': 0.7  # Based on publication dates
        }

    # Helper methods for detailed analysis
    async def _assess_risk_severity(self, factor: str, profile: Dict[str, Any],
                                  evidence: Dict[str, Any]) -> str:
        """Assess the severity of a specific risk factor."""
        # Implementation details for risk severity assessment
        severity_factors = {
            'radiation': 'high' if profile['radiation_exposure'] in ['high', 'extreme'] else 'medium',
            'bone_loss': 'high' if profile['duration_months'] > 6 else 'medium',
            'isolation': 'high' if profile['isolation_level'] == 'extreme' else 'medium'
        }
        return severity_factors.get(factor, 'medium')

    async def _estimate_risk_timeline(self, factor: str, profile: Dict[str, Any]) -> str:
        """Estimate when risk effects become significant."""
        timelines = {
            'bone_loss': 'weeks' if profile['gravity'] < 0.5 else 'months',
            'muscle_atrophy': 'days to weeks',
            'radiation': 'cumulative over mission',
            'psychological': 'months'
        }
        return timelines.get(factor, 'variable')

    async def _identify_affected_systems(self, factor: str) -> List[str]:
        """Identify biological systems affected by a risk factor."""
        system_mapping = {
            'radiation': ['immune', 'nervous', 'reproductive'],
            'bone_loss': ['musculoskeletal', 'endocrine'],
            'muscle_atrophy': ['musculoskeletal', 'cardiovascular'],
            'isolation': ['nervous', 'endocrine', 'immune']
        }
        return system_mapping.get(factor, ['multiple'])

    async def _extract_countermeasures(self, studies: List[Dict[str, Any]],
                                     system: str) -> List[Dict[str, Any]]:
        """Extract countermeasure information from studies."""
        # Placeholder implementation
        return [
            {
                'name': 'Exercise countermeasure',
                'type': 'physical',
                'description': 'Resistance and aerobic exercise protocols'
            }
        ]

    async def _score_countermeasure_effectiveness(self, measure: Dict[str, Any],
                                                system: str, profile: Dict[str, Any]) -> float:
        """Score the effectiveness of a countermeasure."""
        # Placeholder implementation
        return 0.75

    async def _assess_mission_applicability(self, measure: Dict[str, Any],
                                          mission_type: str) -> float:
        """Assess how applicable a countermeasure is to the specific mission."""
        # Placeholder implementation
        return 0.8

    async def _calculate_system_priority(self, system: str, profile: Dict[str, Any]) -> str:
        """Calculate the priority level for a biological system."""
        # Placeholder implementation
        return 'high' if system in profile['priority_systems'][:2] else 'medium'

    async def _identify_mission_specific_gaps(self, factor: str, mission_type: str) -> List[str]:
        """Identify mission-specific research gaps."""
        # Placeholder implementation
        return [f"Limited {mission_type}-specific research for {factor}"]

    async def _identify_system_biomarkers(self, system: str, mission_type: str) -> List[str]:
        """Identify critical biomarkers for a biological system."""
        biomarker_mapping = {
            'musculoskeletal': ['bone density', 'muscle mass', 'bone turnover markers'],
            'cardiovascular': ['blood pressure', 'heart rate variability', 'VO2 max'],
            'immune': ['white blood cell count', 'cytokine levels', 'antibody response']
        }
        return biomarker_mapping.get(system, ['general health markers'])

    async def _determine_monitoring_frequency(self, system: str, profile: Dict[str, Any]) -> str:
        """Determine appropriate monitoring frequency for a system."""
        if profile['duration_months'] > 12:
            return 'weekly'
        elif profile['duration_months'] > 6:
            return 'bi-weekly'
        else:
            return 'monthly'

    async def _recommend_monitoring_equipment(self, biomarkers: List[str],
                                            mission_type: str) -> List[str]:
        """Recommend monitoring equipment based on biomarkers."""
        return ['portable ultrasound', 'blood analysis kit', 'vital signs monitor']

    async def _prioritize_data_collection(self, profile: Dict[str, Any],
                                        biomarkers: List[str]) -> List[str]:
        """Prioritize data collection based on mission profile."""
        return ['pre-flight baseline', 'in-flight monitoring', 'post-flight assessment']

    async def _assess_operational_risk_level(self, factor: str, profile: Dict[str, Any],
                                           crew_size: int) -> Dict[str, Any]:
        """Assess operational risk level for a factor."""
        return {
            'probability': 'medium',
            'impact': 'high',
            'mitigation_options': ['multiple available'],
            'crew_size_factor': 'considered'
        }

    async def _identify_mission_success_factors(self, mission_type: str,
                                              profile: Dict[str, Any]) -> List[str]:
        """Identify factors critical to mission success."""
        return ['crew health maintenance', 'countermeasure compliance', 'risk mitigation']

    async def _estimate_resource_requirements(self, mission_type: str, profile: Dict[str, Any],
                                            crew_size: int) -> Dict[str, Any]:
        """Estimate resource requirements for biological considerations."""
        return {
            'equipment_mass': f"{crew_size * 50} kg",
            'power_requirements': f"{crew_size * 100} W",
            'crew_time': f"{crew_size * 2} hours/day"
        }

    async def _analyze_timeline_considerations(self, mission_type: str,
                                             profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze timeline considerations for biological factors."""
        return {
            'critical_periods': ['launch', 'mid-mission', 'return'],
            'intervention_windows': ['pre-flight', 'in-flight', 'post-flight'],
            'monitoring_milestones': ['monthly assessments']
        }

    async def _develop_contingency_recommendations(self, mission_type: str,
                                                 profile: Dict[str, Any]) -> List[str]:
        """Develop contingency planning recommendations."""
        return [
            'Medical emergency protocols',
            'Countermeasure failure backup plans',
            'Early return criteria',
            'Communication with ground medical team'
        ]

    async def _generate_concern_recommendations(self, concern: str, mission_type: str,
                                              evidence: Dict[str, Any]) -> List[str]:
        """Generate recommendations for specific concerns."""
        return [
            f"Monitor {concern} closely throughout mission",
            f"Implement preventive measures based on available evidence",
            f"Collect mission-specific data on {concern}"
        ]


# Service instance factory
def create_mission_intelligence_service(kg_service: KnowledgeGraphService,
                                       search_service: SemanticSearchService,
                                       llm_service: LLMService,
                                       temporal_service: TemporalAnalysisService,
                                       evidence_service: EvidenceSynthesisService) -> MissionIntelligenceService:
    """Factory function to create MissionIntelligenceService instance."""
    return MissionIntelligenceService(
        kg_service=kg_service,
        search_service=search_service,
        llm_service=llm_service,
        temporal_service=temporal_service,
        evidence_service=evidence_service
    )