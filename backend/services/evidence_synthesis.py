"""
Advanced evidence synthesis and contradiction detection service for the Space Biology Knowledge Engine.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import statistics
import re

from config.database import execute_query, execute_write_query
from services.llm_service import LLMService
from services.semantic_search import SemanticSearchService
from services.text_processing import TextProcessor

logger = logging.getLogger(__name__)


class EvidenceSynthesisService:
    """
    Service for synthesizing evidence across multiple studies and detecting contradictions.
    """

    def __init__(self):
        self.llm_service = LLMService()
        self.search_service = SemanticSearchService()
        self.text_processor = TextProcessor()

    async def synthesize_evidence(self, research_question: str, studies: List[Dict[str, Any]],
                                mission_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Synthesize evidence from multiple studies to answer a research question.

        Args:
            research_question: The research question to address
            studies: List of relevant studies
            mission_context: Mission context for relevance weighting

        Returns:
            Comprehensive evidence synthesis
        """
        try:
            logger.info(f"Synthesizing evidence for: {research_question}")

            # Analyze studies for contradictions
            contradictions = await self.detect_contradictions(studies)

            # Extract and categorize findings
            categorized_findings = await self._categorize_findings(studies)

            # Calculate evidence strength
            evidence_strength = await self._calculate_evidence_strength(studies, research_question)

            # Generate consensus findings
            consensus = await self._generate_consensus(categorized_findings, contradictions, mission_context)

            # Identify knowledge gaps
            gaps = await self._identify_knowledge_gaps(studies, research_question)

            # Calculate confidence levels
            confidence_levels = await self._calculate_confidence_levels(
                studies, contradictions, evidence_strength
            )

            synthesis = {
                'research_question': research_question,
                'studies_analyzed': len(studies),
                'consensus_findings': consensus,
                'contradictions': contradictions,
                'evidence_strength': evidence_strength,
                'confidence_levels': confidence_levels,
                'knowledge_gaps': gaps,
                'recommendations': await self._generate_recommendations(
                    research_question, consensus, confidence_levels, mission_context
                ),
                'meta_analysis': await self._perform_meta_analysis(studies),
                'synthesis_date': datetime.now().isoformat()
            }

            logger.info(f"Evidence synthesis completed: {len(contradictions)} contradictions found")
            return synthesis

        except Exception as e:
            logger.error(f"Evidence synthesis failed: {e}")
            return {
                'error': str(e),
                'research_question': research_question,
                'studies_analyzed': len(studies)
            }

    async def detect_contradictions(self, studies: List[Dict[str, Any]],
                                  confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Detect contradictions between study findings.

        Args:
            studies: List of studies to analyze
            confidence_threshold: Minimum confidence for contradiction detection

        Returns:
            List of detected contradictions with confidence scores
        """
        try:
            contradictions = []

            # Compare studies pairwise
            for i in range(len(studies)):
                for j in range(i + 1, len(studies)):
                    study1 = studies[i]
                    study2 = studies[j]

                    # Skip if studies don't have sufficient overlap
                    if not await self._studies_have_overlap(study1, study2):
                        continue

                    # Analyze for contradictions
                    contradiction = await self._analyze_study_contradiction(
                        study1, study2, confidence_threshold
                    )

                    if contradiction:
                        contradictions.append(contradiction)

            # Group related contradictions
            grouped_contradictions = await self._group_contradictions(contradictions)

            logger.info(f"Detected {len(contradictions)} potential contradictions")
            return grouped_contradictions

        except Exception as e:
            logger.error(f"Contradiction detection failed: {e}")
            return []

    async def _studies_have_overlap(self, study1: Dict[str, Any], study2: Dict[str, Any]) -> bool:
        """Check if two studies have sufficient overlap to be compared."""
        try:
            # Check organism overlap
            organisms1 = set(study1.get('organisms', []))
            organisms2 = set(study2.get('organisms', []))
            organism_overlap = len(organisms1.intersection(organisms2)) > 0

            # Check factor overlap
            factors1 = set(study1.get('factors', []))
            factors2 = set(study2.get('factors', []))
            factor_overlap = len(factors1.intersection(factors2)) > 0

            # Check research type similarity
            type_match = study1.get('research_type') == study2.get('research_type')

            # Check mission relevance overlap
            mission1 = study1.get('mission_relevance', '').lower()
            mission2 = study2.get('mission_relevance', '').lower()
            mission_overlap = mission1 == mission2 or 'space' in mission1 and 'space' in mission2

            # Require at least two types of overlap
            overlap_count = sum([organism_overlap, factor_overlap, type_match, mission_overlap])
            return overlap_count >= 2

        except Exception as e:
            logger.error(f"Failed to check study overlap: {e}")
            return False

    async def _analyze_study_contradiction(self, study1: Dict[str, Any], study2: Dict[str, Any],
                                         confidence_threshold: float) -> Optional[Dict[str, Any]]:
        """Analyze two studies for contradictions."""
        try:
            # Extract key findings
            findings1 = await self._extract_key_findings(study1)
            findings2 = await self._extract_key_findings(study2)

            if not findings1 or not findings2:
                return None

            # Use LLM to detect contradictions
            contradiction_analysis = await self._llm_contradiction_analysis(
                study1, study2, findings1, findings2
            )

            if contradiction_analysis['confidence'] >= confidence_threshold:
                return {
                    'study1_id': study1['id'],
                    'study1_title': study1['title'],
                    'study2_id': study2['id'],
                    'study2_title': study2['title'],
                    'contradiction_type': contradiction_analysis['type'],
                    'contradiction_description': contradiction_analysis['description'],
                    'confidence': contradiction_analysis['confidence'],
                    'findings1': findings1,
                    'findings2': findings2,
                    'potential_explanations': contradiction_analysis.get('explanations', []),
                    'severity': contradiction_analysis.get('severity', 'medium'),
                    'detected_at': datetime.now().isoformat()
                }

            return None

        except Exception as e:
            logger.error(f"Study contradiction analysis failed: {e}")
            return None

    async def _extract_key_findings(self, study: Dict[str, Any]) -> List[str]:
        """Extract key findings from a study."""
        try:
            # Use text processor to extract findings
            description = study.get('description', '')
            methodology = study.get('methodology', '')

            combined_text = f"{description} {methodology}"

            if self.text_processor:
                findings = await self.text_processor.extract_key_findings(combined_text)
                return findings
            else:
                # Fallback: simple pattern matching
                return await self._simple_finding_extraction(combined_text)

        except Exception as e:
            logger.error(f"Key findings extraction failed: {e}")
            return []

    async def _simple_finding_extraction(self, text: str) -> List[str]:
        """Simple pattern-based finding extraction."""
        findings = []

        # Look for conclusion patterns
        patterns = [
            r'(?:results? showed?|findings? indicate|we found|demonstrated that|revealed that)\s+([^.]+)',
            r'(?:significantly|markedly|substantially)\s+([^.]+)',
            r'(?:increased?|decreased?|reduced?|enhanced?|improved?)\s+([^.]+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            findings.extend([match.strip() for match in matches])

        return findings[:5]  # Limit to 5 findings

    async def _llm_contradiction_analysis(self, study1: Dict[str, Any], study2: Dict[str, Any],
                                        findings1: List[str], findings2: List[str]) -> Dict[str, Any]:
        """Use LLM to analyze contradictions between studies."""
        try:
            prompt = f"""
            Analyze these two space biology studies for contradictions:

            Study 1: {study1['title']}
            Organisms: {', '.join(study1.get('organisms', []))}
            Factors: {', '.join(study1.get('factors', []))}
            Findings: {'; '.join(findings1)}

            Study 2: {study2['title']}
            Organisms: {', '.join(study2.get('organisms', []))}
            Factors: {', '.join(study2.get('factors', []))}
            Findings: {'; '.join(findings2)}

            Analyze for contradictions and provide:
            1. Contradiction Type: (direct_opposition, methodological_difference, temporal_variation, population_specific, or no_contradiction)
            2. Confidence (0.0-1.0): How confident are you this is a real contradiction?
            3. Description: Brief explanation of the contradiction
            4. Severity: (high, medium, low) - impact on scientific understanding
            5. Potential Explanations: List possible reasons for the contradiction

            Format as JSON:
            {{
                "type": "contradiction_type",
                "confidence": 0.0,
                "description": "explanation",
                "severity": "level",
                "explanations": ["reason1", "reason2"]
            }}
            """

            response = await self.llm_service.generate_text(prompt, max_tokens=400)

            # Parse LLM response
            try:
                import json
                analysis = json.loads(response)
                return analysis
            except json.JSONDecodeError:
                # Fallback parsing
                return await self._parse_contradiction_response(response)

        except Exception as e:
            logger.error(f"LLM contradiction analysis failed: {e}")
            return {
                'type': 'unknown',
                'confidence': 0.5,
                'description': 'Analysis failed',
                'severity': 'medium',
                'explanations': []
            }

    async def _parse_contradiction_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response when JSON parsing fails."""
        analysis = {
            'type': 'unknown',
            'confidence': 0.5,
            'description': 'Unknown contradiction',
            'severity': 'medium',
            'explanations': []
        }

        # Extract confidence
        confidence_match = re.search(r'confidence[:\s]+(\d*\.?\d+)', response.lower())
        if confidence_match:
            analysis['confidence'] = float(confidence_match.group(1))

        # Extract type
        if 'direct_opposition' in response.lower():
            analysis['type'] = 'direct_opposition'
        elif 'methodological' in response.lower():
            analysis['type'] = 'methodological_difference'
        elif 'temporal' in response.lower():
            analysis['type'] = 'temporal_variation'
        elif 'population' in response.lower():
            analysis['type'] = 'population_specific'
        elif 'no_contradiction' in response.lower():
            analysis['type'] = 'no_contradiction'

        # Extract severity
        if 'high' in response.lower():
            analysis['severity'] = 'high'
        elif 'low' in response.lower():
            analysis['severity'] = 'low'

        return analysis

    async def _group_contradictions(self, contradictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group related contradictions together."""
        try:
            # Group by research topic/organism/factor combinations
            groups = defaultdict(list)

            for contradiction in contradictions:
                # Create grouping key based on overlapping studies
                study1_id = contradiction['study1_id']
                study2_id = contradiction['study2_id']
                contradiction_type = contradiction['contradiction_type']

                group_key = f"{contradiction_type}_{min(study1_id, study2_id)}_{max(study1_id, study2_id)}"
                groups[group_key].append(contradiction)

            # Convert groups to list and add metadata
            grouped = []
            for group_key, group_contradictions in groups.items():
                if len(group_contradictions) == 1:
                    grouped.extend(group_contradictions)
                else:
                    # Merge related contradictions
                    merged = await self._merge_contradiction_group(group_contradictions)
                    grouped.append(merged)

            return grouped

        except Exception as e:
            logger.error(f"Contradiction grouping failed: {e}")
            return contradictions

    async def _merge_contradiction_group(self, contradictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge a group of related contradictions."""
        try:
            # Take the highest confidence contradiction as the base
            base_contradiction = max(contradictions, key=lambda x: x['confidence'])

            # Aggregate information
            all_explanations = []
            for c in contradictions:
                all_explanations.extend(c.get('potential_explanations', []))

            merged = {
                **base_contradiction,
                'related_contradictions': [c['study1_id'] + '_' + c['study2_id'] for c in contradictions],
                'potential_explanations': list(set(all_explanations)),
                'group_size': len(contradictions),
                'average_confidence': statistics.mean([c['confidence'] for c in contradictions])
            }

            return merged

        except Exception as e:
            logger.error(f"Contradiction merging failed: {e}")
            return contradictions[0]

    async def _categorize_findings(self, studies: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize findings by research area and outcome type."""
        try:
            categories = {
                'physiological_effects': [],
                'molecular_changes': [],
                'behavioral_impacts': [],
                'countermeasure_effectiveness': [],
                'risk_factors': [],
                'protective_factors': []
            }

            for study in studies:
                findings = await self._extract_key_findings(study)

                for finding in findings:
                    # Categorize each finding
                    category = await self._classify_finding(finding)

                    finding_obj = {
                        'study_id': study['id'],
                        'study_title': study['title'],
                        'finding': finding,
                        'research_type': study.get('research_type'),
                        'organisms': study.get('organisms', []),
                        'confidence': await self._assess_finding_confidence(study, finding)
                    }

                    categories[category].append(finding_obj)

            return categories

        except Exception as e:
            logger.error(f"Finding categorization failed: {e}")
            return {}

    async def _classify_finding(self, finding: str) -> str:
        """Classify a finding into a research category."""
        finding_lower = finding.lower()

        if any(term in finding_lower for term in ['bone', 'muscle', 'cardiovascular', 'physiological']):
            return 'physiological_effects'
        elif any(term in finding_lower for term in ['gene', 'protein', 'molecular', 'cellular']):
            return 'molecular_changes'
        elif any(term in finding_lower for term in ['behavior', 'cognitive', 'psychological', 'mood']):
            return 'behavioral_impacts'
        elif any(term in finding_lower for term in ['exercise', 'nutrition', 'countermeasure', 'intervention']):
            return 'countermeasure_effectiveness'
        elif any(term in finding_lower for term in ['risk', 'danger', 'hazard', 'adverse']):
            return 'risk_factors'
        elif any(term in finding_lower for term in ['protect', 'benefit', 'positive', 'improve']):
            return 'protective_factors'
        else:
            return 'physiological_effects'  # Default category

    async def _assess_finding_confidence(self, study: Dict[str, Any], finding: str) -> float:
        """Assess confidence level for a specific finding."""
        confidence = 0.5  # Base confidence

        # Higher confidence for human studies
        if any('human' in org.lower() for org in study.get('organisms', [])):
            confidence += 0.2

        # Higher confidence for larger sample sizes
        sample_size = study.get('sample_size')
        if sample_size:
            if sample_size >= 50:
                confidence += 0.2
            elif sample_size >= 20:
                confidence += 0.1

        # Higher confidence for certain research types
        research_type = study.get('research_type', '').lower()
        if research_type in ['genomics', 'proteomics']:
            confidence += 0.1

        # Lower confidence for animal studies on complex behaviors
        if ('behavior' in finding.lower() and
            any(animal in ' '.join(study.get('organisms', [])).lower()
                for animal in ['mouse', 'rat', 'mice'])):
            confidence -= 0.1

        return min(0.95, max(0.1, confidence))

    async def _calculate_evidence_strength(self, studies: List[Dict[str, Any]],
                                         research_question: str) -> Dict[str, Any]:
        """Calculate overall evidence strength for the research question."""
        try:
            strength_metrics = {
                'study_count': len(studies),
                'human_studies': 0,
                'total_sample_size': 0,
                'replication_count': 0,
                'methodological_quality': 0.0,
                'temporal_span': 0,
                'overall_strength': 'weak'
            }

            publication_years = []
            methodological_scores = []

            for study in studies:
                # Count human studies
                if any('human' in org.lower() for org in study.get('organisms', [])):
                    strength_metrics['human_studies'] += 1

                # Sum sample sizes
                sample_size = study.get('sample_size', 0)
                if sample_size:
                    strength_metrics['total_sample_size'] += sample_size

                # Track publication years
                pub_date = study.get('publication_date')
                if pub_date:
                    try:
                        year = datetime.fromisoformat(pub_date).year
                        publication_years.append(year)
                    except:
                        pass

                # Assess methodological quality
                method_score = await self._assess_methodological_quality(study)
                methodological_scores.append(method_score)

            # Calculate temporal span
            if publication_years:
                strength_metrics['temporal_span'] = max(publication_years) - min(publication_years)

            # Calculate average methodological quality
            if methodological_scores:
                strength_metrics['methodological_quality'] = statistics.mean(methodological_scores)

            # Calculate replication (simplified - studies with similar organisms/factors)
            strength_metrics['replication_count'] = await self._count_replications(studies)

            # Determine overall strength
            strength_metrics['overall_strength'] = await self._determine_overall_strength(strength_metrics)

            return strength_metrics

        except Exception as e:
            logger.error(f"Evidence strength calculation failed: {e}")
            return {'error': str(e)}

    async def _assess_methodological_quality(self, study: Dict[str, Any]) -> float:
        """Assess the methodological quality of a study."""
        score = 0.5  # Base score

        # Higher score for controlled studies
        methodology = study.get('methodology', '').lower()
        if any(term in methodology for term in ['controlled', 'randomized', 'blinded']):
            score += 0.3

        # Higher score for larger sample sizes
        sample_size = study.get('sample_size', 0)
        if sample_size >= 100:
            score += 0.2
        elif sample_size >= 50:
            score += 0.1

        # Higher score for human studies
        if any('human' in org.lower() for org in study.get('organisms', [])):
            score += 0.2

        # Higher score for peer-reviewed sources
        source = study.get('source', '').lower()
        if source in ['pmc', 'pubmed']:
            score += 0.1

        return min(1.0, score)

    async def _count_replications(self, studies: List[Dict[str, Any]]) -> int:
        """Count potential replications (studies with similar setups)."""
        replications = 0

        for i in range(len(studies)):
            for j in range(i + 1, len(studies)):
                study1 = studies[i]
                study2 = studies[j]

                # Check for similar experimental setups
                if await self._studies_have_overlap(study1, study2):
                    replications += 1

        return replications

    async def _determine_overall_strength(self, metrics: Dict[str, Any]) -> str:
        """Determine overall evidence strength rating."""
        score = 0

        # Study count contribution
        if metrics['study_count'] >= 10:
            score += 3
        elif metrics['study_count'] >= 5:
            score += 2
        elif metrics['study_count'] >= 3:
            score += 1

        # Human studies contribution
        if metrics['human_studies'] >= 3:
            score += 2
        elif metrics['human_studies'] >= 1:
            score += 1

        # Sample size contribution
        if metrics['total_sample_size'] >= 500:
            score += 2
        elif metrics['total_sample_size'] >= 100:
            score += 1

        # Methodological quality contribution
        if metrics['methodological_quality'] >= 0.8:
            score += 2
        elif metrics['methodological_quality'] >= 0.6:
            score += 1

        # Replication contribution
        if metrics['replication_count'] >= 3:
            score += 1

        # Determine strength level
        if score >= 8:
            return 'strong'
        elif score >= 5:
            return 'moderate'
        elif score >= 3:
            return 'weak'
        else:
            return 'very_weak'

    async def _generate_consensus(self, categorized_findings: Dict[str, List[Dict[str, Any]]],
                                contradictions: List[Dict[str, Any]],
                                mission_context: Optional[str] = None) -> Dict[str, Any]:
        """Generate consensus findings across studies."""
        try:
            consensus = {}

            for category, findings in categorized_findings.items():
                if not findings:
                    continue

                # Group similar findings
                finding_groups = await self._group_similar_findings(findings)

                # Generate consensus for each group
                category_consensus = []
                for group in finding_groups:
                    if len(group) >= 2:  # Require at least 2 supporting studies
                        consensus_finding = await self._synthesize_finding_group(group, contradictions)
                        category_consensus.append(consensus_finding)

                consensus[category] = category_consensus

            return consensus

        except Exception as e:
            logger.error(f"Consensus generation failed: {e}")
            return {}

    async def _group_similar_findings(self, findings: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group similar findings together."""
        groups = []
        processed = set()

        for i, finding in enumerate(findings):
            if i in processed:
                continue

            group = [finding]
            processed.add(i)

            # Find similar findings
            for j, other_finding in enumerate(findings):
                if j in processed or i == j:
                    continue

                # Check semantic similarity
                similarity = await self._calculate_finding_similarity(
                    finding['finding'], other_finding['finding']
                )

                if similarity >= 0.7:  # High similarity threshold
                    group.append(other_finding)
                    processed.add(j)

            groups.append(group)

        return groups

    async def _calculate_finding_similarity(self, finding1: str, finding2: str) -> float:
        """Calculate semantic similarity between two findings."""
        try:
            # Use search service for semantic similarity
            if self.search_service and self.search_service.model:
                import numpy as np

                # Generate embeddings
                embeddings = await self.search_service.generate_embeddings([finding1, finding2])

                if embeddings is not None and len(embeddings) == 2:
                    # Calculate cosine similarity
                    similarity = np.dot(embeddings[0], embeddings[1])
                    return float(similarity)

            # Fallback: simple word overlap
            words1 = set(finding1.lower().split())
            words2 = set(finding2.lower().split())

            if not words1 or not words2:
                return 0.0

            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))

            return overlap / total

        except Exception as e:
            logger.error(f"Finding similarity calculation failed: {e}")
            return 0.0

    async def _synthesize_finding_group(self, findings: List[Dict[str, Any]],
                                      contradictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize a group of similar findings into consensus."""
        try:
            # Calculate consensus confidence
            confidences = [f['confidence'] for f in findings]
            consensus_confidence = statistics.mean(confidences)

            # Check for contradictions within this group
            study_ids = [f['study_id'] for f in findings]
            relevant_contradictions = [
                c for c in contradictions
                if c['study1_id'] in study_ids or c['study2_id'] in study_ids
            ]

            # Adjust confidence based on contradictions
            if relevant_contradictions:
                contradiction_penalty = len(relevant_contradictions) * 0.1
                consensus_confidence = max(0.1, consensus_confidence - contradiction_penalty)

            # Generate consensus statement
            consensus_text = await self._generate_consensus_statement(findings)

            return {
                'consensus_statement': consensus_text,
                'supporting_studies': len(findings),
                'study_ids': study_ids,
                'confidence': consensus_confidence,
                'contradictions_present': len(relevant_contradictions) > 0,
                'evidence_quality': await self._assess_group_evidence_quality(findings)
            }

        except Exception as e:
            logger.error(f"Finding group synthesis failed: {e}")
            return {
                'consensus_statement': 'Synthesis failed',
                'supporting_studies': len(findings),
                'confidence': 0.3
            }

    async def _generate_consensus_statement(self, findings: List[Dict[str, Any]]) -> str:
        """Generate a consensus statement from multiple findings."""
        try:
            finding_texts = [f['finding'] for f in findings]
            study_info = [f"Study: {f['study_title']}" for f in findings]

            prompt = f"""
            Generate a consensus statement from these similar research findings:

            Findings:
            {chr(10).join(f"- {text}" for text in finding_texts)}

            Supporting Studies:
            {chr(10).join(study_info)}

            Create a single, clear consensus statement that:
            1. Synthesizes the common findings
            2. Notes the level of agreement
            3. Mentions any important variations
            4. Uses scientific language appropriate for space biology

            Consensus statement:
            """

            consensus = await self.llm_service.generate_text(prompt, max_tokens=200)
            return consensus.strip()

        except Exception as e:
            logger.error(f"Consensus statement generation failed: {e}")
            return f"Multiple studies ({len(findings)}) report similar findings in this area."

    async def _assess_group_evidence_quality(self, findings: List[Dict[str, Any]]) -> str:
        """Assess the overall evidence quality for a group of findings."""
        avg_confidence = statistics.mean([f['confidence'] for f in findings])
        study_count = len(findings)

        # Check for human studies
        human_studies = sum(1 for f in findings
                          if any('human' in org.lower() for org in f.get('organisms', [])))

        if avg_confidence >= 0.8 and study_count >= 3 and human_studies >= 1:
            return 'high'
        elif avg_confidence >= 0.6 and study_count >= 2:
            return 'moderate'
        elif study_count >= 2:
            return 'low'
        else:
            return 'very_low'

    async def _identify_knowledge_gaps(self, studies: List[Dict[str, Any]],
                                     research_question: str) -> List[Dict[str, Any]]:
        """Identify knowledge gaps based on the evidence review."""
        try:
            gaps = []

            # Analyze research coverage
            coverage_analysis = await self._analyze_research_coverage(studies)

            # Population gaps
            if coverage_analysis['human_studies'] < 2:
                gaps.append({
                    'type': 'population',
                    'description': 'Limited human studies available',
                    'impact': 'high',
                    'recommendation': 'Conduct more human research studies'
                })

            # Temporal gaps
            if coverage_analysis['temporal_span'] < 2:
                gaps.append({
                    'type': 'temporal',
                    'description': 'Limited longitudinal data',
                    'impact': 'medium',
                    'recommendation': 'Conduct longer-term studies'
                })

            # Methodological gaps
            if coverage_analysis['controlled_studies'] < 3:
                gaps.append({
                    'type': 'methodological',
                    'description': 'Limited controlled experimental studies',
                    'impact': 'high',
                    'recommendation': 'Conduct more randomized controlled trials'
                })

            # Mission-specific gaps
            mission_coverage = await self._analyze_mission_coverage(studies)
            for mission_type, count in mission_coverage.items():
                if count < 2:
                    gaps.append({
                        'type': 'mission_specific',
                        'description': f'Limited {mission_type} mission-specific research',
                        'impact': 'medium',
                        'recommendation': f'Conduct more {mission_type}-specific studies'
                    })

            return gaps

        except Exception as e:
            logger.error(f"Knowledge gap identification failed: {e}")
            return []

    async def _analyze_research_coverage(self, studies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze research coverage across different dimensions."""
        coverage = {
            'total_studies': len(studies),
            'human_studies': 0,
            'controlled_studies': 0,
            'temporal_span': 0,
            'sample_size_total': 0
        }

        publication_years = []

        for study in studies:
            # Count human studies
            if any('human' in org.lower() for org in study.get('organisms', [])):
                coverage['human_studies'] += 1

            # Count controlled studies
            methodology = study.get('methodology', '').lower()
            if any(term in methodology for term in ['controlled', 'randomized']):
                coverage['controlled_studies'] += 1

            # Track publication years
            pub_date = study.get('publication_date')
            if pub_date:
                try:
                    year = datetime.fromisoformat(pub_date).year
                    publication_years.append(year)
                except:
                    pass

            # Sum sample sizes
            sample_size = study.get('sample_size', 0)
            if sample_size:
                coverage['sample_size_total'] += sample_size

        # Calculate temporal span
        if publication_years:
            coverage['temporal_span'] = max(publication_years) - min(publication_years)

        return coverage

    async def _analyze_mission_coverage(self, studies: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze coverage by mission type."""
        mission_coverage = {'mars': 0, 'moon': 0, 'iss': 0, 'general': 0}

        for study in studies:
            mission_rel = study.get('mission_relevance', '').lower()
            if 'mars' in mission_rel:
                mission_coverage['mars'] += 1
            elif 'moon' in mission_rel or 'lunar' in mission_rel:
                mission_coverage['moon'] += 1
            elif 'iss' in mission_rel or 'station' in mission_rel:
                mission_coverage['iss'] += 1
            else:
                mission_coverage['general'] += 1

        return mission_coverage

    async def _calculate_confidence_levels(self, studies: List[Dict[str, Any]],
                                         contradictions: List[Dict[str, Any]],
                                         evidence_strength: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence levels for different aspects of the evidence."""
        try:
            confidence_levels = {}

            # Overall confidence based on evidence strength
            strength_rating = evidence_strength.get('overall_strength', 'weak')
            strength_confidence = {
                'strong': 0.9,
                'moderate': 0.7,
                'weak': 0.5,
                'very_weak': 0.3
            }.get(strength_rating, 0.4)

            confidence_levels['overall'] = strength_confidence

            # Confidence by research category
            categorized_findings = await self._categorize_findings(studies)
            for category, findings in categorized_findings.items():
                if findings:
                    avg_confidence = statistics.mean([f['confidence'] for f in findings])

                    # Adjust for contradictions
                    category_contradictions = [
                        c for c in contradictions
                        if any(f['study_id'] in [c['study1_id'], c['study2_id']] for f in findings)
                    ]

                    if category_contradictions:
                        contradiction_penalty = len(category_contradictions) * 0.05
                        avg_confidence = max(0.1, avg_confidence - contradiction_penalty)

                    confidence_levels[category] = avg_confidence

            # Human relevance confidence
            human_studies = sum(1 for s in studies
                              if any('human' in org.lower() for org in s.get('organisms', [])))

            if human_studies >= 3:
                confidence_levels['human_relevance'] = 0.8
            elif human_studies >= 1:
                confidence_levels['human_relevance'] = 0.6
            else:
                confidence_levels['human_relevance'] = 0.3

            return confidence_levels

        except Exception as e:
            logger.error(f"Confidence level calculation failed: {e}")
            return {'overall': 0.5}

    async def _generate_recommendations(self, research_question: str, consensus: Dict[str, Any],
                                      confidence_levels: Dict[str, float],
                                      mission_context: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate evidence-based recommendations."""
        try:
            recommendations = []

            # High confidence recommendations
            high_confidence_areas = [k for k, v in confidence_levels.items() if v >= 0.7]

            for area in high_confidence_areas:
                if area in consensus and consensus[area]:
                    for finding in consensus[area]:
                        if finding['confidence'] >= 0.7:
                            recommendations.append({
                                'type': 'evidence_based',
                                'area': area,
                                'recommendation': await self._generate_specific_recommendation(
                                    finding, mission_context
                                ),
                                'confidence': finding['confidence'],
                                'supporting_studies': finding['supporting_studies']
                            })

            # Research gap recommendations
            medium_confidence_areas = [k for k, v in confidence_levels.items() if 0.4 <= v < 0.7]

            for area in medium_confidence_areas:
                recommendations.append({
                    'type': 'research_needed',
                    'area': area,
                    'recommendation': f"Additional research needed in {area.replace('_', ' ')} to increase confidence",
                    'confidence': confidence_levels[area],
                    'priority': 'medium'
                })

            # Mission-specific recommendations
            if mission_context:
                mission_recs = await self._generate_mission_recommendations(
                    consensus, mission_context, confidence_levels
                )
                recommendations.extend(mission_recs)

            return recommendations

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []

    async def _generate_specific_recommendation(self, finding: Dict[str, Any],
                                             mission_context: Optional[str] = None) -> str:
        """Generate a specific recommendation from a consensus finding."""
        try:
            prompt = f"""
            Based on this consensus finding from space biology research, generate a specific, actionable recommendation:

            Finding: {finding['consensus_statement']}
            Supporting Studies: {finding['supporting_studies']}
            Confidence: {finding['confidence']:.2f}
            Mission Context: {mission_context or 'General space missions'}

            Generate a recommendation that:
            1. Is specific and actionable
            2. Considers the confidence level
            3. Is relevant to space mission planning or operations
            4. Includes implementation guidance

            Recommendation:
            """

            recommendation = await self.llm_service.generate_text(prompt, max_tokens=200)
            return recommendation.strip()

        except Exception as e:
            logger.error(f"Specific recommendation generation failed: {e}")
            return f"Consider implementing measures based on: {finding.get('consensus_statement', 'research findings')}"

    async def _generate_mission_recommendations(self, consensus: Dict[str, Any],
                                              mission_context: str,
                                              confidence_levels: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate mission-specific recommendations."""
        try:
            mission_recs = []

            # High-priority mission recommendations
            if confidence_levels.get('physiological_effects', 0) >= 0.6:
                mission_recs.append({
                    'type': 'mission_critical',
                    'area': 'crew_health',
                    'recommendation': f"Implement comprehensive physiological monitoring and countermeasures for {mission_context} missions",
                    'confidence': confidence_levels['physiological_effects'],
                    'priority': 'high'
                })

            if confidence_levels.get('risk_factors', 0) >= 0.6:
                mission_recs.append({
                    'type': 'mission_critical',
                    'area': 'risk_mitigation',
                    'recommendation': f"Develop specific risk mitigation protocols for identified {mission_context} mission hazards",
                    'confidence': confidence_levels['risk_factors'],
                    'priority': 'high'
                })

            return mission_recs

        except Exception as e:
            logger.error(f"Mission recommendation generation failed: {e}")
            return []

    async def _perform_meta_analysis(self, studies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform a simplified meta-analysis of quantitative data."""
        try:
            meta_analysis = {
                'sample_size_total': 0,
                'effect_sizes': [],
                'heterogeneity': 'unknown',
                'publication_bias': 'unknown',
                'statistical_power': 'unknown'
            }

            # Calculate total sample size
            total_sample = sum(s.get('sample_size', 0) for s in studies if s.get('sample_size'))
            meta_analysis['sample_size_total'] = total_sample

            # Assess heterogeneity (simplified)
            research_types = [s.get('research_type', '') for s in studies]
            unique_types = len(set(research_types))

            if unique_types <= 2:
                meta_analysis['heterogeneity'] = 'low'
            elif unique_types <= 4:
                meta_analysis['heterogeneity'] = 'moderate'
            else:
                meta_analysis['heterogeneity'] = 'high'

            # Assess statistical power (simplified)
            if total_sample >= 500:
                meta_analysis['statistical_power'] = 'high'
            elif total_sample >= 100:
                meta_analysis['statistical_power'] = 'moderate'
            else:
                meta_analysis['statistical_power'] = 'low'

            # Check for publication bias indicators (simplified)
            sources = [s.get('source', '') for s in studies]
            if len(set(sources)) >= 3:
                meta_analysis['publication_bias'] = 'low_risk'
            elif len(set(sources)) >= 2:
                meta_analysis['publication_bias'] = 'moderate_risk'
            else:
                meta_analysis['publication_bias'] = 'high_risk'

            return meta_analysis

        except Exception as e:
            logger.error(f"Meta-analysis failed: {e}")
            return {'error': str(e)}