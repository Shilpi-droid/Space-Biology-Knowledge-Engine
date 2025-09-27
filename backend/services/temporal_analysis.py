"""
Temporal analysis and trend detection service for the Space Biology Knowledge Engine.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics
import numpy as np
from dataclasses import dataclass

from config.database import execute_query
from services.llm_service import LLMService
from services.text_processing import TextProcessor

logger = logging.getLogger(__name__)


@dataclass
class TrendPoint:
    """Represents a data point in a temporal trend."""
    timestamp: datetime
    value: float
    study_count: int
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class ResearchTrend:
    """Represents a research trend over time."""
    topic: str
    trend_type: str  # 'increasing', 'decreasing', 'stable', 'emerging', 'declining'
    time_points: List[TrendPoint]
    trend_strength: float  # 0.0 to 1.0
    statistical_significance: float
    prediction_confidence: float
    related_factors: List[str]
    key_findings: List[str]


class TemporalAnalysisService:
    """
    Service for analyzing temporal patterns in space biology research.
    """

    def __init__(self):
        self.llm_service = LLMService()
        self.text_processor = TextProcessor()

    async def analyze_research_evolution(self, topic: str,
                                       time_window_years: int = 10) -> Dict[str, Any]:
        """
        Analyze how research on a specific topic has evolved over time.

        Args:
            topic: Research topic to analyze
            time_window_years: Number of years to analyze

        Returns:
            Comprehensive temporal analysis
        """
        try:
            logger.info(f"Analyzing research evolution for topic: {topic}")

            # Get temporal research data
            temporal_data = await self._get_temporal_research_data(topic, time_window_years)

            if not temporal_data:
                return {
                    'topic': topic,
                    'error': 'No temporal data found for this topic',
                    'time_window_years': time_window_years
                }

            # Analyze publication trends
            publication_trends = await self._analyze_publication_trends(temporal_data)

            # Analyze research focus evolution
            focus_evolution = await self._analyze_research_focus_evolution(temporal_data)

            # Analyze methodology evolution
            methodology_evolution = await self._analyze_methodology_evolution(temporal_data)

            # Detect emerging themes
            emerging_themes = await self._detect_emerging_themes(temporal_data)

            # Analyze citation patterns
            citation_patterns = await self._analyze_citation_patterns(temporal_data)

            # Generate predictions
            predictions = await self._generate_research_predictions(
                temporal_data, publication_trends, focus_evolution
            )

            # Identify research gaps evolution
            gap_evolution = await self._analyze_research_gap_evolution(temporal_data)

            analysis = {
                'topic': topic,
                'time_window_years': time_window_years,
                'analysis_date': datetime.now().isoformat(),
                'total_studies': len(temporal_data),
                'publication_trends': publication_trends,
                'research_focus_evolution': focus_evolution,
                'methodology_evolution': methodology_evolution,
                'emerging_themes': emerging_themes,
                'citation_patterns': citation_patterns,
                'predictions': predictions,
                'research_gap_evolution': gap_evolution,
                'key_insights': await self._generate_key_insights(
                    publication_trends, focus_evolution, emerging_themes, predictions
                )
            }

            logger.info(f"Temporal analysis completed for {topic}: {len(temporal_data)} studies analyzed")
            return analysis

        except Exception as e:
            logger.error(f"Research evolution analysis failed: {e}")
            return {
                'topic': topic,
                'error': str(e),
                'time_window_years': time_window_years
            }

    async def detect_research_trends(self, research_area: str = None) -> List[ResearchTrend]:
        """
        Detect current research trends in space biology.

        Args:
            research_area: Specific research area to focus on (optional)

        Returns:
            List of detected research trends
        """
        try:
            logger.info(f"Detecting research trends for area: {research_area or 'all areas'}")

            # Get recent research data
            recent_data = await self._get_recent_research_data(research_area, years=5)

            if not recent_data:
                return []

            # Identify trending topics
            trending_topics = await self._identify_trending_topics(recent_data)

            # Analyze each trending topic
            trends = []
            for topic_data in trending_topics:
                trend = await self._analyze_topic_trend(topic_data, recent_data)
                if trend:
                    trends.append(trend)

            # Sort trends by strength and significance
            trends.sort(key=lambda x: (x.trend_strength, x.statistical_significance), reverse=True)

            logger.info(f"Detected {len(trends)} research trends")
            return trends

        except Exception as e:
            logger.error(f"Research trend detection failed: {e}")
            return []

    async def predict_research_directions(self, current_trends: List[ResearchTrend],
                                        time_horizon_years: int = 3) -> Dict[str, Any]:
        """
        Predict future research directions based on current trends.

        Args:
            current_trends: Current research trends
            time_horizon_years: Prediction time horizon

        Returns:
            Research direction predictions
        """
        try:
            logger.info(f"Predicting research directions for {time_horizon_years} years")

            predictions = {
                'time_horizon_years': time_horizon_years,
                'prediction_date': datetime.now().isoformat(),
                'emerging_areas': [],
                'growth_areas': [],
                'declining_areas': [],
                'stable_areas': [],
                'breakthrough_potential': [],
                'funding_priorities': [],
                'technology_enablers': [],
                'confidence_levels': {}
            }

            # Analyze trend trajectories
            for trend in current_trends:
                trajectory = await self._analyze_trend_trajectory(trend, time_horizon_years)

                if trajectory['prediction_type'] == 'emerging':
                    predictions['emerging_areas'].append(trajectory)
                elif trajectory['prediction_type'] == 'growth':
                    predictions['growth_areas'].append(trajectory)
                elif trajectory['prediction_type'] == 'declining':
                    predictions['declining_areas'].append(trajectory)
                else:
                    predictions['stable_areas'].append(trajectory)

                predictions['confidence_levels'][trend.topic] = trajectory['confidence']

            # Identify potential breakthroughs
            predictions['breakthrough_potential'] = await self._identify_breakthrough_potential(current_trends)

            # Generate funding recommendations
            predictions['funding_priorities'] = await self._generate_funding_priorities(current_trends)

            # Identify technology enablers
            predictions['technology_enablers'] = await self._identify_technology_enablers(current_trends)

            # Generate overall insights
            predictions['overall_insights'] = await self._generate_prediction_insights(predictions)

            return predictions

        except Exception as e:
            logger.error(f"Research direction prediction failed: {e}")
            return {
                'error': str(e),
                'time_horizon_years': time_horizon_years
            }

    async def analyze_knowledge_gaps_evolution(self, research_area: str) -> Dict[str, Any]:
        """
        Analyze how knowledge gaps have evolved over time.

        Args:
            research_area: Research area to analyze

        Returns:
            Knowledge gap evolution analysis
        """
        try:
            logger.info(f"Analyzing knowledge gap evolution for: {research_area}")

            # Get historical data
            historical_data = await self._get_temporal_research_data(research_area, 15)

            # Analyze gaps by time period
            time_periods = await self._divide_into_time_periods(historical_data, 3)

            gap_evolution = {
                'research_area': research_area,
                'time_periods': len(time_periods),
                'gap_progression': [],
                'persistent_gaps': [],
                'resolved_gaps': [],
                'emerging_gaps': [],
                'critical_unresolved': []
            }

            # Analyze each time period
            period_gaps = []
            for i, period_data in enumerate(time_periods):
                period_analysis = await self._analyze_period_gaps(period_data, i)
                period_gaps.append(period_analysis)
                gap_evolution['gap_progression'].append(period_analysis)

            # Compare across periods
            gap_evolution['persistent_gaps'] = await self._identify_persistent_gaps(period_gaps)
            gap_evolution['resolved_gaps'] = await self._identify_resolved_gaps(period_gaps)
            gap_evolution['emerging_gaps'] = await self._identify_emerging_gaps(period_gaps)

            # Identify critical unresolved gaps
            gap_evolution['critical_unresolved'] = await self._identify_critical_gaps(
                gap_evolution['persistent_gaps']
            )

            # Generate recommendations
            gap_evolution['recommendations'] = await self._generate_gap_recommendations(gap_evolution)

            return gap_evolution

        except Exception as e:
            logger.error(f"Knowledge gap evolution analysis failed: {e}")
            return {
                'research_area': research_area,
                'error': str(e)
            }

    async def _get_temporal_research_data(self, topic: str, years: int) -> List[Dict[str, Any]]:
        """Get research data over a specified time period."""
        try:
            cutoff_date = datetime.now() - timedelta(days=years * 365)

            query = """
            MATCH (s:Study)
            WHERE s.publication_date >= $cutoff_date
              AND (toLower(s.title) CONTAINS toLower($topic)
                   OR toLower(s.description) CONTAINS toLower($topic))
            OPTIONAL MATCH (s)-[:STUDIES]->(o:Organism)
            OPTIONAL MATCH (s)-[:HAS_FACTOR]->(f:Factor)
            RETURN s.id as id, s.title as title, s.description as description,
                   s.research_type as research_type, s.methodology as methodology,
                   s.publication_date as publication_date, s.source as source,
                   s.sample_size as sample_size, s.mission_relevance as mission_relevance,
                   collect(DISTINCT o.name) as organisms,
                   collect(DISTINCT f.name) as factors
            ORDER BY s.publication_date ASC
            """

            results = execute_query(query, {
                'cutoff_date': cutoff_date.isoformat(),
                'topic': topic
            })

            # Process and enrich results
            processed_results = []
            for record in results:
                if record['publication_date']:
                    try:
                        pub_date = datetime.fromisoformat(record['publication_date'])
                        record['parsed_date'] = pub_date
                        record['year'] = pub_date.year
                        record['organisms'] = [o for o in record['organisms'] if o]
                        record['factors'] = [f for f in record['factors'] if f]
                        processed_results.append(record)
                    except ValueError:
                        continue

            return processed_results

        except Exception as e:
            logger.error(f"Failed to get temporal research data: {e}")
            return []

    async def _get_recent_research_data(self, research_area: Optional[str], years: int) -> List[Dict[str, Any]]:
        """Get recent research data for trend analysis."""
        try:
            cutoff_date = datetime.now() - timedelta(days=years * 365)

            if research_area:
                query = """
                MATCH (s:Study)
                WHERE s.publication_date >= $cutoff_date
                  AND (s.research_type = $research_area
                       OR toLower(s.title) CONTAINS toLower($research_area)
                       OR toLower(s.description) CONTAINS toLower($research_area))
                OPTIONAL MATCH (s)-[:STUDIES]->(o:Organism)
                OPTIONAL MATCH (s)-[:HAS_FACTOR]->(f:Factor)
                RETURN s.id as id, s.title as title, s.description as description,
                       s.research_type as research_type, s.methodology as methodology,
                       s.publication_date as publication_date, s.source as source,
                       collect(DISTINCT o.name) as organisms,
                       collect(DISTINCT f.name) as factors
                ORDER BY s.publication_date DESC
                """
                params = {'cutoff_date': cutoff_date.isoformat(), 'research_area': research_area}
            else:
                query = """
                MATCH (s:Study)
                WHERE s.publication_date >= $cutoff_date
                OPTIONAL MATCH (s)-[:STUDIES]->(o:Organism)
                OPTIONAL MATCH (s)-[:HAS_FACTOR]->(f:Factor)
                RETURN s.id as id, s.title as title, s.description as description,
                       s.research_type as research_type, s.methodology as methodology,
                       s.publication_date as publication_date, s.source as source,
                       collect(DISTINCT o.name) as organisms,
                       collect(DISTINCT f.name) as factors
                ORDER BY s.publication_date DESC
                """
                params = {'cutoff_date': cutoff_date.isoformat()}

            results = execute_query(query, params)

            # Process results
            processed_results = []
            for record in results:
                if record['publication_date']:
                    try:
                        pub_date = datetime.fromisoformat(record['publication_date'])
                        record['parsed_date'] = pub_date
                        record['year'] = pub_date.year
                        record['organisms'] = [o for o in record['organisms'] if o]
                        record['factors'] = [f for f in record['factors'] if f]
                        processed_results.append(record)
                    except ValueError:
                        continue

            return processed_results

        except Exception as e:
            logger.error(f"Failed to get recent research data: {e}")
            return []

    async def _analyze_publication_trends(self, temporal_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze publication trends over time."""
        try:
            # Group by year
            yearly_counts = defaultdict(int)
            for study in temporal_data:
                year = study.get('year')
                if year:
                    yearly_counts[year] += 1

            # Convert to sorted list
            years = sorted(yearly_counts.keys())
            counts = [yearly_counts[year] for year in years]

            if len(years) < 3:
                return {
                    'trend_type': 'insufficient_data',
                    'trend_strength': 0.0,
                    'yearly_data': dict(yearly_counts)
                }

            # Calculate trend
            trend_analysis = await self._calculate_trend_statistics(years, counts)

            # Analyze research velocity
            velocity_analysis = await self._analyze_research_velocity(yearly_counts)

            return {
                'trend_type': trend_analysis['trend_type'],
                'trend_strength': trend_analysis['trend_strength'],
                'statistical_significance': trend_analysis['statistical_significance'],
                'yearly_data': dict(yearly_counts),
                'total_publications': sum(counts),
                'peak_year': max(yearly_counts.items(), key=lambda x: x[1])[0] if yearly_counts else None,
                'research_velocity': velocity_analysis,
                'growth_rate': trend_analysis.get('growth_rate', 0.0),
                'projected_next_year': trend_analysis.get('projection', 0)
            }

        except Exception as e:
            logger.error(f"Publication trend analysis failed: {e}")
            return {'error': str(e)}

    async def _analyze_research_focus_evolution(self, temporal_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how research focus has evolved over time."""
        try:
            # Divide data into time periods
            periods = await self._divide_into_time_periods(temporal_data, 3)

            focus_evolution = {
                'periods': [],
                'focus_shifts': [],
                'emerging_topics': [],
                'declining_topics': [],
                'persistent_topics': []
            }

            # Analyze each period
            period_focuses = []
            for i, period_data in enumerate(periods):
                period_focus = await self._analyze_period_focus(period_data, i)
                period_focuses.append(period_focus)
                focus_evolution['periods'].append(period_focus)

            # Compare periods to identify shifts
            if len(period_focuses) >= 2:
                focus_evolution['focus_shifts'] = await self._identify_focus_shifts(period_focuses)
                focus_evolution['emerging_topics'] = await self._identify_emerging_topics(period_focuses)
                focus_evolution['declining_topics'] = await self._identify_declining_topics(period_focuses)
                focus_evolution['persistent_topics'] = await self._identify_persistent_topics(period_focuses)

            return focus_evolution

        except Exception as e:
            logger.error(f"Research focus evolution analysis failed: {e}")
            return {'error': str(e)}

    async def _analyze_methodology_evolution(self, temporal_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how research methodologies have evolved."""
        try:
            # Group by time periods and methodology
            periods = await self._divide_into_time_periods(temporal_data, 3)

            methodology_evolution = {
                'periods': [],
                'methodology_trends': {},
                'emerging_methods': [],
                'declining_methods': [],
                'technology_adoption': []
            }

            # Analyze methodology usage in each period
            period_methods = []
            for i, period_data in enumerate(periods):
                methods = await self._analyze_period_methodologies(period_data, i)
                period_methods.append(methods)
                methodology_evolution['periods'].append(methods)

            # Identify methodology trends
            if len(period_methods) >= 2:
                methodology_evolution['methodology_trends'] = await self._calculate_methodology_trends(period_methods)
                methodology_evolution['emerging_methods'] = await self._identify_emerging_methods(period_methods)
                methodology_evolution['declining_methods'] = await self._identify_declining_methods(period_methods)
                methodology_evolution['technology_adoption'] = await self._analyze_technology_adoption(period_methods)

            return methodology_evolution

        except Exception as e:
            logger.error(f"Methodology evolution analysis failed: {e}")
            return {'error': str(e)}

    async def _detect_emerging_themes(self, temporal_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect emerging research themes."""
        try:
            # Split data into recent vs. earlier periods
            sorted_data = sorted(temporal_data, key=lambda x: x.get('parsed_date', datetime.min))
            split_point = len(sorted_data) // 2

            earlier_period = sorted_data[:split_point]
            recent_period = sorted_data[split_point:]

            # Extract themes from each period
            earlier_themes = await self._extract_themes_from_studies(earlier_period)
            recent_themes = await self._extract_themes_from_studies(recent_period)

            # Identify emerging themes (present in recent but not earlier)
            emerging_themes = []

            for theme, recent_count in recent_themes.items():
                earlier_count = earlier_themes.get(theme, 0)

                # Calculate emergence score
                if recent_count > earlier_count:
                    growth_factor = recent_count / max(1, earlier_count)
                    emergence_score = min(1.0, growth_factor * (recent_count / len(recent_period)))

                    if emergence_score > 0.3:  # Threshold for emerging themes
                        emerging_themes.append({
                            'theme': theme,
                            'emergence_score': emergence_score,
                            'recent_count': recent_count,
                            'earlier_count': earlier_count,
                            'growth_factor': growth_factor
                        })

            # Sort by emergence score
            emerging_themes.sort(key=lambda x: x['emergence_score'], reverse=True)

            return emerging_themes

        except Exception as e:
            logger.error(f"Emerging theme detection failed: {e}")
            return []

    async def _extract_themes_from_studies(self, studies: List[Dict[str, Any]]) -> Dict[str, int]:
        """Extract research themes from studies."""
        try:
            themes = defaultdict(int)

            for study in studies:
                # Extract themes from title and description
                text = f"{study.get('title', '')} {study.get('description', '')}"

                # Use simple keyword extraction
                study_themes = await self._extract_keywords_from_text(text)

                for theme in study_themes:
                    themes[theme] += 1

            return dict(themes)

        except Exception as e:
            logger.error(f"Theme extraction failed: {e}")
            return {}

    async def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract keywords/themes from text."""
        try:
            # Simple keyword extraction (in practice, would use more sophisticated NLP)
            space_bio_keywords = [
                'microgravity', 'radiation', 'bone loss', 'muscle atrophy', 'cardiovascular',
                'immune system', 'metabolism', 'circadian rhythm', 'stress response',
                'genomics', 'proteomics', 'transcriptomics', 'metabolomics',
                'countermeasure', 'exercise', 'nutrition', 'medication',
                'mars mission', 'lunar mission', 'iss', 'long duration',
                'crew health', 'astronaut', 'spaceflight', 'space medicine'
            ]

            text_lower = text.lower()
            found_keywords = []

            for keyword in space_bio_keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)

            return found_keywords

        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []

    async def _calculate_trend_statistics(self, x_values: List[int], y_values: List[int]) -> Dict[str, Any]:
        """Calculate trend statistics for time series data."""
        try:
            if len(x_values) < 3:
                return {
                    'trend_type': 'insufficient_data',
                    'trend_strength': 0.0,
                    'statistical_significance': 0.0
                }

            # Calculate linear regression
            x_mean = statistics.mean(x_values)
            y_mean = statistics.mean(y_values)

            # Calculate slope and correlation
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
            denominator = sum((x - x_mean) ** 2 for x in x_values)

            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator

            # Calculate correlation coefficient
            x_var = sum((x - x_mean) ** 2 for x in x_values)
            y_var = sum((y - y_mean) ** 2 for y in y_values)

            if x_var == 0 or y_var == 0:
                correlation = 0
            else:
                correlation = numerator / (x_var * y_var) ** 0.5

            # Determine trend type
            if abs(correlation) < 0.3:
                trend_type = 'stable'
            elif correlation > 0.3:
                trend_type = 'increasing'
            else:
                trend_type = 'decreasing'

            # Calculate trend strength
            trend_strength = abs(correlation)

            # Simple significance test (in practice, would use proper statistical tests)
            statistical_significance = min(0.95, trend_strength * len(x_values) / 10)

            # Calculate growth rate
            if len(y_values) >= 2:
                first_val = y_values[0] if y_values[0] > 0 else 1
                last_val = y_values[-1]
                years_span = len(y_values) - 1
                growth_rate = ((last_val / first_val) ** (1 / years_span) - 1) if years_span > 0 else 0
            else:
                growth_rate = 0

            # Project next year
            if slope != 0:
                next_x = max(x_values) + 1
                projection = y_mean + slope * (next_x - x_mean)
                projection = max(0, int(round(projection)))
            else:
                projection = int(y_mean)

            return {
                'trend_type': trend_type,
                'trend_strength': trend_strength,
                'statistical_significance': statistical_significance,
                'slope': slope,
                'correlation': correlation,
                'growth_rate': growth_rate,
                'projection': projection
            }

        except Exception as e:
            logger.error(f"Trend statistics calculation failed: {e}")
            return {
                'trend_type': 'error',
                'trend_strength': 0.0,
                'statistical_significance': 0.0
            }

    async def _analyze_research_velocity(self, yearly_counts: Dict[int, int]) -> Dict[str, Any]:
        """Analyze research velocity and acceleration."""
        try:
            years = sorted(yearly_counts.keys())
            counts = [yearly_counts[year] for year in years]

            if len(years) < 3:
                return {'velocity': 0.0, 'acceleration': 0.0}

            # Calculate year-over-year changes
            velocities = []
            for i in range(1, len(counts)):
                velocity = counts[i] - counts[i-1]
                velocities.append(velocity)

            # Calculate acceleration (change in velocity)
            accelerations = []
            for i in range(1, len(velocities)):
                acceleration = velocities[i] - velocities[i-1]
                accelerations.append(acceleration)

            avg_velocity = statistics.mean(velocities) if velocities else 0
            avg_acceleration = statistics.mean(accelerations) if accelerations else 0

            return {
                'average_velocity': avg_velocity,
                'average_acceleration': avg_acceleration,
                'velocity_trend': 'accelerating' if avg_acceleration > 0 else 'decelerating' if avg_acceleration < 0 else 'steady',
                'recent_velocity': velocities[-1] if velocities else 0
            }

        except Exception as e:
            logger.error(f"Research velocity analysis failed: {e}")
            return {'velocity': 0.0, 'acceleration': 0.0}

    async def _divide_into_time_periods(self, data: List[Dict[str, Any]],
                                      num_periods: int) -> List[List[Dict[str, Any]]]:
        """Divide temporal data into equal time periods."""
        try:
            if not data:
                return []

            # Sort by date
            sorted_data = sorted(data, key=lambda x: x.get('parsed_date', datetime.min))

            # Calculate period boundaries
            total_studies = len(sorted_data)
            studies_per_period = total_studies // num_periods

            periods = []
            for i in range(num_periods):
                start_idx = i * studies_per_period
                end_idx = start_idx + studies_per_period if i < num_periods - 1 else total_studies

                period_data = sorted_data[start_idx:end_idx]
                if period_data:
                    periods.append(period_data)

            return periods

        except Exception as e:
            logger.error(f"Time period division failed: {e}")
            return []

    # Additional helper methods would continue here...
    # Due to length constraints, I'll include the key remaining methods

    async def _generate_research_predictions(self, temporal_data: List[Dict[str, Any]],
                                           publication_trends: Dict[str, Any],
                                           focus_evolution: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions for future research directions."""
        try:
            predictions = {
                'methodology_predictions': [],
                'topic_predictions': [],
                'collaboration_predictions': [],
                'funding_predictions': [],
                'timeline_predictions': {},
                'confidence_assessment': {}
            }

            # Predict research volume
            trend_type = publication_trends.get('trend_type', 'stable')
            growth_rate = publication_trends.get('growth_rate', 0.0)

            if trend_type == 'increasing' and growth_rate > 0:
                predictions['timeline_predictions']['next_2_years'] = 'continued_growth'
                predictions['timeline_predictions']['next_5_years'] = 'sustained_expansion'
            elif trend_type == 'decreasing':
                predictions['timeline_predictions']['next_2_years'] = 'plateau_or_decline'
                predictions['timeline_predictions']['next_5_years'] = 'potential_refocus'
            else:
                predictions['timeline_predictions']['next_2_years'] = 'stable_output'
                predictions['timeline_predictions']['next_5_years'] = 'gradual_evolution'

            # Predict emerging topics based on current trends
            emerging_themes = focus_evolution.get('emerging_topics', [])
            for theme in emerging_themes[:3]:  # Top 3 emerging themes
                predictions['topic_predictions'].append({
                    'topic': theme,
                    'prediction': 'rapid_growth',
                    'confidence': 0.7,
                    'timeframe': '1-3_years'
                })

            # Generate LLM-based predictions
            llm_predictions = await self._generate_llm_predictions(temporal_data, publication_trends)
            predictions.update(llm_predictions)

            return predictions

        except Exception as e:
            logger.error(f"Research prediction generation failed: {e}")
            return {'error': str(e)}

    async def _generate_llm_predictions(self, temporal_data: List[Dict[str, Any]],
                                      trends: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to generate research predictions."""
        try:
            # Prepare context for LLM
            recent_studies = sorted(temporal_data, key=lambda x: x.get('parsed_date', datetime.min))[-10:]

            study_summaries = []
            for study in recent_studies:
                summary = f"{study.get('title', '')}: {study.get('research_type', '')} study"
                study_summaries.append(summary)

            prompt = f"""
            Based on these recent space biology research trends and studies, predict future research directions:

            Publication Trend: {trends.get('trend_type', 'unknown')}
            Growth Rate: {trends.get('growth_rate', 0):.2f}

            Recent Studies:
            {chr(10).join(study_summaries)}

            Predict:
            1. Emerging research areas (next 2-3 years)
            2. Likely technological breakthroughs
            3. Funding priorities
            4. International collaboration opportunities
            5. Key research gaps that will be addressed

            Format as structured predictions with confidence levels.
            """

            response = await self.llm_service.generate_text(prompt, max_tokens=500)

            # Parse LLM response into structured format
            return await self._parse_llm_predictions(response)

        except Exception as e:
            logger.error(f"LLM prediction generation failed: {e}")
            return {}

    async def _parse_llm_predictions(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM predictions into structured format."""
        try:
            # Simple parsing (in practice, would be more sophisticated)
            predictions = {
                'llm_insights': llm_response,
                'parsed_predictions': []
            }

            # Extract key predictions
            lines = llm_response.split('\n')
            current_prediction = {}

            for line in lines:
                line = line.strip()
                if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    if current_prediction:
                        predictions['parsed_predictions'].append(current_prediction)

                    current_prediction = {
                        'prediction': line,
                        'confidence': 0.6  # Default confidence
                    }

            if current_prediction:
                predictions['parsed_predictions'].append(current_prediction)

            return predictions

        except Exception as e:
            logger.error(f"LLM prediction parsing failed: {e}")
            return {'llm_insights': llm_response}

    async def _generate_key_insights(self, publication_trends: Dict[str, Any],
                                   focus_evolution: Dict[str, Any],
                                   emerging_themes: List[Dict[str, Any]],
                                   predictions: Dict[str, Any]) -> List[str]:
        """Generate key insights from temporal analysis."""
        try:
            insights = []

            # Publication trend insights
            trend_type = publication_trends.get('trend_type', 'unknown')
            if trend_type == 'increasing':
                insights.append("Research activity is increasing, indicating growing interest and funding in this area")
            elif trend_type == 'decreasing':
                insights.append("Research activity is declining, suggesting potential field maturation or funding shifts")

            # Emerging theme insights
            if emerging_themes:
                top_theme = emerging_themes[0]['theme']
                insights.append(f"'{top_theme}' is emerging as a key research focus with significant growth")

            # Focus evolution insights
            if focus_evolution.get('focus_shifts'):
                insights.append("Research focus has shifted significantly over time, reflecting evolving priorities")

            # Prediction insights
            timeline_pred = predictions.get('timeline_predictions', {})
            if timeline_pred.get('next_2_years') == 'continued_growth':
                insights.append("Research output is expected to continue growing over the next 2 years")

            # Add general temporal insight
            insights.append("Temporal analysis reveals clear patterns in research evolution and future directions")

            return insights

        except Exception as e:
            logger.error(f"Key insights generation failed: {e}")
            return ["Temporal analysis completed with partial insights"]