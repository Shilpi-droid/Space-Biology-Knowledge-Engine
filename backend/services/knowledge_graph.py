"""
Knowledge graph operations and relationship discovery services.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
from collections import defaultdict

from config.database import get_neo4j_driver, execute_query, execute_write_query, batch_execute
from models.neo4j_models import Study, Organism, Publication, Factor, Relationship, RelationshipTypes
from services.text_processing import TextProcessor

logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """
    Service for managing knowledge graph operations and relationship discovery.
    """

    def __init__(self):
        self.driver = get_neo4j_driver()
        self.text_processor = TextProcessor()

    async def create_study_node(self, study: Study) -> Optional[str]:
        """
        Create a study node in the knowledge graph.

        Args:
            study: Study object to create

        Returns:
            Created node ID or None if creation fails
        """
        try:
            query = """
            CREATE (s:Study $props)
            SET s.created_at = datetime()
            RETURN s.id as node_id
            """

            params = {"props": study.to_cypher_params()}
            result = execute_write_query(query, params)

            if result:
                node_id = result[0]['node_id']
                logger.info(f"Created study node: {node_id}")

                # Create organism relationships
                await self._create_organism_relationships(study)

                # Create factor relationships
                await self._create_factor_relationships(study)

                return node_id

            return None

        except Exception as e:
            logger.error(f"Failed to create study node: {e}")
            return None

    async def _create_organism_relationships(self, study: Study):
        """Create organism nodes and relationships for a study."""
        try:
            for organism_name in study.organisms:
                if not organism_name.strip():
                    continue

                # Create or get organism node
                organism_query = """
                MERGE (o:Organism {name: $name})
                ON CREATE SET o.created_at = datetime(),
                             o.type = $type,
                             o.space_relevance = $space_relevance
                RETURN o.name as name
                """

                organism_params = {
                    "name": organism_name.strip(),
                    "type": self._classify_organism_type(organism_name),
                    "space_relevance": self._assess_space_relevance(organism_name)
                }

                execute_write_query(organism_query, organism_params)

                # Create STUDIES relationship
                relationship_query = """
                MATCH (s:Study {id: $study_id})
                MATCH (o:Organism {name: $organism_name})
                MERGE (s)-[r:STUDIES]->(o)
                ON CREATE SET r.created_at = datetime(),
                             r.confidence = 1.0
                """

                relationship_params = {
                    "study_id": study.id,
                    "organism_name": organism_name.strip()
                }

                execute_write_query(relationship_query, relationship_params)

        except Exception as e:
            logger.error(f"Failed to create organism relationships for study {study.id}: {e}")

    async def _create_factor_relationships(self, study: Study):
        """Create factor nodes and relationships for a study."""
        try:
            for factor_name in study.factors:
                if not factor_name.strip():
                    continue

                # Create or get factor node
                factor_query = """
                MERGE (f:Factor {name: $name})
                ON CREATE SET f.created_at = datetime(),
                             f.type = $type,
                             f.space_relevance = $space_relevance
                RETURN f.name as name
                """

                factor_params = {
                    "name": factor_name.strip(),
                    "type": self._classify_factor_type(factor_name),
                    "space_relevance": self._assess_factor_space_relevance(factor_name)
                }

                execute_write_query(factor_query, factor_params)

                # Create HAS_FACTOR relationship
                relationship_query = """
                MATCH (s:Study {id: $study_id})
                MATCH (f:Factor {name: $factor_name})
                MERGE (s)-[r:HAS_FACTOR]->(f)
                ON CREATE SET r.created_at = datetime(),
                             r.confidence = 1.0
                """

                relationship_params = {
                    "study_id": study.id,
                    "factor_name": factor_name.strip()
                }

                execute_write_query(relationship_query, relationship_params)

        except Exception as e:
            logger.error(f"Failed to create factor relationships for study {study.id}: {e}")

    def _classify_organism_type(self, organism_name: str) -> str:
        """Classify organism type based on name."""
        name_lower = organism_name.lower()

        type_mapping = {
            'human': ['human', 'astronaut', 'homo sapiens'],
            'mouse': ['mouse', 'mice', 'mus musculus'],
            'rat': ['rat', 'rats', 'rattus'],
            'plant': ['plant', 'arabidopsis', 'seedling', 'wheat', 'soybean'],
            'bacteria': ['bacteria', 'e. coli', 'escherichia', 'bacillus'],
            'yeast': ['yeast', 'saccharomyces', 's. cerevisiae'],
            'fly': ['drosophila', 'fruit fly', 'flies'],
            'worm': ['nematode', 'c. elegans', 'caenorhabditis']
        }

        for organism_type, keywords in type_mapping.items():
            if any(keyword in name_lower for keyword in keywords):
                return organism_type

        return 'other'

    def _classify_factor_type(self, factor_name: str) -> str:
        """Classify factor type based on name."""
        name_lower = factor_name.lower()

        type_mapping = {
            'environmental': ['radiation', 'temperature', 'pressure', 'humidity', 'atmosphere'],
            'gravitational': ['microgravity', 'gravity', 'centrifugation', 'hypergravity'],
            'chemical': ['drug', 'chemical', 'compound', 'treatment', 'supplement'],
            'biological': ['age', 'sex', 'strain', 'genotype', 'phenotype'],
            'experimental': ['duration', 'dosage', 'concentration', 'exposure', 'time']
        }

        for factor_type, keywords in type_mapping.items():
            if any(keyword in name_lower for keyword in keywords):
                return factor_type

        return 'other'

    def _assess_space_relevance(self, organism_name: str) -> str:
        """Assess space mission relevance for organisms."""
        name_lower = organism_name.lower()

        if any(keyword in name_lower for keyword in ['human', 'astronaut']):
            return 'direct_mission_relevance'
        elif any(keyword in name_lower for keyword in ['mouse', 'rat']):
            return 'model_organism'
        elif any(keyword in name_lower for keyword in ['bacteria', 'microbe']):
            return 'potential_pathogen_or_beneficial'
        elif any(keyword in name_lower for keyword in ['plant']):
            return 'life_support_system'
        else:
            return 'research_model'

    def _assess_factor_space_relevance(self, factor_name: str) -> str:
        """Assess space mission relevance for factors."""
        name_lower = factor_name.lower()

        if any(keyword in name_lower for keyword in ['microgravity', 'radiation', 'cosmic']):
            return 'primary_space_stressor'
        elif any(keyword in name_lower for keyword in ['isolation', 'confinement', 'stress']):
            return 'psychological_factor'
        elif any(keyword in name_lower for keyword in ['atmosphere', 'pressure', 'oxygen']):
            return 'life_support_factor'
        else:
            return 'experimental_control'

    async def discover_temporal_relationships(self, study_id: str) -> List[str]:
        """
        Discover temporal relationships between studies.

        Args:
            study_id: ID of the study to find temporal relationships for

        Returns:
            List of created relationship IDs
        """
        try:
            # Find studies with similar organisms published before this one
            query = """
            MATCH (s1:Study {id: $study_id})
            MATCH (s1)-[:STUDIES]->(o:Organism)<-[:STUDIES]-(s2:Study)
            WHERE s1 <> s2
              AND s2.publication_date < s1.publication_date
              AND duration.between(date(s2.publication_date), date(s1.publication_date)).years <= 5
            WITH s1, s2, count(o) as shared_organisms
            WHERE shared_organisms >= 1
            RETURN s2.id as predecessor_id, shared_organisms
            ORDER BY s2.publication_date DESC
            LIMIT 5
            """

            result = execute_query(query, {"study_id": study_id})
            relationship_ids = []

            for record in result:
                predecessor_id = record['predecessor_id']
                shared_organisms = record['shared_organisms']

                # Create PRECEDES relationship
                rel_query = """
                MATCH (s1:Study {id: $predecessor_id})
                MATCH (s2:Study {id: $study_id})
                MERGE (s1)-[r:PRECEDES]->(s2)
                ON CREATE SET r.created_at = datetime(),
                             r.confidence = $confidence,
                             r.shared_organisms = $shared_organisms
                RETURN elementId(r) as rel_id
                """

                confidence = min(0.9, 0.5 + (shared_organisms * 0.1))
                rel_params = {
                    "predecessor_id": predecessor_id,
                    "study_id": study_id,
                    "confidence": confidence,
                    "shared_organisms": shared_organisms
                }

                rel_result = execute_write_query(rel_query, rel_params)
                if rel_result:
                    relationship_ids.append(rel_result[0]['rel_id'])

            logger.info(f"Created {len(relationship_ids)} temporal relationships for study {study_id}")
            return relationship_ids

        except Exception as e:
            logger.error(f"Failed to discover temporal relationships: {e}")
            return []

    async def find_contradictions(self) -> List[Dict[str, Any]]:
        """
        Identify potential contradictions between study findings.

        Returns:
            List of potential contradictions with details
        """
        try:
            # Find studies with similar organisms but different findings
            query = """
            MATCH (s1:Study)-[:STUDIES]->(o:Organism)<-[:STUDIES]-(s2:Study)
            WHERE s1.id < s2.id  // Avoid duplicates
              AND s1.research_type = s2.research_type
              AND s1.mission_relevance = s2.mission_relevance
            WITH s1, s2, count(o) as shared_organisms
            WHERE shared_organisms >= 1
            RETURN s1.id as study1_id, s1.title as study1_title,
                   s2.id as study2_id, s2.title as study2_title,
                   shared_organisms
            LIMIT 20
            """

            result = execute_query(query)
            contradictions = []

            for record in result:
                # Use text processing to analyze potential contradictions
                contradiction = await self._analyze_contradiction(
                    record['study1_id'], record['study1_title'],
                    record['study2_id'], record['study2_title']
                )

                if contradiction:
                    contradictions.append(contradiction)

            logger.info(f"Found {len(contradictions)} potential contradictions")
            return contradictions

        except Exception as e:
            logger.error(f"Failed to find contradictions: {e}")
            return []

    async def _analyze_contradiction(self, study1_id: str, study1_title: str,
                                   study2_id: str, study2_title: str) -> Optional[Dict[str, Any]]:
        """
        Analyze two studies for potential contradictions.

        Args:
            study1_id: First study ID
            study1_title: First study title
            study2_id: Second study ID
            study2_title: Second study title

        Returns:
            Contradiction analysis or None if no contradiction found
        """
        try:
            # Get study details
            study_query = """
            MATCH (s:Study {id: $study_id})
            RETURN s.description as description, s.methodology as methodology
            """

            study1_result = execute_query(study_query, {"study_id": study1_id})
            study2_result = execute_query(study_query, {"study_id": study2_id})

            if not study1_result or not study2_result:
                return None

            study1_desc = study1_result[0]['description']
            study2_desc = study2_result[0]['description']

            # Extract key findings from both studies
            findings1 = await self.text_processor.extract_key_findings(study1_desc)
            findings2 = await self.text_processor.extract_key_findings(study2_desc)

            if not findings1 or not findings2:
                return None

            # Simple contradiction detection based on opposing keywords
            contradiction_indicators = [
                ('increase', 'decrease'),
                ('improve', 'worsen'),
                ('beneficial', 'harmful'),
                ('positive', 'negative'),
                ('enhance', 'reduce'),
                ('upregulate', 'downregulate')
            ]

            contradiction_score = 0
            contradiction_details = []

            for finding1 in findings1:
                for finding2 in findings2:
                    for pos_word, neg_word in contradiction_indicators:
                        if (pos_word in finding1.lower() and neg_word in finding2.lower()) or \
                           (neg_word in finding1.lower() and pos_word in finding2.lower()):
                            contradiction_score += 1
                            contradiction_details.append({
                                'finding1': finding1,
                                'finding2': finding2,
                                'type': f"{pos_word}/{neg_word}"
                            })

            if contradiction_score > 0:
                return {
                    'study1_id': study1_id,
                    'study1_title': study1_title,
                    'study2_id': study2_id,
                    'study2_title': study2_title,
                    'contradiction_score': contradiction_score,
                    'details': contradiction_details,
                    'confidence': min(0.9, contradiction_score * 0.2)
                }

            return None

        except Exception as e:
            logger.error(f"Failed to analyze contradiction between {study1_id} and {study2_id}: {e}")
            return None

    async def calculate_centrality(self, node_type: str = "Study") -> Dict[str, float]:
        """
        Calculate centrality measures for nodes in the graph.

        Args:
            node_type: Type of nodes to calculate centrality for

        Returns:
            Dictionary mapping node IDs to centrality scores
        """
        try:
            # Calculate degree centrality
            query = f"""
            MATCH (n:{node_type})
            OPTIONAL MATCH (n)-[r]-(connected)
            WITH n, count(r) as degree
            RETURN n.id as node_id, degree
            ORDER BY degree DESC
            """

            result = execute_query(query)
            max_degree = max([r['degree'] for r in result]) if result else 1

            centrality_scores = {}
            for record in result:
                node_id = record['node_id']
                degree = record['degree']
                # Normalize centrality score
                centrality_scores[node_id] = degree / max_degree if max_degree > 0 else 0

            logger.info(f"Calculated centrality for {len(centrality_scores)} {node_type} nodes")
            return centrality_scores

        except Exception as e:
            logger.error(f"Failed to calculate centrality: {e}")
            return {}

    async def query_subgraph(self, center_node_id: str, depth: int = 2,
                           node_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract a subgraph around a center node.

        Args:
            center_node_id: ID of the center node
            depth: Maximum traversal depth
            node_types: Optional list of node types to include

        Returns:
            Subgraph data with nodes and relationships
        """
        try:
            # Build node type filter
            type_filter = ""
            if node_types:
                type_labels = "|".join(node_types)
                type_filter = f"WHERE labels(connected)[0] IN {node_types}"

            query = f"""
            MATCH (center {{id: $center_id}})
            MATCH path = (center)-[*1..{depth}]-(connected)
            {type_filter}
            WITH path, connected
            LIMIT 100
            RETURN
                nodes(path) as path_nodes,
                relationships(path) as path_relationships
            """

            result = execute_query(query, {"center_id": center_node_id})

            # Process results into nodes and edges
            nodes = {}
            edges = []

            for record in result:
                path_nodes = record['path_nodes']
                path_relationships = record['path_relationships']

                # Process nodes
                for node in path_nodes:
                    node_id = node.get('id')
                    if node_id and node_id not in nodes:
                        nodes[node_id] = {
                            'id': node_id,
                            'labels': list(node.labels),
                            'properties': dict(node)
                        }

                # Process relationships
                for rel in path_relationships:
                    edge = {
                        'start_node': rel.start_node.get('id'),
                        'end_node': rel.end_node.get('id'),
                        'type': rel.type,
                        'properties': dict(rel)
                    }
                    edges.append(edge)

            subgraph = {
                'center_node_id': center_node_id,
                'depth': depth,
                'nodes': list(nodes.values()),
                'edges': edges,
                'node_count': len(nodes),
                'edge_count': len(edges)
            }

            logger.info(f"Extracted subgraph: {len(nodes)} nodes, {len(edges)} edges")
            return subgraph

        except Exception as e:
            logger.error(f"Failed to query subgraph: {e}")
            return {
                'center_node_id': center_node_id,
                'nodes': [],
                'edges': [],
                'error': str(e)
            }

    async def create_similarity_relationships(self) -> int:
        """
        Create similarity relationships between studies based on shared characteristics.

        Returns:
            Number of relationships created
        """
        try:
            # Find pairs of studies with high similarity
            query = """
            MATCH (s1:Study)-[:STUDIES]->(o:Organism)<-[:STUDIES]-(s2:Study)
            WHERE s1.id < s2.id
            WITH s1, s2, count(o) as shared_organisms

            MATCH (s1)-[:HAS_FACTOR]->(f:Factor)<-[:HAS_FACTOR]-(s2)
            WITH s1, s2, shared_organisms, count(f) as shared_factors

            WHERE shared_organisms >= 1 OR shared_factors >= 1

            WITH s1, s2, shared_organisms, shared_factors,
                 (shared_organisms + shared_factors) as total_shared
            WHERE total_shared >= 2

            RETURN s1.id as study1_id, s2.id as study2_id,
                   shared_organisms, shared_factors, total_shared
            ORDER BY total_shared DESC
            LIMIT 50
            """

            result = execute_query(query)
            created_count = 0

            for record in result:
                study1_id = record['study1_id']
                study2_id = record['study2_id']
                total_shared = record['total_shared']

                # Calculate similarity confidence
                confidence = min(0.95, 0.5 + (total_shared * 0.1))

                # Create SIMILAR_TO relationship
                rel_query = """
                MATCH (s1:Study {id: $study1_id})
                MATCH (s2:Study {id: $study2_id})
                MERGE (s1)-[r:SIMILAR_TO]-(s2)
                ON CREATE SET r.created_at = datetime(),
                             r.confidence = $confidence,
                             r.shared_organisms = $shared_organisms,
                             r.shared_factors = $shared_factors,
                             r.similarity_score = $total_shared
                """

                rel_params = {
                    "study1_id": study1_id,
                    "study2_id": study2_id,
                    "confidence": confidence,
                    "shared_organisms": record['shared_organisms'],
                    "shared_factors": record['shared_factors'],
                    "total_shared": total_shared
                }

                execute_write_query(rel_query, rel_params)
                created_count += 1

            logger.info(f"Created {created_count} similarity relationships")
            return created_count

        except Exception as e:
            logger.error(f"Failed to create similarity relationships: {e}")
            return 0

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the knowledge graph.

        Returns:
            Dictionary with graph statistics
        """
        try:
            stats = {}

            # Node counts
            node_count_query = """
            MATCH (n)
            RETURN labels(n)[0] as label, count(n) as count
            """
            node_results = execute_query(node_count_query)
            stats['node_counts'] = {r['label']: r['count'] for r in node_results}

            # Relationship counts
            rel_count_query = """
            MATCH ()-[r]->()
            RETURN type(r) as relationship_type, count(r) as count
            """
            rel_results = execute_query(rel_count_query)
            stats['relationship_counts'] = {r['relationship_type']: r['count'] for r in rel_results}

            # Graph density (for studies)
            density_query = """
            MATCH (s:Study)
            WITH count(s) as node_count
            MATCH ()-[r:SIMILAR_TO|PRECEDES|CONTRADICTS|SUPPORTS]->()
            WITH node_count, count(r) as edge_count
            RETURN node_count, edge_count,
                   toFloat(edge_count) / (node_count * (node_count - 1)) as density
            """
            density_result = execute_query(density_query)
            if density_result:
                stats['graph_density'] = density_result[0]['density']

            # Most connected nodes
            connected_query = """
            MATCH (n:Study)
            OPTIONAL MATCH (n)-[r]-(connected)
            WITH n, count(r) as degree
            ORDER BY degree DESC
            LIMIT 10
            RETURN n.id as node_id, n.title as title, degree
            """
            connected_results = execute_query(connected_query)
            stats['most_connected_studies'] = connected_results

            # Research type distribution
            research_type_query = """
            MATCH (s:Study)
            WHERE s.research_type IS NOT NULL AND s.research_type <> ''
            RETURN s.research_type as research_type, count(s) as count
            ORDER BY count DESC
            """
            research_type_results = execute_query(research_type_query)
            stats['research_type_distribution'] = {
                r['research_type']: r['count'] for r in research_type_results
            }

            stats['last_updated'] = datetime.now().isoformat()
            return stats

        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {'error': str(e)}