"""
Neo4j graph database schemas and node models for the Space Biology Knowledge Engine.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import uuid


@dataclass
class Study:
    """
    Represents a research study node in the knowledge graph.

    Properties correspond to Neo4j Study node schema.
    """
    id: str
    title: str
    description: str = ""
    source: str = ""  # OSDR, PMC, NTRS
    organisms: List[str] = field(default_factory=list)
    factors: List[str] = field(default_factory=list)
    methodology: str = ""
    research_type: str = ""  # genomics, proteomics, physiology, etc.
    mission_relevance: str = ""  # Mars, Moon, ISS, etc.
    duration: Optional[int] = None  # study duration in days
    sample_size: Optional[int] = None
    publication_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate study data after initialization."""
        if not self.id:
            self.id = str(uuid.uuid4())

        if not self.title:
            raise ValueError("Study title is required")

        # Clean and normalize text fields
        self.title = self.title.strip()
        self.description = self.description.strip()

        # Validate organisms list
        self.organisms = [org.strip() for org in self.organisms if org.strip()]

        # Validate factors list
        self.factors = [factor.strip() for factor in self.factors if factor.strip()]

    def to_cypher_params(self) -> Dict[str, Any]:
        """
        Convert Study instance to parameters for Cypher CREATE statement.

        Returns:
            Dictionary with Neo4j-compatible property values
        """
        params = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "methodology": self.methodology,
            "research_type": self.research_type,
            "mission_relevance": self.mission_relevance,
            "duration": self.duration,
            "sample_size": self.sample_size,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

        # Handle optional datetime
        if self.publication_date:
            params["publication_date"] = self.publication_date.isoformat()

        return params

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Study":
        """Create Study instance from dictionary data."""
        # Handle datetime strings
        for date_field in ["publication_date", "created_at", "updated_at"]:
            if date_field in data and isinstance(data[date_field], str):
                try:
                    data[date_field] = datetime.fromisoformat(data[date_field])
                except ValueError:
                    data[date_field] = None

        return cls(**data)


@dataclass
class Organism:
    """
    Represents an organism node in the knowledge graph.
    """
    name: str
    scientific_name: str = ""
    organism_type: str = ""  # bacteria, plant, animal, human, etc.
    taxonomy_id: Optional[str] = None
    description: str = ""
    space_relevance: str = ""  # model organism, pathogen, food source, etc.
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate organism data."""
        if not self.name:
            raise ValueError("Organism name is required")

        self.name = self.name.strip()
        self.scientific_name = self.scientific_name.strip()

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to Cypher parameters."""
        return {
            "name": self.name,
            "scientific_name": self.scientific_name,
            "type": self.organism_type,
            "taxonomy_id": self.taxonomy_id,
            "description": self.description,
            "space_relevance": self.space_relevance,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Publication:
    """
    Represents a publication node in the knowledge graph.
    """
    title: str
    doi: Optional[str] = None
    pmid: Optional[str] = None
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    publication_date: Optional[datetime] = None
    source: str = ""  # PMC, NTRS, etc.
    url: str = ""
    keywords: List[str] = field(default_factory=list)
    research_areas: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate publication data."""
        if not self.title:
            raise ValueError("Publication title is required")

        self.title = self.title.strip()
        self.abstract = self.abstract.strip()

        # Clean authors list
        self.authors = [author.strip() for author in self.authors if author.strip()]

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to Cypher parameters."""
        params = {
            "title": self.title,
            "doi": self.doi,
            "pmid": self.pmid,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "source": self.source,
            "url": self.url,
            "keywords": self.keywords,
            "research_areas": self.research_areas,
            "created_at": self.created_at.isoformat()
        }

        if self.publication_date:
            params["publication_date"] = self.publication_date.isoformat()

        return params


@dataclass
class Factor:
    """
    Represents an experimental factor node in the knowledge graph.
    """
    name: str
    factor_type: str = ""  # environmental, experimental, biological
    description: str = ""
    units: str = ""
    value_range: str = ""
    space_relevance: str = ""  # radiation, microgravity, isolation, etc.
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate factor data."""
        if not self.name:
            raise ValueError("Factor name is required")

        self.name = self.name.strip()

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to Cypher parameters."""
        return {
            "name": self.name,
            "type": self.factor_type,
            "description": self.description,
            "units": self.units,
            "value_range": self.value_range,
            "space_relevance": self.space_relevance,
            "created_at": self.created_at.isoformat()
        }


class RelationshipTypes:
    """Constants for relationship types in the knowledge graph."""

    # Study relationships
    STUDIES = "STUDIES"  # Study -> Organism
    HAS_FACTOR = "HAS_FACTOR"  # Study -> Factor
    CITES = "CITES"  # Study -> Publication
    PUBLISHED_AS = "PUBLISHED_AS"  # Study -> Publication

    # Research relationships
    CONTRADICTS = "CONTRADICTS"  # Study -> Study
    SUPPORTS = "SUPPORTS"  # Study -> Study
    EXTENDS = "EXTENDS"  # Study -> Study
    PRECEDES = "PRECEDES"  # Study -> Study (temporal)

    # Classification relationships
    BELONGS_TO = "BELONGS_TO"  # Organism -> TaxonomyGroup
    ASSOCIATED_WITH = "ASSOCIATED_WITH"  # Factor -> Factor

    # Mission relationships
    RELEVANT_TO = "RELEVANT_TO"  # Study -> Mission
    ADDRESSES = "ADDRESSES"  # Study -> RiskFactor


@dataclass
class Relationship:
    """
    Represents a relationship between nodes in the knowledge graph.
    """
    start_node_id: str
    end_node_id: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # 0.0 to 1.0
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate relationship data."""
        if not self.start_node_id or not self.end_node_id:
            raise ValueError("Start and end node IDs are required")

        if not self.relationship_type:
            raise ValueError("Relationship type is required")

        # Validate confidence score
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    def to_cypher_query(self, start_label: str, end_label: str) -> tuple:
        """
        Generate Cypher query for creating this relationship.

        Args:
            start_label: Neo4j label for start node
            end_label: Neo4j label for end node

        Returns:
            Tuple of (query_string, parameters)
        """
        query = f"""
        MATCH (start:{start_label} {{id: $start_id}})
        MATCH (end:{end_label} {{id: $end_id}})
        CREATE (start)-[r:{self.relationship_type}]->(end)
        SET r += $properties
        SET r.confidence = $confidence
        SET r.created_at = $created_at
        RETURN r
        """

        params = {
            "start_id": self.start_node_id,
            "end_id": self.end_node_id,
            "properties": self.properties,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat()
        }

        return query, params


class GraphQueries:
    """Common Cypher queries for the knowledge graph."""

    @staticmethod
    def create_study(study: Study) -> tuple:
        """Generate query to create a study node."""
        query = """
        CREATE (s:Study $props)
        RETURN s
        """
        return query, study.to_cypher_params()

    @staticmethod
    def create_organism(organism: Organism) -> tuple:
        """Generate query to create an organism node."""
        query = """
        MERGE (o:Organism {name: $name})
        SET o += $props
        RETURN o
        """
        params = organism.to_cypher_params()
        return query, {"name": organism.name, "props": params}

    @staticmethod
    def create_publication(publication: Publication) -> tuple:
        """Generate query to create a publication node."""
        query = """
        MERGE (p:Publication {doi: $doi})
        SET p += $props
        RETURN p
        """
        params = publication.to_cypher_params()
        return query, {"doi": publication.doi, "props": params}

    @staticmethod
    def find_similar_studies(study_id: str, limit: int = 10) -> tuple:
        """Find studies with similar organisms or factors."""
        query = """
        MATCH (s1:Study {id: $study_id})-[:STUDIES]->(o:Organism)<-[:STUDIES]-(s2:Study)
        WHERE s1 <> s2
        WITH s2, count(o) as shared_organisms
        MATCH (s1)-[:HAS_FACTOR]->(f:Factor)<-[:HAS_FACTOR]-(s2)
        WITH s2, shared_organisms, count(f) as shared_factors
        RETURN s2, shared_organisms, shared_factors,
               (shared_organisms + shared_factors) as similarity_score
        ORDER BY similarity_score DESC
        LIMIT $limit
        """
        return query, {"study_id": study_id, "limit": limit}

    @staticmethod
    def get_study_network(study_id: str, depth: int = 2) -> tuple:
        """Get the network of related studies within specified depth."""
        query = """
        MATCH path = (s:Study {id: $study_id})-[*1..$depth]-(connected)
        WHERE connected:Study OR connected:Organism OR connected:Publication OR connected:Factor
        RETURN path
        """
        return query, {"study_id": study_id, "depth": depth}