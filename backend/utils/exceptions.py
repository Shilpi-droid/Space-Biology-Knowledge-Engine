"""
Custom exceptions for the Space Biology Knowledge Engine.
"""


class SpaceBiologyEngineError(Exception):
    """Base exception for the Space Biology Knowledge Engine."""
    pass


class DataIngestionError(SpaceBiologyEngineError):
    """Exception raised during data ingestion processes."""
    pass


class DatabaseConnectionError(SpaceBiologyEngineError):
    """Exception raised when database connection fails."""
    pass


class EmbeddingError(SpaceBiologyEngineError):
    """Exception raised during embedding generation or search."""
    pass


class LLMServiceError(SpaceBiologyEngineError):
    """Exception raised during LLM service operations."""
    pass


class ValidationError(SpaceBiologyEngineError):
    """Exception raised for data validation errors."""
    pass


class SearchError(SpaceBiologyEngineError):
    """Exception raised during search operations."""
    pass


class AgentError(SpaceBiologyEngineError):
    """Exception raised during AI agent operations."""
    pass


class ConfigurationError(SpaceBiologyEngineError):
    """Exception raised for configuration-related issues."""
    pass


class KnowledgeGraphError(SpaceBiologyEngineError):
    """Exception raised during knowledge graph operations."""
    pass