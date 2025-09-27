"""
Environment configuration and constants for the Space Biology Knowledge Engine.
"""

import os

from pathlib import Path
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # API Keys
    openai_api_key: Optional[str] = None

    # External API Endpoints
    osdr_base_url: str = "https://osdr.nasa.gov/osdr/data"
    pmc_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    ntrs_base_url: str = "https://ntrs.nasa.gov/api/citations"

    # Rate Limiting
    api_delay: float = 1.0  # seconds between requests
    max_requests_per_minute: int = 60

    # Embedding Model Configuration
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # LLM Configuration
    default_llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 500

    # Search Configuration
    semantic_search_k: int = 10
    similarity_threshold: float = 0.5

    # Application Configuration
    app_name: str = "Space Biology Knowledge Engine"
    app_version: str = "1.0.0"
    debug_mode: bool = False

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_data_paths() -> dict:
    """
    Get data directory paths and ensure they exist.

    Returns:
        Dictionary of Path objects for various data directories
    """
    base_dir = Path(__file__).parent.parent
    paths = {
        "base": base_dir,
        "data": base_dir / "data",
        "raw": base_dir / "data" / "raw",
        "processed": base_dir / "data" / "processed",
        "embeddings": base_dir / "data" / "embeddings",
        "cache": base_dir / "data" / "cache",
        "logs": base_dir / "data" / "logs",
        "static": base_dir / "static"
    }

    # Create directories if they don't exist
    for name, path in paths.items():
        if name != "base":  # Don't try to create base directory
            path.mkdir(parents=True, exist_ok=True)

    return paths


def validate_config() -> bool:
    """
    Validate that required configuration is present.

    Returns:
        True if configuration is valid, False otherwise
    """
    settings = get_settings()

    # Check required settings
    required_fields = ["neo4j_uri", "neo4j_user", "neo4j_password"]
    missing_fields = []

    for field in required_fields:
        value = getattr(settings, field, None)
        if not value:
            missing_fields.append(field)

    if missing_fields:
        print(f"Missing required configuration: {', '.join(missing_fields)}")
        return False

    return True


def get_database_urls() -> dict:
    """
    Get database connection URLs.

    Returns:
        Dictionary with database connection information
    """
    settings = get_settings()
    return {
        "neo4j_uri": settings.neo4j_uri,
        "neo4j_user": settings.neo4j_user,
        "neo4j_password": settings.neo4j_password
    }


def get_api_keys() -> dict:
    """
    Get API keys for external services.

    Returns:
        Dictionary with API keys (excludes None values)
    """
    settings = get_settings()
    keys = {
        "openai": settings.openai_api_key
    }

    # Return only non-None values
    return {k: v for k, v in keys.items() if v is not None}