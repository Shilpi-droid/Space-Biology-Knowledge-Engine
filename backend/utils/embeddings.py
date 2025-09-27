"""
Embedding utility functions for the Space Biology Knowledge Engine.
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Global model instance
_embedding_model: Optional[SentenceTransformer] = None


async def load_embedding_model() -> Optional[SentenceTransformer]:
    """
    Load the embedding model.

    Returns:
        Loaded SentenceTransformer model or None if loading fails
    """
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    settings = get_settings()

    try:
        # Load model in a thread to avoid blocking
        _embedding_model = await asyncio.to_thread(
            SentenceTransformer, settings.embedding_model_name
        )
        logger.info(f"Embedding model loaded: {settings.embedding_model_name}")
        return _embedding_model

    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return None


def get_embedding_model() -> Optional[SentenceTransformer]:
    """
    Get the current embedding model instance.

    Returns:
        SentenceTransformer model or None if not loaded
    """
    return _embedding_model


async def embed_text(text: str) -> Optional[np.ndarray]:
    """
    Generate embedding for a single text.

    Args:
        text: Text to embed

    Returns:
        Embedding vector or None if generation fails
    """
    model = await load_embedding_model()
    if not model:
        return None

    try:
        embedding = await asyncio.to_thread(
            model.encode, [text], normalize_embeddings=True
        )
        return embedding[0]

    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return None


async def embed_text_batch(texts: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
    """
    Generate embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        batch_size: Batch size for processing

    Returns:
        Array of embeddings or None if generation fails
    """
    model = await load_embedding_model()
    if not model:
        return None

    try:
        embeddings = await asyncio.to_thread(
            model.encode,
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings

    except Exception as e:
        logger.error(f"Failed to generate batch embeddings: {e}")
        return None


async def save_embeddings(embeddings: np.ndarray, file_path: Path) -> bool:
    """
    Save embeddings to disk.

    Args:
        embeddings: Embedding array to save
        file_path: Path to save the embeddings

    Returns:
        True if successful, False otherwise
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(np.save, file_path, embeddings)
        logger.info(f"Embeddings saved to {file_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save embeddings: {e}")
        return False


async def load_embeddings(file_path: Path) -> Optional[np.ndarray]:
    """
    Load embeddings from disk.

    Args:
        file_path: Path to load embeddings from

    Returns:
        Loaded embeddings or None if loading fails
    """
    try:
        if not file_path.exists():
            logger.warning(f"Embeddings file not found: {file_path}")
            return None

        embeddings = await asyncio.to_thread(np.load, file_path)
        logger.info(f"Embeddings loaded from {file_path}")
        return embeddings

    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return None