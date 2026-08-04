"""
Embedding utilities using Sentence Transformers.

Responsibilities:
- Load embedding model
- Generate embeddings
- Batch embedding generation
"""

from __future__ import annotations

from typing import List

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


# ==========================================================
# Load Embedding Model (Cached)
# ==========================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model only once.
    """

    model = SentenceTransformer(
        "BAAI/bge-base-en-v1.5"
    )

    return model


# ==========================================================
# Single Embedding
# ==========================================================

def embed_text(text: str) -> np.ndarray:
    """
    Generate embedding for a single text.
    """

    model = load_embedding_model()

    embedding = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embedding.astype("float32")


# ==========================================================
# Batch Embeddings
# ==========================================================

def embed_documents(documents: List[str]) -> np.ndarray:
    """
    Generate embeddings for multiple documents.
    """

    model = load_embedding_model()

    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embeddings.astype("float32")


# ==========================================================
# Embedding Dimension
# ==========================================================

def embedding_dimension() -> int:
    """
    Return embedding dimension.
    """

    model = load_embedding_model()

    return model.get_sentence_embedding_dimension()


# ==========================================================
# Similarity
# ==========================================================

def cosine_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray
) -> float:
    """
    Compute cosine similarity between two embeddings.
    """

    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)

    return float(np.dot(embedding1, embedding2))


# ==========================================================
# Model Information
# ==========================================================

def model_info() -> dict:
    """
    Return embedding model details.
    """

    model = load_embedding_model()

    return {
        "model_name": "BAAI/bge-base-en-v1.5",
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "max_sequence_length": model.max_seq_length,
    }
