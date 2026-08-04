"""
FAISS Vector Store Utilities

Responsibilities:
- Create FAISS index
- Store document embeddings
- Perform similarity search
- Save and load vector database
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

from utils.embeddings import (
    embed_documents,
    embed_text,
    embedding_dimension,
)
from utils.helper import VECTOR_DB_DIR, create_project_directories


INDEX_FILE = VECTOR_DB_DIR / "research.index"
METADATA_FILE = VECTOR_DB_DIR / "metadata.pkl"


class FAISSVectorStore:
    """
    Simple FAISS vector store with metadata support.
    """

    def __init__(self):

        create_project_directories()

        self.dimension = embedding_dimension()

        self.index = faiss.IndexFlatIP(self.dimension)

        self.metadata: List[Dict] = []

    # ======================================================
    # Add Documents
    # ======================================================

    def add_documents(
        self,
        texts: List[str],
        metadata: List[Dict],
    ) -> None:
        """
        Add document chunks to the vector database.
        """

        if not texts:
            return

        embeddings = embed_documents(texts)

        self.index.add(embeddings)

        self.metadata.extend(metadata)

    # ======================================================
    # Search
    # ======================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Perform semantic similarity search.
        """

        if self.index.ntotal == 0:
            return []

        query_embedding = embed_text(query)

        distances, indices = self.index.search(
            np.array([query_embedding]),
            top_k,
        )

        results = []

        for score, idx in zip(distances[0], indices[0]):

            if idx == -1:
                continue

            item = self.metadata[idx].copy()

            item["score"] = float(score)

            results.append(item)

        return results

    # ======================================================
    # Save
    # ======================================================

    def save(self) -> None:
        """
        Save index and metadata.
        """

        create_project_directories()

        faiss.write_index(
            self.index,
            str(INDEX_FILE),
        )

        with open(METADATA_FILE, "wb") as f:
            pickle.dump(self.metadata, f)

    # ======================================================
    # Load
    # ======================================================

    def load(self) -> None:
        """
        Load saved index.
        """

        if INDEX_FILE.exists():
            self.index = faiss.read_index(str(INDEX_FILE))

        if METADATA_FILE.exists():

            with open(METADATA_FILE, "rb") as f:
                self.metadata = pickle.load(f)

    # ======================================================
    # Clear
    # ======================================================

    def clear(self) -> None:
        """
        Remove all vectors.
        """

        self.index = faiss.IndexFlatIP(self.dimension)

        self.metadata = []

    # ======================================================
    # Statistics
    # ======================================================

    def stats(self) -> Dict:
        """
        Return database statistics.
        """

        return {
            "vectors": self.index.ntotal,
            "dimension": self.dimension,
            "metadata_entries": len(self.metadata),
        }


# ==========================================================
# Build Vector Store
# ==========================================================

def build_vector_store(
    chunks: List[Dict],
) -> FAISSVectorStore:
    """
    Build FAISS vector database from document chunks.

    Expected chunk format:
    {
        "text": "...",
        "source": "...",
        "page": 1
    }
    """

    vector_store = FAISSVectorStore()

    texts = [chunk["text"] for chunk in chunks]

    metadata = [
        {
            "text": chunk["text"],
            "source": chunk.get("source", ""),
            "page": chunk.get("page", 0),
        }
        for chunk in chunks
    ]

    vector_store.add_documents(
        texts=texts,
        metadata=metadata,
    )

    return vector_store
