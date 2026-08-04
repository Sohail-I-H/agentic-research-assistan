"""
FAISS Vector Store Utilities

Responsibilities:
- Create and manage FAISS index
- Store precomputed embeddings
- Perform semantic search
- Save/load vector database
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

from utils.embeddings import (
    embed_text,
    embedding_dimension,
)
from utils.helper import (
    VECTOR_DB_DIR,
    create_project_directories,
)


INDEX_FILE = VECTOR_DB_DIR / "research.index"
METADATA_FILE = VECTOR_DB_DIR / "metadata.pkl"


class FAISSVectorStore:

    def __init__(self):

        create_project_directories()

        self.dimension = embedding_dimension()

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.metadata = []

    # ======================================================
    # Add Embeddings
    # ======================================================

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
    ):

        if len(embeddings) == 0:
            return

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

        if self.index.ntotal == 0:
            return []

        query_embedding = embed_text(query)

        scores, indices = self.index.search(
            np.array([query_embedding]),
            top_k,
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0],
        ):

            if idx == -1:
                continue

            item = self.metadata[idx].copy()

            item["score"] = float(score)

            results.append(item)

        return results

    # ======================================================
    # Save
    # ======================================================

    def save(self):

        create_project_directories()

        faiss.write_index(
            self.index,
            str(INDEX_FILE),
        )

        with open(
            METADATA_FILE,
            "wb",
        ) as f:

            pickle.dump(
                self.metadata,
                f,
            )

    # ======================================================
    # Load
    # ======================================================

    def load(self):

        if INDEX_FILE.exists():

            self.index = faiss.read_index(
                str(INDEX_FILE)
            )

        if METADATA_FILE.exists():

            with open(
                METADATA_FILE,
                "rb",
            ) as f:

                self.metadata = pickle.load(f)

    # ======================================================
    # Clear
    # ======================================================

    def clear(self):

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.metadata = []

    # ======================================================
    # Statistics
    # ======================================================

    def stats(self):

        return {

            "vectors": self.index.ntotal,

            "dimension": self.dimension,

            "metadata_entries": len(
                self.metadata
            ),
        }


# ==========================================================
# Build Vector Store
# ==========================================================

def build_vector_store(
    embeddings: np.ndarray,
    chunks: List[Dict],
) -> FAISSVectorStore:
    """
    Build FAISS vector database using
    precomputed embeddings.
    """

    store = FAISSVectorStore()

    metadata = []

    for chunk in chunks:

        metadata.append(

            {

                "text": chunk["text"],

                "source": chunk.get(
                    "source",
                    "",
                ),

                "page": chunk.get(
                    "page",
                    0,
                ),

                "title": chunk.get(
                    "title",
                    "",
                ),

                "author": chunk.get(
                    "author",
                    "",
                ),

            }

        )

    store.add_embeddings(
        embeddings,
        metadata,
    )

    return store
