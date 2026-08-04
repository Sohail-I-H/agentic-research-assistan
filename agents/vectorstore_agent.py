"""
Vector Store Agent

Responsibilities:
- Build FAISS vector database
- Build BM25 index
- Persist vector database
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from core.state import AgentState
from utils.helper import (
    add_log,
    update_step,
)
from utils.vectorstore import build_vector_store


def vector_node(
    state: AgentState,
) -> AgentState:
    """
    Build vector databases for retrieval.
    """

    update_step(
        state,
        "Vector Store",
    )

    add_log(
        state,
        "Building vector database...",
    )

    chunks = state.get(
        "chunks",
        [],
    )

    embeddings = state.get(
        "embeddings",
        None,
    )

    if (
        not chunks
        or embeddings is None
    ):

        add_log(
            state,
            "Chunks or embeddings missing.",
        )

        return state

    try:

        vector_store = build_vector_store(
            embeddings,
            chunks,
        )

        vector_store.save()

        tokenized_docs = [

            chunk["text"].lower().split()

            for chunk in chunks

        ]

        bm25 = BM25Okapi(
            tokenized_docs
        )

        state["vector_store"] = vector_store

        state["bm25_index"] = bm25

        add_log(
            state,
            f"Indexed {len(chunks)} chunks.",
        )

    except Exception as e:

        state["error"] = str(e)

        add_log(
            state,
            f"Vector Store Error: {e}",
        )

    update_step(
        state,
        "Retrieval",
    )

    return state
