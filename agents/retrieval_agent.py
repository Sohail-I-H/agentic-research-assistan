"""
Retrieval Agent

Responsibilities:
- Dense Retrieval (FAISS)
- Sparse Retrieval (BM25)
- Reciprocal Rank Fusion (RRF)
- Return top documents for downstream agents
"""

from __future__ import annotations

from typing import Dict, List

from core.state import AgentState
from utils.helper import (
    add_log,
    update_step,
)


# ==========================================================
# BM25 Retrieval
# ==========================================================

def bm25_search(
    query: str,
    bm25,
    chunks: List[Dict],
    top_k: int = 10,
) -> List[Dict]:
    """
    Retrieve documents using BM25.
    """

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []

    for score, chunk in ranked[:top_k]:

        item = chunk.copy()

        item["bm25_score"] = float(score)

        results.append(item)

    return results


# ==========================================================
# Dense Retrieval
# ==========================================================

def dense_search(
    query: str,
    vector_store,
    top_k: int = 10,
) -> List[Dict]:

    return vector_store.search(
        query=query,
        top_k=top_k,
    )


# ==========================================================
# Reciprocal Rank Fusion
# ==========================================================

def reciprocal_rank_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    k: int = 60,
) -> List[Dict]:
    """
    Merge rankings using Reciprocal Rank Fusion.
    """

    scores = {}

    lookup = {}

    for rank, doc in enumerate(dense_results):

        key = (
            doc["source"],
            doc["page"],
            doc["text"][:100],
        )

        lookup[key] = doc

        scores[key] = scores.get(key, 0.0) + (
            1 / (k + rank + 1)
        )

    for rank, doc in enumerate(sparse_results):

        key = (
            doc["source"],
            doc["page"],
            doc["text"][:100],
        )

        lookup[key] = doc

        scores[key] = scores.get(key, 0.0) + (
            1 / (k + rank + 1)
        )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []

    for key, score in ranked:

        item = lookup[key].copy()

        item["rrf_score"] = score

        results.append(item)

    return results


# ==========================================================
# LangGraph Node
# ==========================================================

def retrieval_node(
    state: AgentState,
) -> AgentState:
    """
    Hybrid Retrieval Node.
    """

    update_step(
        state,
        "Hybrid Retrieval",
    )

    add_log(
        state,
        "Running hybrid retrieval...",
    )

    query = state["query"]

    vector_store = state.get(
        "vector_store"
    )

    bm25 = state.get(
        "bm25_index"
    )

    chunks = state.get(
        "chunks",
        [],
    )

    if (
        vector_store is None
        or bm25 is None
    ):

        add_log(
            state,
            "Vector databases unavailable.",
        )

        return state

    try:

        dense = dense_search(
            query,
            vector_store,
        )

        sparse = bm25_search(
            query,
            bm25,
            chunks,
        )

        results = reciprocal_rank_fusion(
            dense,
            sparse,
        )

        state["retrieved_docs"] = results

        add_log(
            state,
            f"Retrieved {len(results)} documents."
        )

    except Exception as e:

        state["error"] = str(e)

        add_log(
            state,
            f"Retrieval Error: {e}"
        )

    update_step(
        state,
        "Reranker",
    )

    return state
