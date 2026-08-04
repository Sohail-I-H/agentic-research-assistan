"""
Reranker Agent

Responsibilities:
- Re-rank retrieved documents using a Cross Encoder.
- Improve retrieval quality before LLM reasoning.
"""

from __future__ import annotations

from typing import List, Dict

import streamlit as st
from sentence_transformers import CrossEncoder

from core.state import AgentState
from utils.helper import add_log, update_step


# ==========================================================
# Load Cross Encoder
# ==========================================================

@st.cache_resource(show_spinner="Loading reranker model...")
def load_reranker() -> CrossEncoder:
    """
    Load reranker only once.
    """

    return CrossEncoder(
        "BAAI/bge-reranker-base"
    )


# ==========================================================
# Rerank Function
# ==========================================================

def rerank_documents(
    query: str,
    documents: List[Dict],
    top_k: int = 10,
) -> List[Dict]:
    """
    Rerank retrieved documents.
    """

    if not documents:
        return []

    model = load_reranker()

    pairs = [
        (
            query,
            doc["text"],
        )
        for doc in documents
    ]

    scores = model.predict(pairs)

    reranked = []

    for score, doc in zip(scores, documents):

        item = doc.copy()

        item["rerank_score"] = float(score)

        reranked.append(item)

    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    return reranked[:top_k]


# ==========================================================
# LangGraph Node
# ==========================================================

def reranker_node(
    state: AgentState,
) -> AgentState:
    """
    Rerank retrieved documents.
    """

    update_step(
        state,
        "Reranker",
    )

    add_log(
        state,
        "Re-ranking retrieved documents...",
    )

    query = state["query"]

    retrieved = state.get(
        "retrieved_docs",
        [],
    )

    if not retrieved:

        add_log(
            state,
            "No retrieved documents found.",
        )

        state["reranked_docs"] = []

        return state

    try:

        reranked = rerank_documents(
            query=query,
            documents=retrieved,
            top_k=10,
        )

        state["reranked_docs"] = reranked

        add_log(
            state,
            f"Top {len(reranked)} documents selected.",
        )

    except Exception as e:

        state["error"] = str(e)

        add_log(
            state,
            f"Reranker Error: {e}",
        )

    update_step(
        state,
        "Reviewer",
    )

    return state
