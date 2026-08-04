"""
Embedding Agent

Responsibilities:
- Generate embeddings for document chunks
- Store embeddings in shared state
"""

from __future__ import annotations

import numpy as np

from core.state import AgentState
from utils.embeddings import embed_documents
from utils.helper import add_log, update_step


# ==========================================================
# LangGraph Node
# ==========================================================

def embedding_node(
    state: AgentState,
) -> AgentState:
    """
    Generate embeddings for all document chunks.
    """

    update_step(
        state,
        "Embedding Agent",
    )

    add_log(
        state,
        "Generating embeddings...",
    )

    chunks = state.get(
        "chunks",
        [],
    )

    if not chunks:

        state["embeddings"] = np.empty(
            (0, 0),
            dtype="float32",
        )

        add_log(
            state,
            "No chunks available for embedding.",
        )

        return state

    try:

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = embed_documents(
            texts
        )

        state["embeddings"] = embeddings

        add_log(
            state,
            f"Generated embeddings for {len(texts)} chunks.",
        )

    except Exception as e:

        state["error"] = str(e)

        add_log(
            state,
            f"Embedding Agent Error: {e}",
        )

    update_step(
        state,
        "Vector Store",
    )

    return state
