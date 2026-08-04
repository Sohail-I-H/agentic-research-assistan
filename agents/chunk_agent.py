"""
Chunk Agent

Responsibilities:
- Split parsed documents into chunks
- Preserve metadata
- Prepare chunks for embedding
"""

from __future__ import annotations

from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter

from core.state import AgentState
from utils.helper import add_log, update_step


# ==========================================================
# Text Splitter
# ==========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ],
)


# ==========================================================
# Chunk Single Document
# ==========================================================

def chunk_document(document: Dict) -> List[Dict]:
    """
    Convert one parsed document into metadata-rich chunks.
    """

    chunks = []

    metadata = document.get("metadata", {})

    source = document.get("source", "")

    pages = document.get("pages", [])

    for page in pages:

        page_number = page["page"]

        page_text = page["text"]

        if not page_text.strip():
            continue

        split_chunks = text_splitter.split_text(page_text)

        for chunk in split_chunks:

            chunks.append(
                {
                    "text": chunk,
                    "source": source,
                    "page": page_number,
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                }
            )

    return chunks


# ==========================================================
# Chunk Multiple Documents
# ==========================================================

def chunk_documents(
    documents: List[Dict],
) -> List[Dict]:
    """
    Chunk multiple parsed documents.
    """

    all_chunks = []

    for document in documents:

        all_chunks.extend(
            chunk_document(document)
        )

    return all_chunks


# ==========================================================
# LangGraph Node
# ==========================================================

def chunk_node(
    state: AgentState,
) -> AgentState:
    """
    LangGraph Chunk Node.
    """

    update_step(
        state,
        "Chunk Agent",
    )

    add_log(
        state,
        "Splitting documents into chunks...",
    )

    parsed_documents = state.get(
        "parsed_documents",
        [],
    )

    if not parsed_documents:

        add_log(
            state,
            "No parsed documents available.",
        )

        state["chunks"] = []

        return state

    try:

        chunks = chunk_documents(
            parsed_documents
        )

        state["chunks"] = chunks

        add_log(
            state,
            f"Created {len(chunks)} chunks.",
        )

    except Exception as e:

        state["error"] = str(e)

        add_log(
            state,
            f"Chunk Agent Error: {e}",
        )

    update_step(
        state,
        "Embedding Agent",
    )

    return state
