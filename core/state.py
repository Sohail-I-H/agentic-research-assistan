"""
Shared state for the LangGraph workflow.
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # ==========================================================
    # User Input
    # ==========================================================
    query: str
    uploaded_files: List[str]

    # ==========================================================
    # Supervisor
    # ==========================================================
    execution_plan: str

    # ==========================================================
    # Search
    # ==========================================================
    search_results: List[Dict[str, Any]]
    paper_links: List[str]

    # ==========================================================
    # Parser
    # ==========================================================
    parsed_documents: List[Dict[str, Any]]

    # ==========================================================
    # Chunking
    # ==========================================================
    chunks: List[Dict[str, Any]]

    # ==========================================================
    # Embeddings
    # ==========================================================
    embeddings: Any

    # ==========================================================
    # Vector Stores
    # ==========================================================
    vector_store: Any
    bm25_index: Any

    # ==========================================================
    # Retrieval
    # ==========================================================
    retrieved_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]

    # ==========================================================
    # Literature Review
    # ==========================================================
    comparison_matrix: List[Dict[str, Any]]

    research_gaps: str

    novel_ideas: str

    generated_paper: str

    citations: str

    # ==========================================================
    # Workflow
    # ==========================================================
    logs: List[str]

    current_step: str

    error: Optional[str]

    report_markdown: str

    report_pdf: bytes
