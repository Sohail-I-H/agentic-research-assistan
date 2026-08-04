from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    """
    Shared state used by every LangGraph agent.

    Each node reads from this state, updates it,
    and passes it to the next node.
    """

    # =============================
    # User Input
    # =============================
    query: str
    uploaded_files: List[str]

    # =============================
    # Search Results
    # =============================
    search_results: List[Dict[str, Any]]
    paper_links: List[str]

    # =============================
    # Parsed PDF Data
    # =============================
    parsed_documents: List[Dict[str, Any]]

    # =============================
    # Chunked Documents
    # =============================
    chunks: List[Any]

    # =============================
    # Vector Store
    # =============================
    vector_store: Optional[Any]

    # =============================
    # Retrieval
    # =============================
    retrieved_docs: List[Any]

    # =============================
    # Reranked Results
    # =============================
    reranked_docs: List[Any]

    # =============================
    # Literature Review
    # =============================
    comparison_matrix: List[Dict[str, Any]]

    # =============================
    # Research Gap Analysis
    # =============================
    research_gaps: str

    # =============================
    # Novel Research Ideas
    # =============================
    novel_ideas: str

    # =============================
    # Generated Paper
    # =============================
    generated_paper: str

    # =============================
    # References
    # =============================
    citations: List[str]

    # =============================
    # Agent Logs
    # =============================
    logs: List[str]

    # =============================
    # Current Workflow Status
    # =============================
    current_step: str

    # =============================
    # Error Handling
    # =============================
    error: Optional[str]
