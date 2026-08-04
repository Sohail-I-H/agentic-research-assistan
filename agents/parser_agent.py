"""
Parser Agent

Responsibilities:
- Parse uploaded PDFs
- Extract metadata
- Extract page-wise text
- Store structured documents in shared state
"""

from __future__ import annotations

from core.state import AgentState
from utils.helper import add_log, update_step
from utils.pdf_utils import parse_multiple_pdfs


def parser_node(state: AgentState) -> AgentState:
    """
    Parse all uploaded PDFs and store structured documents.

    Input:
        state["uploaded_files"]

    Output:
        state["parsed_documents"]
    """

    update_step(state, "PDF Parser")
    add_log(state, "Parsing uploaded PDF documents...")

    uploaded_files = state.get("uploaded_files", [])

    if not uploaded_files:
        add_log(state, "No uploaded PDFs found.")
        state["parsed_documents"] = []
        return state

    try:

        parsed_documents = parse_multiple_pdfs(uploaded_files)

        state["parsed_documents"] = parsed_documents

        add_log(
            state,
            f"Successfully parsed {len(parsed_documents)} PDF(s)."
        )

    except Exception as e:

        state["error"] = str(e)

        add_log(
            state,
            f"Parser Agent Error: {e}"
        )

    update_step(state, "Chunk Agent")

    return state
