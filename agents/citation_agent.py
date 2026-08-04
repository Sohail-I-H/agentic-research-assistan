"""
Citation Agent

Generates IEEE style references.
"""

from __future__ import annotations

import streamlit as st

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from core.prompts import CITATION_PROMPT
from core.state import AgentState
from utils.helper import add_log, update_step


@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    )


def build_context(state: AgentState) -> str:

    context = ""

    for paper in state.get("search_results", []):

        context += f"""
Title: {paper.get("title","")}

Authors: {paper.get("authors","")}

Year: {paper.get("year","")}

Source: {paper.get("source","")}

URL: {paper.get("url","")}

"""

    return context


def citation_node(state: AgentState) -> AgentState:

    update_step(state, "Citation")

    add_log(state, "Generating citations...")

    try:

        llm = load_llm()

        context = build_context(state)

        prompt = f"""
{CITATION_PROMPT}

Generate IEEE references for the following papers.

{context}

Return only the references.
"""

        response = llm.invoke(
            [
                HumanMessage(
                    content=prompt
                )
            ]
        )

        citations = response.content

        state["citations"] = citations

        add_log(
            state,
            "Citations generated successfully."
        )

    except Exception as e:

        state["error"] = str(e)

        add_log(
            state,
            f"Citation Error: {e}"
        )

    update_step(state, "Completed")

    return state
