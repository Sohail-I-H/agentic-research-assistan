"""
Writer Agent

Generates the final IEEE-style research paper.
"""

from __future__ import annotations

import streamlit as st

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from core.prompts import WRITER_PROMPT
from core.state import AgentState
from utils.helper import add_log, update_step


@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
    )


def build_context(state: AgentState) -> str:
    """
    Build context for the writer.
    """

    context = f"""
Research Topic

{state.get("query","")}


==============================

Literature Review

==============================

{state.get("literature_review","")}


==============================

Research Gaps

==============================

{state.get("research_gaps","")}


==============================

Novel Ideas

==============================

{state.get("novel_ideas","")}
"""

    return context


def writer_node(state: AgentState) -> AgentState:
    """
    Generate complete IEEE-style research paper.
    """

    update_step(state, "Writer")

    add_log(state, "Generating research paper...")

    try:

        llm = load_llm()

        context = build_context(state)

        prompt = f"""
{WRITER_PROMPT}

Use the following research information.

{context}

Generate a professional IEEE-style paper containing:

# Title

# Abstract

# Keywords

# Introduction

# Literature Review

# Proposed Methodology

# Expected Results

# Future Work

# Conclusion
"""

        response = llm.invoke(
            [
                HumanMessage(
                    content=prompt
                )
            ]
        )

        state["generated_paper"] = response.content

        add_log(
            state,
            "Research paper generated successfully."
        )

    except Exception as e:

        state["error"] = str(e)

        add_log(
            state,
            f"Writer Error: {e}"
        )

    update_step(state, "Citation")

    return state
