"""
Planner Agent

Responsibilities:
- Identify research gaps
- Generate novel research ideas
- Suggest future work
"""

from __future__ import annotations

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from core.prompts import PLANNER_PROMPT
from core.state import AgentState
from utils.helper import (
    add_log,
    update_step,
)


# ==========================================================
# Load LLM
# ==========================================================

@st.cache_resource
def load_llm():

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
    )


# ==========================================================
# Planner Node
# ==========================================================

def planner_node(
    state: AgentState,
) -> AgentState:
    """
    Generate research gaps and novel ideas.
    """

    update_step(
        state,
        "Planner",
    )

    add_log(
        state,
        "Analyzing literature for research opportunities...",
    )

    literature_review = state.get(
        "literature_review",
        "",
    )

    comparison = state.get(
        "comparison_matrix",
        [],
    )

    if not literature_review:

        add_log(
            state,
            "Literature review unavailable.",
        )

        return state

    try:

        llm = load_llm()

        comparison_text = ""

        for index, paper in enumerate(
            comparison,
            start=1,
        ):

            comparison_text += f"""
Paper {index}

Title:
{paper.get("Title","")}

Source:
{paper.get("Source","")}

Page:
{paper.get("Page","")}

"""

        prompt = f"""
{PLANNER_PROMPT}

Research Topic

{state["query"]}


=========================
Literature Review
=========================

{literature_review}


=========================
Comparison Matrix
=========================

{comparison_text}


Generate the following:

1. Research Gaps

2. Novel Research Ideas

3. Future Work

4. Suggested Methodology

5. Novelty Score (1-10)

Return the response using clear markdown headings.
"""

        response = llm.invoke(
            [
                HumanMessage(
                    content=prompt
                )
            ]
        )

        output = response.content

        state["research_gaps"] = output

        state["novel_ideas"] = output

        add_log(
            state,
            "Research planning completed.",
        )

    except Exception as e:

        state["error"] = str(e)

        add_log(
            state,
            f"Planner Error: {e}",
        )

    update_step(
        state,
        "Writer",
    )

    return state
