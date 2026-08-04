"""
Reviewer Agent

Responsibilities:
- Analyze retrieved papers
- Generate literature review
- Create comparison matrix
"""

from __future__ import annotations

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from core.prompts import REVIEWER_PROMPT
from core.state import AgentState
from utils.helper import add_log, update_step


# ==========================================================
# Load LLM
# ==========================================================

@st.cache_resource
def load_llm():

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )


# ==========================================================
# Build Context
# ==========================================================

def build_context(documents):

    context = []

    for doc in documents:

        context.append(

            f"""
Title:
{doc.get('title','Unknown')}

Source:
{doc.get('source','')}

Page:
{doc.get('page','')}

Content:
{doc.get('text','')}
"""
        )

    return "\n\n".join(context)


# ==========================================================
# Reviewer Node
# ==========================================================

def reviewer_node(
    state: AgentState,
) -> AgentState:

    update_step(
        state,
        "Reviewer",
    )

    add_log(
        state,
        "Analyzing retrieved literature...",
    )

    docs = state.get(
        "reranked_docs",
        [],
    )

    if not docs:

        add_log(
            state,
            "No documents available.",
        )

        return state

    try:

        llm = load_llm()

        context = build_context(
            docs
        )

        prompt = f"""
{REVIEWER_PROMPT}

Research Topic:

{state["query"]}

Research Context:

{context}

Return:

1. Literature Review

2. Comparison Matrix

3. Strengths

4. Weaknesses

5. Research Trends
"""

        response = llm.invoke(
            [
                HumanMessage(
                    content=prompt
                )
            ]
        )

        state["literature_review"] = response.content

        comparison = []

        for doc in docs:

            comparison.append(

                {

                    "Title": doc.get(
                        "title",
                        "Unknown",
                    ),

                    "Source": doc.get(
                        "source",
                        "",
                    ),

                    "Page": doc.get(
                        "page",
                        "",
                    ),

                }

            )

        state[
            "comparison_matrix"
        ] = comparison

        add_log(
            state,
            "Literature review completed.",
        )

    except Exception as e:

        state["error"] = str(e)

        add_log(
            state,
            f"Reviewer Error: {e}",
        )

    update_step(
        state,
        "Planner",
    )

    return state
