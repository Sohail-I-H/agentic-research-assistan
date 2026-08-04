"""
Supervisor Agent

Responsibilities:
- Understand user request
- Plan workflow
- Initialize state
- Coordinate downstream agents
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from core.prompts import SUPERVISOR_PROMPT
from core.state import AgentState
from utils.helper import add_log, update_step


# ==========================================================
# Load LLM
# ==========================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
)


# ==========================================================
# Supervisor Node
# ==========================================================

def supervisor_node(state: AgentState) -> AgentState:
    """
    Entry point of the LangGraph workflow.
    """

    update_step(state, "Supervisor")

    add_log(state, "Supervisor Agent Started")

    query = state.get("query", "").strip()

    if not query:

        state["error"] = "Research topic is empty."

        add_log(state, "No research topic provided.")

        return state

    prompt = f"""
{SUPERVISOR_PROMPT}

User Research Topic:

{query}

Return a concise execution plan.
"""

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    execution_plan = response.content

    state["execution_plan"] = execution_plan

    add_log(state, "Execution plan generated.")

    update_step(state, "Search Agent")

    return state
