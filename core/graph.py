"""
LangGraph Workflow
"""

from langgraph.graph import StateGraph, END

from core.state import AgentState

from agents.supervisor import supervisor_node
from agents.search_agent import search_node
from agents.parser_agent import parser_node
from agents.chunk_agent import chunk_node
from agents.embedding_agent import embedding_node
from agents.vectorstore_agent import vector_node
from agents.retrieval_agent import retrieval_node
from agents.reranker_agent import reranker_node
from agents.reviewer_agent import reviewer_node
from agents.planner_agent import planner_node
from agents.writer_agent import writer_node
from agents.citation_agent import citation_node


# ==========================================================
# Build Graph
# ==========================================================

workflow = StateGraph(AgentState)

# ==========================================================
# Nodes
# ==========================================================

workflow.add_node("Supervisor", supervisor_node)

workflow.add_node("Search", search_node)

workflow.add_node("Parser", parser_node)

workflow.add_node("Chunk", chunk_node)

workflow.add_node("Embedding", embedding_node)

workflow.add_node("VectorStore", vector_node)

workflow.add_node("Retrieval", retrieval_node)

workflow.add_node("Reranker", reranker_node)

workflow.add_node("Reviewer", reviewer_node)

workflow.add_node("Planner", planner_node)

workflow.add_node("Writer", writer_node)

workflow.add_node("Citation", citation_node)

# ==========================================================
# Flow
# ==========================================================

workflow.set_entry_point("Supervisor")

workflow.add_edge("Supervisor", "Search")

workflow.add_edge("Search", "Parser")

workflow.add_edge("Parser", "Chunk")

workflow.add_edge("Chunk", "Embedding")

workflow.add_edge("Embedding", "VectorStore")

workflow.add_edge("VectorStore", "Retrieval")

workflow.add_edge("Retrieval", "Reranker")

workflow.add_edge("Reranker", "Reviewer")

workflow.add_edge("Reviewer", "Planner")

workflow.add_edge("Planner", "Writer")

workflow.add_edge("Writer", "Citation")

workflow.add_edge("Citation", END)

# ==========================================================
# Compile
# ==========================================================

graph = workflow.compile()
