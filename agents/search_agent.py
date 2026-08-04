"""
Search Agent

Responsibilities:
- Search arXiv
- Search Tavily
- Merge results
- Remove duplicates
"""

from __future__ import annotations

import arxiv
import streamlit as st

from tavily import TavilyClient

from core.state import AgentState
from utils.helper import (
    add_log,
    remove_duplicates,
    update_step,
)


# ==========================================================
# Tavily Client
# ==========================================================

def get_tavily_client():
    """
    Initialize Tavily client using Streamlit Secrets.
    """

    return TavilyClient(
        api_key=st.secrets["TAVILY_API_KEY"]
    )


# ==========================================================
# arXiv Search
# ==========================================================

def search_arxiv(
    query: str,
    max_results: int = 5,
):
    """
    Search arXiv.
    """

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []

    for result in search.results():

        papers.append(
            {
                "title": result.title,
                "authors": ", ".join(
                    author.name
                    for author in result.authors
                ),
                "summary": result.summary,
                "year": result.published.year,
                "url": result.entry_id,
                "pdf": result.pdf_url,
                "source": "arXiv",
            }
        )

    return papers


# ==========================================================
# Tavily Search
# ==========================================================

def search_tavily(
    query: str,
    max_results: int = 5,
):
    """
    Search web using Tavily.
    """

    client = get_tavily_client()

    response = client.search(
        query=query,
        max_results=max_results,
    )

    papers = []

    for item in response.get("results", []):

        papers.append(
            {
                "title": item.get("title", ""),
                "authors": "Unknown",
                "summary": item.get("content", ""),
                "year": "",
                "url": item.get("url", ""),
                "pdf": "",
                "source": "Tavily",
            }
        )

    return papers


# ==========================================================
# Merge Results
# ==========================================================

def merge_results(
    arxiv_results,
    tavily_results,
):
    """
    Merge and deduplicate search results.
    """

    merged = arxiv_results + tavily_results

    seen = set()

    unique = []

    for paper in merged:

        title = paper["title"].lower()

        if title not in seen:

            seen.add(title)

            unique.append(paper)

    return unique


# ==========================================================
# LangGraph Node
# ==========================================================

def search_node(
    state: AgentState,
) -> AgentState:
    """
    Search node for LangGraph.
    """

    update_step(
        state,
        "Paper Search",
    )

    add_log(
        state,
        "Searching research papers...",
    )

    query = state["query"]

    try:

        arxiv_results = search_arxiv(query)

        tavily_results = search_tavily(query)

        papers = merge_results(
            arxiv_results,
            tavily_results,
        )

        state["search_results"] = papers

        links = []

        for paper in papers:

            if paper["url"]:
                links.append(paper["url"])

            if paper["pdf"]:
                links.append(paper["pdf"])

        state["paper_links"] = remove_duplicates(
            links
        )

        add_log(
            state,
            f"Found {len(papers)} papers."
        )

    except Exception as e:

        state["error"] = str(e)

        add_log(
            state,
            f"Search failed: {e}"
        )

    update_step(
        state,
        "PDF Parser",
    )

    return state
