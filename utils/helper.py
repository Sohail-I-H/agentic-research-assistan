"""
Common helper utilities used across the project.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import List

# ==========================================================
# Project Directories
# ==========================================================

UPLOAD_DIR = Path("data/uploads")
VECTOR_DB_DIR = Path("data/faiss_db")


def create_project_directories() -> None:
    """
    Create required project directories if they do not exist.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Logging
# ==========================================================

def add_log(state: dict, message: str) -> dict:
    """
    Append a log message to the shared state.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")

    state.setdefault("logs", [])
    state["logs"].append(f"[{timestamp}] {message}")

    return state


# ==========================================================
# Workflow Status
# ==========================================================

def update_step(state: dict, step: str) -> dict:
    """
    Update the current workflow step.
    """
    state["current_step"] = step
    return state


# ==========================================================
# Error Handling
# ==========================================================

def set_error(state: dict, error_message: str) -> dict:
    """
    Store an error message inside the workflow state.
    """
    state["error"] = error_message
    add_log(state, f"ERROR: {error_message}")
    return state


# ==========================================================
# Timestamp
# ==========================================================

def current_timestamp() -> str:
    """
    Return current timestamp.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==========================================================
# Text Cleaning
# ==========================================================

def clean_text(text: str) -> str:
    """
    Basic cleanup for extracted PDF text.
    """

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


# ==========================================================
# File Validation
# ==========================================================

def allowed_pdf(filename: str) -> bool:
    """
    Validate uploaded PDF file.
    """

    return filename.lower().endswith(".pdf")


# ==========================================================
# Save Uploaded File
# ==========================================================

def save_uploaded_file(uploaded_file) -> str:
    """
    Save a Streamlit uploaded PDF.

    Returns
    -------
    str
        Path of saved file.
    """

    create_project_directories()

    save_path = UPLOAD_DIR / uploaded_file.name

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(save_path)


# ==========================================================
# Deduplicate List
# ==========================================================

def remove_duplicates(items: List[str]) -> List[str]:
    """
    Remove duplicate strings while preserving order.
    """

    seen = set()
    unique = []

    for item in items:
        if item not in seen:
            unique.append(item)
            seen.add(item)

    return unique


# ==========================================================
# Safe Filename
# ==========================================================

def safe_filename(name: str) -> str:
    """
    Convert arbitrary text into a filesystem-safe filename.
    """

    name = re.sub(r"[^\w\-_. ]", "", name)
    name = name.replace(" ", "_")

    return name[:120]


# ==========================================================
# Reset Workflow
# ==========================================================

def reset_state(state: dict) -> dict:
    """
    Clear transient workflow values while preserving user input.
    """

    state["logs"] = []
    state["current_step"] = "Idle"
    state["error"] = None

    state["search_results"] = []
    state["paper_links"] = []

    state["parsed_documents"] = []

    state["chunks"] = []

    state["retrieved_docs"] = []
    state["reranked_docs"] = []

    state["comparison_matrix"] = []

    state["research_gaps"] = ""

    state["novel_ideas"] = ""

    state["generated_paper"] = ""

    state["citations"] = []

    return state
