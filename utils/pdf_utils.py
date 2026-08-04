"""
PDF utilities using PyMuPDF.

Responsibilities:
- Read uploaded PDFs
- Extract page-wise text
- Extract metadata
- Return structured documents for downstream agents
"""

from __future__ import annotations

import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List

from utils.helper import clean_text


# ==========================================================
# Metadata Extraction
# ==========================================================

def extract_metadata(pdf_path: str) -> Dict:
    """
    Extract metadata from a PDF file.
    """

    document = fitz.open(pdf_path)

    metadata = document.metadata or {}

    result = {
        "title": metadata.get("title") or Path(pdf_path).stem,
        "author": metadata.get("author", "Unknown"),
        "subject": metadata.get("subject", ""),
        "keywords": metadata.get("keywords", ""),
        "creator": metadata.get("creator", ""),
        "producer": metadata.get("producer", ""),
        "page_count": document.page_count,
    }

    document.close()

    return result


# ==========================================================
# Extract Full Text
# ==========================================================

def extract_text(pdf_path: str) -> str:
    """
    Extract complete text from a PDF.
    """

    document = fitz.open(pdf_path)

    pages = []

    for page in document:
        text = page.get_text("text")

        if text:
            pages.append(clean_text(text))

    document.close()

    return "\n".join(pages)


# ==========================================================
# Page-wise Extraction
# ==========================================================

def extract_pages(pdf_path: str) -> List[Dict]:
    """
    Extract text page by page.
    """

    document = fitz.open(pdf_path)

    pages = []

    for index, page in enumerate(document):

        text = page.get_text("text")

        pages.append(
            {
                "page": index + 1,
                "text": clean_text(text),
            }
        )

    document.close()

    return pages


# ==========================================================
# Structured Document
# ==========================================================

def parse_pdf(pdf_path: str) -> Dict:
    """
    Parse PDF into a structured dictionary.
    """

    return {
        "metadata": extract_metadata(pdf_path),
        "pages": extract_pages(pdf_path),
        "text": extract_text(pdf_path),
        "source": pdf_path,
    }


# ==========================================================
# Multiple PDFs
# ==========================================================

def parse_multiple_pdfs(pdf_files: List[str]) -> List[Dict]:
    """
    Parse multiple uploaded PDFs.
    """

    documents = []

    for pdf in pdf_files:
        try:
            documents.append(parse_pdf(pdf))
        except Exception as e:
            print(f"Failed to parse {pdf}: {e}")

    return documents


# ==========================================================
# Statistics
# ==========================================================

def pdf_statistics(document: Dict) -> Dict:
    """
    Generate simple statistics for a parsed document.
    """

    text = document.get("text", "")

    words = len(text.split())

    chars = len(text)

    pages = len(document.get("pages", []))

    return {
        "pages": pages,
        "words": words,
        "characters": chars,
    }


# ==========================================================
# Preview
# ==========================================================

def preview_document(document: Dict, max_chars: int = 1000) -> str:
    """
    Return a short preview of the extracted text.
    """

    text = document.get("text", "")

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."
