import streamlit as st

from core.graph import graph
from utils.helper import create_project_directories, save_uploaded_file


st.set_page_config(
    page_title="Agentic Academic Research Assistant",
    page_icon="📚",
    layout="wide",
)

create_project_directories()

st.title("📚 Agentic Academic Research Assistant")

st.markdown(
    """
Generate literature reviews, research gaps,
novel ideas and an IEEE style research paper.
"""
)

# =====================================================
# User Input
# =====================================================

query = st.text_input(
    "Research Topic"
)

uploaded_files = st.file_uploader(
    "Upload Research Papers (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
)

# =====================================================
# Run Button
# =====================================================

if st.button("Generate Research"):

    if not query:

        st.warning("Please enter a research topic.")

        st.stop()

    with st.spinner("Running Multi-Agent Workflow..."):

        saved_files = []

        if uploaded_files:

            for pdf in uploaded_files:

                path = save_uploaded_file(pdf)

                saved_files.append(path)

        state = {

            "query": query,

            "uploaded_files": saved_files,

            "execution_plan": "",

            "search_results": [],

            "paper_links": [],

            "parsed_documents": [],

            "chunks": [],

            "embeddings": None,

            "vector_store": None,

            "bm25_index": None,

            "retrieved_docs": [],

            "reranked_docs": [],

            "literature_review": "",

            "comparison_matrix": [],

            "research_gaps": "",

            "novel_ideas": "",

            "generated_paper": "",

            "citations": "",

            "logs": [],

            "current_step": "",

            "error": None,

        }

        result = graph.invoke(state)

    st.success("Workflow Completed!")

    # ===========================================
    # Logs
    # ===========================================

    with st.expander("Workflow Logs"):

        for log in result.get("logs", []):

            st.write(log)

    # ===========================================
    # Literature Review
    # ===========================================

    st.header("📖 Literature Review")

    st.write(result.get("literature_review", "Not generated"))

    # ===========================================
    # Research Gaps
    # ===========================================

    st.header("🔍 Research Gaps")

    result.get("research_gaps", "Not generated")

    # ===========================================
    # Novel Ideas
    # ===========================================

    st.header("💡 Novel Ideas")

    result.get("novel_ideas", "Not generated")

    # ===========================================
    # Paper
    # ===========================================

    st.header("📝 Generated Research Paper")

    result.get("generated_paper", "Not generated")

    st.download_button(
        "Download Paper",
        result["generated_paper"],
        file_name="research_paper.md",
    )

    # ===========================================
    # Citations
    # ===========================================

    st.header("📚 References")

    result.get("citations", "Not generated")

    if result["error"]:

        st.error(result["error"])
