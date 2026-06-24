import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.hybrid_search import hybrid_search, load_all_chunks, build_bm25_index
from generation.synthesizer import synthesize

# Variables
SOURCE_PREVIEW_LENGTH = 500

# --- Page Config ---
st.set_page_config(
    page_title="SEC Filing RAG",
    page_icon="📈",
    layout="wide"
)

# --- Load Components (cached so they don't reload on every query) ---
@st.cache_resource
def load_components():
    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("BAAI/bge-m3")
    chunks = load_all_chunks()
    bm25 = build_bm25_index(chunks)
    return client, model, bm25, chunks


# --- UI ---
st.title("📈 SEC Filing RAG")
st.markdown("Query and compare SEC 10-K filings using natural language.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This tool uses a multi-document RAG pipeline to answer questions 
    about SEC 10-K filings.
    
    **Current Companies:**
    - Apple (AAPL)
    
    **Pipeline:**
    - Hybrid search (dense + BM25)
    - Reciprocal Rank Fusion
    - Gemini 2.5 Flash generation
    """)

    st.header("Example Questions")
    example_questions = [
        "What are Apple's biggest risk factors?",
        "How does Apple describe its AI strategy?",
        "What new products did Apple announce in 2025?",
        "How does Apple describe competition in its markets?",
        "What does Apple say about tariffs and trade policy?"
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.query = q

# Main area
query = st.text_input(
    "Ask a question about SEC filings:",
    value=st.session_state.get("query", ""),
    placeholder="e.g. What are Apple's biggest risk factors related to AI?"
)

if st.button("Search", type="primary"):
    if query:
        with st.spinner("Retrieving and generating answer..."):
            client, model, bm25, chunks = load_components()
            retrieved = hybrid_search(client, model, bm25, chunks, query)
            result = synthesize(query, retrieved)

        # Answer
        st.subheader("Answer")
        st.markdown(result["answer"])

        # Sources
        st.subheader("Sources Used")
        for source in result["sources"]:
            with st.expander(
                f"[Source {source['source_num']}] "
                f"{source['company'].upper()} | {source['section']}"
            ):
                chunk = next(
                    (c for c in chunks if c["chunk_id"] == source["chunk_id"]),
                    None
                )
                if chunk:
                    st.text(chunk["text"][:SOURCE_PREVIEW_LENGTH] + "...")
    else:
        st.warning("Please enter a question.")