"""
Ask the Assistant — GraphRAG Q&A Interface
============================================
The main chat interface. Every answer is grounded in retrieved
knowledge graph data and textbook passages.
Source attribution panel shows exactly which sources were used.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from backend.pipeline import BiochemPipeline

st.set_page_config(
    page_title="Ask the Assistant",
    page_icon="💬",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .answer-box {
        background: #0e1117;
        border: 1px solid #1e2530;
        border-left: 4px solid #00c49a;
        border-radius: 6px;
        padding: 1.5rem;
        margin: 1rem 0;
        line-height: 1.7;
    }
    .source-tag {
        display: inline-block;
        background: #00c49a15;
        border: 1px solid #00c49a33;
        color: #00c49a;
        border-radius: 4px;
        padding: 0.15rem 0.5rem;
        font-size: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        margin: 0.2rem;
    }
    .graph-tag {
        background: #4f8ef715;
        border-color: #4f8ef733;
        color: #4f8ef7;
    }
</style>
""", unsafe_allow_html=True)

st.title("💬 Ask the Assistant")
st.caption("All answers are grounded in KEGG pathway data and verified textbook passages.")

# Initialize pipeline (cached)
@st.cache_resource(show_spinner="Initializing GraphRAG pipeline...")
def load_pipeline():
    return BiochemPipeline()

pipeline = load_pipeline()

# Example questions
st.markdown("**Try asking:**")
examples = [
    "What are the steps of glycolysis?",
    "How does pyruvate enter the TCA cycle?",
    "What inhibits phosphofructokinase?",
    "How is ATP generated in beta-oxidation?",
    "What is the role of NAD+ in the TCA cycle?",
]
cols = st.columns(len(examples))
for i, example in enumerate(examples):
    if cols[i].button(example, key=f"ex_{i}", use_container_width=True):
        st.session_state["query"] = example

st.markdown("---")

# Query input
query = st.text_input(
    "Your question:",
    value=st.session_state.get("query", ""),
    placeholder="e.g. How does high glucose inhibit the TCA cycle?",
    key="query_input"
)

if st.button("Ask", type="primary", use_container_width=False):
    if query.strip():
        with st.spinner("Retrieving from knowledge graph and textbooks..."):
            result = pipeline.query(query)

        # Answer
        st.markdown("### Answer")
        st.markdown(
            f'<div class="answer-box">{result["answer"]}</div>',
            unsafe_allow_html=True
        )

        # Retrieval stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Graph Entities Found", result["graph_hits"])
        col2.metric("Textbook Passages Used", result["vector_hits"])
        col3.metric("Model", result["model"])

        # Source attribution
        st.markdown("### Sources Used")
        if result["sources_used"]:
            for source in result["sources_used"]:
                if source["type"] == "textbook":
                    st.markdown(
                        f'<span class="source-tag">📖 {source["source"]} — p.{source["page"]} '
                        f'(relevance: {source["score"]:.3f})</span>',
                        unsafe_allow_html=True
                    )
                elif source["type"] == "knowledge_graph":
                    st.markdown(
                        f'<span class="source-tag graph-tag">🔗 KEGG Graph: {source["entity"]} '
                        f'({source["node_type"]})</span>',
                        unsafe_allow_html=True
                    )
        else:
            st.warning("No sources retrieved — answer may be less reliable.")

        # History
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].append({"q": query, "a": result["answer"]})

    else:
        st.warning("Please enter a question.")

# Chat history
if st.session_state.get("history"):
    st.markdown("---")
    st.markdown("### Previous Questions")
    for item in reversed(st.session_state["history"][:-1]):
        with st.expander(item["q"]):
            st.write(item["a"])