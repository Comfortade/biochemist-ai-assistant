"""
Biochemist AI Assistant — Home Page
=====================================
Entry point for the Streamlit multi-page app.
"""

import streamlit as st

st.set_page_config(
    page_title="Biochemist AI Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    
    .main-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.8rem;
        font-weight: 600;
        color: #00c49a;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #8892a4;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .feature-card {
        background: #0e1117;
        border: 1px solid #1e2530;
        border-left: 3px solid #00c49a;
        border-radius: 6px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        color: #00c49a;
        margin-bottom: 0.4rem;
    }
    
    .feature-desc {
        font-size: 0.9rem;
        color: #8892a4;
        line-height: 1.5;
    }

    .arch-box {
        background: #0e1117;
        border: 1px solid #1e2530;
        border-radius: 6px;
        padding: 1.5rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        color: #00c49a;
        line-height: 1.8;
    }

    .badge {
        display: inline-block;
        background: #00c49a22;
        color: #00c49a;
        border: 1px solid #00c49a44;
        border-radius: 4px;
        padding: 0.2rem 0.6rem;
        font-size: 0.78rem;
        font-family: 'IBM Plex Mono', monospace;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧬 Biochemist AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">GraphRAG-powered metabolic pathway intelligence — grounded, cited, hallucination-free</div>', unsafe_allow_html=True)

# Badges
st.markdown("""
<div>
    <span class="badge">GraphRAG</span>
    <span class="badge">KEGG Knowledge Graph</span>
    <span class="badge">BioBERT Embeddings</span>
    <span class="badge">FAISS Vector Search</span>
    <span class="badge">Gemini 1.5 Flash</span>
    <span class="badge">NetworkX</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### What this system does")
    
    features = [
        ("🗺️ Pathway Explorer", "Visualize and traverse metabolic pathways interactively. Nodes are metabolites and enzymes; edges are reactions sourced directly from KEGG."),
        ("💬 Ask the Assistant", "Ask any biochemistry question. The system retrieves relevant graph subgraphs and textbook passages before generating a grounded answer — no invented steps, no hallucinated enzymes."),
        ("🔬 Molecule Viewer", "Render 2D chemical structures from SMILES notation for any metabolite in the knowledge graph."),
    ]
    
    for title, desc in features:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### Architecture")
    st.markdown("""
    <div class="arch-box">
    User Query<br>
    &nbsp;&nbsp;&nbsp;&nbsp;│<br>
    &nbsp;&nbsp;&nbsp;&nbsp;▼<br>
    Graph Search (KEGG KG)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;│<br>
    &nbsp;&nbsp;&nbsp;&nbsp;▼<br>
    Vector Search (BioBERT)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;│<br>
    &nbsp;&nbsp;&nbsp;&nbsp;▼<br>
    Context Fusion<br>
    &nbsp;&nbsp;&nbsp;&nbsp;│<br>
    &nbsp;&nbsp;&nbsp;&nbsp;▼<br>
    Constrained Gemini LLM<br>
    &nbsp;&nbsp;&nbsp;&nbsp;│<br>
    &nbsp;&nbsp;&nbsp;&nbsp;▼<br>
    Grounded Answer + Citations
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Data Sources")
col3, col4, col5 = st.columns(3)
with col3:
    st.info("**KEGG REST API**\nPathway maps, reaction equations, enzyme metadata")
with col4:
    st.info("**Biochemistry Free For All**\nAhern & Rajagopal — open-access textbook")
with col5:
    st.info("**LibreTexts Biochemistry**\nOpen-access community textbook")