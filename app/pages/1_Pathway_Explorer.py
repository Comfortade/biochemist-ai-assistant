"""
Pathway Explorer — Interactive Knowledge Graph Visualization
=============================================================
Renders the KEGG metabolic knowledge graph using streamlit-agraph.
Users can select pathways, click nodes to inspect metabolites/enzymes,
and trace reaction sequences visually.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from backend.pipeline import BiochemPipeline
from backend.graph.pathway_graph import graph_query
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(
    page_title="Pathway Explorer",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Pathway Explorer")
st.caption("Interactive visualization of KEGG metabolic pathway knowledge graph.")

@st.cache_resource(show_spinner="Loading knowledge graph...")
def load_pipeline():
    return BiochemPipeline()

pipeline = load_pipeline()
G = pipeline.get_graph()

if G is None:
    st.error("Knowledge graph not loaded. Check pipeline initialization.")
    st.stop()

st.success(f"Knowledge graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Controls
col1, col2 = st.columns([2, 1])
with col1:
    search_entity = st.text_input(
        "Search for a metabolite or enzyme:",
        placeholder="e.g. pyruvate, citrate, R00200"
    )
with col2:
    depth = st.slider("Traversal depth", min_value=1, max_value=3, value=2)

max_nodes = st.slider("Max nodes to display", min_value=20, max_value=200, value=80)

# Build visualization
st.markdown("---")

if search_entity:
    result = graph_query(G, search_entity, depth=depth)
    if "error" in result:
        st.warning(result["error"])
        subgraph_nodes = list(G.nodes())[:max_nodes]
    else:
        import networkx as nx
        ego = nx.ego_graph(G, result["entity"], radius=depth)
        subgraph_nodes = list(ego.nodes())[:max_nodes]
        st.info(
            f"Showing subgraph around **{result['entity']}** — "
            f"{len(subgraph_nodes)} nodes, depth {depth}"
        )
else:
    subgraph_nodes = list(G.nodes())[:max_nodes]
    st.info(f"Showing first {max_nodes} nodes. Search above to focus on a specific entity.")

# Build agraph nodes and edges
COLOR_MAP = {
    "compound": "#00c49a",
    "reaction": "#4f8ef7",
    "enzyme": "#f7a24f",
    "unknown": "#8892a4",
}

nodes = []
edges = []
node_set = set(subgraph_nodes)

for node in subgraph_nodes:
    node_data = G.nodes[node]
    node_type = node_data.get("node_type", "unknown")
    color = COLOR_MAP.get(node_type, "#8892a4")
    size = 20 if node_type == "reaction" else 15
    nodes.append(
        Node(
            id=str(node),
            label=str(node)[:12],
            size=size,
            color=color,
            title=f"{node} ({node_type})",
        )
    )

for u, v, data in G.edges(data=True):
    if u in node_set and v in node_set:
        edges.append(
            Edge(
                source=str(u),
                target=str(v),
                label=data.get("role", ""),
                color="#1e2530",
            )
        )

config = Config(
    width="100%",
    height=600,
    directed=True,
    physics=True,
    hierarchical=False,
    nodeHighlightBehavior=True,
    highlightColor="#00c49a",
    collapsible=False,
)

agraph(nodes=nodes, edges=edges, config=config)

# Legend
st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.markdown("🟢 **Compound** (metabolite)")
col2.markdown("🔵 **Reaction** (KEGG reaction ID)")
col3.markdown("🟠 **Enzyme**")

# Node inspector
if search_entity:
    result = graph_query(G, search_entity, depth=1)
    if "error" not in result:
        st.markdown("### Node Details")
        st.json(result)