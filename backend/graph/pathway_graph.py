"""
Pathway Knowledge Graph Builder:
Parses KEGG KGML files and constructs a directed NetworkX graph
representing metabolic pathways.

Graph structure:
- Nodes: metabolites (compounds) and enzymes
- Edges: reactions connecting substrate → enzyme → product
- Edge attributes: reaction ID, equation, cofactors, reversibility

This graph is the foundation of the GraphRAG retrieval system.
Querying it ensures factually grounded, step-complete pathway answers.
"""

import xml.etree.ElementTree as ET
import networkx as nx
import json
import logging
from pathlib import Path
from typing import Optional
from backend.graph.kegg_loader import fetch_compound_info, fetch_reaction_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_kgml(kgml_string: str, pathway_name: str) -> nx.DiGraph:

    G = nx.DiGraph(name=pathway_name)
    root = ET.fromstring(kgml_string)

    # Build entry ID → name/type lookup from KGML entries
    entries = {}
    for entry in root.findall("entry"):
        entry_id = entry.get("id")
        entry_name = entry.get("name", "").replace("cpd:", "").replace("hsa:", "")
        entry_type = entry.get("type", "")
        entries[entry_id] = {"name": entry_name, "type": entry_type}

    # Parse reactions and build edges
    for reaction in root.findall("reaction"):
        rxn_id = reaction.get("name", "").replace("rn:", "")
        rxn_type = reaction.get("type", "reversible")

        substrates = [
            s.get("name", "").replace("cpd:", "")
            for s in reaction.findall("substrate")
        ]
        products = [
            p.get("name", "").replace("cpd:", "")
            for p in reaction.findall("product")
        ]

        # Add reaction node
        G.add_node(rxn_id, node_type="reaction", reversible=(rxn_type == "reversible"))

        # Add substrate → reaction edges
        for substrate in substrates:
            G.add_node(substrate, node_type="compound")
            G.add_edge(substrate, rxn_id, role="substrate")

        # Add reaction → product edges
        for product in products:
            G.add_node(product, node_type="compound")
            G.add_edge(rxn_id, product, role="product")

    logger.info(
        f"Built graph for {pathway_name}: "
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    return G


def build_full_graph(kgml_dir: str = "data/raw") -> nx.DiGraph:
    """
    Load all saved KGML files and merge them into one master knowledge graph.
    The merged graph allows cross-pathway queries
    (e.g., how glycolysis feeds into TCA cycle).
    """
    master_graph = nx.DiGraph(name="biochemistry_master")
    kgml_dir = Path(kgml_dir)

    for kgml_file in kgml_dir.glob("*.kgml"):
        pathway_name = kgml_file.stem
        kgml_string = kgml_file.read_text(encoding="utf-8")
        pathway_graph = parse_kgml(kgml_string, pathway_name)
        master_graph = nx.compose(master_graph, pathway_graph)
        logger.info(f"Merged {pathway_name} into master graph")

    logger.info(
        f"Master graph: {master_graph.number_of_nodes()} nodes, "
        f"{master_graph.number_of_edges()} edges"
    )
    return master_graph


def graph_query(G: nx.DiGraph, entity: str, depth: int = 2) -> dict:
    """
    Retrieve the local subgraph around a given entity (metabolite or enzyme).
    Used during RAG retrieval to extract relevant pathway context.
    
    Args:
        G: The master knowledge graph
        entity: Compound or reaction ID to query around
        depth: How many hops to traverse from the entity
    
    Returns:
        Dict with neighbors, edges, and path information
    """
    if entity not in G:
        # Try partial match
        matches = [n for n in G.nodes if entity.lower() in str(n).lower()]
        if not matches:
            return {"error": f"Entity '{entity}' not found in knowledge graph"}
        entity = matches[0]
        logger.info(f"Partial match found: {entity}")

    # Get ego graph (subgraph within N hops)
    ego = nx.ego_graph(G, entity, radius=depth)
    
    predecessors = list(G.predecessors(entity))
    successors = list(G.successors(entity))

    return {
        "entity": entity,
        "node_type": G.nodes[entity].get("node_type", "unknown"),
        "predecessors": predecessors,
        "successors": successors,
        "subgraph_nodes": list(ego.nodes(data=True)),
        "subgraph_edges": list(ego.edges(data=True)),
        "total_connections": len(predecessors) + len(successors),
    }


def save_graph(G: nx.DiGraph, output_path: str = "data/embeddings/graph.json"):
    """Serialize the graph to JSON for caching."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(G)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Graph saved to {output_path}")


def load_graph(input_path: str = "data/embeddings/graph.json") -> Optional[nx.DiGraph]:
    """Load a previously saved graph from JSON."""
    path = Path(input_path)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    G = nx.node_link_graph(data)
    logger.info(f"Graph loaded from {input_path}")
    return G


if __name__ == "__main__":
    from backend.graph.kegg_loader import save_raw_kgml
    save_raw_kgml()
    G = build_full_graph()
    save_graph(G)
    print(f"Graph ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")