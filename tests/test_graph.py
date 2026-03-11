"""
Tests for Knowledge Graph construction and querying.
Run with: pytest tests/test_graph.py -v
"""

import pytest
import networkx as nx
from unittest.mock import patch, MagicMock
from backend.graph.pathway_graph import parse_kgml, graph_query, build_full_graph


SAMPLE_KGML = """<?xml version="1.0"?>
<pathway name="path:hsa00010" org="hsa" number="00010" title="Glycolysis">
    <entry id="1" name="cpd:C00267" type="compound"/>
    <entry id="2" name="cpd:C00092" type="compound"/>
    <entry id="3" name="cpd:C00085" type="compound"/>
    <reaction id="4" name="rn:R00299" type="irreversible">
        <substrate name="cpd:C00267"/>
        <product name="cpd:C00092"/>
    </reaction>
    <reaction id="5" name="rn:R01015" type="reversible">
        <substrate name="cpd:C00092"/>
        <product name="cpd:C00085"/>
    </reaction>
</pathway>"""


def test_parse_kgml_returns_digraph():
    G = parse_kgml(SAMPLE_KGML, "test_pathway")
    assert isinstance(G, nx.DiGraph)


def test_parse_kgml_correct_node_count():
    G = parse_kgml(SAMPLE_KGML, "test_pathway")
    # 3 compounds + 2 reactions = 5 nodes
    assert G.number_of_nodes() == 5


def test_parse_kgml_correct_edge_count():
    G = parse_kgml(SAMPLE_KGML, "test_pathway")
    # Each reaction: 1 substrate edge + 1 product edge = 4 edges
    assert G.number_of_edges() == 4


def test_parse_kgml_node_types():
    G = parse_kgml(SAMPLE_KGML, "test_pathway")
    reaction_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "reaction"]
    compound_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "compound"]
    assert len(reaction_nodes) == 2
    assert len(compound_nodes) == 3


def test_graph_query_found():
    G = parse_kgml(SAMPLE_KGML, "test_pathway")
    result = graph_query(G, "R00299")
    assert "error" not in result
    assert result["entity"] == "R00299"


def test_graph_query_not_found():
    G = parse_kgml(SAMPLE_KGML, "test_pathway")
    result = graph_query(G, "nonexistent_entity_xyz")
    assert "error" in result


def test_graph_query_has_successors():
    G = parse_kgml(SAMPLE_KGML, "test_pathway")
    result = graph_query(G, "R00299")
    assert "successors" in result
    assert len(result["successors"]) > 0


def test_graph_is_directed():
    G = parse_kgml(SAMPLE_KGML, "test_pathway")
    assert nx.is_directed(G)