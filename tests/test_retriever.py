"""
Tests for hybrid retrieval logic.
Run with: pytest tests/test_retriever.py -v
"""

import pytest
import networkx as nx
from unittest.mock import MagicMock, patch
from backend.rag.retriever import extract_biochem_entities, hybrid_retrieve
from backend.graph.pathway_graph import parse_kgml


SAMPLE_KGML = """<?xml version="1.0"?>
<pathway name="path:hsa00010" org="hsa" number="00010" title="Glycolysis">
    <entry id="1" name="cpd:C00267" type="compound"/>
    <entry id="2" name="cpd:C00022" type="compound"/>
    <reaction id="3" name="rn:R00299" type="irreversible">
        <substrate name="cpd:C00267"/>
        <product name="cpd:C00022"/>
    </reaction>
</pathway>"""


def test_extract_entities_glycolysis():
    query = "What happens to glucose in glycolysis?"
    entities = extract_biochem_entities(query)
    assert "glucose" in entities
    assert "glycolysis" in entities


def test_extract_entities_tca():
    query = "How does pyruvate enter the TCA cycle?"
    entities = extract_biochem_entities(query)
    assert "pyruvate" in entities
    assert "tca" in entities


def test_extract_entities_empty():
    query = "What is biochemistry?"
    entities = extract_biochem_entities(query)
    assert isinstance(entities, list)


def test_hybrid_retrieve_structure():
    G = parse_kgml(SAMPLE_KGML, "test")
    mock_index = MagicMock()
    mock_index.search.return_value = (
        [[0.9, 0.8]],
        [[0, 1]]
    )
    mock_chunks = [
        {"text": "Glucose is phosphorylated by hexokinase.", "source": "test", "page": 1, "chunk_id": "t_p1_c0"},
        {"text": "Pyruvate is the end product of glycolysis.", "source": "test", "page": 2, "chunk_id": "t_p2_c0"},
    ]

    with patch("backend.rag.retriever.SentenceTransformer") as mock_st:
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 768).astype("float32")
        mock_st.return_value = mock_model

        result = hybrid_retrieve("What is glycolysis?", G, mock_index, mock_chunks)

    assert "query" in result
    assert "graph_context" in result
    assert "vector_context" in result
    assert "has_graph_data" in result
    assert "has_vector_data" in result


def test_hybrid_retrieve_returns_vector_results():
    G = nx.DiGraph()
    mock_index = MagicMock()
    mock_index.search.return_value = ([[0.95]], [[0]])
    mock_chunks = [
        {"text": "ATP is produced in glycolysis.", "source": "test", "page": 5, "chunk_id": "t_p5_c0"}
    ]

    with patch("backend.rag.retriever.SentenceTransformer") as mock_st:
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 768).astype("float32")
        mock_st.return_value = mock_model

        result = hybrid_retrieve("How is ATP produced?", G, mock_index, mock_chunks)

    assert len(result["vector_context"]) == 1