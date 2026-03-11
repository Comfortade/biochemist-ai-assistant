"""
End-to-end pipeline integration tests.
Run with: pytest tests/test_pipeline.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
import networkx as nx


def test_pipeline_query_returns_expected_keys():
    """Pipeline query result must always contain these keys."""
    with patch("backend.pipeline.BiochemPipeline._initialize"):
        from backend.pipeline import BiochemPipeline
        pipeline = BiochemPipeline.__new__(BiochemPipeline)
        pipeline.graph = nx.DiGraph()
        pipeline.index = MagicMock()
        pipeline.chunks = []
        pipeline.is_ready = True

        with patch("backend.pipeline.hybrid_retrieve") as mock_retrieve:
            with patch("backend.pipeline.ask_gemini") as mock_gemini:
                mock_retrieve.return_value = {
                    "query": "test",
                    "graph_context": [],
                    "vector_context": [],
                    "has_graph_data": False,
                    "has_vector_data": False,
                }
                mock_gemini.return_value = {
                    "answer": "Test answer.",
                    "sources_used": [],
                    "had_graph_data": False,
                    "had_vector_data": False,
                    "model": "gemini-1.5-flash",
                }

                result = pipeline.query("What is glycolysis?")

    assert "answer" in result
    assert "sources_used" in result
    assert "graph_hits" in result
    assert "vector_hits" in result


def test_pipeline_not_ready_returns_safe_response():
    """Pipeline should return a safe message when not initialized."""
    with patch("backend.pipeline.BiochemPipeline._initialize"):
        from backend.pipeline import BiochemPipeline
        pipeline = BiochemPipeline.__new__(BiochemPipeline)
        pipeline.is_ready = False
        pipeline.graph = None
        pipeline.index = None
        pipeline.chunks = []

        result = pipeline.query("What is ATP?")

    assert "answer" in result
    assert "not ready" in result["answer"].lower()


def test_pipeline_answer_is_string():
    """Answer must always be a string, never None."""
    with patch("backend.pipeline.BiochemPipeline._initialize"):
        from backend.pipeline import BiochemPipeline
        pipeline = BiochemPipeline.__new__(BiochemPipeline)
        pipeline.graph = nx.DiGraph()
        pipeline.index = MagicMock()
        pipeline.chunks = []
        pipeline.is_ready = True

        with patch("backend.pipeline.hybrid_retrieve") as mock_retrieve:
            with patch("backend.pipeline.ask_gemini") as mock_gemini:
                mock_retrieve.return_value = {
                    "query": "test",
                    "graph_context": [],
                    "vector_context": [],
                    "has_graph_data": False,
                    "has_vector_data": False,
                }
                mock_gemini.return_value = {
                    "answer": "Some answer.",
                    "sources_used": [],
                    "had_graph_data": False,
                    "had_vector_data": False,
                    "model": "gemini-1.5-flash",
                }

                result = pipeline.query("test question")

    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0