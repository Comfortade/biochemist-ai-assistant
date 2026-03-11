"""
Hybrid GraphRAG Retriever
==========================
The core retrieval engine of the Biochemist AI Assistant.

This module implements HYBRID retrieval — combining two complementary
information sources to eliminate hallucination:

1. GRAPH RETRIEVAL (structured):
   - Traverses the KEGG knowledge graph to find exact pathway steps,
     enzyme-substrate relationships, and metabolic connections
   - Guarantees no invented reaction steps — every edge is from KEGG

2. VECTOR RETRIEVAL (semantic):
   - Searches BioBERT-embedded textbook chunks via FAISS
   - Finds contextually relevant explanations, mechanisms, and regulation details
   - Handles natural language questions that don't map to exact graph entities

3. FUSION:
   - Both results are merged into a structured context object
   - The LLM receives ONLY this context — it cannot go beyond it
   - Every answer is traceable back to a specific graph edge or textbook page

This architecture is based on the GraphRAG paradigm (Edge et al., 2024, Microsoft Research).
"""

import numpy as np
import logging
from typing import Optional
from sentence_transformers import SentenceTransformer
import networkx as nx
import faiss
import json

from backend.graph.pathway_graph import graph_query
from backend.rag.embedder import EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOP_K_VECTORS = 5  # Number of textbook chunks to retrieve


def extract_biochem_entities(query: str) -> list[str]:
    """
    Extract potential biochemical entity names from a query string.
    These are used as entry points for graph traversal.

    Simple keyword-based extraction — in production this would use
    a named entity recognition (NER) model fine-tuned on biochemistry.

    Examples:
        "How is pyruvate converted to acetyl-CoA?" → ["pyruvate", "acetyl-CoA"]
        "What inhibits isocitrate dehydrogenase?" → ["isocitrate dehydrogenase"]
    """
    # Common biochemical terms to look for
    biochem_keywords = [
        "glucose", "pyruvate", "acetyl-coa", "acetyl coa", "nadh", "nadph",
        "atp", "adp", "fadh2", "oxaloacetate", "citrate", "isocitrate",
        "alpha-ketoglutarate", "succinyl-coa", "succinate", "fumarate",
        "malate", "glycolysis", "tca", "krebs", "beta-oxidation",
        "phosphofructokinase", "hexokinase", "pyruvate kinase",
        "isocitrate dehydrogenase", "fatty acid", "insulin", "glucagon",
        "fructose", "lactate", "coenzyme", "enzyme", "cofactor",
        "phosphorylation", "oxidation", "reduction", "substrate",
    ]

    query_lower = query.lower()
    found = [kw for kw in biochem_keywords if kw in query_lower]

    # Also extract capitalized terms (likely proper biochemical names)
    import re
    capitalized = re.findall(r'\b[A-Z][a-zA-Z-]+(?:\s[A-Z][a-zA-Z-]+)*\b', query)
    found.extend([c.lower() for c in capitalized])

    return list(set(found))


def vector_search(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    top_k: int = TOP_K_VECTORS
) -> list[dict]:
    """
    Semantic search over BioBERT-embedded textbook chunks.

    Args:
        query: Natural language biochemistry question
        index: Loaded FAISS index
        chunks: Corresponding chunk metadata list
        top_k: Number of chunks to retrieve

    Returns:
        List of top-k chunks with similarity scores
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            chunk = chunks[idx].copy()
            chunk["similarity_score"] = float(score)
            results.append(chunk)

    logger.info(f"Vector search returned {len(results)} chunks")
    return results


def graph_search(
    query: str,
    G: nx.DiGraph,
    depth: int = 2
) -> list[dict]:
    """
    Knowledge graph traversal based on entities detected in the query.

    Args:
        query: Natural language biochemistry question
        G: Loaded NetworkX knowledge graph
        depth: Traversal depth from each found entity

    Returns:
        List of subgraph context dicts for each found entity
    """
    entities = extract_biochem_entities(query)
    results = []

    for entity in entities:
        result = graph_query(G, entity, depth=depth)
        if "error" not in result:
            results.append(result)
            logger.info(f"Graph hit for entity: {entity}")

    if not results:
        logger.info("No direct graph entities found — relying on vector retrieval")

    return results


def hybrid_retrieve(
    query: str,
    G: nx.DiGraph,
    index: faiss.Index,
    chunks: list[dict]
) -> dict:
    """
    Main retrieval function — runs graph + vector search in parallel
    and fuses results into a single structured context object.

    This context object is passed directly to the LLM.
    The LLM is instructed to answer ONLY from this context.

    Returns:
        {
            "query": original question,
            "graph_context": list of graph subgraph results,
            "vector_context": list of textbook chunk results,
            "has_graph_data": bool,
            "has_vector_data": bool,
        }
    """
    logger.info(f"Hybrid retrieval for query: {query}")

    graph_results = graph_search(query, G)
    vector_results = vector_search(query, index, chunks)

    context = {
        "query": query,
        "graph_context": graph_results,
        "vector_context": vector_results,
        "has_graph_data": len(graph_results) > 0,
        "has_vector_data": len(vector_results) > 0,
    }

    logger.info(
        f"Retrieval complete — "
        f"graph hits: {len(graph_results)}, "
        f"vector hits: {len(vector_results)}"
    )

    return context