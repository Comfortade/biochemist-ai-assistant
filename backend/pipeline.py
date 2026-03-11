"""
GraphRAG Pipeline — End-to-End Orchestrator
=============================================
This module ties together all backend components into a single
callable pipeline that powers the Streamlit frontend.

Flow:
    User Query
        │
        ▼
    Graph Search (KEGG NetworkX KG)
        │
        ▼
    Vector Search (BioBERT + FAISS)
        │
        ▼
    Context Fusion
        │
        ▼
    Constrained Gemini LLM
        │
        ▼
    Grounded Answer + Source Citations

Usage:
    from backend.pipeline import BiochemPipeline
    pipeline = BiochemPipeline()
    result = pipeline.query("How does glucose enter glycolysis?")
    print(result["answer"])
"""

import logging
from pathlib import Path
import networkx as nx
import faiss

from backend.graph.kegg_loader import save_raw_kgml
from backend.graph.pathway_graph import build_full_graph, save_graph, load_graph
from backend.rag.chunker import process_all_textbooks
from backend.rag.embedder import build_faiss_index, load_faiss_index
from backend.rag.retriever import hybrid_retrieve
from backend.llm.gemini_client import ask_gemini

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiochemPipeline:
    """
    Main pipeline class. Initializes all components on first load
    and caches them in memory for fast subsequent queries.

    On first run: fetches KEGG data, processes PDFs, builds FAISS index.
    On subsequent runs: loads everything from disk cache.
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        embeddings_dir: str = "data/embeddings",
        chunks_dir: str = "data/chunks",
        force_rebuild: bool = False
    ):
        self.data_dir = data_dir
        self.embeddings_dir = embeddings_dir
        self.chunks_dir = chunks_dir
        self.force_rebuild = force_rebuild

        self.graph: nx.DiGraph = None
        self.index: faiss.Index = None
        self.chunks: list[dict] = []
        self.is_ready = False

        self._initialize()

    def _initialize(self):
        """
        Load or build all pipeline components.
        Checks disk cache first to avoid reprocessing on every startup.
        """
        logger.info("Initializing BiochemPipeline...")

        # --- Knowledge Graph ---
        graph_path = Path(self.embeddings_dir) / "graph.json"
        if not self.force_rebuild and graph_path.exists():
            self.graph = load_graph(str(graph_path))
            logger.info("Loaded graph from cache")
        else:
            logger.info("Building knowledge graph from KEGG...")
            save_raw_kgml(self.data_dir)
            self.graph = build_full_graph(self.data_dir)
            save_graph(self.graph, str(graph_path))

        # --- FAISS Vector Index ---
        index_path = Path(self.embeddings_dir) / "biochem.index"
        metadata_path = Path(self.embeddings_dir) / "chunks_metadata.json"

        if not self.force_rebuild and index_path.exists() and metadata_path.exists():
            self.index, self.chunks = load_faiss_index(
                str(index_path), str(metadata_path)
            )
            logger.info("Loaded FAISS index from cache")
        else:
            logger.info("Building FAISS index from textbooks...")
            chunks = process_all_textbooks(self.data_dir, self.chunks_dir)
            if chunks:
                self.index, self.chunks = build_faiss_index(
                    chunks, self.embeddings_dir
                )
            else:
                logger.warning(
                    "No PDF chunks found. Add PDFs to data/raw/ and rebuild."
                )

        self.is_ready = self.graph is not None
        logger.info(f"Pipeline ready: {self.is_ready}")

    def query(self, question: str) -> dict:
        """
        Run a full GraphRAG query through the pipeline.

        Args:
            question: Natural language biochemistry question

        Returns:
            {
                "answer": grounded LLM answer,
                "sources_used": list of cited sources,
                "graph_hits": number of graph entities found,
                "vector_hits": number of textbook chunks retrieved,
                "had_graph_data": bool,
                "had_vector_data": bool,
            }
        """
        if not self.is_ready:
            return {
                "answer": "Pipeline not ready. Check logs for initialization errors.",
                "sources_used": [],
                "graph_hits": 0,
                "vector_hits": 0,
                "had_graph_data": False,
                "had_vector_data": False,
            }

        logger.info(f"Processing query: {question}")

        # Step 1: Hybrid retrieval
        context = hybrid_retrieve(
            query=question,
            G=self.graph,
            index=self.index if self.index else None,
            chunks=self.chunks,
        )

        # Step 2: Grounded LLM answer
        result = ask_gemini(question, context)

        # Step 3: Enrich result with retrieval metadata
        result["graph_hits"] = len(context.get("graph_context", []))
        result["vector_hits"] = len(context.get("vector_context", []))

        return result

    def get_graph(self) -> nx.DiGraph:
        """Return the loaded knowledge graph for visualization."""
        return self.graph

    def rebuild(self):
        """Force a full rebuild of all indexes — use after adding new PDFs."""
        self.force_rebuild = True
        self._initialize()
        self.force_rebuild = False