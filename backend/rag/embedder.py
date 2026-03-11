"""
BioBERT Embedder + FAISS Index Builder
========================================
Converts text chunks into dense vector embeddings using a
biomedical-domain sentence transformer model, then indexes
them in FAISS for fast similarity search.

Model: pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
- Fine-tuned on biomedical NLI and semantic similarity tasks
- Significantly outperforms generic models on biochemistry text
- Produces 768-dimensional embeddings

Why FAISS:
- Facebook AI Similarity Search — industry standard for vector retrieval
- Sub-millisecond search over millions of vectors
- No external server needed — runs entirely in memory/on disk
"""

import faiss
import json
import numpy as np
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
EMBEDDING_DIM = 768
BATCH_SIZE = 32


def load_chunks(chunks_path: str = "data/chunks/all_chunks.json") -> list[dict]:
    """Load preprocessed text chunks from disk."""
    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {chunks_path}. "
            "Run chunker.py first to process your PDFs."
        )
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks


def build_faiss_index(
    chunks: list[dict],
    output_dir: str = "data/embeddings"
) -> tuple[faiss.Index, list[dict]]:
    """
    Embed all chunks and build a FAISS index.

    Process:
    1. Load BioBERT sentence transformer
    2. Encode all chunk texts in batches
    3. Normalize vectors (for cosine similarity)
    4. Build FAISS IndexFlatIP (inner product = cosine on normalized vectors)
    5. Save index + metadata to disk

    Returns:
        (faiss_index, chunks_with_metadata)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]
    logger.info(f"Embedding {len(texts)} chunks in batches of {BATCH_SIZE}...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Required for cosine similarity via inner product
    )

    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Build FAISS index (IndexFlatIP = exact search with inner product)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings.astype(np.float32))
    logger.info(f"FAISS index built with {index.ntotal} vectors")

    # Save index to disk
    index_path = Path(output_dir) / "biochem.index"
    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS index saved to {index_path}")

    # Save chunk metadata (text + source + page) separately
    metadata_path = Path(output_dir) / "chunks_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved to {metadata_path}")

    return index, chunks


def load_faiss_index(
    index_path: str = "data/embeddings/biochem.index",
    metadata_path: str = "data/embeddings/chunks_metadata.json"
) -> tuple[Optional[faiss.Index], list[dict]]:
    """Load a previously built FAISS index and its metadata from disk."""
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    if not index_path.exists() or not metadata_path.exists():
        logger.warning("FAISS index not found. Run embedder.py first.")
        return None, []

    index = faiss.read_index(str(index_path))
    with open(metadata_path, encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
    return index, chunks


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string for retrieval.
    Uses the same model as indexing to ensure vector space alignment.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embedding.astype(np.float32)


if __name__ == "__main__":
    from backend.rag.chunker import process_all_textbooks
    chunks = process_all_textbooks()
    if chunks:
        index, metadata = build_faiss_index(chunks)
        print(f"Index ready: {index.ntotal} vectors indexed")
    else:
        print("No chunks found. Add PDFs to data/raw/ first.")