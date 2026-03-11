"""
Textbook Chunker
=================
Splits biochemistry PDF textbooks into overlapping text chunks
for embedding and vector retrieval.

Strategy:
- Chunk size: 500 words with 50-word overlap
- Overlap prevents context loss at chunk boundaries
- Each chunk retains its source (book title, page number)
  so the assistant can cite exactly where information came from.

Supported sources:
- Biochemistry Free For All (Ahern & Rajagopal)
- LibreTexts Biochemistry
"""

import fitz  # PyMuPDF
import json
import logging
import re
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE = 500      # words per chunk
CHUNK_OVERLAP = 50    # words overlap between consecutive chunks


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from each page of a PDF.
    Returns list of {page_number, text} dicts.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = clean_text(text)
        if len(text.strip()) > 100:  # skip near-empty pages
            pages.append({"page": page_num, "text": text})

    logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
    return pages


def clean_text(text: str) -> str:
    """
    Remove noise from extracted PDF text:
    - Multiple blank lines
    - Page headers/footers patterns
    - Hyphenated line breaks
    """
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def chunk_text(text: str, page_num: int, source: str) -> list[dict]:
    """
    Split a page's text into overlapping word-level chunks.
    Each chunk includes metadata for citation in the UI.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        if len(chunk_words) > 50:  # skip tiny trailing chunks
            chunks.append({
                "text": chunk_text,
                "source": source,
                "page": page_num,
                "word_count": len(chunk_words),
                "chunk_id": f"{source}_p{page_num}_c{len(chunks)}",
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def process_pdf(pdf_path: str, source_name: str) -> list[dict]:
    """
    Full pipeline: PDF → pages → cleaned text → overlapping chunks.
    """
    pages = extract_text_from_pdf(pdf_path)
    all_chunks = []
    for page in pages:
        chunks = chunk_text(page["text"], page["page"], source_name)
        all_chunks.extend(chunks)

    logger.info(f"Created {len(all_chunks)} chunks from {source_name}")
    return all_chunks


def process_all_textbooks(
    data_dir: str = "data/raw",
    output_dir: str = "data/chunks"
) -> list[dict]:
    """
    Process all PDFs in data/raw and save chunks to data/chunks.
    Automatically detects PDFs and assigns source names from filenames.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)
    all_chunks = []

    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {data_dir}")
        logger.info("Place your textbook PDFs in the data/raw/ folder")
        return []

    for pdf_file in pdf_files:
        source_name = pdf_file.stem.replace(" ", "_").lower()
        chunks = process_pdf(str(pdf_file), source_name)
        all_chunks.extend(chunks)

        # Save per-book chunks
        output_path = Path(output_dir) / f"{source_name}_chunks.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")

    # Save combined chunks
    combined_path = Path(output_dir) / "all_chunks.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"Total chunks across all textbooks: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    chunks = process_all_textbooks()
    if chunks:
        print(f"Sample chunk:\n{chunks[0]['text'][:300]}...")
        print(f"Total chunks: {len(chunks)}")