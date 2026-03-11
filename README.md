# Biochemist AI Assistant

> A GraphRAG-powered biochemistry assistant that answers metabolic pathway questions without hallucinating. Every answer is grounded in verified KEGG pathway data and peer-reviewed textbook passages.

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![KEGG](https://img.shields.io/badge/Data-KEGG_REST_API-green?style=flat-square)](https://rest.kegg.jp)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

##  The Problem This Solves

General-purpose LLMs hallucinate biochemical facts. They invent enzyme names, skip pathway steps, and fabricate reaction equations with high confidence. This is dangerous for students, researchers, and anyone using AI for scientific reasoning.

This project solves that by implementing **GraphRAG** — a retrieval architecture that forces the LLM to reason *only* over verified, structured data retrieved at query time. The model cannot go beyond what the retrieval system provides.

---

##  Architecture
```
User Query
    │
    ├──► Graph Search — traverses KEGG Knowledge Graph (NetworkX)
    │         Finds exact pathway steps, enzyme-substrate relationships,
    │         reaction equations, and metabolic connections
    │
    ├──► Vector Search — semantic search over BioBERT-embedded textbook chunks (FAISS)
    │         Retrieves contextually relevant passages from biochemistry textbooks
    │         using a biomedical domain-specific embedding model
    │
    ▼
Context Fusion — structured context object combining both retrieval results
    │
    ▼
Constrained Gemini 1.5 Flash
    │    System prompt forbids the model from adding any information
    │    not present in the retrieved context
    │
    ▼
Grounded Answer + Source Citations
    (every claim traceable to a KEGG edge or textbook page)
```

---

##  Features

| Feature | Description |
|---|---|
|  **Pathway Explorer** | Interactive graph visualization of KEGG metabolic pathways (Glycolysis, TCA Cycle, Beta-Oxidation, Pentose Phosphate) |
|  **Ask the Assistant** | Natural language Q&A grounded in knowledge graph + textbook retrieval |
|  **Molecule Viewer** | 2D chemical structure rendering for any metabolite via PubChem |
|  **Source Attribution** | Every answer shows exactly which graph nodes and textbook pages were used |
|  **Anti-Hallucination** | LLM is architecturally prevented from inventing steps or reactions |

---

##  Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Knowledge Graph | NetworkX + KEGG REST API | Structured pathway data, graph traversal |
| Vector Store | FAISS (Facebook AI Similarity Search) | Sub-millisecond semantic search |
| Embeddings | BioBERT (pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb) | Biomedical domain-specific text embeddings |
| LLM | Google Gemini 1.5 Flash | Constrained answer generation |
| PDF Processing | PyMuPDF | Textbook text extraction |
| Frontend | Streamlit + streamlit-agraph | Interactive UI and graph visualization |
| Testing | pytest | Unit and integration tests |

---

##  Project Structure
```
biochemist-ai-assistant/
├── app/
│   ├── Home.py                        # Landing page
│   └── pages/
│       ├── 1_Pathway_Explorer.py      # Interactive graph visualization
│       ├── 2_Ask_the_Assistant.py     # GraphRAG Q&A interface
│       └── 3_Molecule_Viewer.py       # Chemical structure renderer
├── backend/
│   ├── graph/
│   │   ├── kegg_loader.py             # KEGG REST API data fetching
│   │   └── pathway_graph.py           # NetworkX knowledge graph builder
│   ├── rag/
│   │   ├── chunker.py                 # PDF text extraction and chunking
│   │   ├── embedder.py                # BioBERT embedding + FAISS indexing
│   │   └── retriever.py               # Hybrid graph + vector retrieval
│   ├── llm/
│   │   └── gemini_client.py           # Constrained Gemini API client
│   └── pipeline.py                    # End-to-end orchestrator
├── data/
│   ├── raw/                           # PDF textbooks + KEGG KGML files
│   ├── chunks/                        # Processed text chunks (JSON)
│   └── embeddings/                    # FAISS index + graph cache
├── tests/
│   ├── test_graph.py                  # Knowledge graph unit tests
│   ├── test_retriever.py              # Retrieval unit tests
│   └── test_pipeline.py              # End-to-end integration tests
├── .env                               # API keys (never committed)
├── requirements.txt                   # Python dependencies
└── streamlit_app.py                   # Streamlit entry point
```

---

##  Getting Started

### Prerequisites
- Python 3.11 or 3.12
- A Google Gemini API key ([get one free here](https://aistudio.google.com/app/apikey))

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/biochemist-ai-assistant.git
cd biochemist-ai-assistant
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt

# Windows only — if torch fails:
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Configure environment variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Add textbook PDFs
Place your biochemistry PDF textbooks in `data/raw/`:
```
data/raw/biochemistry_free_for_all.pdf
data/raw/libretexts_biochemistry.pdf
```
> The system automatically detects and processes any PDF in this folder.

### 6. Initialize the pipeline
On first run, this fetches KEGG pathway data, chunks your PDFs, and builds the FAISS index. Takes approximately 5–10 minutes.
```bash
python -c "from backend.pipeline import BiochemPipeline; BiochemPipeline()"
```

### 7. Run the app
```bash
streamlit run app/Home.py
```

---

##  Running Tests
```bash
pytest tests/ -v
```

Expected output:
```
tests/test_graph.py::test_parse_kgml_returns_digraph PASSED
tests/test_graph.py::test_parse_kgml_correct_node_count PASSED
tests/test_graph.py::test_parse_kgml_correct_edge_count PASSED
...
```

---

##  Data Sources

- **[KEGG REST API](https://rest.kegg.jp)** — Kyoto Encyclopedia of Genes and Genomes. Pathway maps, reaction equations, compound metadata. Freely accessible, no API key required.
- **[Biochemistry Free For All](https://biochem.science.oregonstate.edu/)** — Ahern & Rajagopal, Oregon State University. Open-access biochemistry textbook.
- **[LibreTexts Biochemistry](https://bio.libretexts.org/)** — Open-access community biochemistry textbook.

---

##  Pathways Covered

| Pathway | KEGG ID | Description |
|---|---|---|
| Glycolysis / Gluconeogenesis | hsa00010 | 10-step glucose catabolism pathway |
| TCA Cycle | hsa00020 | Citric acid cycle, central metabolic hub |
| Pentose Phosphate Pathway | hsa00030 | NADPH production and nucleotide synthesis |
| Fatty Acid Beta-Oxidation | hsa00061 | Fatty acid catabolism |

---

##  Roadmap

- [ ] Add PubMed abstract retrieval for research literature grounding
- [ ] Integrate BRENDA enzyme database for kinetics data
- [ ] Add pathway comparison feature (e.g. aerobic vs anaerobic)
- [ ] Export answers as formatted PDF reports
- [ ] Add support for disease-associated pathway perturbations (KEGG DISEASE)

---




---

##  License

MIT License — see [LICENSE](LICENSE) for details.