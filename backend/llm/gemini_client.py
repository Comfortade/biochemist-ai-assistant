"""
Gemini LLM Client — Constrained Biochemistry Reasoner
=======================================================
Wraps the Google Gemini API with strict prompting constraints
that prevent hallucination of biochemical facts.

Core anti-hallucination mechanism:
- The system prompt explicitly forbids the model from using
  any knowledge outside the provided context
- Every answer must cite either a graph node/edge or a textbook chunk
- If the retrieved context does not contain enough information,
  the model is instructed to say so rather than speculate

This is the key architectural decision that makes this system
trustworthy for biochemistry education and research use.
"""

import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.0-flash"

SYSTEM_PROMPT = """You are a precise biochemistry assistant with expertise in metabolic pathways, 
enzymology, and molecular biology. 

STRICT RULES — you must follow these without exception:

1. ONLY use information provided in the CONTEXT section below.
2. NEVER invent, hallucinate, or add reaction steps not present in the context.
3. If the context does not contain enough information to answer, say exactly:
   "The retrieved context does not contain sufficient information to answer this question reliably."
4. When describing a pathway, list steps IN ORDER as they appear in the graph context.
5. Always cite your source: either the graph (KEGG pathway data) or the textbook chunk 
   (source name + page number).
6. Use precise biochemical terminology — enzyme names, compound names, cofactors.
7. If asked about regulation, only describe regulators explicitly present in the context.

FORMAT your response as:
- A direct answer to the question
- Step-by-step pathway details if relevant (numbered)
- Source citations at the end

You are grounded in verified data. Your value is precision, not creativity."""


def format_context_for_prompt(retrieval_context: dict) -> str:
    """
    Convert the hybrid retrieval context dict into a
    clean string block for injection into the prompt.
    """
    sections = []

    # Format graph context
    if retrieval_context.get("has_graph_data"):
        sections.append("=== KNOWLEDGE GRAPH DATA (KEGG) ===")
        for hit in retrieval_context["graph_context"]:
            sections.append(f"Entity: {hit['entity']} (type: {hit['node_type']})")
            if hit.get("predecessors"):
                sections.append(f"  Preceded by: {', '.join(hit['predecessors'])}")
            if hit.get("successors"):
                sections.append(f"  Followed by: {', '.join(hit['successors'])}")
            sections.append("")

    # Format vector context
    if retrieval_context.get("has_vector_data"):
        sections.append("=== TEXTBOOK PASSAGES ===")
        for i, chunk in enumerate(retrieval_context["vector_context"], 1):
            source = chunk.get("source", "unknown")
            page = chunk.get("page", "?")
            score = chunk.get("similarity_score", 0)
            sections.append(
                f"[Passage {i} | Source: {source} | Page: {page} | "
                f"Relevance: {score:.3f}]"
            )
            sections.append(chunk["text"][:600])  # cap length per chunk
            sections.append("")

    if not sections:
        return "No relevant context was retrieved for this query."

    return "\n".join(sections)


def ask_gemini(query: str, retrieval_context: dict) -> dict:
    """
    Send a grounded query to Gemini with strict context constraints.

    Args:
        query: The user's biochemistry question
        retrieval_context: Output from hybrid_retrieve()

    Returns:
        {
            "answer": str,
            "sources_used": list,
            "had_graph_data": bool,
            "had_vector_data": bool,
            "model": str,
        }
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment. "
            "Add it to your .env file."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    context_block = format_context_for_prompt(retrieval_context)

    full_prompt = f"""{SYSTEM_PROMPT}

=== CONTEXT ===
{context_block}

=== QUESTION ===
{query}

=== YOUR ANSWER ==="""

    logger.info(f"Sending query to Gemini: {query[:80]}...")

    try:
        response = model.generate_content(full_prompt)
        answer = response.text

        # Extract sources cited
        sources = []
        for chunk in retrieval_context.get("vector_context", []):
            sources.append({
                "type": "textbook",
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "score": chunk.get("similarity_score"),
            })
        for hit in retrieval_context.get("graph_context", []):
            sources.append({
                "type": "knowledge_graph",
                "entity": hit.get("entity"),
                "node_type": hit.get("node_type"),
            })

        logger.info("Gemini response received successfully")
        return {
            "answer": answer,
            "sources_used": sources,
            "had_graph_data": retrieval_context.get("has_graph_data", False),
            "had_vector_data": retrieval_context.get("has_vector_data", False),
            "model": GEMINI_MODEL,
        }

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return {
            "answer": f"Error contacting Gemini API: {str(e)}",
            "sources_used": [],
            "had_graph_data": False,
            "had_vector_data": False,
            "model": GEMINI_MODEL,
        }