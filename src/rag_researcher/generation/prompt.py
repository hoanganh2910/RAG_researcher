from __future__ import annotations

from rag_researcher.retrieval.vector_store import SearchResult

RAG_PROMPT_TEMPLATE = """You are RAG_researcher's local RAG assistant. Please answer the question based only on the provided context.
If the context is insufficient to answer the question, explicitly state "Cannot determine based on current information."

Question:
{query}

Context:
{context}

Requirements:
1. Answer in English.
2. Prioritize citing facts from the context.
3. Do not make up information outside the context.
"""


def build_rag_prompt(query: str, chunks: list[SearchResult]) -> str:
    return RAG_PROMPT_TEMPLATE.format(query=query, context=format_context(chunks))


def format_context(chunks: list[SearchResult]) -> str:
    if not chunks:
        return "No available context."

    parts = []
    for index, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Chunk {index} | chunk_id={chunk.chunk_id} | document_id={chunk.document_id}]\n"
            f"{chunk.content}"
        )

    return "\n\n".join(parts)
