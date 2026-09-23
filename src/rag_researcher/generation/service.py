from __future__ import annotations

from collections.abc import Iterator

from rag_researcher.config import settings
from rag_researcher.generation.prompt import build_rag_prompt
from rag_researcher.retrieval.search import SearchResponse, hybrid_search
from rag_researcher.runtime import get_answer_generator

DEEPSEEK_NOT_CONFIGURED_MESSAGE = "DEEPSEEK_API_KEY is not configured."


def resolve_retrieval_flags(
    *,
    use_hyde: bool | None,
    use_rerank: bool | None,
) -> tuple[bool, bool]:
    return (
        settings.hyde_enabled if use_hyde is None else use_hyde,
        settings.rerank_enabled if use_rerank is None else use_rerank,
    )


def retrieve_for_answer(
    *,
    query: str,
    top_k: int,
    use_hyde: bool,
    use_rerank: bool,
) -> SearchResponse:
    return hybrid_search(
        query=query,
        elasticsearch_url=settings.elasticsearch_url,
        index_name=settings.elasticsearch_index,
        model_name=settings.embedding_model,
        top_k=top_k,
        use_hyde=use_hyde,
        use_rerank=use_rerank,
        reranker_model=settings.reranker_model,
    )


def generate_answer(query: str, retrieval: SearchResponse) -> str:
    generator = get_answer_generator()
    if not generator.is_configured():
        return DEEPSEEK_NOT_CONFIGURED_MESSAGE
    prompt = build_rag_prompt(query, retrieval.results)
    return generator.generate(prompt).answer


def stream_answer(query: str, retrieval: SearchResponse) -> Iterator[str]:
    generator = get_answer_generator()
    if not generator.is_configured():
        raise DeepSeekNotConfiguredError(DEEPSEEK_NOT_CONFIGURED_MESSAGE)
    prompt = build_rag_prompt(query, retrieval.results)
    yield from generator.stream_generate(prompt)


class DeepSeekNotConfiguredError(RuntimeError):
    pass
