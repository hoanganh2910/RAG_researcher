from __future__ import annotations

from functools import lru_cache

from rag_researcher.config import settings
from rag_researcher.embedding.encoder import BGEEncoder, OllamaEncoder
from rag_researcher.generation.generator import DeepSeekGenerator, OllamaDecisionGenerator, OllamaAnswerGenerator
from rag_researcher.cache.redis_cache import RedisCache
from rag_researcher.observability.langfuse_tracer import LangfuseTracer, get_langfuse_tracer
from rag_researcher.retrieval.vector_store import ElasticsearchVectorStore
from rag_researcher.reranking.reranker import BGEReranker

@lru_cache(maxsize=4)
def get_embedding_encoder(
    model_name: str = "Qwen/Qwen3-Embedding-4B",
    expected_dimension: int | None = None,
) -> BGEEncoder | OllamaEncoder:
    dimension = expected_dimension or settings.embedding_dimensions
    
    if model_name.startswith("ollama:"):
        actual_model = model_name.removeprefix("ollama:")
        return OllamaEncoder(
            model_name=actual_model,
            base_url=settings.ollama_url,
            expected_dimension=dimension,
        )
        
    return BGEEncoder(
        model_name,
        expected_dimension=dimension,
    )


@lru_cache(maxsize=8)
def get_elasticsearch_vector_store(
    url: str,
    index_name: str = "rag_researcher_chunks",
    embedding_dims: int | None = None,
) -> ElasticsearchVectorStore:
    return ElasticsearchVectorStore(
        url=url,
        index_name=index_name,
        embedding_dims=embedding_dims or settings.embedding_dimensions,
    )


@lru_cache(maxsize=8)
def get_answer_generator() -> DeepSeekGenerator | OllamaAnswerGenerator:
    if settings.deepseek_api_key:
        return DeepSeekGenerator(api_key=settings.deepseek_api_key, model=settings.deepseek_model)
    
    from rag_researcher.generation.generator import OllamaAnswerGenerator
    return OllamaAnswerGenerator(
        url=settings.ollama_url,
        model=settings.agent_decision_model,
        timeout=120,
        keep_alive=settings.ollama_keep_alive,
    )


def get_ollama_decision_generator(
    url: str,
    model: str,
    timeout: int,
    keep_alive: str,
) -> DeepSeekGenerator | OllamaDecisionGenerator:
    if model == "deepseek":
        if not settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required to use deepseek as the decision model.")
        return DeepSeekGenerator(
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            timeout=min(timeout, 15),  # fail fast for decision tasks
        )

    return OllamaDecisionGenerator(
        url=url,
        model=model,
        timeout=timeout,
        keep_alive=keep_alive,
    )


@lru_cache(maxsize=4)
def get_redis_cache(url: str) -> RedisCache:
    return RedisCache(url=url)


@lru_cache(maxsize=1)
def get_observability_tracer() -> LangfuseTracer:
    return get_langfuse_tracer()


@lru_cache(maxsize=4)
def get_bge_reranker(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    max_content_chars: int | None = None,
) -> BGEReranker:
    return BGEReranker(
        model_name,
        max_content_chars=max_content_chars or settings.rerank_max_content_chars,
    )
