from __future__ import annotations

from contextlib import asynccontextmanager
import sys
from pathlib import Path
from typing import AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI

from app.api.v1.router import router as v1_router
from rag_researcher.config import settings
from rag_researcher.logging import configure_logging
from rag_researcher.runtime import (
    get_embedding_encoder,
    get_answer_generator,
    get_elasticsearch_vector_store,
    get_observability_tracer,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    startup_load_runtime(app)
    yield
    app.state.observability_tracer.flush()


app = FastAPI(title="RAG_researcher", lifespan=lifespan)
app.include_router(v1_router)


def startup_load_runtime(target_app: FastAPI = app) -> None:
    configure_logging()
    target_app.state.embedding_encoder = get_embedding_encoder(settings.embedding_model)
    target_app.state.elasticsearch_vector_store = get_elasticsearch_vector_store(
        settings.elasticsearch_url,
        settings.elasticsearch_index,
    )
    target_app.state.answer_generator = get_answer_generator()
    target_app.state.observability_tracer = get_observability_tracer()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "RAG_researcher running"}
