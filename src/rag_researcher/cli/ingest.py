from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from rag_researcher.config import settings
from rag_researcher.ingestion.loaders import load_documents
from rag_researcher.ingestion.repository import ensure_documents_table, insert_documents


async def ingest(path: str) -> int:
    documents = load_documents(path)
    await ensure_documents_table(settings.postgres_dsn)
    inserted_count = await insert_documents(settings.postgres_dsn, documents)

    print(f"Loaded documents: {len(documents)}")
    print(f"Inserted documents: {inserted_count}")
    print(f"Skipped duplicates: {len(documents) - inserted_count}")

    return inserted_count


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    load_dotenv()

    if len(sys.argv) != 2:
        print("Usage: uv run rag_researcher-ingest data/sample_docs")
        raise SystemExit(2)

    asyncio.run(ingest(sys.argv[1]))
