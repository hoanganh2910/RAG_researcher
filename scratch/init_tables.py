import asyncio
import sys
from dotenv import load_dotenv

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from rag_researcher.config import settings
from rag_researcher.papers.repository import ensure_papers_table
from rag_researcher.ingestion.repository import ensure_documents_table
from rag_researcher.chunking.repository import ensure_document_chunks_table
from rag_researcher.embedding.repository import ensure_chunk_embeddings_table

async def main():
    load_dotenv()
    dsn = settings.postgres_dsn
    print("Initializing database tables...")
    await ensure_papers_table(dsn)
    await ensure_documents_table(dsn)
    await ensure_document_chunks_table(dsn)
    await ensure_chunk_embeddings_table(dsn)
    print("Database tables initialized successfully.")

if __name__ == "__main__":
    asyncio.run(main())
