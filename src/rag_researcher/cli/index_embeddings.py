from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from rag_researcher.pipeline.steps import index_embeddings

__all__ = ["index_embeddings"]


def main() -> None:
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    load_dotenv()
    parser = argparse.ArgumentParser(description="Index RAG_researcher chunk embeddings into Elasticsearch")
    parser.set_defaults(recreate=True)
    parser.add_argument("--recreate", dest="recreate", action="store_true", help="Delete and recreate the Elasticsearch index before indexing")
    parser.add_argument("--no-recreate", dest="recreate", action="store_false", help="Upsert into the existing Elasticsearch index")
    args = parser.parse_args()
    asyncio.run(index_embeddings(recreate=args.recreate))
