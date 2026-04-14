"""
ChromaDB client singleton and helpers.
Wraps the synchronous ChromaDB SDK in asyncio.to_thread() to avoid blocking the event loop.
"""
import asyncio
import chromadb
from chromadb.config import Settings as ChromaSettings
from config import settings


def get_chroma_client() -> chromadb.HttpClient:
    """Create a ChromaDB HTTP client pointing at the configured host."""
    return chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


async def query_collection_async(
    client: chromadb.HttpClient,
    collection_name: str,
    query_texts: list[str],
    n_results: int = 5,
    where: dict | None = None,
) -> dict:
    """Async wrapper around ChromaDB collection.query()."""
    def _query():
        col = client.get_collection(collection_name)
        kwargs: dict = {
            "query_texts": query_texts,
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        return col.query(**kwargs)

    return await asyncio.to_thread(_query)


async def chroma_health_check(client: chromadb.HttpClient) -> bool:
    """Check if ChromaDB is reachable."""
    try:
        await asyncio.to_thread(client.heartbeat)
        return True
    except Exception:
        return False
