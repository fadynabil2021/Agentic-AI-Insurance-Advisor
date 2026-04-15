"""
ChromaDB client singleton and helpers.
Wraps the synchronous ChromaDB SDK in asyncio.to_thread() to avoid blocking the event loop.
"""
import asyncio
import chromadb
from chromadb.config import Settings as ChromaSettings
from config import settings


_chroma_client: chromadb.HttpClient | None = None


def get_chroma_client() -> chromadb.HttpClient | None:
    """Create a ChromaDB HTTP client pointing at the configured host."""
    global _chroma_client
    if _chroma_client is None:
        try:
            _chroma_client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        except Exception:
            return None
    return _chroma_client


async def query_collection_async(
    client: chromadb.HttpClient | None,
    collection_name: str,
    query_texts: list[str],
    n_results: int = 5,
    where: dict | None = None,
) -> dict | None:
    """Async wrapper around ChromaDB collection.query()."""
    if client is None:
        return None

    def _query():
        col = client.get_collection(collection_name)
        kwargs: dict = {
            "query_texts": query_texts,
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        return col.query(**kwargs)

    try:
        return await asyncio.to_thread(_query)
    except Exception:
        return None


async def chroma_health_check(client: chromadb.HttpClient | None) -> bool:
    """Check if ChromaDB is reachable."""
    if client is None:
        return False
    try:
        await asyncio.to_thread(client.heartbeat)
        return True
    except Exception:
        return False
