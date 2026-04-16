"""
Pinecone client for vector storage and retrieval.
Replaces ChromaDB for cloud-native deployment.
"""
from pinecone import Pinecone, ServerlessSpec
from typing import Optional, List, Dict, Any
import asyncio


_pinecone_client: Optional[Pinecone] = None


def get_pinecone_client(api_key: str) -> Optional[Pinecone]:
    """Get or create the Pinecone client singleton."""
    global _pinecone_client
    if _pinecone_client is None:
        try:
            _pinecone_client = Pinecone(api_key=api_key)
        except Exception:
            return None
    return _pinecone_client


def get_or_create_index(
    client: Pinecone,
    index_name: str,
    dimension: int = 768,
    metric: str = "cosine",
    environment: str = "us-east-1",
) -> Any:
    """Get existing index or create a new one."""
    try:
        # Check if index exists
        existing_indexes = client.list_indexes().names()
        if index_name not in existing_indexes:
            print(f"[pinecone] Creating index: {index_name}")
            client.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region=environment),
            )
            # Wait for index to be ready
            while not client.describe_index(index_name).status["ready"]:
                asyncio.sleep(1)

        return client.Index(index_name)
    except Exception as e:
        print(f"[pinecone] Error getting/creating index: {e}")
        return None


async def upsert_vectors_async(
    index: Any,
    vectors: List[tuple],
    namespace: str = "",
) -> bool:
    """
    Async wrapper for Pinecone upsert.
    vectors: List of (id, vector, metadata) tuples
    """
    try:
        await asyncio.to_thread(
            index.upsert,
            vectors=vectors,
            namespace=namespace,
        )
        return True
    except Exception as e:
        print(f"[pinecone] Upsert failed: {e}")
        return False


async def query_vectors_async(
    index: Any,
    query_vector: List[float],
    top_k: int = 5,
    namespace: str = "",
    filter_dict: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Async wrapper for Pinecone query.
    Returns dict with 'matches' containing results.
    """
    try:
        result = await asyncio.to_thread(
            index.query,
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict,
            include_metadata=True,
            include_values=False,
        )
        return result
    except Exception as e:
        print(f"[pinecone] Query failed: {e}")
        return None


async def delete_index_async(index: Any, ids: List[str], namespace: str = "") -> bool:
    """Delete vectors by ID."""
    try:
        await asyncio.to_thread(index.delete, ids=ids, namespace=namespace)
        return True
    except Exception:
        return False


async def pinecone_health_check(api_key: str, index_name: str) -> bool:
    """Check if Pinecone is reachable and index exists."""
    try:
        client = Pinecone(api_key=api_key)
        indexes = client.list_indexes().names()
        return index_name in indexes
    except Exception:
        return False
