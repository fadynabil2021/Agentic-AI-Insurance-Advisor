"""
GET /api/v1/health — system health check endpoint.
Checks Ollama, ChromaDB, Langfuse, and Redis connectivity.
"""
from fastapi import APIRouter, Request
from models.schemas import HealthResponse
from clients.langfuse_client import langfuse_health_check
from clients.redis_client import redis_health_check
from config import settings
import httpx
import asyncio

router = APIRouter(tags=["Health"])


async def check_ollama() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{settings.OLLAMA_HOST}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


async def check_chromadb() -> bool:
    try:
        port = settings.CHROMA_PORT
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"http://{settings.CHROMA_HOST}:{port}/api/v1/heartbeat")
            return r.status_code == 200
    except Exception:
        return False


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check connectivity of all dependent services."""
    results = await asyncio.gather(
        check_ollama(),
        check_chromadb(),
        langfuse_health_check(),
        redis_health_check(),
        return_exceptions=True,
    )

    service_status = {
        "ollama":   bool(results[0]) if not isinstance(results[0], Exception) else False,
        "chromadb": bool(results[1]) if not isinstance(results[1], Exception) else False,
        "langfuse": bool(results[2]) if not isinstance(results[2], Exception) else False,
        "redis":    bool(results[3]) if not isinstance(results[3], Exception) else False,
    }

    # Consider healthy if at least the app is running (services can be connected later)
    healthy_count = sum(1 for v in service_status.values() if v)
    if healthy_count >= 2:  # At least 2 services connected
        overall = "healthy"
    elif healthy_count >= 1:
        overall = "degraded"
    else:
        overall = "degraded"  # App is running but no external services

    return HealthResponse(
        status=overall,
        version="1.1.0",
        services=service_status,
    )
