"""
GET /api/v1/health — system health check endpoint.
Checks Gemini API, Pinecone, Langfuse Cloud, and Upstash Redis connectivity.
"""
from fastapi import APIRouter, Request
from models.schemas import HealthResponse
from clients.langfuse_client import langfuse_health_check
from config import settings
import asyncio

router = APIRouter(tags=["Health"])


async def check_gemini() -> bool:
    """Check if Gemini API is reachable."""
    try:
        from clients.gemini_client import GeminiClient
        if not settings.GEMINI_API_KEY:
            print("[health] WARNING: GEMINI_API_KEY is missing")
            return False
        client = GeminiClient(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
        )
        is_healthy = await client.health_check()
        if not is_healthy:
            print("[health] WARNING: Gemini health check failed (API reachable but returned no content)")
        return is_healthy
    except Exception as e:
        print(f"[health] ERROR: Gemini check failed — {e}")
        return False


async def check_pinecone() -> bool:
    """Check if Pinecone index exists and is accessible."""
    try:
        from pinecone import Pinecone
        if not settings.PINECONE_API_KEY:
            print("[health] WARNING: PINECONE_API_KEY is missing")
            return False
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        indexes = pc.list_indexes().names()
        return settings.PINECONE_INDEX_NAME in indexes
    except Exception as e:
        print(f"[health] ERROR: Pinecone check failed — {e}")
        return False


async def check_upstash_redis() -> bool:
    """Check if Upstash Redis is reachable."""
    try:
        from upstash_redis.asyncio import Redis
        if not settings.UPSTASH_REDIS_REST_URL:
            print("[health] WARNING: UPSTASH_REDIS_REST_URL is missing")
            return False
        redis_client = Redis(
            url=settings.UPSTASH_REDIS_REST_URL,
            token=settings.UPSTASH_REDIS_REST_TOKEN,
        )
        result = await redis_client.ping()
        return result == "PONG"
    except Exception as e:
        print(f"[health] ERROR: Upstash Redis check failed — {e}")
        return False


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check connectivity of all dependent services."""
    results = await asyncio.gather(
        check_gemini(),
        check_pinecone(),
        langfuse_health_check(),
        check_upstash_redis(),
        return_exceptions=True,
    )

    service_status = {
        "gemini":   bool(results[0]) if not isinstance(results[0], Exception) else False,
        "pinecone": bool(results[1]) if not isinstance(results[1], Exception) else False,
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
        version="2.0.0",
        services=service_status,
    )
