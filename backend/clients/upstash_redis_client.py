"""
Upstash Redis client for cloud-native caching.
Self-contained singleton — callers use get_cached_response / cache_response directly.
Credentials are read from settings (UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN).
"""
from upstash_redis.asyncio import Redis
from typing import Optional
import hashlib
import json

from config import settings

_redis_client: Optional[Redis] = None


def _get_client() -> Optional[Redis]:
    """Lazy-initialise the Upstash Redis singleton."""
    global _redis_client
    if _redis_client is None:
        if not settings.UPSTASH_REDIS_REST_URL or not settings.UPSTASH_REDIS_REST_TOKEN:
            return None
        try:
            _redis_client = Redis(
                url=settings.UPSTASH_REDIS_REST_URL,
                token=settings.UPSTASH_REDIS_REST_TOKEN,
            )
        except Exception:
            return None
    return _redis_client


def make_cache_key(user_request: str) -> str:
    return f"insurance:query:{hashlib.sha256(user_request.encode()).hexdigest()[:16]}"


async def get_cached_response(user_request: str) -> Optional[dict]:
    """Return cached response dict if present, else None."""
    try:
        client = _get_client()
        if client is None:
            return None
        raw = await client.get(make_cache_key(user_request))
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


async def cache_response(user_request: str, response: dict) -> None:
    """Store response in Upstash Redis with TTL."""
    try:
        client = _get_client()
        if client is None:
            return
        await client.setex(
            make_cache_key(user_request),
            settings.CACHE_TTL_SECONDS,
            json.dumps(response),
        )
    except Exception:
        pass


async def redis_health_check() -> bool:
    """Ping Upstash Redis to check connectivity."""
    try:
        client = _get_client()
        if client is None:
            return False
        result = await client.ping()
        return result == "PONG"
    except Exception:
        return False
