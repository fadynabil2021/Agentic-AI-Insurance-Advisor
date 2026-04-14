"""
Redis client wrapper for response caching and deduplication.
"""
import hashlib
import json
import redis.asyncio as aioredis
from config import settings

_redis_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis_client


def make_cache_key(user_request: str) -> str:
    """Create a cache key from the user request string."""
    return f"insurance:query:{hashlib.sha256(user_request.encode()).hexdigest()[:16]}"


async def get_cached_response(user_request: str) -> dict | None:
    """Return cached response if present, else None."""
    try:
        r = get_redis()
        raw = await r.get(make_cache_key(user_request))
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


async def cache_response(user_request: str, response: dict) -> None:
    """Store response in Redis with TTL."""
    try:
        r = get_redis()
        await r.setex(
            make_cache_key(user_request),
            settings.CACHE_TTL_SECONDS,
            json.dumps(response),
        )
    except Exception:
        pass


async def redis_health_check() -> bool:
    """Ping Redis to check connectivity."""
    try:
        r = get_redis()
        return await r.ping()
    except Exception:
        return False
