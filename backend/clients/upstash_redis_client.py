"""
Upstash Redis client for serverless Redis access.
Replaces local Redis for cloud-native deployment.
"""
from upstash_redis.asyncio import Redis as AsyncRedis
from typing import Optional
import hashlib
import json


_redis_client: Optional[AsyncRedis] = None


def get_upstash_redis(url: str, token: str) -> Optional[AsyncRedis]:
    """Get or create the Upstash Redis client singleton."""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = AsyncRedis(url=url, token=token)
        except Exception:
            return None
    return _redis_client


def make_cache_key(user_request: str) -> str:
    """Create a cache key from the user request string."""
    return f"insurance:query:{hashlib.sha256(user_request.encode()).hexdigest()[:16]}"


async def get_cached_response(user_request: str, redis_client: AsyncRedis) -> Optional[dict]:
    """Return cached response if present, else None."""
    try:
        raw = await redis_client.get(make_cache_key(user_request))
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


async def cache_response(user_request: str, response: dict, redis_client: AsyncRedis, ttl: int = 600) -> None:
    """Store response in Redis with TTL."""
    try:
        await redis_client.setex(
            make_cache_key(user_request),
            ttl,
            json.dumps(response),
        )
    except Exception:
        pass


async def redis_health_check(redis_client: Optional[AsyncRedis]) -> bool:
    """Ping Redis to check connectivity."""
    try:
        if redis_client is None:
            return False
        result = await redis_client.ping()
        return result == "PONG"
    except Exception:
        return False
