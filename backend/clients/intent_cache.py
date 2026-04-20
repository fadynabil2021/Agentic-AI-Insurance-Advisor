"""
In-memory LRU cache for intent parsing results.
Avoids repeated LLM calls for identical queries within a short time window.
"""
from collections import OrderedDict
import hashlib
import time
from typing import Optional, Dict, Any

class IntentCache:
    """Simple in-memory LRU cache with TTL for intent parsing."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds

    def _make_key(self, user_request: str) -> str:
        """Create a cache key from the user request."""
        return hashlib.sha256(user_request.encode()).hexdigest()[:16]

    async def get(self, user_request: str) -> Optional[Dict[str, Any]]:
        """Get cached intent if present and not expired."""
        key = self._make_key(user_request)
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return value
            else:
                # Expired - remove
                del self._cache[key]
        return None

    async def set(self, user_request: str, intent_data: Dict[str, Any]) -> None:
        """Store intent data in cache."""
        key = self._make_key(user_request)
        timestamp = time.time()

        # Evict oldest if at capacity
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (intent_data, timestamp)

    async def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()


# Global singleton instance
intent_cache = IntentCache(max_size=1000, ttl_seconds=300)
