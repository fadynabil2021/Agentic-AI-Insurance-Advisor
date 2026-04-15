"""
Langfuse client singleton.
All calls are wrapped in try/except so Langfuse outages never block agent responses.
"""
from langfuse import Langfuse
from config import settings

_langfuse_client: Langfuse | None = None


def get_langfuse() -> Langfuse | None:
    global _langfuse_client
    if _langfuse_client is None:
        try:
            _langfuse_client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
        except Exception:
            return None
    return _langfuse_client


def safe_score(trace_id: str, name: str, value: float) -> None:
    """Log a score to Langfuse, silently ignoring any errors."""
    try:
        lf = get_langfuse()
        if lf is not None:
            lf.score(trace_id=trace_id, name=name, value=value)
    except Exception:
        pass


def safe_create_trace(name: str, trace_id: str, session_id: str | None = None, metadata: dict | None = None):
    """Create a Langfuse trace, returning None on failure."""
    try:
        lf = get_langfuse()
        if lf is None:
            return None
        return lf.trace(id=trace_id, session_id=session_id, name=name, metadata=metadata or {})
    except Exception:
        return None


def safe_create_span(trace, name: str, input_data: dict | None = None):
    """Create a Langfuse span under the given trace, returning None on failure."""
    try:
        if trace is None:
            return None
        return trace.span(name=name, input=input_data or {})
    except Exception:
        return None


async def langfuse_health_check() -> bool:
    """Check if Langfuse is reachable via its health endpoint."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{settings.LANGFUSE_HOST}/api/public/health")
            return r.status_code == 200
    except Exception:
        return False
