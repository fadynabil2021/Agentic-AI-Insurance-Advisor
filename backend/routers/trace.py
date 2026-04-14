"""
GET /api/v1/trace/{trace_id} — fetch trace details from Langfuse.
"""
from fastapi import APIRouter, HTTPException
from models.schemas import TraceResponse
from clients.langfuse_client import get_langfuse
from config import settings

router = APIRouter(tags=["Trace"])


@router.get("/trace/{trace_id}", response_model=TraceResponse)
async def get_trace(trace_id: str):
    """
    Retrieve trace details from Langfuse for a given trace ID.
    Returns span tree, scores, latency, and token usage.
    """
    try:
        lf = get_langfuse()
        # Langfuse Python SDK fetch trace
        trace = lf.get_trace(trace_id)

        steps = []
        if hasattr(trace, "observations"):
            for obs in trace.observations:
                steps.append({
                    "name": getattr(obs, "name", ""),
                    "type": getattr(obs, "type", "span"),
                    "start_time": str(getattr(obs, "start_time", "")),
                    "end_time": str(getattr(obs, "end_time", "")),
                    "input": getattr(obs, "input", {}),
                    "output": getattr(obs, "output", {}),
                    "latency_ms": getattr(obs, "latency", 0),
                })

        scores = {}
        if hasattr(trace, "scores"):
            for score in trace.scores:
                scores[score.name] = score.value

        return TraceResponse(
            trace_id=trace_id,
            langfuse_url=f"{settings.LANGFUSE_HOST.replace('langfuse:3000', 'localhost:3001')}/trace/{trace_id}",
            steps=steps,
            scores=scores,
            total_latency_ms=getattr(trace, "latency", 0) or 0,
            token_usage=getattr(trace, "usage", {}) or {},
        )
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Trace not found or Langfuse unavailable: {str(e)}",
        )
