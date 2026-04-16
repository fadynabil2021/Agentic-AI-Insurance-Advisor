"""
POST /api/v1/query — main agent endpoint.
Accepts a natural-language insurance query, runs the LangGraph, returns AgentResponse.
"""
import time
import uuid
from fastapi import APIRouter, Request, HTTPException
from models.schemas import QueryRequest, AgentResponse, Recommendation, ComparisonMatrix
from clients.upstash_redis_client import get_cached_response, cache_response
from clients.langfuse_client import safe_create_trace
from agent.graph import make_initial_state
from agent.state import ConfidenceLevel

router = APIRouter(tags=["Query"])


def _state_to_response(state: dict, latency_ms: int) -> AgentResponse:
    """Map final AgentState to the API AgentResponse model."""
    rec = state.get("recommendation")
    recommendation = None
    if rec:
        pr = rec.get("price_range", [0, 0])
        if isinstance(pr, str):
            import ast
            try:
                pr = ast.literal_eval(pr)
            except Exception:
                pr = [0, 0]
        recommendation = Recommendation(
            plan_name=rec.get("plan_name", "Unknown"),
            network=rec.get("network", "B"),
            price_range=pr if isinstance(pr, list) else [0, 0],
        )

    cm = state.get("comparison_matrix")
    comparison_matrix = None
    if cm:
        comparison_matrix = ComparisonMatrix(
            packages=cm.get("packages", []),
            dimensions=cm.get("dimensions", {}),
            recommendation=cm.get("recommendation"),
            recommendation_reason=cm.get("recommendation_reason"),
        )

    conf = state.get("confidence")
    if hasattr(conf, "value"):
        conf = conf.value
    confidence_str = str(conf) if conf else "none"

    return AgentResponse(
        user_request=state.get("user_request", ""),
        thread_id=state.get("thread_id", ""),
        plan={"steps": state.get("execution_plan") or []},
        tools_used=state.get("tools_used") or [],
        recommendation=recommendation,
        comparison_matrix=comparison_matrix,
        reasoning=state.get("reasoning"),
        confidence=confidence_str,
        execution_trace=state.get("execution_trace") or [],
        fallback_or_risk_note=state.get("fallback_or_risk_note"),
        requires_clarification=state.get("requires_clarification", False),
        clarification_question=state.get("clarification_question"),
        eval_scores=state.get("eval_scores"),
        latency_ms=latency_ms,
    )


@router.post("/query", response_model=AgentResponse)
async def run_query(request: QueryRequest, req: Request):
    """
    Main agent endpoint.
    1. Check Redis cache for duplicate request
    2. Run LangGraph
    3. Cache result
    4. Return AgentResponse
    """
    start = time.time()

    # Check cache
    cached = await get_cached_response(request.user_request)
    if cached:
        cached["latency_ms"] = int((time.time() - start) * 1000)
        return AgentResponse(**cached)

    # Generate or reuse thread_id
    thread_id = request.thread_id or str(uuid.uuid4())
    current_trace_id = str(uuid.uuid4())

    # Create Langfuse trace (non-blocking) — now threaded into graph state
    trace = safe_create_trace(
        name="insurance_advisor_query",
        trace_id=current_trace_id,
        session_id=thread_id,
        metadata={"user_request": request.user_request},
    )

    # Build initial state
    initial_state = make_initial_state(
        user_request=request.user_request,
        thread_id=thread_id,
    )

    # Get compiled graph from app state
    compiled_graph = req.app.state.compiled_graph

    # Check if graph was initialized (services available)
    if compiled_graph is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Agent graph not initialized. External services (Gemini, Pinecone) may be unavailable.",
        )

    try:
        config = {
            "configurable": {
                "thread_id": thread_id,
                "langfuse_trace": trace,
            }
        }
        final_state = await compiled_graph.ainvoke(initial_state, config=config)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Agent execution failed: {str(e)}",
        )

    latency_ms = int((time.time() - start) * 1000)
    response = _state_to_response(final_state, latency_ms)

    # Cache successful responses
    try:
        await cache_response(request.user_request, response.model_dump())
    except Exception:
        pass

    return response
