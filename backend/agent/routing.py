"""
Conditional routing functions for the LangGraph state graph.
These are pure deterministic Python — NO LLM calls.
"""
from agent.state import AgentState, QueryType


def route_after_validation(state: AgentState) -> str:
    """
    Routes after the validator node:
    - 'unsupported' → fallback (query outside domain)
    - 'incomplete'  → clarifier (missing required fields)
    - 'complete'    → planner (all required fields present)
    """
    error = state.get("error")
    if error and error.get("type") == "UNSUPPORTED_QUERY":
        return "unsupported"

    # Also handle parse errors from intent_parser
    if error and error.get("type") == "PARSE_ERROR":
        return "unsupported"

    missing = state.get("missing_fields") or []
    if len(missing) > 0:
        return "incomplete"

    return "complete"


def route_after_scoring(state: AgentState) -> str:
    """
    Routes after the scoring tool node:
    - 'retry'    → retrieval_tool (if no results and under retry limit)
    - 'compare'  → comparison_tool (if query_type == COMPARE)
    - 'fallback' → fallback (no scoring results)
    - 'recommend'→ output_composer (happy path)
    """
    from config import settings

    retry_count = state.get("retry_count", 0)
    retrieval_results = state.get("retrieval_results") or []
    scoring_results = state.get("scoring_results") or []

    # Retry if no packages found and under retry limit
    has_packages = any(r.get("type") == "package" for r in retrieval_results)
    if not has_packages and retry_count < settings.MAX_RETRIES:
        return "retry"

    # Fallback if scoring yielded nothing (unless it's an explain query)
    if not scoring_results:
        if state.get("query_type") == QueryType.EXPLAIN:
            return "recommend"
        return "fallback"

    # Compare flow
    if state.get("query_type") == QueryType.COMPARE:
        return "compare"

    return "recommend"
