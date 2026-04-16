"""
Fallback Node — handles all failure modes gracefully.
Uses static response templates — no LLM.
"""
from agent.state import AgentState, ConfidenceLevel
from clients.langfuse_client import safe_create_span

FALLBACK_MESSAGES = {
    "UNSUPPORTED_QUERY": (
        "This system is designed for insurance plan recommendations. "
        "Your query appears to be outside this scope. "
        "Please ask about plan selection, comparison, or cost optimization."
    ),
    "EMPTY_RETRIEVAL": (
        "Unable to find relevant packages matching your criteria. "
        "Please verify the industry and region details, "
        "or contact an advisor for custom options."
    ),
    "LOW_CONFIDENCE": (
        "The system was able to generate a recommendation but confidence is low "
        "due to conflicting constraints. "
        "Please review the reasoning carefully before proceeding."
    ),
    "PARSE_ERROR": (
        "Unable to understand the request. "
        "Please rephrase your question with the industry type, region, "
        "and any budget or coverage preferences."
    ),
    "DEFAULT": (
        "An unexpected error occurred while processing your request. "
        "Please try again or rephrase your query."
    ),
}


def fallback_node(state: AgentState, langfuse_trace) -> AgentState:
    """
    Final safety net — always returns a structured (non-null) response
    with confidence=none and an informative fallback note.
    """
    span = safe_create_span(langfuse_trace, "fallback", {
        "error": state.get("error"),
    })

    error = state.get("error") or {}
    error_type = error.get("type", "DEFAULT")
    
    # Use specific error message if available, otherwise use template
    message = error.get("message") or FALLBACK_MESSAGES.get(error_type, FALLBACK_MESSAGES["DEFAULT"])

    state["recommendation"] = None
    state["reasoning"] = None
    state["confidence"] = ConfidenceLevel.NONE
    state["fallback_or_risk_note"] = message
    state["requires_clarification"] = False

    state["execution_trace"].append(
        f"fallback: error_type={error_type}, message='{message[:80]}...'"
    )

    if span:
        try:
            span.end(output={"error_type": error_type, "fallback_message": message})
        except Exception:
            pass

    return state
