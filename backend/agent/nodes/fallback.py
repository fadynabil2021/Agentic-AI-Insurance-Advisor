"""
Fallback Node — handles all failure modes gracefully.
Uses static response templates — no LLM.
"""
from agent.state import AgentState, ConfidenceLevel
from clients.langfuse_client import safe_create_span

FALLBACK_MESSAGES = {
    "UNSUPPORTED_QUERY": (
        "This request is outside the supported service area. "
        "We only provide insurance recommendations for: "
        "• Regions: Riyadh, Jeddah, Dammam (Saudi Arabia only) "
        "• Industries: Healthcare, Construction, Retail "
        "Please resubmit with valid region and industry details."
    ),
    "EMPTY_RETRIEVAL": (
        "No insurance packages matched your criteria. "
        "This may happen with unusual combinations of industry, region, and budget. "
        "Try adjusting your requirements or contact an advisor for custom quotes."
    ),
    "LOW_CONFIDENCE": (
        "Recommendation generated with low confidence due to conflicting requirements. "
        "Review the reasoning bullets carefully — you may want to adjust your priorities."
    ),
    "PARSE_ERROR": (
        "Could not parse your request. "
        "Include: industry (healthcare/construction/retail), region (Riyadh/Jeddah/Dammam), "
        "and budget level (low/medium/high) for best results."
    ),
    "DEFAULT": (
        "Unable to process this request. "
        "Please rephrase or provide more details about your insurance needs."
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
    message = FALLBACK_MESSAGES.get(error_type, FALLBACK_MESSAGES["DEFAULT"])

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
