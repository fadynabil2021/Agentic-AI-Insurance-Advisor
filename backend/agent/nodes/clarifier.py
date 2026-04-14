"""
Clarifier Node — returns a structured clarification request when required fields are missing.
Uses template-based responses (no LLM dependency).
Routes to END so the user can resubmit with the missing information.
"""
from agent.state import AgentState, ConfidenceLevel
from clients.langfuse_client import safe_create_span


def clarifier_node(state: AgentState, langfuse_trace) -> AgentState:
    """
    Node: Generate a clarification question for the user.
    Always routes to END — user must resubmit with missing info.
    """
    span = safe_create_span(langfuse_trace, "clarifier", {
        "missing_fields": state.get("missing_fields"),
    })

    # Ensure these fields are always set for an incomplete response
    state["recommendation"] = None
    state["reasoning"] = None
    state["confidence"] = ConfidenceLevel.NONE
    state["requires_clarification"] = True
    state["fallback_or_risk_note"] = (
        "Recommendation cannot be generated without "
        + " and ".join(state.get("missing_fields") or ["required"])
        + " information."
    )

    # clarification_question was set by validator — just confirm it's present
    if not state.get("clarification_question"):
        state["clarification_question"] = (
            "Could you provide more details about your company's industry and region?"
        )

    state["execution_trace"].append(
        f"clarifier: generated clarification question for missing fields "
        f"{state.get('missing_fields')}"
    )

    if span:
        try:
            span.end(output={"clarification_question": state.get("clarification_question")})
        except Exception:
            pass

    return state
