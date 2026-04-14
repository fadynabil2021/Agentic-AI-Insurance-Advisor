"""
Validator Node — deterministic required-field check.
No LLM involved. Pure Python logic.
"""
from agent.state import AgentState, QueryType
from clients.langfuse_client import safe_create_span

REQUIRED_FIELDS: dict[str, list[str]] = {
    "recommend":   ["industry", "region"],
    "cheapest":    ["industry", "region"],
    "compare":     ["industry", "region"],
    "explain":     [],
    "clarify":     [],
    "unsupported": [],
}


def build_clarification_question(missing: list[str]) -> str:
    questions = []
    field_prompts = {
        "industry": "What industry is your company in? (e.g., healthcare, construction, retail)",
        "region": "Which region are you based in? (e.g., Riyadh, Jeddah, Dammam)",
        "budget": "What is your approximate budget level? (low / medium / high)",
        "compare_packages": "Which two packages would you like to compare? (e.g., Standard and Premium)",
    }
    for i, field in enumerate(missing, 1):
        prompt = field_prompts.get(field, f"Please provide your {field}")
        questions.append(f"({i}) {prompt}")

    intro = "To give you an accurate recommendation, I need a few more details:"
    optional = "\nOptionally, you can also share your approximate budget level (low/medium/high) and any coverage priorities."
    return f"{intro}\n" + "\n".join(questions) + optional


def validator_node(state: AgentState, langfuse_trace) -> AgentState:
    """
    Node 2: Validate that required fields are present for the detected query type.
    Sets requires_clarification and clarification_question if fields are missing.
    """
    span = safe_create_span(langfuse_trace, "validator", {
        "query_type": state.get("query_type"),
        "entities": state.get("extracted_entities"),
    })

    qt = state.get("query_type")

    # Handle error states passed through from intent_parser
    error = state.get("error")
    if error and error.get("type") in ("PARSE_ERROR",):
        state["execution_trace"].append("validator: upstream parse error, routing to fallback")
        if span:
            try:
                span.end(output={"routed_to": "fallback"})
            except Exception:
                pass
        return state

    if qt == QueryType.UNSUPPORTED or qt == "unsupported":
        state["error"] = {
            "type": "UNSUPPORTED_QUERY",
            "message": "Request is outside the supported domain (insurance plan selection).",
        }
        state["execution_trace"].append("validator: query flagged UNSUPPORTED")
        if span:
            try:
                span.end(output={"routed_to": "fallback"})
            except Exception:
                pass
        return state

    required = REQUIRED_FIELDS.get(qt or "recommend", [])
    entities = state.get("extracted_entities") or {}
    missing = [f for f in required if not entities.get(f)]
    state["missing_fields"] = missing

    if missing:
        state["requires_clarification"] = True
        state["clarification_question"] = build_clarification_question(missing)
        state["execution_trace"].append(
            f"validator: missing required fields {missing}, will request clarification"
        )
    else:
        state["requires_clarification"] = False
        state["execution_trace"].append("validator: all required fields present, proceeding to planner")

    if span:
        try:
            span.end(output={"missing": missing, "requires_clarification": state["requires_clarification"]})
        except Exception:
            pass

    return state
