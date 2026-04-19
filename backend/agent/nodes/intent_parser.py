"""
Intent Parser Node — uses Gemini API to extract structured entities
from the user's natural language query.
"""
import json
from agent.state import AgentState, QueryType
from clients.gemini_client import GeminiClient, safe_json_parse
from clients.langfuse_client import safe_create_span

INTENT_SYSTEM_PROMPT = """You are an intent extraction engine for an insurance advisor system.
Extract structured information from the user query.

Return valid JSON matching this schema. Wrap your final JSON answer inside <answer> and </answer> tags (no preamble, no markdown, no explanation outside the tags):
{
  "query_type": one of ["recommend", "compare", "explain", "cheapest", "clarify", "unsupported"],
  "industry": string or null,
  "region": string or null,
  "budget": one of ["low", "medium", "high"] or null,
  "priority": string or null,
  "employees": integer or null,
  "dependents_ratio": float or null,
  "compare_packages": list of strings or null
}

Rules:
- If the request is not about insurance plan selection, set query_type to "unsupported"
- If the request asks for cheapest / lowest cost, set query_type to "cheapest"
- If the request asks to compare two plans, set query_type to "compare"
- If the request asks to explain a previous recommendation, set query_type to "explain"
- Do NOT infer fields that are not stated or strongly implied by the request
- Return null for any field you cannot determine
- region values: normalize to lowercase (riyadh, jeddah, dammam)
- industry values: normalize to lowercase (healthcare, construction, retail)
- budget values: low, medium, high only"""

# Required fields per query type (for missing_fields detection)
REQUIRED_FIELDS_PER_QUERY_TYPE: dict[str, list[str]] = {
    "recommend": ["industry", "region"],
    "cheapest":  ["industry", "region"],
    "compare":   ["industry", "region"],
    "explain":   [],
    "clarify":   [],
    "unsupported": [],
}


async def intent_parser_node(state: AgentState, gemini_client: GeminiClient, langfuse_trace) -> AgentState:
    """
    Node 1: Parse user intent and extract entities.
    LLM call: Gemini API.
    """
    span = safe_create_span(langfuse_trace, "intent_parser", {"user_request": state["user_request"]})

    raw = None
    parsed = None
    for attempt in range(3):  # up to 2 retries
        try:
            raw = await gemini_client.chat(
                messages=[
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": state["user_request"]},
                ],
                temperature=0.1,
            )
            parsed = safe_json_parse(raw)
            if parsed is not None:
                break
        except Exception as e:
            state["execution_trace"].append(f"intent_parser: attempt {attempt+1} failed — {e}")

    if parsed is None:
        state["error"] = {"type": "PARSE_ERROR", "message": "Failed to parse intent after 3 attempts"}
        state["execution_trace"].append("intent_parser: FAILED to parse intent, routing to fallback")
        if span:
            try:
                span.end(output={"error": state["error"]})
            except Exception:
                pass
        return state

    # Map parsed output to state
    state["query_type"] = parsed.get("query_type", "unsupported")
    entities = {
        k: v for k, v in parsed.items()
        if k != "query_type" and v is not None
    }
    state["extracted_entities"] = entities

    # Identify missing required fields
    qt = state["query_type"]
    required = REQUIRED_FIELDS_PER_QUERY_TYPE.get(qt, [])
    state["missing_fields"] = [f for f in required if not entities.get(f)]

    state["execution_trace"].append(
        f"intent_parser: query_type={qt}, "
        f"entities={entities}, "
        f"missing={state['missing_fields']}"
    )

    if span:
        try:
            span.end(output={"query_type": qt, "entities": entities, "missing": state["missing_fields"]})
        except Exception:
            pass

    return state
