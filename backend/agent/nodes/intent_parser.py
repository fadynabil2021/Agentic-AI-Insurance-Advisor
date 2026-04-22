"""
Intent Parser Node — uses Gemini API to extract structured entities
from the user's natural language query.
"""
import json
from agent.state import AgentState, QueryType
from clients.gemini_client import GeminiClient, safe_json_parse
from clients.langfuse_client import safe_create_span

INTENT_SYSTEM_PROMPT = """You are a strictly bound intent extraction engine for a Saudi Arabian Insurance Advisor.
Your ONLY purpose is to help with business insurance plan selection within Saudi Arabia.

CONSTRAINTS (CRITICAL):
1. SUPPORTED REGIONS: ONLY "riyadh", "jeddah", "dammam".
2. SUPPORTED INDUSTRIES: ONLY "healthcare", "construction", "retail".
3. If the user asks about ANY region outside Saudi Arabia (e.g., London, Cairo, New York), set query_type to "unsupported".
4. If the user asks about ANY industry not listed above (e.g., food delivery, tourism, agriculture), set query_type to "unsupported".
5. If the request is not related to insurance plan selection, set query_type to "unsupported".

Extraction Schema (Return ONLY JSON inside <answer></answer> tags):
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

Inference Rules:
- INFER budget/priority from tone: "best/top/premium" -> high/maximum; "cheapest/budget/low" -> low/lowest cost.
- industry normalization: "hospital/clinic" -> healthcare; "builder/engineer" -> construction; "shop/store" -> retail.
- region normalization: lowercase (riyadh, jeddah, dammam)."""

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

    # Check for cached intent parsing (same query = same intent)
    from clients.intent_cache import intent_cache
    user_request = state["user_request"]
    cached_intent = await intent_cache.get(user_request)
    if cached_intent:
        state["query_type"] = cached_intent.get("query_type", "unsupported")
        state["extracted_entities"] = cached_intent.get("entities", {})
        state["missing_fields"] = cached_intent.get("missing_fields", [])
        state["execution_trace"].append(f"intent_parser: CACHE HIT for query")
        if span:
            try:
                span.end(output={"cached": True, "query_type": state["query_type"]})
            except Exception:
                pass
        return state

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
                max_tokens=256,  # Reduced for faster response
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

    # Cache the parsed intent for future identical queries
    await intent_cache.set(user_request, {
        "query_type": qt,
        "entities": entities,
        "missing_fields": state["missing_fields"],
    })

    if span:
        try:
            span.end(output={"query_type": qt, "entities": entities, "missing": state["missing_fields"]})
        except Exception:
            pass

    return state
