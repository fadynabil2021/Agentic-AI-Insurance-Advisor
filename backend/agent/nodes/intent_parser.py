"""
Intent Parser Node — uses Gemini API to extract structured entities
from the user's natural language query.
"""
import json
from agent.state import AgentState, QueryType
from clients.gemini_client import GeminiClient, safe_json_parse
from clients.langfuse_client import safe_create_span

INTENT_SYSTEM_PROMPT = """You are an intent extraction engine for a Saudi Arabian Insurance Advisor.

SUPPORTED REGIONS (Saudi Arabia only): riyadh, jeddah, dammam
SUPPORTED INDUSTRIES: healthcare, construction, retail

YOUR TASK:
1. Extract industry and region from the user's query
2. Infer budget/priority from tone: "best/top/premium" -> high; "cheapest/budget/low" -> low
3. Return ONLY valid JSON inside <answer></answer> tags

QUERY TYPE RULES:
- "recommend" — user asks for best/top recommendation
- "cheapest" — user asks for lowest cost option
- "compare" — user wants to compare packages
- "explain" — user asks about plans/coverage details
- "clarify" — user asks a question needing clarification
- "unsupported" — region outside Saudi Arabia OR industry not supported OR not about insurance

EXAMPLES:
Input: "Recommend the best plan for a healthcare company in Riyadh"
Output: <answer>{"query_type": "recommend", "industry": "healthcare", "region": "riyadh", "budget": "high", "priority": "best"}</answer>

Input: "Cheapest insurance for construction in Jeddah"
Output: <answer>{"query_type": "cheapest", "industry": "construction", "region": "jeddah", "budget": "low"}</answer>

Input: "Best plan for healthcare in London"
Output: <answer>{"query_type": "unsupported", "industry": null, "region": null}</answer>

Input: "Insurance for tourism company in Riyadh"
Output: <answer>{"query_type": "unsupported", "industry": null, "region": null}</answer>

NORMALIZATION:
- industry: "hospital/clinic/medical" -> "healthcare"; "builder/engineering" -> "construction"; "shop/store/retail" -> "retail"
- region: lowercase (riyadh, jeddah, dammam)"""

# Required fields per query type (for missing_fields detection)
REQUIRED_FIELDS_PER_QUERY_TYPE: dict[str, list[str]] = {
    "recommend": ["industry", "region"],
    "cheapest":  ["industry", "region"],
    "compare":   ["industry", "region"],
    "explain":   [],
    "clarify":   [],
    "unsupported": [],
}

# Supported values for validation
SUPPORTED_REGIONS = {"riyadh", "jeddah", "dammam"}
SUPPORTED_INDUSTRIES = {"healthcare", "construction", "retail"}


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
    parse_attempts = 0
    for attempt in range(3):  # up to 2 retries
        try:
            raw = await gemini_client.chat(
                messages=[
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": state["user_request"]},
                ],
                temperature=0.1,
                max_tokens=256,
            )
            parse_attempts += 1
            print(f"[intent_parser] Attempt {parse_attempts}, raw response: {raw[:300]}...")
            parsed = safe_json_parse(raw)
            if parsed is not None:
                print(f"[intent_parser] SUCCESS on attempt {parse_attempts}: {parsed}")
                break
            else:
                print(f"[intent_parser] Failed to parse JSON on attempt {parse_attempts}")
        except Exception as e:
            state["execution_trace"].append(f"intent_parser: attempt {attempt+1} failed — {e}")
            print(f"[intent_parser] Exception on attempt {attempt+1}: {e}")

    if parsed is None:
        state["error"] = {"type": "PARSE_ERROR", "message": "Failed to parse intent after 3 attempts"}
        state["execution_trace"].append("intent_parser: FAILED to parse intent, routing to fallback")
        print(f"[intent_parser] ALL ATTEMPTS FAILED - routing to fallback")
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

    # POST-PARSING VALIDATION: Programmatically validate region and industry
    # This ensures unsupported regions/industries are caught even if LLM misses them
    region = entities.get("region", "").lower() if entities.get("region") else ""
    industry = entities.get("industry", "").lower() if entities.get("industry") else ""

    if region and region not in SUPPORTED_REGIONS:
        state["query_type"] = QueryType.UNSUPPORTED
        state["extracted_entities"] = {}
        state["execution_trace"].append(f"intent_parser: region '{region}' not supported, routing to fallback")
    elif industry and industry not in SUPPORTED_INDUSTRIES:
        state["query_type"] = QueryType.UNSUPPORTED
        state["extracted_entities"] = {}
        state["execution_trace"].append(f"intent_parser: industry '{industry}' not supported, routing to fallback")

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
