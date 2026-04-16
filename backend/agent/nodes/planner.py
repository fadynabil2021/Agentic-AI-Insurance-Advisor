"""
Planner Node — selects the execution plan (ordered list of tool names).
Uses a static lookup table for known query types (fast, deterministic).
 Falls back to Gemini API only for ambiguous cases not covered by the table.
"""
from agent.state import AgentState, QueryType
from clients.gemini_client import GeminiClient
from clients.langfuse_client import safe_create_span

# Deterministic plan for each known query type
STATIC_PLANS: dict[str, list[str]] = {
    "recommend": ["retrieval_tool", "scoring_tool", "output_composer"],
    "cheapest":  ["retrieval_tool", "scoring_tool", "output_composer"],
    "compare":   ["retrieval_tool", "scoring_tool", "comparison_tool", "output_composer"],
    "explain":   ["output_composer"],
}


async def planner_node(state: AgentState, gemini_client: GeminiClient, langfuse_trace) -> AgentState:
    """
    Node 3: Decide which tools to invoke and in what order.
    Deterministic for all known query types; LLM only for edge cases.
    """
    span = safe_create_span(langfuse_trace, "planner", {
        "query_type": state.get("query_type"),
        "entities": state.get("extracted_entities"),
    })

    qt = state.get("query_type", "recommend")

    if qt in STATIC_PLANS:
        plan = STATIC_PLANS[qt]
        state["execution_trace"].append(f"planner: deterministic plan selected={plan} for query_type={qt}")
    else:
        # LLM-assisted planning for ambiguous cases
        try:
            prompt = (
                f"Given query_type='{qt}' and entities={state.get('extracted_entities')}, "
                "return a JSON list of tools to call in order from: "
                "[retrieval_tool, scoring_tool, comparison_tool, output_composer]. "
                "Return ONLY a JSON array."
            )
            raw = await gemini_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            import json
            # Extract JSON array
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1:
                plan = json.loads(raw[start:end+1])
            else:
                plan = STATIC_PLANS["recommend"]  # safe fallback
        except Exception:
            plan = STATIC_PLANS["recommend"]
        state["execution_trace"].append(f"planner: LLM-assisted plan={plan}")

    state["execution_plan"] = plan

    if span:
        try:
            span.end(output={"plan": plan})
        except Exception:
            pass

    return state
