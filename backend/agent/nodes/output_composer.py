"""
Output Composer Node — synthesizes the final structured recommendation.
LLM (Gemini API) is used ONLY for the reasoning narrative bullets.
All structured fields (plan_name, network, price_range) are built deterministically.
"""
import json
from agent.state import AgentState, ConfidenceLevel
from clients.gemini_client import GeminiClient
from clients.langfuse_client import safe_create_span

REASONING_SYSTEM_PROMPT = """Based ONLY on the provided scoring data, generate 3-5 concise reasoning bullets explaining why the selected package was chosen.

Rules:
- Each bullet MUST reference at least one specific data point (score, rule, entity value)
- Do NOT introduce new information not present in the scoring data
- Return ONLY a JSON array of strings, e.g.: ["Reason 1", "Reason 2", "Reason 3"]
- No preamble, no markdown, no tags, no explanation - just valid JSON."""


def build_risk_note(top_pkg_name: str, scoring_results: list[dict], entities: dict) -> str:
    """
    Build a deterministic risk/tradeoff note.
    Always present — draws from scoring results.
    """
    top = top_pkg_name.lower()
    others = [r for r in scoring_results if r["package"].get("name", "").lower() != top]
    industry = entities.get("industry", "")
    dep_ratio = entities.get("dependents_ratio")

    notes = []

    if top == "basic" and industry.lower() == "healthcare":
        notes.append("⚠ Basic is not recommended for healthcare — consider Standard for better network coverage.")
    elif top == "standard":
        prem = next((r for r in others if r["package"].get("name", "").lower() == "premium"), None)
        if prem:
            notes.append(
                f"Standard is recommended. Premium (score {prem['score']}/100) is available "
                "if the customer has high dependents ratio or specific Network A requirements."
            )
        if industry.lower() == "healthcare":
            notes.append("Basic is not suitable for healthcare industry.")
    elif top == "premium":
        std = next((r for r in others if r["package"].get("name", "").lower() == "standard"), None)
        if std:
            notes.append(
                f"Premium selected for best coverage. Standard (score {std['score']}/100) "
                "is available at lower cost if budget is a priority."
            )

    if dep_ratio and dep_ratio > 0.5:
        notes.append(
            f"High dependents ratio ({dep_ratio:.0%}) increases benefits cost — "
            "ensure selected plan covers all dependents adequately."
        )

    if not notes:
        notes.append(
            f"{top_pkg_name.title()} selected based on scoring rules. "
            "Review the execution trace for full scoring breakdown."
        )

    return " | ".join(notes)


def parse_reasoning_list(raw: str) -> list[str]:
    """Extract a list of strings from raw Gemma 4 output."""
    text = raw.strip()
    
    # Try to extract content between <answer> tags
    import re
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    elif text.startswith("```"):
        # Fallback for markdown fences
        lines = text.split("\n")
        text = "\n".join(l for l in lines[1:] if l.strip() != "```").strip()
        
    # Just split by newlines and clean up
    lines = [l.lstrip("•-* ").strip() for l in text.split("\n") if l.strip()]
    return lines[:5] if lines else ["Recommendation generated based on scoring rules."]


async def output_composer_node(
    state: AgentState,
    gemini_client: GeminiClient,
    langfuse_trace,
) -> AgentState:
    """
    Node 7: Compose the final structured recommendation.
    Deterministic fields + LLM reasoning narrative.
    """
    span = safe_create_span(langfuse_trace, "output_composer", {
        "query_type": state.get("query_type"),
    })

    state["tools_used"].append("output_composer")
    qt = state.get("query_type", "recommend")

    # Handle "explain" query type — pull from prior scoring if available
    scoring_results = state.get("scoring_results") or []

    if not scoring_results and qt != "explain":
        state["error"] = {"type": "EMPTY_RETRIEVAL", "message": "No scored packages available for composition"}
        state["execution_trace"].append("output_composer: no scoring results — routing to fallback state")
        if span:
            try:
                span.end(output={"error": "no scoring results"})
            except Exception:
                pass
        return state

    if scoring_results:
        top_result = scoring_results[0]
        top_pkg = top_result["package"]
    else:
        # explain mode with no prior scoring — use cached state info
        top_pkg = {"name": "Standard", "network": "B", "price_range": [6000, 7500], "coverage": "Medium"}
        top_result = {"package": top_pkg, "score": 80, "reasons": ["Based on prior session scoring."]}

    # ── Deterministic fields ───────────────────────────────────────────────────
    price_range = top_pkg.get("price_range", [0, 0])
    if isinstance(price_range, str):
        # Handle string format like "[6000, 7500]"
        import ast
        try:
            price_range = ast.literal_eval(price_range)
        except Exception:
            price_range = [0, 0]

    state["recommendation"] = {
        "plan_name": top_pkg.get("name", "Standard"),
        "network": top_pkg.get("network", "B"),
        "price_range": price_range if isinstance(price_range, list) else [0, 0],
    }

    # ── LLM reasoning narrative ────────────────────────────────────────────────
    entities = state.get("extracted_entities") or {}
    scoring_breakdown = state.get("scoring_breakdown") or {}
    top_name_lower = top_pkg.get("name", "Standard").lower()

    # FAST PATH: Use deterministic reasoning from scoring breakdown (skip LLM call)
    # This significantly reduces latency for most queries
    reasons = scoring_breakdown.get(top_name_lower, {}).get("reasons", [])
    if reasons:
        # Convert scoring reasons to user-friendly reasoning bullets
        reasoning = []
        for r in reasons[:5]:
            # Clean up penalty/bonus prefixes for cleaner output
            clean_reason = r
            if clean_reason.startswith("PENALTY("):
                clean_reason = clean_reason.split("): ", 1)[-1] if "): " in clean_reason else clean_reason
            elif clean_reason.startswith("BONUS("):
                clean_reason = clean_reason.split("): ", 1)[-1] if "): " in clean_reason else clean_reason
            elif clean_reason.startswith("NOTE("):
                clean_reason = clean_reason.split("): ", 1)[-1] if "): " in clean_reason else clean_reason
            reasoning.append(clean_reason)
        state["execution_trace"].append(f"output_composer: using deterministic reasoning ({len(reasoning)} bullets)")
    else:
        # FALLBACK: LLM-generated reasoning (slower, but more detailed)
        reasoning_context = {
            "package": top_pkg.get("name"),
            "entities": entities,
            "scoring_reasons": scoring_breakdown.get(top_name_lower, {}).get("reasons", []),
            "all_scores": {
                r["package"].get("name", "?"): r["score"]
                for r in scoring_results
            },
            "query_type": qt,
        }

        try:
            raw_reasoning = await gemini_client.chat(
                messages=[
                    {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(reasoning_context)},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            import json as json_lib
            try:
                reasoning = json_lib.loads(raw_reasoning.strip())
                if not isinstance(reasoning, list):
                    reasoning = parse_reasoning_list(raw_reasoning)
            except json_lib.JSONDecodeError:
                reasoning = parse_reasoning_list(raw_reasoning)
        except Exception:
            reasoning = [
                f"{top_pkg.get('name')} selected with score {top_result['score']}/100.",
                f"Query type: {qt}. Industry: {entities.get('industry', 'N/A')}. Region: {entities.get('region', 'N/A')}.",
            ]

    state["reasoning"] = reasoning

    # ── Deterministic risk note ────────────────────────────────────────────────
    state["fallback_or_risk_note"] = build_risk_note(
        top_pkg.get("name", "Standard"),
        scoring_results,
        entities,
    )

    state["execution_trace"].append(
        f"output_composer: composed recommendation={state['recommendation']['plan_name']}, "
        f"confidence={state.get('confidence')}, "
        f"reasoning_bullets={len(reasoning)}"
    )

    if span:
        try:
            span.end(output={
                "recommendation": state["recommendation"],
                "confidence": str(state.get("confidence")),
                "reasoning_count": len(reasoning),
            })
        except Exception:
            pass

    return state
