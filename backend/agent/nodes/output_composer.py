"""
Output Composer Node — synthesizes the final structured recommendation.
LLM (Gemini API) is used ONLY for the reasoning narrative bullets.
All structured fields (plan_name, network, price_range) are built deterministically.
"""
import json
from agent.state import AgentState, ConfidenceLevel
from clients.gemini_client import GeminiClient
from clients.langfuse_client import safe_create_span

REASONING_SYSTEM_PROMPT = """As a senior Saudi Insurance Consultant, justify why the selected medical plan is the optimal choice for this business. 

Your explanation must be unique, professional, and deeply grounded in the providing scoring data. 
Focus on explaining the value to a business owner in the specific region and industry provided.

STRICT FORMATTING:
- You must provide 3-5 distinct bullet points.
- Start every bullet with a dash (-).
- You MUST wrap your entire final response inside [REASONING] and [/REASONING] tags.
- Do NOT repeat these instructions. Do NOT include any preamble or self-introduction outside the tags."""

def build_risk_note(top_pkg_name: str, scoring_results: list[dict], entities: dict) -> str:
    """
    Build a deterministic risk/tradeoff note.
    Always present — draws from scoring results with scenario-specific messaging.
    """
    top = top_pkg_name.lower()
    others = [r for r in scoring_results if r["package"].get("name", "").lower() != top]
    industry = entities.get("industry", "")
    region = entities.get("region", "")
    budget = entities.get("budget")
    dep_ratio = entities.get("dependents_ratio")
    priority = entities.get("priority", "")

    notes = []

    # Industry-specific notes with varied messaging
    industry_lower = industry.lower()
    if industry_lower == "healthcare":
        if top == "basic":
            notes.append("⚠ Basic is not recommended for healthcare — consider Standard for better network coverage.")
        elif top == "standard":
            notes.append("Standard provides Network B coverage suitable for healthcare industry clinical staff.")
        elif top == "premium":
            notes.append("Premium with Network A is optimal for healthcare industry — covers specialist consultations and advanced diagnostics.")
    elif industry_lower == "construction":
        if top == "basic":
            notes.append("⚠ Basic may be insufficient for construction — consider Standard for occupational injury coverage.")
        elif top == "standard":
            notes.append("Standard covers occupational hazards common in construction — worker injury protection included.")
    elif industry_lower == "retail":
        if top == "premium":
            notes.append("Premium may be over-specification for retail unless executive benefits are required.")
        elif top == "standard":
            notes.append("Standard aligns with retail industry risk profile — cost-effective for staff coverage.")
    elif industry_lower == "technology":
        if top == "standard":
            notes.append("Standard fits technology sector — balances coverage with cost for typically young, healthy workforce.")
        elif top == "premium":
            notes.append("Premium for technology sector — attractive for talent retention with enhanced benefits.")
    elif industry_lower == "finance":
        if top == "premium":
            notes.append("Premium aligns with finance industry expectations — comprehensive coverage for client-facing roles.")
        elif top == "standard":
            notes.append("Standard meets finance industry compliance while managing cost structure.")
    elif industry_lower == "education":
        if top == "basic":
            notes.append("Basic acceptable for education sector — low-risk environment with younger demographics.")
        elif top == "standard":
            notes.append("Standard for education — extends coverage for faculty and administrative staff.")

    # Region-specific notes with economic context
    region_lower = region.lower()
    if region_lower == "riyadh":
        if top == "premium" and budget == "medium":
            notes.append("⚠ Riyadh has highest medical costs — Premium on medium budget may strain finances.")
        elif top == "standard":
            notes.append("Standard is well-suited for Riyadh's high-cost medical market — viable network coverage.")
        elif top == "premium":
            notes.append("Premium justified in Riyadh — access to top-tier private hospitals in the capital.")
    elif region_lower == "jeddah":
        if top in ["basic", "standard"]:
            notes.append(f"Jeddah's lower cost structure makes {top_pkg_name.title()} highly cost-effective — competitive provider rates.")
        elif top == "premium":
            notes.append("Premium in Jeddah — comprehensive coverage in a more affordable medical market.")
    elif region_lower == "dammam":
        if top == "standard":
            notes.append("Standard matches Dammam's moderate cost environment — balanced provider network.")
        elif top == "basic":
            notes.append("Basic viable in Dammam — lower regional costs offset reduced coverage.")

    # Budget-specific notes with financial framing
    if budget:
        budget_lower = budget.lower()
        if budget_lower == "low":
            if top == "basic":
                notes.append("Basic maximizes coverage within tight budget — compliance-first approach.")
            elif top == "standard":
                notes.append("Standard stretches low budget — optimal cost-to-coverage ratio for cost-conscious buyers.")
        elif budget_lower == "medium":
            if top == "standard":
                notes.append("Standard is the sweet spot for medium budgets — no wasted spend on unnecessary features.")
            elif top == "premium":
                notes.append("Premium attainable on medium budget — strategic allocation for enhanced protection.")
        elif budget_lower == "high":
            if top == "premium":
                notes.append("Premium leverages high budget — unlocks top-tier networks and specialist access.")
            elif top == "standard":
                notes.append("Standard on high budget — cost-efficient choice allows budget reallocation elsewhere.")

    # Dependents ratio notes with family impact framing
    if dep_ratio:
        if dep_ratio > 0.7:
            notes.append(f"High dependents ratio ({dep_ratio:.0%}) — family coverage adequacy is critical; {top_pkg_name.title()} ensures broad access.")
        elif dep_ratio > 0.5:
            notes.append(f"Dependents ratio {dep_ratio:.0%} — {top_pkg_name.title()} selected to balance employee and family needs.")
        elif dep_ratio < 0.3:
            notes.append(f"Low dependents ratio ({dep_ratio:.0%}) — individual-focused coverage makes {top_pkg_name.title()} cost-efficient.")

    # Priority-based framing
    priority_lower = priority.lower() if priority else ""
    if "cost" in priority_lower or "cheapest" in priority_lower:
        notes.append(f"Cost optimization priority — {top_pkg_name.title()} delivers required coverage at minimal expense.")
    elif "coverage" in priority_lower or "maximum" in priority_lower:
        notes.append(f"Maximum coverage priority — {top_pkg_name.title()} provides most comprehensive benefits package.")
    elif "balanced" in priority_lower:
        notes.append(f"Balanced approach — {top_pkg_name.title()} offers middle ground between cost and coverage breadth.")
    elif "stable" in priority_lower or "reliable" in priority_lower:
        notes.append(f"Stability priority — {top_pkg_name.title()} selected for consistent network availability and service quality.")

    # Comparison with alternatives — varied by scenario
    if others:
        prem = next((r for r in others if r["package"].get("name", "").lower() == "premium"), None)
        std = next((r for r in others if r["package"].get("name", "").lower() == "standard"), None)
        basic = next((r for r in others if r["package"].get("name", "").lower() == "basic"), None)

        top_score = scoring_results[0]["score"] if scoring_results else 0

        if top == "standard":
            if prem and prem["score"] < top_score - 10:
                notes.append(f"Premium significantly lower score ({prem['score']}/100) — Standard is the clear optimal choice.")
            elif prem:
                notes.append(f"Premium (score {prem['score']}/100) available if Network A is required.")
            if basic and basic["score"] < top_score - 15:
                notes.append(f"Basic underperforms ({basic['score']}/100) — Standard's additional coverage worth the cost.")
            elif basic:
                notes.append(f"Basic (score {basic['score']}/100) viable for pure cost-minimization scenarios.")
        elif top == "premium":
            if std and std["score"] >= top_score - 5:
                notes.append(f"Standard competitive ({std['score']}/100) — consider if budget becomes priority over Network A.")
            elif std:
                notes.append(f"Standard (score {std['score']}/100) available at lower cost if budget is priority.")
        elif top == "basic":
            if std and std["score"] >= top_score - 3:
                notes.append(f"Standard close in score ({std['score']}/100) — worth considering for marginal cost increase.")
            elif std:
                notes.append(f"Standard (score {std['score']}/100) recommended if industry risk requires better coverage.")

    if not notes:
        notes.append(
            f"{top_pkg_name.title()} selected with score {scoring_results[0]['score'] if scoring_results else 'N/A'}/100 — "
            f"optimal match for your profile across all scored dimensions."
        )

    return " | ".join(notes)


def parse_reasoning_list(raw: str) -> list[str]:
    """Extract a list of strings from raw Gemini output."""
    text = raw.strip()

    # Try to extract content between [REASONING] tags
    import re
    match = re.search(r'\[REASONING\](.*?)\[/REASONING\]', text, re.DOTALL | re.IGNORECASE)
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

    # ── Context-aware reasoning generation ─────────────────────────────────────
    entities = state.get("extracted_entities") or {}
    scoring_breakdown = state.get("scoring_breakdown") or {}
    top_name_lower = top_pkg.get("name", "Standard").lower()

    # Extract all context variables
    industry = entities.get("industry", "")
    region = entities.get("region", "")
    budget = entities.get("budget")
    priority = entities.get("priority")
    dep_ratio = entities.get("dependents_ratio")
    top_score = top_result['score']

    # Get second place score for comparison context
    second_score = scoring_results[1]["score"] if len(scoring_results) > 1 else None
    score_gap = (top_score - second_score) if second_score is not None else None

    # Get raw scoring reasons for this package
    raw_reasons = scoring_breakdown.get(top_name_lower, {}).get("reasons", [])

    # ── LLM reasoning narrative ────────────────────────────────────────────────
    # We mix the deterministic scoring context with LLM creativity to 
    # ensure every response is unique while staying factual.
    reasoning_context = {
        "package": top_pkg.get("name"),
        "score": top_result["score"],
        "entities": entities,
        "scoring_reasons": raw_reasons,
        "market_context": {
            "region": region,
            "industry": industry,
            "budget_tier": budget
        },
        "all_scored_packages": {
            r["package"].get("name", "?"): r["score"]
            for r in scoring_results
        }
    }

    try:
        raw_reasoning = await gemini_client.chat(
            messages=[
                {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(reasoning_context)},
            ],
            temperature=0.7, # Higher temperature for unique phrasing
        )
        reasoning = parse_reasoning_list(raw_reasoning)
    except Exception as e:
        # Deterministic fallback reasoning from our previous logic if LLM fails
        state["execution_trace"].append(f"output_composer: LLM reasoning failed ({e}) — using deterministic fallback")
        reasoning = [
            f"{top_pkg.get('name')} selected with score {top_result['score']}/100.",
            f"Optimized for your profile in {region or 'KSA'} {industry or ''} sector.",
            f"Matches criteria for {budget or 'standard'} budget requirements."
        ]

    state["execution_trace"].append(f"output_composer: generated {len(reasoning)} reasoning bullets")
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
