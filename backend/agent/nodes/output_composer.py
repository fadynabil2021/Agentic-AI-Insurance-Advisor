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


def _reframe_scoring_reason(reason: str, prefix_type: str | None, entities: dict, pkg_name: str) -> str:
    """
    Reframe a raw scoring reason into user-friendly language with varied framing.
    Different prefix types get different rhetorical treatments.
    """
    # Map technical terms to user-friendly language
    reason_lower = reason.lower()

    # Budget-related reframing
    if "budget" in reason_lower:
        budget = entities.get("budget", "")
        if prefix_type == "PENALTY":
            if budget == "low":
                return f"Cost constraint: {pkg_name} fits within tight budget parameters while maintaining compliance."
            elif budget == "medium":
                return f"Budget optimization: {pkg_name} maximizes coverage value for medium budget allocation."
            else:
                return f"Financial alignment: {pkg_name} matches stated budget tier requirements."
        elif prefix_type == "BONUS":
            return f"Budget efficiency: {pkg_name} delivers optimal cost-to-coverage ratio for {budget or 'this'} budget."

    # Industry-related reframing
    if "industry" in reason_lower:
        industry = entities.get("industry", "")
        if "high-risk" in reason_lower or "not suitable" in reason_lower:
            return f"Industry protection: {pkg_name} meets the coverage demands of {industry} sector risks."
        elif "over-spec" in reason_lower:
            return f"Right-sized coverage: Avoiding over-insurance for {industry} — {pkg_name} matches actual risk profile."
        else:
            return f"Industry alignment: {pkg_name} addresses specific coverage needs of {industry} sector."

    # Region-related reframing
    if "region" in reason_lower or "cost pressure" in reason_lower:
        region = entities.get("region", "")
        if "high" in reason_lower or "riyadh" in reason_lower.lower():
            return f"Regional economics: {pkg_name} is financially viable given {region}'s elevated medical costs."
        else:
            return f"Geographic fit: {pkg_name} aligns with {region}'s healthcare cost structure."

    # Priority-related reframing
    if "priority" in reason_lower or "align" in reason_lower:
        priority = entities.get("priority", "")
        if prefix_type == "PENALTY":
            return f"Goal mismatch: {pkg_name} doesn't fully align with '{priority}' objective."
        else:
            return f"Priority match: {pkg_name} directly supports your '{priority}' goal."

    # Dependents-related reframing
    if "dependents" in reason_lower:
        dep_ratio = entities.get("dependents_ratio", 0)
        if dep_ratio > 0.6:
            return f"Family coverage: {pkg_name} provides adequate protection for {dep_ratio:.0%} dependents ratio."
        else:
            return f"Dependents consideration: {pkg_name} balances employee and family healthcare needs."

    # Default: return the reason as-is but cleaned up
    return reason.replace("not compatible", "not the best fit").replace("insufficient", "lacks required coverage")


def _generate_industry_reasoning(industry: str, pkg_name: str, pkg_display: str) -> list[str]:
    """Generate industry-specific reasoning with varied language."""
    industry_lower = industry.lower()
    reasons = []

    industry_profiles = {
        "healthcare": {
            "basic": "Healthcare sector requires robust network access — clinical staff need reliable coverage.",
            "standard": f"{pkg_display} covers Network B — minimum requirement for healthcare industry compliance.",
            "premium": f"{pkg_display} with Network A provides healthcare workers access to specialist care and diagnostics.",
        },
        "construction": {
            "basic": "Construction work carries injury risk — basic coverage may leave gaps in worker protection.",
            "standard": f"{pkg_display} includes occupational hazard coverage essential for construction industry.",
            "premium": f"{pkg_display} offers comprehensive injury coverage for high-physical-demand construction roles.",
        },
        "retail": {
            "basic": "Retail environments are lower-risk — basic coverage often sufficient for staff needs.",
            "standard": f"{pkg_display} balances cost and coverage for retail sector's medium-low risk profile.",
            "premium": "Premium coverage in retail typically reserved for executive-level benefit packages.",
        },
        "technology": {
            "basic": "Tech sector employees often younger demographics — may prioritize network quality over breadth.",
            "standard": f"{pkg_display} fits tech companies — cost-effective for typically healthy workforce.",
            "premium": f"{pkg_display} serves as talent retention tool — enhanced benefits attract top tech talent.",
        },
        "finance": {
            "basic": "Finance sector client-facing roles typically expect comprehensive benefits.",
            "standard": f"{pkg_display} meets finance industry standards while managing operational costs.",
            "premium": f"{pkg_display} aligns with finance sector expectations for executive and client-facing staff.",
        },
        "education": {
            "basic": f"{pkg_display} works for education sector — lower-risk environment with younger age demographics.",
            "standard": f"{pkg_display} extends coverage for education faculty and administrative personnel.",
            "premium": "Premium in education often reserved for senior administration or specialized faculty.",
        },
    }

    if industry_lower in industry_profiles:
        pkg_key = pkg_name if pkg_name in industry_profiles[industry_lower] else "standard"
        reason = industry_profiles[industry_lower].get(pkg_key)
        if reason:
            reasons.append(reason)

    return reasons


def _generate_region_reasoning(region: str, pkg_name: str, budget: str | None, pkg_display: str) -> list[str]:
    """Generate region-specific reasoning with economic and network context."""
    region_lower = region.lower()
    reasons = []

    region_profiles = {
        "riyadh": {
            "basic": "Riyadh's high medical costs make budget planning critical — verify network adequacy.",
            "standard": f"Riyadh market: {pkg_display} provides viable coverage in Saudi Arabia's highest-cost medical market.",
            "premium": f"Riyadh's premium hospitals accessible with {pkg_display} — justified given capital's cost structure.",
        },
        "jeddah": {
            "basic": f"Jeddah's competitive medical market makes {pkg_display} cost-effective — lower regional rates.",
            "standard": f"{pkg_display} leverages Jeddah's lower healthcare costs — strong value in this market.",
            "premium": f"Jeddah's affordable premium care makes {pkg_display} attractive — comprehensive coverage at moderate cost.",
        },
        "dammam": {
            "basic": f"Dammam's moderate cost environment — {pkg_display} balances affordability with access.",
            "standard": f"{pkg_display} matches Dammam's mid-range medical costs — balanced provider network.",
            "premium": "Premium coverage in Dammam provides Eastern Province specialist access.",
        },
    }

    if region_lower in region_profiles:
        pkg_key = pkg_name if pkg_name in region_profiles[region_lower] else "standard"
        reason = region_profiles[region_lower].get(pkg_key)
        if reason:
            # Add budget context if medium budget in high-cost region
            if region_lower == "riyadh" and budget == "medium" and pkg_name == "premium":
                reason += " Note: Premium on medium budget requires careful cost management in this market."
            reasons.append(reason)

    return reasons


def _generate_budget_reasoning(budget: str, pkg_name: str, pkg_display: str) -> list[str]:
    """Generate budget-specific reasoning with financial framing."""
    budget_lower = budget.lower()
    reasons = []

    budget_frames = {
        "low": {
            "basic": f"Cost minimization: {pkg_display} achieves regulatory compliance at lowest possible cost.",
            "standard": f"Value optimization: {pkg_display} stretches limited budget — best coverage per riyal spent.",
            "premium": f"Budget stretch: {pkg_display} on low budget requires tradeoffs — verify long-term affordability.",
        },
        "medium": {
            "basic": f"Conservative allocation: {pkg_display} leaves budget room for other business priorities.",
            "standard": f"Sweet spot: {pkg_display} on medium budget — optimal balance of cost and coverage breadth.",
            "premium": f"Strategic investment: {pkg_display} attainable on medium budget — enhanced protection without premium pricing.",
        },
        "high": {
            "basic": f"Under-utilization: {pkg_display} on high budget leaves value on the table — consider enhanced coverage.",
            "standard": f"Fiscal discipline: {pkg_display} on high budget — cost-efficient choice allows budget reallocation.",
            "premium": f"Maximum leverage: {pkg_display} utilizes high budget for top-tier networks and specialist access.",
        },
    }

    if budget_lower in budget_frames:
        pkg_key = pkg_name if pkg_name in budget_frames[budget_lower] else "standard"
        reason = budget_frames[budget_lower].get(pkg_key)
        if reason:
            reasons.append(reason)

    return reasons


def _generate_priority_reasoning(priority: str, pkg_name: str, pkg_display: str) -> list[str]:
    """Generate priority-based reasoning that reflects the user's stated goals."""
    priority_lower = priority.lower()
    reasons = []

    # Cost-focused priorities
    if "cost" in priority_lower or "cheapest" in priority_lower or "lowest" in priority_lower:
        if pkg_name == "basic":
            reasons.append(f"Cost-first selection: {pkg_display} achieves absolute minimum cost while meeting requirements.")
        elif pkg_name == "standard":
            reasons.append(f"Cost-conscious choice: {pkg_display} offers best coverage-to-cost ratio for budget-aware buyers.")
        else:
            reasons.append(f"Cost consideration: {pkg_display} selected despite higher cost — other factors drove decision.")

    # Coverage-focused priorities
    elif "coverage" in priority_lower or "maximum" in priority_lower or "comprehensive" in priority_lower:
        if pkg_name == "premium":
            reasons.append(f"Coverage maximized: {pkg_display} delivers the most extensive network and benefits available.")
        elif pkg_name == "standard":
            reasons.append(f"Coverage balance: {pkg_display} provides solid coverage breadth without premium pricing.")
        else:
            reasons.append(f"Coverage tradeoff: {pkg_display} selected — consider if coverage breadth meets your standards.")

    # Balance-focused priorities
    elif "balanced" in priority_lower or "middle" in priority_lower:
        if pkg_name == "standard":
            reasons.append(f"Balanced approach: {pkg_display} is the middle option — neither cost-cutting nor over-specification.")
        else:
            reasons.append(f"Balance consideration: {pkg_display} selected — weighs cost against coverage priorities.")

    # Stability/reliability priorities
    elif "stable" in priority_lower or "reliable" in priority_lower or "consistent" in priority_lower:
        if pkg_name == "standard":
            reasons.append(f"Reliability focus: {pkg_display} offers consistent network availability and service quality.")
        elif pkg_name == "premium":
            reasons.append(f"Maximum reliability: {pkg_display} provides priority access and premium provider relationships.")
        else:
            reasons.append(f"Stability note: {pkg_display} selected — verify network stability meets reliability standards.")

    # Best value priorities
    elif "best" in priority_lower or "value" in priority_lower or "optimal" in priority_lower:
        if pkg_name == "standard":
            reasons.append(f"Value optimization: {pkg_display} delivers best overall value — coverage quality per cost unit.")
        else:
            reasons.append(f"Value consideration: {pkg_display} selected based on value scoring across all dimensions.")

    return reasons


def _generate_dependents_reasoning(dep_ratio: float, pkg_name: str, pkg_display: str) -> list[str]:
    """Generate dependents ratio reasoning with family impact framing."""
    reasons = []

    if dep_ratio > 0.7:
        if pkg_name == "basic":
            reasons.append(f"High dependents alert: {dep_ratio:.0%} dependents ratio — {pkg_display} may lack breadth for family needs.")
        elif pkg_name == "standard":
            reasons.append(f"Family coverage: {dep_ratio:.0%} dependents — {pkg_display} provides adequate network for family healthcare.")
        else:
            reasons.append(f"Comprehensive family protection: {dep_ratio:.0%} dependents — {pkg_display} ensures broad access for all members.")
    elif dep_ratio > 0.4:
        if pkg_name == "basic":
            reasons.append(f"Moderate dependents: {dep_ratio:.0%} ratio — {pkg_display} covers essentials but verify specialist access.")
        elif pkg_name == "standard":
            reasons.append(f"Balanced family coverage: {dep_ratio:.0%} dependents — {pkg_display} suits mixed individual and family needs.")
        else:
            reasons.append(f"Enhanced family benefits: {dep_ratio:.0%} dependents — {pkg_display} provides comprehensive family coverage.")
    else:
        if pkg_name == "basic":
            reasons.append(f"Low dependents: {dep_ratio:.0%} ratio — {pkg_display} cost-efficient for primarily individual coverage.")
        elif pkg_name == "standard":
            reasons.append(f"Individual-focused: {dep_ratio:.0%} dependents — {pkg_display} optimizes for employee-first coverage.")
        else:
            reasons.append(f"Premium individual coverage: {dep_ratio:.0%} dependents — {pkg_display} prioritizes employee benefits quality.")

    return reasons


def _generate_comparative_reasoning(top_score: int, second_score: int | None, score_gap: int | None, pkg_name: str) -> list[str]:
    """Generate reasoning based on score comparison with alternatives."""
    reasons = []

    if second_score is None or score_gap is None:
        return reasons

    # Strong win — clear differentiation
    if score_gap >= 20:
        reasons.append(f"Clear winner: {pkg_name.title()} ({top_score}/100) significantly outperforms alternatives — {second_score}/100 for second place.")
    # Moderate win — solid but not dominant
    elif score_gap >= 10:
        reasons.append(f"Solid choice: {pkg_name.title()} ({top_score}/100) edges out competition ({second_score}/100) — scoring rules favor this profile match.")
    # Close race — acknowledge tradeoffs
    elif score_gap >= 5:
        reasons.append(f"Competitive field: {pkg_name.title()} ({top_score}/100) narrowly leads ({second_score}/100) — alternative packages worth considering.")
    # Photo finish — be transparent about closeness
    else:
        reasons.append(f"Close call: {pkg_name.title()} ({top_score}/100) vs ({second_score}/100) — minimal score difference; decision may come down to specific priorities.")

    return reasons


def _generate_score_reasoning(score: int, pkg_name: str) -> list[str]:
    """Generate score-based reasoning when other contextual reasons are insufficient."""
    reasons = []

    if score >= 85:
        reasons.append(f"Excellent match: Score of {score}/100 indicates {pkg_name.title()} is highly aligned with your profile across all dimensions.")
    elif score >= 70:
        reasons.append(f"Strong match: {score}/100 score reflects {pkg_name.title()} is a solid fit for your stated requirements.")
    elif score >= 55:
        reasons.append(f"Adequate match: {score}/100 suggests {pkg_name.title()} meets core requirements with some tradeoffs.")
    else:
        reasons.append(f"Best available: {score}/100 indicates constraints limit options — {pkg_name.title()} is the strongest available match.")

    return reasons


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

    reasoning = []
    seen_reasons = set()

    # ── Strategy: Build reasoning from multiple sources with varied framing ────
    # Priority 1: Scoring rule reasons (most authoritative)
    # Priority 2: Context-specific inferences (industry, region, budget, etc.)
    # Priority 3: Comparative analysis (score gaps, alternatives)
    # Priority 4: Score-based confidence statements

    # 1. Process scoring rule reasons with contextual reframing
    raw_reasons = scoring_breakdown.get(top_name_lower, {}).get("reasons", [])
    for r in raw_reasons:
        clean_reason = r
        prefix_type = None

        if r.startswith("PENALTY("):
            prefix_type = "PENALTY"
            clean_reason = r.split("): ", 1)[-1] if "): " in r else r
        elif r.startswith("BONUS("):
            prefix_type = "BONUS"
            clean_reason = r.split("): ", 1)[-1] if "): " in r else r
        elif r.startswith("NOTE("):
            prefix_type = "NOTE"
            clean_reason = r.split("): ", 1)[-1] if "): " in r else r

        if clean_reason in seen_reasons:
            continue
        seen_reasons.add(clean_reason)

        # Reframe with varied language based on context
        reframed = _reframe_scoring_reason(clean_reason, prefix_type, entities, top_name_lower)
        if reframed and reframed not in seen_reasons:
            reasoning.append(reframed)
            seen_reasons.add(reframed)

    # 2. Add industry-specific reasoning (varied by industry + plan combo)
    if industry:
        industry_reasons = _generate_industry_reasoning(industry, top_name_lower, top_pkg.get("name"))
        for reason in industry_reasons:
            if reason not in seen_reasons and len(reasoning) < 5:
                reasoning.append(reason)
                seen_reasons.add(reason)

    # 3. Add region-specific reasoning (economic and network factors)
    if region:
        region_reasons = _generate_region_reasoning(region, top_name_lower, budget, top_pkg.get("name"))
        for reason in region_reasons:
            if reason not in seen_reasons and len(reasoning) < 5:
                reasoning.append(reason)
                seen_reasons.add(reason)

    # 4. Add budget-specific reasoning (financial framing)
    if budget:
        budget_reasons = _generate_budget_reasoning(budget, top_name_lower, top_pkg.get("name"))
        for reason in budget_reasons:
            if reason not in seen_reasons and len(reasoning) < 5:
                reasoning.append(reason)
                seen_reasons.add(reason)

    # 5. Add priority-based reasoning (goal alignment)
    if priority:
        priority_reasons = _generate_priority_reasoning(priority, top_name_lower, top_pkg.get("name"))
        for reason in priority_reasons:
            if reason not in seen_reasons and len(reasoning) < 5:
                reasoning.append(reason)
                seen_reasons.add(reason)

    # 6. Add dependents ratio reasoning (family impact)
    if dep_ratio is not None:
        dep_reasons = _generate_dependents_reasoning(dep_ratio, top_name_lower, top_pkg.get("name"))
        for reason in dep_reasons:
            if reason not in seen_reasons and len(reasoning) < 5:
                reasoning.append(reason)
                seen_reasons.add(reason)

    # 7. Add comparative reasoning (score gap analysis)
    if score_gap is not None:
        comparative_reasons = _generate_comparative_reasoning(top_score, second_score, score_gap, top_name_lower)
        for reason in comparative_reasons:
            if reason not in seen_reasons and len(reasoning) < 5:
                reasoning.append(reason)
                seen_reasons.add(reason)

    # 8. Ensure minimum 3 reasoning bullets with score-based statements
    if len(reasoning) < 3:
        score_reasons = _generate_score_reasoning(top_score, top_name_lower)
        for reason in score_reasons:
            if reason not in seen_reasons and len(reasoning) < 3:
                reasoning.append(reason)
                seen_reasons.add(reason)

    reasoning = reasoning[:5]  # Cap at 5 bullets

    # If still empty, provide a default
    if not reasoning:
        reasoning = [
            f"{top_pkg.get('name')} selected based on comprehensive scoring analysis.",
            f"Score: {top_score}/100 reflects strong alignment with your profile.",
            f"Query type: {qt}. Key factors: industry ({industry or 'N/A'}), region ({region or 'N/A'}).",
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
