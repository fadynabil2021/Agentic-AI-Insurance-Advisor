"""
Scoring Tool Node — deterministic rules engine (NO LLM).
Applies business rules from PRD Section 7.5 to score each retrieved package.
This is the most critical tool: 100% auditable Python, no non-determinism.

Design decisions (team-lead level):
  - All constants imported from scoring_constants.py (single source of truth).
  - Weighted penalties with documented rationale, not flat magic numbers.
  - Confidence computed from BOTH top score AND score gap to #2.
  - Longest-match-first priority matching to avoid substring ambiguity.
  - No dead code, no redundant dict construction.
"""
from agent.state import AgentState, ConfidenceLevel
from agent.tools.scoring_constants import (
    INDUSTRY_RISK,
    REGION_COST,
    BUDGET_PACKAGE_COMPATIBILITY,
    PRIORITY_PACKAGE_MAP,
    PRIORITY_KEYS_BY_LENGTH,
    WEIGHT_BUDGET_INCOMPATIBLE,
    WEIGHT_INDUSTRY_HIGH_RISK_BASIC,
    WEIGHT_REGION_COST_PRESSURE,
    WEIGHT_PRIORITY_MISALIGN,
    WEIGHT_DEPENDENTS_BASIC,
    WEIGHT_INDUSTRY_OVER_SPEC,
    BONUS_PRIORITY_MATCH,
    CONFIDENCE_SCORE_HIGH,
    CONFIDENCE_SCORE_MEDIUM_HIGH,
    CONFIDENCE_SCORE_MEDIUM,
    CONFIDENCE_GAP_STRONG,
    CONFIDENCE_GAP_NARROW,
)
from clients.langfuse_client import safe_create_span


def _match_priority(priority: str) -> list[str]:
    """
    Match user priority string to preferred packages using longest-match-first.

    This avoids the fragile dict-iteration approach where "cheapest" could match
    before "cheapest acceptable" depending on insertion order.
    """
    priority_lower = priority.lower()
    for key in PRIORITY_KEYS_BY_LENGTH:
        if key in priority_lower:
            return PRIORITY_PACKAGE_MAP[key]
    return []


def _compute_confidence(top_score: int, second_score: int | None) -> ConfidenceLevel:
    """
    Compute confidence from both the top score AND the gap to second place.

    Rationale: A top score of 85 with second at 84 is less confident than
    a top score of 75 with second at 50. The gap measures how clearly the
    scoring rules differentiate the recommended package.
    """
    # Step 1: Base confidence from absolute score
    if top_score >= CONFIDENCE_SCORE_HIGH:
        base = ConfidenceLevel.HIGH
    elif top_score >= CONFIDENCE_SCORE_MEDIUM_HIGH:
        base = ConfidenceLevel.MEDIUM_HIGH
    elif top_score >= CONFIDENCE_SCORE_MEDIUM:
        base = ConfidenceLevel.MEDIUM
    else:
        base = ConfidenceLevel.LOW

    # Step 2: Adjust for score gap (only when we have 2+ packages)
    if second_score is not None:
        gap = top_score - second_score
        if gap < CONFIDENCE_GAP_NARROW:
            # Narrow gap — downgrade confidence by one tier
            downgrade_map = {
                ConfidenceLevel.HIGH: ConfidenceLevel.MEDIUM_HIGH,
                ConfidenceLevel.MEDIUM_HIGH: ConfidenceLevel.MEDIUM,
                ConfidenceLevel.MEDIUM: ConfidenceLevel.LOW,
                ConfidenceLevel.LOW: ConfidenceLevel.LOW,
            }
            return downgrade_map[base]

    return base


def run_scoring(packages: list[dict], entities: dict) -> list[dict]:
    """
    Core scoring logic — callable directly (used by MCP server and node).

    Returns sorted list of {"package": dict, "score": int, "reasons": list[str]}.
    Each package starts at 100 and penalties/bonuses are applied per rule.
    """
    results: list[dict] = []

    for pkg in packages:
        # Skip non-package documents (rules, snippets)
        pkg_type = pkg.get("type")
        if pkg_type is not None and pkg_type != "package":
            continue

        score = 100
        pkg_name = (pkg.get("name") or "").lower()
        reasons: list[str] = []

        # ── Rule 1: Budget compatibility ────────────────────────────────────
        budget = entities.get("budget")
        # Only apply budget penalty if budget was explicitly provided or strongly inferred
        if budget:
            budget = budget.lower()
            compatible = BUDGET_PACKAGE_COMPATIBILITY.get(
                budget, ["basic", "standard", "premium"]
            )
            if pkg_name not in compatible:
                score -= WEIGHT_BUDGET_INCOMPATIBLE
                reasons.append(
                    f"PENALTY(−{WEIGHT_BUDGET_INCOMPATIBLE}): "
                    f"{pkg_name} not compatible with {budget} budget"
                )

        # ── Rule 2: Industry risk ────────────────────────────────────────────
        industry = (entities.get("industry") or "").lower()
        risk = INDUSTRY_RISK.get(industry, "medium")
        if risk == "high" and pkg_name == "basic":
            score -= WEIGHT_INDUSTRY_HIGH_RISK_BASIC
            reasons.append(
                f"PENALTY(−{WEIGHT_INDUSTRY_HIGH_RISK_BASIC}): "
                f"Basic not suitable for high-risk industry ({industry})"
            )
        if risk == "medium-low" and pkg_name == "premium":
            score -= WEIGHT_INDUSTRY_OVER_SPEC
            reasons.append(
                f"NOTE(−{WEIGHT_INDUSTRY_OVER_SPEC}): "
                f"Premium may be over-spec for medium-low risk industry ({industry})"
            )

        # ── Rule 3: Region cost pressure ─────────────────────────────────────
        region = (entities.get("region") or "").lower()
        cost_pressure = REGION_COST.get(region, 1)
        if cost_pressure >= 3 and pkg_name == "premium" and budget == "medium":
            score -= WEIGHT_REGION_COST_PRESSURE
            reasons.append(
                f"PENALTY(−{WEIGHT_REGION_COST_PRESSURE}): "
                f"High region cost pressure ({region}) limits Premium viability "
                f"on medium budget"
            )

        # ── Rule 4: Priority alignment ───────────────────────────────────────
        priority = entities.get("priority")
        # Only apply priority penalty/bonus if priority was explicitly provided
        if priority:
            priority = priority.lower()
            preferred = _match_priority(priority)

            # Also handle cheapest query type when no explicit priority matched
            qt = entities.get("_query_type", "")
            if qt == "cheapest" and not preferred:
                preferred = ["basic", "standard"]

            if preferred and pkg_name not in preferred:
                score -= WEIGHT_PRIORITY_MISALIGN
                reasons.append(
                    f"PENALTY(−{WEIGHT_PRIORITY_MISALIGN}): "
                    f"{pkg_name} does not align with priority '{priority}'"
                )
            elif preferred and pkg_name in preferred:
                score += BONUS_PRIORITY_MATCH
                reasons.append(
                    f"BONUS(+{BONUS_PRIORITY_MATCH}): "
                    f"{pkg_name} matches priority '{priority}'"
                )

        # ── Rule 5: Dependents ratio ─────────────────────────────────────────
        dep_ratio = entities.get("dependents_ratio")
        if dep_ratio and dep_ratio > 0.5 and pkg_name == "basic":
            score -= WEIGHT_DEPENDENTS_BASIC
            reasons.append(
                f"PENALTY(−{WEIGHT_DEPENDENTS_BASIC}): "
                f"High dependents ratio ({dep_ratio:.0%}) increases benefits cost; "
                f"Basic insufficient"
            )

        score = max(0, min(100, score))  # clamp 0–100
        results.append({"package": pkg, "score": score, "reasons": reasons})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def scoring_tool_node(state: AgentState, langfuse_trace) -> AgentState:
    """
    Node 5: Apply deterministic scoring rules to all retrieved packages.
    Sets scoring_results, scoring_breakdown, and confidence.
    """
    span = safe_create_span(langfuse_trace, "scoring_tool", {
        "entities": state.get("extracted_entities"),
        "packages_count": len([
            r for r in (state.get("retrieval_results") or [])
            if r.get("type") == "package"
        ]),
    })

    state["tools_used"].append("scoring_tool")
    entities = dict(state.get("extracted_entities") or {})

    # Inject query_type into entities for priority detection
    if state.get("query_type"):
        entities["_query_type"] = state["query_type"]

    retrieval_results = state.get("retrieval_results") or []
    packages = [
        r for r in retrieval_results
        if r.get("type") in ("package", None)
    ]

    results = run_scoring(packages, entities)

    # Build breakdown from results (single computation, no duplication)
    breakdown = {
        r["package"].get("name", "").lower(): {
            "score": r["score"],
            "reasons": r["reasons"],
        }
        for r in results
    }

    state["scoring_results"] = results
    state["scoring_breakdown"] = breakdown

    if results:
        top = results[0]
        top_name = top["package"].get("name", "unknown")
        top_score = top["score"]
        second_score = results[1]["score"] if len(results) > 1 else None

        state["execution_trace"].append(
            f"scoring_tool: top package='{top_name}' score={top_score}, "
            f"gap_to_second={top_score - second_score if second_score is not None else 'N/A'}, "
            f"all_scores={[(r['package'].get('name', '?'), r['score']) for r in results]}"
        )

        # Confidence based on both absolute score and gap
        state["confidence"] = _compute_confidence(top_score, second_score)
    else:
        state["execution_trace"].append("scoring_tool: WARNING — no packages scored")
        state["confidence"] = ConfidenceLevel.NONE
        state["error"] = {
            "type": "EMPTY_RETRIEVAL",
            "message": "No packages found matching criteria",
        }

    if span:
        try:
            span.end(output={
                "top_package": results[0]["package"].get("name") if results else None,
                "top_score": results[0]["score"] if results else 0,
                "confidence": state["confidence"],
            })
        except Exception:
            pass

    return state
