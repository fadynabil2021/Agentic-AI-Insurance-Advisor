"""
Comparison Tool Node — builds a structured comparison matrix when query_type == COMPARE.
Table construction is deterministic; optional Gemma 4 call for recommendation_reason narrative.
"""
from agent.state import AgentState
from agent.tools.scoring_constants import INDUSTRY_RISK
from clients.langfuse_client import safe_create_span
from clients.ollama_client import OllamaClient

DIMENSION_LABELS = ["network", "price_range", "coverage", "score", "budget_fit", "risk_fit"]


def _budget_fit_label(score: int, budget: str, pkg_name: str) -> str:
    """Determine budget fit label from score and budget tier."""
    compatible = {
        "low":    ["basic", "standard"],
        "medium": ["basic", "standard", "premium"],
        "high":   ["standard", "premium"],
    }
    fits = compatible.get(budget or "medium", ["basic", "standard", "premium"])
    if pkg_name.lower() in fits:
        if score >= 70:
            return f"Yes ({budget} budget)"
        return f"Borderline ({budget} budget)"
    return f"No ({budget} budget)"


def _risk_fit_label(industry: str, pkg_name: str) -> str:
    """Assess risk fit for a package given industry."""
    risk = INDUSTRY_RISK.get(industry.lower() if industry else "", "medium")
    if pkg_name.lower() == "basic" and risk == "high":
        return "Not suitable"
    if pkg_name.lower() == "premium" and risk == "medium-low":
        return "Over-spec"
    return "Yes"


async def comparison_tool_node(
    state: AgentState,
    ollama_client: OllamaClient,
    langfuse_trace,
) -> AgentState:
    """
    Node 6 (conditional): Build a comparison matrix between two or more packages.
    Only invoked when query_type == COMPARE.
    """
    span = safe_create_span(langfuse_trace, "comparison_tool", {
        "packages_to_compare": state.get("extracted_entities", {}).get("compare_packages"),
    })

    state["tools_used"].append("comparison_tool")
    entities = state.get("extracted_entities") or {}
    scoring_results = state.get("scoring_results") or []
    budget = entities.get("budget", "medium") or "medium"
    industry = entities.get("industry", "") or ""

    # Build deterministic dimensions table
    packages_in_results = [r["package"] for r in scoring_results]
    score_map = {r["package"].get("name", "").lower(): r["score"] for r in scoring_results}

    if not packages_in_results:
        state["execution_trace"].append("comparison_tool: no packages to compare — skipping")
        if span:
            try:
                span.end(output={"error": "no packages"})
            except Exception:
                pass
        return state

    pkg_names = [p.get("name", "?") for p in packages_in_results]

    dimensions: dict = {}

    # Network
    dimensions["network"] = {p.get("name", "?"): p.get("network", "B") for p in packages_in_results}

    # Price range
    def fmt_price(p: dict) -> str:
        pr = p.get("price_range", [0, 0])
        if isinstance(pr, list) and len(pr) == 2:
            return f"{pr[0]:,}–{pr[1]:,} SAR"
        return str(pr)

    dimensions["price_range"] = {p.get("name", "?"): fmt_price(p) for p in packages_in_results}

    # Coverage
    dimensions["coverage"] = {p.get("name", "?"): p.get("coverage", "Medium") for p in packages_in_results}

    # Score
    dimensions["score"] = {
        p.get("name", "?"): score_map.get(p.get("name", "").lower(), 0)
        for p in packages_in_results
    }

    # Budget fit
    dimensions["budget_fit"] = {
        p.get("name", "?"): _budget_fit_label(
            score_map.get(p.get("name", "").lower(), 0), budget, p.get("name", "")
        )
        for p in packages_in_results
    }

    # Risk fit
    dimensions["risk_fit"] = {
        p.get("name", "?"): _risk_fit_label(industry, p.get("name", ""))
        for p in packages_in_results
    }

    # Top recommendation = highest scoring package
    top = scoring_results[0]["package"].get("name") if scoring_results else pkg_names[0]

    # Generate recommendation reason via Gemma 4
    reason = ""
    try:
        prompt_data = {
            "packages": pkg_names,
            "scores": {n: score_map.get(n.lower(), 0) for n in pkg_names},
            "industry": industry,
            "budget": budget,
            "dimensions": dimensions,
        }
        import json
        raw_reason = await ollama_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an insurance advisor. Given comparison data, write ONE concise sentence "
                        "explaining why the top-scoring package is recommended over the others. "
                        "Reference specific data points (scores, budget, industry risk). "
                        "Return ONLY the sentence, no JSON, no preamble."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt_data)},
            ],
            temperature=0.1,
        )
        reason = raw_reason.strip()
    except Exception:
        # Deterministic fallback reason
        second = pkg_names[1] if len(pkg_names) > 1 else ""
        top_score = score_map.get(top.lower(), 0)
        second_score = score_map.get(second.lower(), 0) if second else 0
        reason = (
            f"{top} scored {top_score}/100 vs {second}'s {second_score}/100. "
            f"Primary difference: budget compatibility and industry risk alignment."
        )

    matrix = {
        "packages": pkg_names,
        "dimensions": dimensions,
        "recommendation": top,
        "recommendation_reason": reason,
    }

    state["comparison_matrix"] = matrix
    state["execution_trace"].append(
        f"comparison_tool: built {len(DIMENSION_LABELS)}-dimension matrix "
        f"for packages={pkg_names}, recommended={top}"
    )

    if span:
        try:
            span.end(output={"packages": pkg_names, "recommended": top})
        except Exception:
            pass

    return state
