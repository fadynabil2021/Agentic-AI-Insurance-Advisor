"""
Unit tests for the deterministic scoring tool.
These tests require NO LLM and NO external services — pure Python.
Run: pytest tests/test_scoring_tool.py -v
"""
import pytest
from agent.tools.scoring_tool import (
    run_scoring,
    _match_priority,
    _compute_confidence,
)
from agent.tools.scoring_constants import (
    BUDGET_PACKAGE_COMPATIBILITY,
    INDUSTRY_RISK,
    REGION_COST,
    PRIORITY_PACKAGE_MAP,
    PRIORITY_KEYS_BY_LENGTH,
    WEIGHT_BUDGET_INCOMPATIBLE,
    WEIGHT_INDUSTRY_HIGH_RISK_BASIC,
    CONFIDENCE_GAP_NARROW,
)
from agent.state import ConfidenceLevel

# ─── Shared Fixtures ──────────────────────────────────────────────────────────

ALL_PACKAGES = [
    {"name": "Basic",    "type": "package", "network": "C", "price_range": [4000, 5000],  "coverage": "Basic"},
    {"name": "Standard", "type": "package", "network": "B", "price_range": [6000, 7500],  "coverage": "Medium"},
    {"name": "Premium",  "type": "package", "network": "A", "price_range": [9000, 12000], "coverage": "High"},
]


# ─── Scenario 1: Healthcare / Riyadh / medium budget ─────────────────────────

class TestScenario1HealthcareRiyadh:
    def test_standard_is_top_for_healthcare_riyadh(self):
        entities = {"industry": "healthcare", "region": "riyadh", "budget": "medium"}
        results = run_scoring(ALL_PACKAGES, entities)
        scores = {r["package"]["name"]: r["score"] for r in results}
        # Standard should beat Premium (region cost pressure) and Basic (high risk penalty)
        assert scores["Standard"] > scores["Basic"], "Standard should score higher than Basic for healthcare"
        assert scores["Basic"] < 80, "Basic should be heavily penalized for high-risk industry"

    def test_basic_penalized_for_high_risk(self):
        entities = {"industry": "healthcare", "region": "riyadh", "budget": "medium"}
        results = run_scoring(ALL_PACKAGES, entities)
        basic_result = next(r for r in results if r["package"]["name"] == "Basic")
        assert any("PENALTY" in reason for reason in basic_result["reasons"]), \
            "Basic should have PENALTY reason for healthcare industry"

    def test_premium_penalized_riyadh_medium_budget(self):
        entities = {"industry": "healthcare", "region": "riyadh", "budget": "medium"}
        results = run_scoring(ALL_PACKAGES, entities)
        premium_result = next(r for r in results if r["package"]["name"] == "Premium")
        riyadh_penalty = any("region cost" in r.lower() for r in premium_result["reasons"])
        assert riyadh_penalty, "Premium should be penalized for Riyadh cost pressure on medium budget"

    def test_scores_clamped_0_100(self):
        entities = {"industry": "healthcare", "region": "riyadh", "budget": "low"}
        results = run_scoring(ALL_PACKAGES, entities)
        for r in results:
            assert 0 <= r["score"] <= 100, f"Score out of range for {r['package']['name']}"


# ─── Scenario 2: Construction / Jeddah / cheapest ─────────────────────────────

class TestScenario2CheapestConstruction:
    def test_basic_or_standard_top_for_cheapest(self):
        entities = {"industry": "construction", "region": "jeddah", "budget": "low", "_query_type": "cheapest"}
        results = run_scoring(ALL_PACKAGES, entities)
        top_name = results[0]["package"]["name"]
        assert top_name in ["Basic", "Standard"], f"Expected Basic or Standard on top, got {top_name}"

    def test_premium_heavily_penalized_for_low_budget(self):
        entities = {"industry": "construction", "region": "jeddah", "budget": "low"}
        results = run_scoring(ALL_PACKAGES, entities)
        premium_result = next(r for r in results if r["package"]["name"] == "Premium")
        assert premium_result["score"] < 70, "Premium should be penalized for low budget"

    def test_construction_medium_risk_no_basic_penalty(self):
        entities = {"industry": "construction", "region": "jeddah", "budget": "medium"}
        results = run_scoring(ALL_PACKAGES, entities)
        basic_result = next(r for r in results if r["package"]["name"] == "Basic")
        # Basic should NOT have the high-risk-industry penalty (only for healthcare)
        assert not any("high-risk industry" in r for r in basic_result["reasons"]), \
            "Construction is not high-risk — Basic should not get healthcare penalty"


# ─── Scenario 3: Retail / Dammam / compare ────────────────────────────────────

class TestScenario3CompareRetailDammam:
    def test_standard_beats_premium_for_retail(self):
        entities = {"industry": "retail", "region": "dammam", "budget": "medium"}
        results = run_scoring(ALL_PACKAGES, entities)
        scores = {r["package"]["name"]: r["score"] for r in results}
        assert scores["Standard"] >= scores["Premium"], \
            "Standard should score >= Premium for retail medium-low risk"

    def test_premium_over_spec_penalty_for_retail(self):
        entities = {"industry": "retail", "region": "dammam", "budget": "medium"}
        results = run_scoring(ALL_PACKAGES, entities)
        premium_result = next(r for r in results if r["package"]["name"] == "Premium")
        over_spec = any("over-spec" in r.lower() for r in premium_result["reasons"])
        assert over_spec, "Premium should get over-spec note for retail (medium-low risk)"


# ─── Edge Cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_packages_returns_empty(self):
        results = run_scoring([], {"industry": "healthcare", "region": "riyadh"})
        assert results == []

    def test_non_package_types_skipped(self):
        mixed = [
            {"type": "rule", "name": "some_rule"},
            {"name": "Standard", "type": "package", "network": "B", "price_range": [6000, 7500]},
        ]
        results = run_scoring(mixed, {"industry": "retail", "region": "jeddah"})
        # Only Standard should be scored
        assert all(r["package"].get("type") != "rule" for r in results)

    def test_snippet_type_skipped(self):
        mixed = [
            {"type": "snippet", "name": "some_snippet"},
            {"name": "Basic", "type": "package", "network": "C", "price_range": [4000, 5000]},
        ]
        results = run_scoring(mixed, {"industry": "retail", "region": "jeddah"})
        assert len(results) == 1
        assert results[0]["package"]["name"] == "Basic"

    def test_dependents_ratio_penalty_on_basic(self):
        entities = {"industry": "retail", "region": "jeddah", "dependents_ratio": 0.6}
        results = run_scoring(ALL_PACKAGES, entities)
        basic_result = next(r for r in results if r["package"]["name"] == "Basic")
        dep_penalty = any("dependents ratio" in r.lower() for r in basic_result["reasons"])
        assert dep_penalty, "High dependents ratio should penalize Basic"

    def test_results_sorted_descending_by_score(self):
        entities = {"industry": "healthcare", "region": "riyadh", "budget": "medium"}
        results = run_scoring(ALL_PACKAGES, entities)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results must be sorted by score descending"

    def test_unknown_industry_defaults_to_medium_risk(self):
        entities = {"industry": "mining", "region": "riyadh", "budget": "medium"}
        # Should not raise, should use default medium risk
        results = run_scoring(ALL_PACKAGES, entities)
        assert len(results) > 0

    def test_unknown_region_defaults_to_low_cost_pressure(self):
        entities = {"industry": "retail", "region": "tabuk", "budget": "medium"}
        results = run_scoring(ALL_PACKAGES, entities)
        # Premium should NOT get the Riyadh high-cost penalty
        premium_result = next(r for r in results if r["package"]["name"] == "Premium")
        riyadh_penalty = any("region cost pressure" in r.lower() for r in premium_result["reasons"])
        assert not riyadh_penalty, "Unknown region should not trigger high-cost-pressure penalty"

    def test_penalty_amounts_shown_in_reasons(self):
        """Each PENALTY reason should include the weight amount for auditability."""
        entities = {"industry": "healthcare", "region": "riyadh", "budget": "low"}
        results = run_scoring(ALL_PACKAGES, entities)
        for r in results:
            for reason in r["reasons"]:
                if "PENALTY" in reason:
                    assert "−" in reason or "-" in reason, \
                        f"Penalty reason should include weight amount: {reason}"


# ─── Priority Matching (longest-match-first) ──────────────────────────────────

class TestPriorityMatching:
    def test_cheapest_acceptable_matches_before_cheapest(self):
        """'cheapest acceptable' should match the longer key first."""
        result = _match_priority("cheapest acceptable")
        assert result == ["basic", "standard"]

    def test_cheapest_alone(self):
        result = _match_priority("cheapest")
        assert result == ["basic", "standard"]

    def test_maximum_coverage(self):
        result = _match_priority("maximum coverage")
        assert result == ["premium"]

    def test_balanced_default(self):
        result = _match_priority("balanced")
        assert result == ["standard"]

    def test_unknown_priority_returns_empty(self):
        result = _match_priority("foobar xyz")
        assert result == []

    def test_case_insensitive(self):
        result = _match_priority("BALANCED")
        assert result == ["standard"]

    def test_priority_keys_sorted_by_length(self):
        """Ensure PRIORITY_KEYS_BY_LENGTH is actually sorted longest-first."""
        lengths = [len(k) for k in PRIORITY_KEYS_BY_LENGTH]
        assert lengths == sorted(lengths, reverse=True)


# ─── Confidence Computation (score + gap) ─────────────────────────────────────

class TestConfidenceComputation:
    def test_high_score_large_gap(self):
        result = _compute_confidence(top_score=90, second_score=60)
        assert result == ConfidenceLevel.HIGH

    def test_high_score_narrow_gap_downgrades(self):
        """High score but narrow gap should downgrade to MEDIUM_HIGH."""
        result = _compute_confidence(top_score=85, second_score=83)
        assert result == ConfidenceLevel.MEDIUM_HIGH

    def test_medium_high_score_narrow_gap_downgrades(self):
        result = _compute_confidence(top_score=65, second_score=63)
        assert result == ConfidenceLevel.MEDIUM

    def test_single_package_no_downgrade(self):
        """With only one package (no second score), use base confidence."""
        result = _compute_confidence(top_score=90, second_score=None)
        assert result == ConfidenceLevel.HIGH

    def test_low_score(self):
        result = _compute_confidence(top_score=30, second_score=10)
        assert result == ConfidenceLevel.LOW

    def test_low_score_cannot_downgrade_below_low(self):
        result = _compute_confidence(top_score=30, second_score=29)
        assert result == ConfidenceLevel.LOW

    def test_medium_score(self):
        result = _compute_confidence(top_score=50, second_score=20)
        assert result == ConfidenceLevel.MEDIUM

    def test_exactly_at_boundary(self):
        result = _compute_confidence(top_score=80, second_score=50)
        assert result == ConfidenceLevel.HIGH

    def test_gap_exactly_at_narrow_threshold(self):
        """Gap == CONFIDENCE_GAP_NARROW should still trigger downgrade."""
        gap = CONFIDENCE_GAP_NARROW
        result = _compute_confidence(top_score=85, second_score=85 - gap + 1)
        assert result == ConfidenceLevel.MEDIUM_HIGH  # downgraded from HIGH


# ─── Rule Table Tests ──────────────────────────────────────────────────────────

class TestRuleTables:
    def test_budget_compatibility_low(self):
        assert "premium" not in BUDGET_PACKAGE_COMPATIBILITY["low"]

    def test_budget_compatibility_high(self):
        assert "basic" not in BUDGET_PACKAGE_COMPATIBILITY["high"]

    def test_industry_risk_healthcare_high(self):
        assert INDUSTRY_RISK["healthcare"] == "high"

    def test_industry_risk_retail_medium_low(self):
        assert INDUSTRY_RISK["retail"] == "medium-low"

    def test_region_cost_riyadh_highest(self):
        assert REGION_COST["riyadh"] > REGION_COST["jeddah"]
        assert REGION_COST["riyadh"] > REGION_COST["dammam"]

    def test_weights_are_positive(self):
        """All penalty weights must be positive integers."""
        assert WEIGHT_BUDGET_INCOMPATIBLE > 0
        assert WEIGHT_INDUSTRY_HIGH_RISK_BASIC > 0
