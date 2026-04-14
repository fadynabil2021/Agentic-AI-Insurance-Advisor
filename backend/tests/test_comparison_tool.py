"""
Unit tests for the deterministic comparison tool.
Tests the matrix construction logic without LLM calls.
Run: pytest tests/test_comparison_tool.py -v
"""
import pytest
from agent.tools.comparison_tool import (
    _budget_fit_label,
    _risk_fit_label,
    DIMENSION_LABELS,
)


# ─── Budget Fit Label ─────────────────────────────────────────────────────────

class TestBudgetFitLabel:
    def test_compatible_high_score(self):
        result = _budget_fit_label(score=85, budget="medium", pkg_name="Standard")
        assert "Yes" in result
        assert "medium" in result

    def test_compatible_borderline_score(self):
        result = _budget_fit_label(score=55, budget="medium", pkg_name="Standard")
        assert "Borderline" in result

    def test_incompatible_premium_on_low_budget(self):
        result = _budget_fit_label(score=40, budget="low", pkg_name="Premium")
        assert "No" in result

    def test_incompatible_basic_on_high_budget(self):
        result = _budget_fit_label(score=80, budget="high", pkg_name="Basic")
        assert "No" in result

    def test_unknown_budget_defaults_to_compatible(self):
        result = _budget_fit_label(score=80, budget="unknown", pkg_name="Standard")
        assert "Yes" in result

    def test_none_budget_defaults_to_medium(self):
        result = _budget_fit_label(score=80, budget=None, pkg_name="Standard")
        assert "Yes" in result or "medium" in result.lower()


# ─── Risk Fit Label ──────────────────────────────────────────────────────────

class TestRiskFitLabel:
    def test_basic_not_suitable_for_healthcare(self):
        result = _risk_fit_label(industry="healthcare", pkg_name="Basic")
        assert result == "Not suitable"

    def test_premium_over_spec_for_retail(self):
        result = _risk_fit_label(industry="retail", pkg_name="Premium")
        assert result == "Over-spec"

    def test_standard_always_suitable(self):
        for industry in ["healthcare", "construction", "retail", "technology"]:
            result = _risk_fit_label(industry=industry, pkg_name="Standard")
            assert result == "Yes", f"Standard should be suitable for {industry}"

    def test_premium_suitable_for_healthcare(self):
        result = _risk_fit_label(industry="healthcare", pkg_name="Premium")
        assert result == "Yes"

    def test_basic_suitable_for_retail(self):
        result = _risk_fit_label(industry="retail", pkg_name="Basic")
        assert result == "Yes"

    def test_empty_industry_defaults_to_medium(self):
        result = _risk_fit_label(industry="", pkg_name="Basic")
        assert result == "Yes"

    def test_none_industry_does_not_crash(self):
        result = _risk_fit_label(industry=None, pkg_name="Standard")
        assert result == "Yes"


# ─── Dimension Labels ────────────────────────────────────────────────────────

class TestDimensionLabels:
    def test_all_expected_dimensions_present(self):
        expected = {"network", "price_range", "coverage", "score", "budget_fit", "risk_fit"}
        assert set(DIMENSION_LABELS) == expected

    def test_dimension_count(self):
        assert len(DIMENSION_LABELS) == 6
