"""
Shared scoring constants — single source of truth for all scoring rules.
Used by scoring_tool.py and comparison_tool.py.

All business rules from PRD Section 7.5 are centralized here.
Weights are configurable and documented.
"""

# ─── Industry Risk Classification ─────────────────────────────────────────────
# Maps industry → risk level. Used for penalty/bonus calculations.

INDUSTRY_RISK: dict[str, str] = {
    "healthcare":    "high",
    "construction":  "medium",
    "retail":        "medium-low",
    "manufacturing": "medium",
    "technology":    "medium-low",
    "finance":       "medium",
    "education":     "low",
}

# ─── Region Cost Pressure ─────────────────────────────────────────────────────
# Maps region → cost pressure score (1=lowest, 3=highest).

REGION_COST: dict[str, int] = {
    "riyadh":  3,   # highest cost pressure
    "dammam":  2,
    "jeddah":  1,   # lowest
}

# ─── Budget ↔ Package Compatibility ───────────────────────────────────────────
# Defines which packages are compatible with each budget tier.

BUDGET_PACKAGE_COMPATIBILITY: dict[str, list[str]] = {
    "low":    ["basic", "standard"],
    "medium": ["basic", "standard", "premium"],
    "high":   ["standard", "premium"],
}

# ─── Priority → Preferred Packages ────────────────────────────────────────────
# Ordered from most specific (longest key) to least specific for safe matching.
# The scorer uses longest-match-first to avoid ambiguity.

PRIORITY_PACKAGE_MAP: dict[str, list[str]] = {
    "cheapest acceptable": ["basic", "standard"],
    "maximum coverage":    ["premium"],
    "best coverage":       ["premium", "standard"],
    "stable service":      ["standard", "premium"],
    "cost effective":      ["basic", "standard"],
    "lowest cost":         ["basic", "standard"],
    "cheapest":            ["basic", "standard"],
    "balanced":            ["standard"],
}

# Precomputed: sorted keys from longest to shortest for safe substring matching
PRIORITY_KEYS_BY_LENGTH: list[str] = sorted(
    PRIORITY_PACKAGE_MAP.keys(), key=len, reverse=True
)

# ─── Scoring Weights ──────────────────────────────────────────────────────────
# Each penalty/bonus is documented with its rationale and weight.
# Weights sum to context — they are deducted from a base score of 100.
#
# Weight tiers:
#   SEVERE  (35-40):  Hard constraint violation — this package is unsuitable.
#   HIGH    (20-25):  Important mismatch — significant fitness reduction.
#   MEDIUM  (10-15):  Soft constraint — notable but not disqualifying.
#   LOW     (5-10):   Informational — minor tradeoff.
#   BONUS   (5-15):   Positive alignment — package matches stated priority.

WEIGHT_BUDGET_INCOMPATIBLE:      int = 35   # package outside budget tier
WEIGHT_INDUSTRY_HIGH_RISK_BASIC: int = 25   # Basic plan for high-risk industry
WEIGHT_REGION_COST_PRESSURE:     int = 15   # Premium in high-cost region on medium budget
WEIGHT_PRIORITY_MISALIGN:        int = 20   # package doesn't match user's stated priority
WEIGHT_DEPENDENTS_BASIC:         int = 15   # high dependents ratio on Basic plan
WEIGHT_INDUSTRY_OVER_SPEC:       int = 10   # Premium for medium-low risk (over-specification)
BONUS_PRIORITY_MATCH:            int = 10   # package matches user's stated priority

# ─── Confidence Thresholds ────────────────────────────────────────────────────
# Confidence is determined by BOTH the top score AND the gap to the second score.
# A narrow gap means lower confidence even if the top score is high.

CONFIDENCE_SCORE_HIGH:       int = 80
CONFIDENCE_SCORE_MEDIUM_HIGH: int = 60
CONFIDENCE_SCORE_MEDIUM:     int = 40
CONFIDENCE_GAP_STRONG:       int = 15   # gap >= 15 → no confidence downgrade
CONFIDENCE_GAP_NARROW:       int = 5    # gap < 5 → downgrade by one tier
