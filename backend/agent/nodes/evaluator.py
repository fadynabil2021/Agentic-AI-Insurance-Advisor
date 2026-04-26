"""
Evaluator Node — measures quality of the final recommendation.
5 deterministic dimensions + 2 LLM-judge dimensions (Gemini 1.5 Pro).
Logs all scores to Langfuse.
"""
import json
from agent.state import AgentState, ConfidenceLevel
from clients.gemini_client import GeminiClient, safe_json_parse
from clients.langfuse_client import safe_create_span, safe_score

GROUNDING_PROMPT = """You are an evaluation judge for an insurance recommendation system.
Given the scoring data (ground truth) and the generated reasoning (claims),
check if any claim in the reasoning is NOT supported by or CONTRADICTS the scoring data.

Ground truth scoring data: {scoring_breakdown}
Generated reasoning bullets: {reasoning}

Return ONLY valid JSON (no preamble, no markdown):
{{
  "grounding_score": <float 0.0 to 1.0>,
  "hallucination_detected": <true or false>,
  "hallucination_detail": <string or null>
}}"""

CONFIDENCE_ORDER = ["none", "low", "medium", "medium-high", "high"]


def _confidence_rank(level: str | None) -> int:
    return CONFIDENCE_ORDER.index(level) if level in CONFIDENCE_ORDER else 0


def _compute_deterministic_scores(state: AgentState) -> dict:
    """Compute the 5 deterministic eval dimensions."""
    scores = {}

    # 1. task_success — did the agent complete its primary task?
    rec = state.get("recommendation")
    requires_clar = state.get("requires_clarification", False)
    scores["task_success"] = 1.0 if (rec is not None or requires_clar) else 0.0

    # 2. plan_completeness — are all required response fields non-null?
    required_present = [
        state.get("confidence") is not None,
        state.get("execution_trace") and len(state["execution_trace"]) > 0,
        state.get("tools_used") is not None,
    ]
    scores["plan_completeness"] = sum(required_present) / len(required_present)

    # 3. tool_coverage — were all planned tools actually executed?
    plan = state.get("execution_plan") or []
    used = state.get("tools_used") or []
    if plan:
        covered = sum(1 for t in plan if t in used)
        scores["tool_coverage"] = covered / len(plan)
    else:
        scores["tool_coverage"] = 1.0  # no plan = no miss

    # 4. confidence_calibration — is confidence consistent with top score?
    scoring_results = state.get("scoring_results") or []
    confidence = state.get("confidence")
    if scoring_results and confidence:
        top_score = scoring_results[0]["score"]
        expected_conf = (
            "high" if top_score >= 80
            else "medium-high" if top_score >= 60
            else "medium" if top_score >= 40
            else "low"
        )
        actual_rank = _confidence_rank(str(confidence.value if hasattr(confidence, "value") else confidence))
        expected_rank = _confidence_rank(expected_conf)
        scores["confidence_calibration"] = 1.0 if abs(actual_rank - expected_rank) <= 1 else 0.0
    else:
        scores["confidence_calibration"] = 1.0

    # 5. trace_integrity — at least 4 entries in execution_trace
    trace = state.get("execution_trace") or []
    scores["trace_integrity"] = 1.0 if len(trace) >= 4 else len(trace) / 4.0

    return scores


async def evaluator_node(
    state: AgentState,
    gemini_client: GeminiClient,
    langfuse_trace,
) -> AgentState:
    """
    Node 8 (final): Evaluate recommendation quality and log to Langfuse.
    """
    span = safe_create_span(langfuse_trace, "evaluator", {
        "recommendation": state.get("recommendation"),
        "confidence": str(state.get("confidence")),
    })

    eval_scores = _compute_deterministic_scores(state)

    # ── LLM Judge: grounding + hallucination check ────────────────────────────
    reasoning = state.get("reasoning") or []
    scoring_breakdown = state.get("scoring_breakdown") or {}

    if reasoning and scoring_breakdown:
        try:
            prompt = GROUNDING_PROMPT.format(
                scoring_breakdown=json.dumps(scoring_breakdown),
                reasoning=json.dumps(reasoning),
            )
            raw = await gemini_client.chat(
                messages=[
                    {"role": "system", "content": "You are an impartial evaluation judge. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            parsed = safe_json_parse(raw)
            if parsed:
                eval_scores["grounding_score"] = float(parsed.get("grounding_score", 0.8))
                eval_scores["hallucination_detected"] = bool(parsed.get("hallucination_detected", False))
                eval_scores["hallucination_detail"] = parsed.get("hallucination_detail")

                # Auto-downgrade confidence if hallucination detected
                if eval_scores["hallucination_detected"]:
                    conf = state.get("confidence")
                    conf_str = str(conf.value if hasattr(conf, "value") else conf or "none")
                    rank = _confidence_rank(conf_str)
                    if rank > 0:
                        downgraded_str = CONFIDENCE_ORDER[rank - 1]
                        # Map back to ConfidenceLevel enum for type consistency
                        _str_to_enum = {
                            "high": ConfidenceLevel.HIGH,
                            "medium-high": ConfidenceLevel.MEDIUM_HIGH,
                            "medium": ConfidenceLevel.MEDIUM,
                            "low": ConfidenceLevel.LOW,
                            "none": ConfidenceLevel.NONE,
                        }
                        state["confidence"] = _str_to_enum.get(
                            downgraded_str, ConfidenceLevel.LOW
                        )
            else:
                eval_scores["grounding_score"] = 0.8
                eval_scores["hallucination_detected"] = False
        except Exception:
            eval_scores["grounding_score"] = 0.8
            eval_scores["hallucination_detected"] = False
    else:
        eval_scores["grounding_score"] = 1.0  # no reasoning = no hallucination risk
        eval_scores["hallucination_detected"] = False

    state["eval_scores"] = eval_scores

    # ── Log all scores to Langfuse ────────────────────────────────────────────
    trace_id = langfuse_trace.id if langfuse_trace else None
    if trace_id:
        for score_name, value in eval_scores.items():
            if isinstance(value, (int, float)):
                safe_score(trace_id, score_name, float(value))
            elif isinstance(value, bool):
                safe_score(trace_id, score_name, float(value))

    state["execution_trace"].append(
        f"evaluator: task_success={eval_scores.get('task_success')}, "
        f"grounding={eval_scores.get('grounding_score'):.2f}, "
        f"hallucination={eval_scores.get('hallucination_detected')}"
    )

    if span:
        try:
            span.end(output=eval_scores)
        except Exception:
            pass

    return state
