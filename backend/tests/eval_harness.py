"""
Evaluation harness — runs all 8 predefined scenarios against the compiled LangGraph.
Logs each run to Langfuse and writes a JSON report.
"""
import asyncio
import json
import datetime
import sys
import os
import uuid
from typing import Any
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.graph import make_initial_state

EVAL_SCENARIOS = [
    {
        "id": "S1_balanced",
        "user_request": "Recommend the best plan for a healthcare company in Riyadh.",
        "expected": {
            "query_type": "recommend",
            "recommendation_in": ["Standard", "Premium"],
            "must_mention_industry": "healthcare",
            "confidence_min": "medium",
            "task_success": 1,
        },
    },
    {
        "id": "S2_cheapest",
        "user_request": "Give me the cheapest acceptable option for a construction customer in Jeddah.",
        "expected": {
            "query_type": "cheapest",
            "recommendation_in": ["Basic", "Standard"],
            "confidence_min": "medium",
            "task_success": 1,
        },
    },
    {
        "id": "S3_compare",
        "user_request": "Compare Standard and Premium for a retail customer in Dammam.",
        "expected": {
            "query_type": "compare",
            "comparison_matrix_present": True,
            "task_success": 1,
        },
    },
    {
        "id": "S4_missing_info",
        "user_request": "Recommend a plan for my company.",
        "expected": {
            "requires_clarification": True,
            "recommendation_null": True,
            "task_success": 1,
        },
    },
    {
        "id": "S5_explain",
        "user_request": "Why did you choose that recommendation?",
        "expected": {
            "reasoning_present": True,
            "execution_trace_length_min": 3,
            "task_success": 1,
        },
    },
    {
        "id": "E1_conflicting_constraints",
        "user_request": "I want the best possible coverage but my budget is very low for a construction company in Riyadh.",
        "expected": {
            "recommendation_in": ["Basic", "Standard"],
            "fallback_note_present": True,
            "confidence_max": "medium",
        },
    },
    {
        "id": "E2_unsupported",
        "user_request": "What is the weather in Riyadh today?",
        "expected": {
            "query_type": "unsupported",
            "recommendation_null": True,
            "fallback_note_present": True,
        },
    },
    {
        "id": "E3_empty_retrieval",
        "user_request": "Recommend a plan for a mining company in Tabuk.",
        "expected": {
            "fallback_note_present": True,
        },
    },
]

CONFIDENCE_ORDER = ["none", "low", "medium", "medium-high", "high"]


def _conf_rank(level: Any) -> int:
    # Handle Enum objects or their string representations
    if hasattr(level, "value"):
        s = str(level.value).lower()
    else:
        s = str(level).lower() if level else "none"
        
    # Handle "ConfidenceLevel.HIGH" format
    if "." in s:
        s = s.split(".")[-1]
        
    return CONFIDENCE_ORDER.index(s) if s in CONFIDENCE_ORDER else 0


def _check_scenario(result: dict, expected: dict) -> tuple[bool, list[str]]:
    """Returns (passed, list_of_failures)."""
    failures = []

    # recommendation_in
    if "recommendation_in" in expected:
        rec = result.get("recommendation")
        plan_name = rec.get("plan_name") if rec else None
        if plan_name not in expected["recommendation_in"]:
            failures.append(
                f"Expected plan in {expected['recommendation_in']}, got '{plan_name}'"
            )

    # recommendation_null
    if expected.get("recommendation_null"):
        if result.get("recommendation") is not None:
            failures.append("Expected recommendation=null")

    # requires_clarification
    if "requires_clarification" in expected:
        if result.get("requires_clarification") != expected["requires_clarification"]:
            failures.append(
                f"Expected requires_clarification={expected['requires_clarification']}, "
                f"got {result.get('requires_clarification')}"
            )

    # comparison_matrix_present
    if expected.get("comparison_matrix_present"):
        if not result.get("comparison_matrix"):
            failures.append("Expected comparison_matrix to be present")

    # fallback_note_present
    if expected.get("fallback_note_present"):
        if not result.get("fallback_or_risk_note"):
            failures.append("Expected fallback_or_risk_note to be present")

    # task_success
    if "task_success" in expected:
        eval_scores = result.get("eval_scores") or {}
        ts = eval_scores.get("task_success", 0)
        if ts < expected["task_success"]:
            failures.append(f"Expected task_success>={expected['task_success']}, got {ts}")

    # confidence_min
    if "confidence_min" in expected:
        conf = result.get("confidence", "none")
        if _conf_rank(conf) < _conf_rank(expected["confidence_min"]):
            failures.append(
                f"Expected confidence >= {expected['confidence_min']}, got {conf}"
            )

    # confidence_max
    if "confidence_max" in expected:
        conf = result.get("confidence", "none")
        if _conf_rank(conf) > _conf_rank(expected["confidence_max"]):
            failures.append(
                f"Expected confidence <= {expected['confidence_max']}, got {conf}"
            )

    # reasoning_present
    if expected.get("reasoning_present"):
        reasoning = result.get("reasoning") or []
        if len(reasoning) == 0:
            failures.append("Expected reasoning to be present")

    # execution_trace_length_min
    if "execution_trace_length_min" in expected:
        trace = result.get("execution_trace") or []
        if len(trace) < expected["execution_trace_length_min"]:
            failures.append(
                f"Expected execution_trace >= {expected['execution_trace_length_min']} entries, "
                f"got {len(trace)}"
            )

    return len(failures) == 0, failures


async def run_harness(
    compiled_graph,
    scenario_ids: Any = "all",
    report_dir: str = "/app/docs",
) -> tuple[list[dict], str]:
    """Run evaluation scenarios and return (results, report_path)."""
    scenarios = EVAL_SCENARIOS
    if scenario_ids != "all" and isinstance(scenario_ids, list):
        scenarios = [s for s in EVAL_SCENARIOS if s["id"] in scenario_ids]

    run_id = f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\nRunning evaluation harness — {len(scenarios)} scenarios [{run_id}]")
    print("─" * 60)

    results = []

    for scenario in scenarios:
        sid = scenario["id"]
        user_request = scenario["user_request"]
        expected = scenario["expected"]

        thread_id = str(uuid.uuid4())
        initial_state = make_initial_state(user_request=user_request, thread_id=thread_id)

        import time
        start = time.time()
        try:
            config = {"configurable": {"thread_id": thread_id}}
            final_state = await compiled_graph.ainvoke(initial_state, config=config)
            latency_ms = int((time.time() - start) * 1000)

            # Convert state to dict for checking
            conf = final_state.get("confidence")
            if hasattr(conf, "value"):
                conf_str = str(conf.value)
            else:
                conf_str = str(conf) if conf else "none"

            result = {
                "recommendation": final_state.get("recommendation"),
                "confidence": conf_str,
                "reasoning": final_state.get("reasoning"),
                "comparison_matrix": final_state.get("comparison_matrix"),
                "execution_trace": final_state.get("execution_trace") or [],
                "requires_clarification": final_state.get("requires_clarification", False),
                "fallback_or_risk_note": final_state.get("fallback_or_risk_note"),
                "eval_scores": final_state.get("eval_scores") or {},
                "tools_used": final_state.get("tools_used") or [],
            }

            passed, failures = _check_scenario(result, expected)

            eval_scores = final_state.get("eval_scores") or {}
            grounding = eval_scores.get("grounding_score", 0.0)

            status = "PASS" if passed else "FAIL"
            print(
                f"[{status}] {sid:<30} "
                f"confidence={result['confidence']:<12} "
                f"grounding={grounding:.2f}  "
                f"latency={latency_ms}ms"
            )
            if failures:
                for f in failures:
                    print(f"         → {f}")

            results.append({
                "scenario_id": sid,
                "user_request": user_request,
                "passed": passed,
                "failures": failures,
                "recommendation": result["recommendation"],
                "confidence": result["confidence"],
                "scores": eval_scores,
                "latency_ms": latency_ms,
                "trace_id": thread_id,
            })

        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            print(f"[ERROR] {sid}: {e}")
            results.append({
                "scenario_id": sid,
                "user_request": user_request,
                "passed": False,
                "failures": [f"Exception: {str(e)}"],
                "scores": {},
                "latency_ms": latency_ms,
            })

    # Summary
    passed_count = sum(1 for r in results if r["passed"])
    print("\n" + "─" * 60)
    print(f"Results:  {passed_count}/{len(results)} passed  ({100*passed_count//len(results)}%)")

    all_grounding = [r["scores"].get("grounding_score", 0) for r in results if "grounding_score" in r.get("scores", {})]
    if all_grounding:
        print(f"Avg grounding score:  {sum(all_grounding)/len(all_grounding):.3f}")

    hallucinations = sum(1 for r in results if r.get("scores", {}).get("hallucination_detected", False))
    print(f"Hallucinations detected: {hallucinations}")

    avg_latency = sum(r.get("latency_ms", 0) for r in results) // len(results) if results else 0
    print(f"Avg latency: {avg_latency}ms")

    # Write report
    report = {
        "run_id": run_id,
        "total_scenarios": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "pass_rate": passed_count / len(results) if results else 0,
        "avg_scores": {},
        "results": results,
    }

    # Compute aggregate scores
    score_keys = ["task_success", "grounding_score", "hallucination_detected",
                  "plan_completeness", "confidence_calibration", "tool_coverage", "trace_integrity"]
    for key in score_keys:
        vals = [r.get("scores", {}).get(key) for r in results if key in r.get("scores", {})]
        if vals:
            report["avg_scores"][key] = round(sum(float(v) for v in vals) / len(vals), 3)

    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Full report: {report_path}\n")
    return results, report_path


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/app")
    from clients.ollama_client import OllamaClient
    from agent.graph import build_graph
    from config import settings

    async def main():
        ollama = OllamaClient(
            host=settings.OLLAMA_HOST,
            model=settings.OLLAMA_MODEL,
            embed_model=settings.OLLAMA_EMBED_MODEL,
            timeout=settings.OLLAMA_TIMEOUT,
        )
        graph = build_graph(ollama)
        await run_harness(graph)
        await ollama.aclose()

    asyncio.run(main())
