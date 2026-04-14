"""
POST /api/v1/eval/run — trigger the evaluation harness from the API.
"""
import datetime
from fastapi import APIRouter, Request
from models.schemas import EvalRunRequest, EvalRunResponse

router = APIRouter(tags=["Evaluation"])


@router.post("/eval/run", response_model=EvalRunResponse)
async def run_evaluation(request: EvalRunRequest, req: Request):
    """
    Trigger evaluation harness. Runs all 8 (or specified) scenarios.
    Each run is logged to Langfuse and results written to docs/evaluation_report.json.
    """
    # Import here to avoid circular imports at startup
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from tests.eval_harness import run_harness

    compiled_graph = req.app.state.compiled_graph
    scenario_ids = request.scenario_ids

    results, report_path = await run_harness(
        compiled_graph=compiled_graph,
        scenario_ids=scenario_ids,
    )

    passed = sum(1 for r in results if r.get("passed"))
    failed = len(results) - passed
    run_id = f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return EvalRunResponse(
        run_id=run_id,
        total_runs=len(results),
        passed=passed,
        failed=failed,
        pass_rate=passed / len(results) if results else 0.0,
        results=results,
        report_path=report_path,
    )
