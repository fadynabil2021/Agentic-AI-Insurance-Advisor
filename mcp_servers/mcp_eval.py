from mcp.server.fastmcp import FastMCP
import os
import sys
import asyncio
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from tests.eval_harness import run_harness
from clients.ollama_client import OllamaClient
from agent.graph import build_graph
from config import settings

mcp = FastMCP("EvaluationTrigger")

@mcp.tool()
async def run_eval_harness(scenario_ids: str = "all") -> str:
    """
    Trigger the evaluation harness.
    Pass 'all' to run all scenarios, or a comma-separated list of scenario IDs (e.g. 'S1_balanced,E2_unsupported').
    Note: Requires Ollama and ChromaDB to be running.
    """
    try:
        s_ids = scenario_ids
        if scenario_ids != "all":
            s_ids = [s.strip() for s in scenario_ids.split(',')]

        ollama = OllamaClient(
            host=settings.OLLAMA_HOST,
            model=settings.OLLAMA_MODEL,
            embed_model=settings.OLLAMA_EMBED_MODEL,
            timeout=settings.OLLAMA_TIMEOUT,
        )
        graph = build_graph(ollama)
        
        results, report_path = await run_harness(
            compiled_graph=graph,
            scenario_ids=s_ids,
            report_dir="/tmp/docs" # write locally
        )
        await ollama.aclose()
        
        passed = sum(1 for r in results if r["passed"])
        summary = f"Eval complete. Passed: {passed}/{len(results)}. Report available at {report_path}"
        return summary
    except Exception as e:
        return f"Eval Harness error: {str(e)}"

if __name__ == "__main__":
    mcp.run()
