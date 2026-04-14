from mcp.server.fastmcp import FastMCP
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from agent.tools.scoring_tool import run_scoring

mcp = FastMCP("RuleSimulator")

# Mock packages for simulation
MOCK_PACKAGES = [
    {"name": "Basic", "type": "package", "network": "C"},
    {"name": "Standard", "type": "package", "network": "B"},
    {"name": "Premium", "type": "package", "network": "A"},
]

@mcp.tool()
def simulate_scoring(entities_json: str) -> str:
    """
    Run the deterministic scoring rules on the provided customer entities.
    entities_json should be a JSON string like: {"industry": "healthcare", "region": "riyadh", "budget": "medium"}
    """
    try:
        entities = json.loads(entities_json)
        results = run_scoring(MOCK_PACKAGES, entities)
        return json.dumps(results, indent=2)
    except json.JSONDecodeError:
        return "Invalid JSON provided for entities."
    except Exception as e:
        return f"Simulation error: {e}"

if __name__ == "__main__":
    mcp.run()
