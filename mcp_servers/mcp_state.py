from mcp.server.fastmcp import FastMCP
import os
import sys

# Add backend to path so we can import from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from clients.langfuse_client import get_langfuse
from clients.redis_client import get_cached_response
import asyncio

mcp = FastMCP("StateInspector")

@mcp.tool()
async def get_session_state(user_request: str) -> str:
    """Retrieve the cached agent response from Redis for a given user request."""
    try:
        data = await get_cached_response(user_request)
        if not data:
            return "No cached state found for that request."
        import json
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error connecting to Redis: {e}"

@mcp.tool()
def get_langfuse_trace(trace_id: str) -> str:
    """Retrieve full trace details from Langfuse for a given trace_id."""
    try:
        lf = get_langfuse()
        trace = lf.get_trace(trace_id)
        return trace.json() if hasattr(trace, 'json') else str(trace)
    except Exception as e:
        return f"Error retrieving trace {trace_id}: {e}"

if __name__ == "__main__":
    mcp.run()
