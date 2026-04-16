"""
LangGraph StateGraph — the central orchestrator.
Compiles all 9 nodes + conditional routing into a single graph.
Uses MemorySaver checkpointer for per-run state isolation.

Design decision: Langfuse trace is passed via config["configurable"]["langfuse_trace"]
rather than in AgentState, because the trace object (StatefulTraceClient) is NOT
msgpack-serializable and would break MemorySaver checkpointing.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.routing import route_after_validation, route_after_scoring
from clients.gemini_client import GeminiClient
from config import settings


def _get_trace(config: dict):
    """Extract Langfuse trace from LangGraph config, safely."""
    configurable = (config or {}).get("configurable", {})
    return configurable.get("langfuse_trace")


def build_graph(gemini_client: GeminiClient) -> any:
    """
    Build and compile the LangGraph StateGraph.
    Each node wrapper extracts the Langfuse trace from config["configurable"],
    keeping the non-serializable object out of the checkpointed state.
    """
    from agent.nodes.intent_parser import intent_parser_node
    from agent.nodes.validator import validator_node
    from agent.nodes.planner import planner_node
    from agent.nodes.clarifier import clarifier_node
    from agent.nodes.fallback import fallback_node
    from agent.nodes.output_composer import output_composer_node
    from agent.nodes.evaluator import evaluator_node
    from agent.tools.retrieval_tool import retrieval_tool_node
    from agent.tools.scoring_tool import scoring_tool_node
    from agent.tools.comparison_tool import comparison_tool_node

    # ── Node wrappers (inject clients via closure, trace via config) ───────

    async def _intent_parser(state: AgentState, config: dict) -> AgentState:
        return await intent_parser_node(state, gemini_client, _get_trace(config))

    def _validator(state: AgentState, config: dict) -> AgentState:
        return validator_node(state, _get_trace(config))

    async def _planner(state: AgentState, config: dict) -> AgentState:
        return await planner_node(state, gemini_client, _get_trace(config))

    def _clarifier(state: AgentState, config: dict) -> AgentState:
        return clarifier_node(state, _get_trace(config))

    def _fallback(state: AgentState, config: dict) -> AgentState:
        return fallback_node(state, _get_trace(config))

    async def _retrieval_tool(state: AgentState, config: dict) -> AgentState:
        return await retrieval_tool_node(state, _get_trace(config))

    def _scoring_tool(state: AgentState, config: dict) -> AgentState:
        return scoring_tool_node(state, _get_trace(config))

    async def _comparison_tool(state: AgentState, config: dict) -> AgentState:
        return await comparison_tool_node(state, gemini_client, _get_trace(config))

    async def _output_composer(state: AgentState, config: dict) -> AgentState:
        return await output_composer_node(state, gemini_client, _get_trace(config))

    async def _evaluator(state: AgentState, config: dict) -> AgentState:
        return await evaluator_node(state, gemini_client, _get_trace(config))

    # ── Build graph ────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("intent_parser",   _intent_parser)
    graph.add_node("validator",       _validator)
    graph.add_node("planner",         _planner)
    graph.add_node("clarifier",       _clarifier)
    graph.add_node("fallback",        _fallback)
    graph.add_node("retrieval_tool",  _retrieval_tool)
    graph.add_node("scoring_tool",    _scoring_tool)
    graph.add_node("comparison_tool", _comparison_tool)
    graph.add_node("output_composer", _output_composer)
    graph.add_node("evaluator",       _evaluator)

    # Entry point
    graph.set_entry_point("intent_parser")

    # Fixed edges
    graph.add_edge("intent_parser", "validator")

    # Conditional: after validator
    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "complete":    "planner",
            "incomplete":  "clarifier",
            "unsupported": "fallback",
        },
    )

    graph.add_edge("clarifier", "evaluator")
    graph.add_edge("fallback",  "evaluator")

    graph.add_edge("planner",        "retrieval_tool")
    graph.add_edge("retrieval_tool", "scoring_tool")

    # Conditional: after scoring
    graph.add_conditional_edges(
        "scoring_tool",
        route_after_scoring,
        {
            "compare":   "comparison_tool",
            "recommend": "output_composer",
            "retry":     "retrieval_tool",   # retry cycle
            "fallback":  "fallback",
        },
    )

    graph.add_edge("comparison_tool", "output_composer")
    graph.add_edge("output_composer", "evaluator")
    graph.add_edge("evaluator",       END)

    # Compile with in-memory checkpointing
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    return compiled


def make_initial_state(
    user_request: str,
    thread_id: str,
) -> AgentState:
    """Create a clean initial AgentState for a new graph run."""
    return AgentState(
        user_request=user_request,
        thread_id=thread_id,
        query_type=None,
        extracted_entities=None,
        missing_fields=None,
        execution_plan=None,
        tools_used=[],
        retrieval_results=None,
        retrieval_query=None,
        scoring_results=None,
        scoring_breakdown=None,
        comparison_matrix=None,
        recommendation=None,
        reasoning=None,
        confidence=None,
        execution_trace=[],
        fallback_or_risk_note=None,
        eval_scores=None,
        retry_count=0,
        error=None,
        requires_clarification=False,
        clarification_question=None,
    )
