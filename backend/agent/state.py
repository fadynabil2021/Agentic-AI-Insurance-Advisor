"""
AgentState TypedDict and supporting enums.
This is the single shared state object passed between ALL LangGraph nodes.
Every field is explicitly typed — no hidden state.
"""
from typing import TypedDict, Optional, List, Any
from enum import Enum


class QueryType(str, Enum):
    RECOMMEND = "recommend"
    COMPARE = "compare"
    EXPLAIN = "explain"
    CHEAPEST = "cheapest"
    CLARIFY = "clarify"
    UNSUPPORTED = "unsupported"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM_HIGH = "medium-high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class AgentState(TypedDict):
    # ─── Input ──────────────────────────────────────────────────
    user_request: str
    thread_id: str  # = Langfuse trace ID

    # ─── Intent Parsing ─────────────────────────────────────────
    query_type: Optional[QueryType]
    extracted_entities: Optional[dict]   # {industry, region, budget, priority, employees}
    missing_fields: Optional[List[str]]

    # ─── Planning ───────────────────────────────────────────────
    execution_plan: Optional[List[str]]   # ordered list of tool names
    tools_used: List[str]                 # grows as tools are called

    # ─── Retrieval ──────────────────────────────────────────────
    retrieval_results: Optional[List[dict]]  # matched packages + rules
    retrieval_query: Optional[str]           # expanded query used

    # ─── Scoring ────────────────────────────────────────────────
    scoring_results: Optional[List[dict]]    # packages with scores
    scoring_breakdown: Optional[dict]        # per-rule scores

    # ─── Comparison ─────────────────────────────────────────────
    comparison_matrix: Optional[dict]        # only set if query_type == COMPARE

    # ─── Output ─────────────────────────────────────────────────
    recommendation: Optional[dict]           # plan_name, network, price_range
    reasoning: Optional[List[str]]           # 3-5 reasoning drivers
    confidence: Optional[ConfidenceLevel]
    execution_trace: List[str]               # human-readable step log
    fallback_or_risk_note: Optional[str]

    # ─── Evaluation ─────────────────────────────────────────────
    eval_scores: Optional[dict]              # grounding, completeness, hallucination

    # ─── Control Flow ───────────────────────────────────────────
    retry_count: int                         # starts at 0
    error: Optional[dict]                    # {type, message}
    requires_clarification: bool
    clarification_question: Optional[str]
