"""
Pydantic request/response models for the FastAPI backend.
All models mirror the AgentState fields for clean serialization.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Any


# ─── Request ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    user_request: str = Field(..., min_length=5, max_length=2000)
    thread_id: Optional[str] = None  # for follow-up queries in same session


# ─── Recommendation ───────────────────────────────────────────────────────────

class Recommendation(BaseModel):
    plan_name: str
    network: str
    price_range: List[int]  # [min, max]


# ─── Comparison Matrix ────────────────────────────────────────────────────────

class ComparisonMatrix(BaseModel):
    packages: List[str]
    dimensions: dict
    recommendation: Optional[str] = None
    recommendation_reason: Optional[str] = None


# ─── Main Agent Response ──────────────────────────────────────────────────────

class AgentResponse(BaseModel):
    user_request: str
    thread_id: str
    plan: dict  # {steps: List[str]}
    tools_used: List[str]
    recommendation: Optional[Recommendation] = None
    comparison_matrix: Optional[ComparisonMatrix] = None
    reasoning: Optional[List[str]] = None
    confidence: str
    execution_trace: List[str]
    fallback_or_risk_note: Optional[str] = None
    requires_clarification: bool = False
    clarification_question: Optional[str] = None
    eval_scores: Optional[dict] = None
    latency_ms: Optional[int] = None


# ─── Trace Response ───────────────────────────────────────────────────────────

class TraceResponse(BaseModel):
    trace_id: str
    langfuse_url: str
    steps: List[dict]
    scores: dict
    total_latency_ms: int
    token_usage: dict


# ─── Health Response ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict  # {ollama: bool, chromadb: bool, langfuse: bool, redis: bool}


# ─── Eval Request / Response ─────────────────────────────────────────────────

class EvalRunRequest(BaseModel):
    scenario_ids: Any = "all"  # List[str] or "all"


class EvalRunResponse(BaseModel):
    run_id: str
    total_runs: int
    passed: int
    failed: int
    pass_rate: float
    results: List[dict]
    report_path: str
