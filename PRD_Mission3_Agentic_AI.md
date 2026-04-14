# Product Requirements Document
## Mission 3 – Elite Assessment: Agentic Insurance Advisor System

**Version:** 1.0.1
**Author:** Principal Engineering Lead
**Date:** 2026-04-07
**Classification:** Internal — Implementation Reference
**Status:** Final — Ready for Development

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Goals & Success Criteria](#2-system-goals--success-criteria)
3. [Confirmed Technical Stack](#3-confirmed-technical-stack)
4. [Architecture Overview](#4-architecture-overview)
5. [LangGraph Orchestration Design](#5-langgraph-orchestration-design)
6. [Agent State Model](#6-agent-state-model)
7. [Component Specifications](#7-component-specifications)
   - 7.1 Intent Parser Node
   - 7.2 Planner Node
   - 7.3 Retrieval Tool
   - 7.4 Scoring Tool (Deterministic)
   - 7.5 Comparison Tool
   - 7.6 Validation Tool
   - 7.7 Fallback Tool
   - 7.8 Output Composer Node
   - 7.9 Evaluator Node
8. [Pydantic Data Models](#8-pydantic-data-models)
9. [MCP Server Design (Antigravity IDE)](#9-mcp-server-design-antigravity-ide)
10. [FastAPI Backend Specification](#10-fastapi-backend-specification)
11. [Next.js Frontend Specification](#11-nextjs-frontend-specification)
12. [Evaluation Strategy (Langfuse)](#12-evaluation-strategy-langfuse)
13. [Ollama + Gemma 4 (gemma4:31b-cloud) Integration](#13-ollama--gemma-4-gemma431b-cloud-integration)
14. [Docker Compose Infrastructure](#14-docker-compose-infrastructure)
15. [Repository Structure](#15-repository-structure)
16. [Implementation Sequencing & Milestones](#16-implementation-sequencing--milestones)
17. [Edge Cases & Failure Handling](#17-edge-cases--failure-handling)
18. [Non-Functional Requirements](#18-non-functional-requirements)
19. [Risk Register](#19-risk-register)
20. [Appendix: Expected Outputs for All 5 Test Scenarios](#20-appendix-expected-outputs-for-all-5-test-scenarios)

---

## 1. Executive Summary

This document is the complete Product Requirements Document (PRD) for building the **Agentic Insurance Advisor System** as specified in Mission 3 of the Elite Assessment. The system is a production-grade agentic AI that accepts natural-language insurance plan queries, decomposes them into a structured plan, executes multiple specialized tools, and returns a fully grounded, structured recommendation with execution trace and confidence score.

The system is not a chatbot. It is a stateful, graph-based agentic workflow with:
- A **LangGraph** orchestrator managing state across multi-step reasoning
- A **Gemma 4 (gemma4:31b-cloud)** model via **Ollama** powering all LLM-dependent nodes (intent parsing, planning, response composition)
- Fully **deterministic** scoring and rules logic — no LLM in the hot path for business rules
- A **Langfuse** self-hosted observability stack measuring quality, hallucination, and task success
- A **FastAPI** backend exposing the agent via REST and WebSocket
- A **Next.js** chat UI for demo and walkthrough
- **MCP servers** exposing each tool to the **Antigravity IDE** for development-time agentic access

The system is containerized entirely in **Docker Compose** for local-only deployment, reproducible from a single `docker compose up` command.

---

## 2. System Goals & Success Criteria

### 2.1 Primary Goals

| Goal | Measure |
|------|---------|
| Correct structured recommendation for all 5 visible test scenarios | 5/5 pass |
| Real tool execution (not simulated) | Execution trace shows actual tool calls with I/O |
| Graceful failure on all hidden edge cases | No hallucinated certainty; fallback note always present |
| Full observability of every agent step | Langfuse trace covers every node transition |
| Reproducible setup from cold start | `docker compose up` brings full system online in < 5 min |
| Assignment scorecard ≥ 85/100 | Per Section 14 of the PDF |

### 2.2 Non-Goals

- Production cloud deployment (Docker Compose only)
- Multi-language support (English only per spec)
- Authentication / multi-tenant isolation
- Streaming LLM tokens to the frontend (WebSocket sends complete structured outputs)

### 2.3 Success Criteria per Scorecard Section

| Scorecard Area | Weight | Target | Key Deliverable |
|----------------|--------|--------|-----------------|
| Agentic Architecture | 25% | 23/25 | LangGraph graph with 8 nodes, state management, tool wiring |
| Tooling & Runtime | 20% | 19/20 | Justified framework choice, MCP servers, intentional orchestration |
| Evaluation & Reliability | 20% | 19/20 | Langfuse traces, eval harness, ≥5 evaluated runs |
| Model Selection | 15% | 14/15 | gemma4:31b-cloud + Ollama justified, deterministic fallback for rules |
| Implementation Quality | 10% | 9/10 | Clean Python, typed, tested, reproducible |
| Communication Quality | 10% | 9/10 | README, architecture diagram, technical report |

---

## 3. Confirmed Technical Stack

All decisions below are confirmed from pre-PRD elicitation. No assumptions remain.

### 3.1 Decision Table

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Agent Orchestration | **LangGraph 0.2.x** | Graph-based stateful workflows; conditional branching; built-in retry; best fit for non-linear agentic plans |
| LLM Runtime | **Ollama** (local) + **gemma4:31b-cloud** | Fully local, no API costs, no data egress; 31b parameter model with strong instruction-following; runs on local GPU |
| Backend | **FastAPI 0.115.x** + Python 3.12 | Async-native, OpenAPI auto-docs, WebSocket support, production-grade |
| Frontend | **Next.js 15** + React 19 + Tailwind CSS | App Router, server components, fast iteration |
| Vector Store (Retrieval) | **ChromaDB** (local, persistent) | Zero-config, Docker-native, persistent storage, Python-native client |
| Observability | **Langfuse** (self-hosted) | Open-source, Docker Compose native, trace + eval + cost tracking |
| Evaluation DB | **PostgreSQL 16** | Used by Langfuse for persistent trace storage |
| Dev IDE Tooling | **Antigravity IDE** (VS Code + Claude Sonnet 4.6 agent) + **MCP servers** | Each tool exposed as an MCP server so the Antigravity Claude agent can invoke tools directly during development and demo |
| Container Runtime | **Docker Compose v2** | Single-command local deployment; all services containerized |
| Cache | **Redis 7** | Rate limiting, request deduplication, short-lived result caching |
| Data Validation | **Pydantic v2** | All inputs, outputs, and intermediate states are typed |

### 3.2 Why NOT Other Frameworks

| Rejected Option | Reason |
|----------------|--------|
| CrewAI | Role-based multi-agent model adds overhead; our problem is a single-workflow system, not a team of autonomous agents |
| AutoGen | Conversational multi-agent loop; overkill and harder to control state for a deterministic business-rules workflow |
| Semantic Kernel | C#-first; Python SDK is secondary; tight Microsoft ecosystem coupling |
| Raw OpenAI SDK | No built-in state management, graph traversal, or retry primitives; we'd be rebuilding LangGraph |
| GPT-4o / Azure OpenAI | Requires external API calls; data leaves the machine; cost per run; Ollama+gemma4:31b-cloud satisfies the spec locally |

### 3.3 Why LangGraph Specifically

LangGraph models the agent as a **directed graph** (StateGraph) where:
- Each **node** is a Python async function that reads and writes to shared `AgentState`
- Each **edge** is a routing decision (conditional or unconditional)
- State is passed by value, not by reference, preventing hidden side effects
- Built-in support for **cycles** (retry loops), **checkpointing**, and **interrupt/resume**
- The execution trace is a first-class concept — every node transition is logged with timestamps

This is exactly what the assignment requires: planning, tool invocation, state handling across steps, controlled final answer synthesis, and retry/repair behavior.

---

## 4. Architecture Overview

### 4.1 System Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ANTIGRAVITY IDE                              │
│   Claude Sonnet 4.6 Agent Window ──► MCP Servers (tools exposed)    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ MCP Protocol (stdio / HTTP)
┌──────────────────────────▼──────────────────────────────────────────┐
│                      NEXT.JS FRONTEND (port 3000)                   │
│   Chat UI │ Trace Viewer │ Recommendation Card │ Confidence Badge    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ HTTP REST + WebSocket
┌──────────────────────────▼──────────────────────────────────────────┐
│                    FASTAPI BACKEND (port 8000)                      │
│   /api/v1/query  │  /api/v1/trace/{id}  │  /api/v1/health          │
│                           │                                         │
│              ┌────────────▼────────────┐                            │
│              │   LANGGRAPH ORCHESTRATOR │                            │
│              │                          │                            │
│   ┌──────────▼──────────────────────────────────────────────┐      │
│   │                    AGENT STATE GRAPH                      │      │
│   │                                                           │      │
│   │  [intent_parser] → [validator] → [planner]               │      │
│   │         ↓                ↓             ↓                  │      │
│   │  [fallback]      [clarifier]   [retrieval_tool]           │      │
│   │                                      ↓                    │      │
│   │                             [scoring_tool]                │      │
│   │                                  ↓     ↓                  │      │
│   │                        [comparison_tool] (conditional)    │      │
│   │                                      ↓                    │      │
│   │                           [output_composer]               │      │
│   │                                      ↓                    │      │
│   │                              [evaluator]                  │      │
│   └──────────────────────────────────────────────────────────┘      │
│              │                                                        │
└──────────────┼────────────────────────────────────────────────────  ┘
               │
   ┌───────────┼────────────────────────────────────────────┐
   │           │         INFRASTRUCTURE SERVICES            │
   │  ┌────────▼──────┐  ┌──────────┐  ┌────────────────┐  │
   │  │   OLLAMA      │  │ CHROMADB │  │   LANGFUSE     │  │
   │  │   Gemma 4     │  │ (vector) │  │ + PostgreSQL   │  │
   │  │   port 11434  │  │ port 8001│  │   port 3001    │  │
   │  └───────────────┘  └──────────┘  └────────────────┘  │
   │                         ┌──────────────────────────┐   │
   │                         │    REDIS (port 6379)      │   │
   │                         └──────────────────────────┘   │
   └────────────────────────────────────────────────────────┘
```

### 4.2 Request Lifecycle

```
User types query in Next.js Chat UI
    │
    ▼
POST /api/v1/query  (FastAPI)
    │
    ├── Redis: check duplicate request cache
    ├── Langfuse: open top-level trace
    │
    ▼
LangGraph: run_graph(user_request)
    │
    ├── Node 1: intent_parser     → extract intent, entities, query_type
    ├── Node 2: validator          → check completeness, detect missing fields
    │     ├── [INCOMPLETE] → clarifier_node → return clarification request
    │     └── [COMPLETE]   → planner_node
    ├── Node 3: planner            → select tools, set execution_plan in state
    ├── Node 4: retrieval_tool     → query ChromaDB, return relevant packages + rules
    ├── Node 5: scoring_tool       → deterministic rules engine, score packages
    ├── Node 6: comparison_tool    → (conditional) only if query_type == COMPARE
    ├── Node 7: output_composer    → synthesize recommendation with Gemma 4
    ├── Node 8: evaluator          → score quality, detect hallucination, log to Langfuse
    │
    ▼
Return structured JSON response
    │
    ▼
Next.js renders Recommendation Card + Trace Panel
    │
    ▼
Langfuse records trace, scores, latency, token usage
```

---

## 5. LangGraph Orchestration Design

### 5.1 Graph Definition (Python pseudocode — full implementation follows in Section 7)

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

graph = StateGraph(AgentState)

# Register nodes
graph.add_node("intent_parser", intent_parser_node)
graph.add_node("validator", validator_node)
graph.add_node("clarifier", clarifier_node)
graph.add_node("planner", planner_node)
graph.add_node("retrieval_tool", retrieval_tool_node)
graph.add_node("scoring_tool", scoring_tool_node)
graph.add_node("comparison_tool", comparison_tool_node)
graph.add_node("output_composer", output_composer_node)
graph.add_node("evaluator", evaluator_node)
graph.add_node("fallback", fallback_node)

# Entry point
graph.set_entry_point("intent_parser")

# Edges
graph.add_edge("intent_parser", "validator")

graph.add_conditional_edges(
    "validator",
    route_after_validation,
    {
        "complete": "planner",
        "incomplete": "clarifier",
        "unsupported": "fallback",
    }
)

graph.add_edge("clarifier", END)  # Returns clarification; user must resubmit
graph.add_edge("fallback", END)

graph.add_edge("planner", "retrieval_tool")
graph.add_edge("retrieval_tool", "scoring_tool")

graph.add_conditional_edges(
    "scoring_tool",
    route_after_scoring,
    {
        "compare": "comparison_tool",
        "recommend": "output_composer",
        "retry": "retrieval_tool",   # retry if retrieval was empty
        "fallback": "fallback",
    }
)

graph.add_edge("comparison_tool", "output_composer")
graph.add_edge("output_composer", "evaluator")
graph.add_edge("evaluator", END)

# Compile with in-memory checkpointing (upgrade to SQLite saver for persistence)
checkpointer = MemorySaver()
compiled_graph = graph.compile(checkpointer=checkpointer)
```

### 5.2 Conditional Routing Logic

#### `route_after_validation(state: AgentState) → str`

```
if state.error and state.error.type == "UNSUPPORTED_QUERY":
    return "unsupported"
if state.missing_fields and len(state.missing_fields) > 0:
    return "incomplete"
return "complete"
```

#### `route_after_scoring(state: AgentState) → str`

```
if state.retry_count < MAX_RETRIES and state.retrieval_results == []:
    state.retry_count += 1
    return "retry"
if state.query_type == QueryType.COMPARE:
    return "compare"
if state.scoring_results is None:
    return "fallback"
return "recommend"
```

### 5.3 Retry Behavior

- Max retries: `MAX_RETRIES = 2`
- On retry: retrieval tool receives expanded query (query + synonyms, industry variations)
- If still empty after 2 retries: route to fallback with `confidence = "low"` and a safe failure message
- All retry attempts are logged in `execution_trace` and in Langfuse

### 5.4 State Checkpointing

- **Development:** `MemorySaver` (in-memory, fast, reset per process)
- **Production-ready upgrade path:** `SqliteSaver` (persistent, thread-safe, drop-in replacement)
- Each graph run gets a unique `thread_id` (= `trace_id`) so the state can be resumed or inspected

---

## 6. Agent State Model

### 6.1 `AgentState` TypedDict

This is the single shared state object passed between all nodes. Every field is explicitly typed.

```python
from typing import TypedDict, Optional, List, Literal
from enum import Enum

class QueryType(str, Enum):
    RECOMMEND    = "recommend"
    COMPARE      = "compare"
    EXPLAIN      = "explain"
    CHEAPEST     = "cheapest"
    CLARIFY      = "clarify"
    UNSUPPORTED  = "unsupported"

class ConfidenceLevel(str, Enum):
    HIGH         = "high"
    MEDIUM_HIGH  = "medium-high"
    MEDIUM       = "medium"
    LOW          = "low"
    NONE         = "none"

class AgentState(TypedDict):
    # ─── Input ────────────────────────────────────────────
    user_request:        str
    thread_id:           str          # = Langfuse trace ID

    # ─── Intent Parsing ───────────────────────────────────
    query_type:          Optional[QueryType]
    extracted_entities:  Optional[dict]   # {industry, region, budget, priority, employees}
    missing_fields:      Optional[List[str]]

    # ─── Planning ─────────────────────────────────────────
    execution_plan:      Optional[List[str]]  # ordered list of tool names
    tools_used:          List[str]            # grows as tools are called

    # ─── Retrieval ────────────────────────────────────────
    retrieval_results:   Optional[List[dict]]  # matched packages + rules
    retrieval_query:     Optional[str]         # expanded query used

    # ─── Scoring ──────────────────────────────────────────
    scoring_results:     Optional[List[dict]]  # packages with scores
    scoring_breakdown:   Optional[dict]        # per-rule scores

    # ─── Comparison ───────────────────────────────────────
    comparison_matrix:   Optional[dict]        # only set if query_type == COMPARE

    # ─── Output ───────────────────────────────────────────
    recommendation:      Optional[dict]        # plan_name, network, price_range
    reasoning:           Optional[List[str]]   # 3-5 reasoning drivers
    confidence:          Optional[ConfidenceLevel]
    execution_trace:     List[str]             # human-readable step log
    fallback_or_risk_note: Optional[str]

    # ─── Evaluation ───────────────────────────────────────
    eval_scores:         Optional[dict]        # grounding, completeness, hallucination

    # ─── Control Flow ─────────────────────────────────────
    retry_count:         int                   # starts at 0
    error:               Optional[dict]        # {type, message}
    requires_clarification: bool
    clarification_question: Optional[str]
```

---

## 7. Component Specifications

### 7.1 Intent Parser Node

**Purpose:** Convert the raw natural language user request into structured entities the rest of the graph can use.

**Model:** gemma4:31b-cloud via Ollama
**Input:** `state.user_request`
**Output:** Writes to `state.query_type`, `state.extracted_entities`, `state.missing_fields`

**System Prompt:**

```
You are an intent extraction engine for an insurance advisor system.
Extract structured information from the user query.

Return ONLY valid JSON matching this schema:
{
  "query_type": one of ["recommend", "compare", "explain", "cheapest", "clarify", "unsupported"],
  "industry": string or null,
  "region": string or null,
  "budget": one of ["low", "medium", "high"] or null,
  "priority": string or null,
  "employees": integer or null,
  "dependents_ratio": float or null,
  "compare_packages": list of strings or null
}

Rules:
- If the request is not about insurance plan selection, set query_type to "unsupported"
- If the request asks for cheapest / lowest cost, set query_type to "cheapest"
- If the request asks to compare two plans, set query_type to "compare"
- Do NOT infer fields that are not stated or strongly implied by the request
- Return null for any field you cannot determine
```

**Failure Handling:**
- If Gemma 4 returns invalid JSON: retry parse up to 2 times with stricter prompt
- If still invalid after retries: set `state.error = {type: "PARSE_ERROR", message: "..."}` and route to fallback
- Log parse attempt count in execution_trace

**Code Skeleton:**

```python
async def intent_parser_node(state: AgentState) -> AgentState:
    with langfuse.start_as_current_span("intent_parser"):
        prompt = build_intent_prompt(state["user_request"])
        raw = await ollama_client.chat(model="gemma4:31b-cloud", messages=[
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": state["user_request"]}
        ])
        parsed = safe_json_parse(raw, retries=2)
        
        if parsed is None:
            state["error"] = {"type": "PARSE_ERROR", "message": "Failed to parse intent"}
            state["execution_trace"].append("intent_parser: FAILED to parse intent, routing to fallback")
            return state
        
        state["query_type"] = parsed.get("query_type")
        state["extracted_entities"] = {k: v for k, v in parsed.items() if k != "query_type"}
        state["missing_fields"] = [k for k, v in state["extracted_entities"].items()
                                    if v is None and k in REQUIRED_FIELDS_PER_QUERY_TYPE[state["query_type"]]]
        state["execution_trace"].append(
            f"intent_parser: identified query_type={state['query_type']}, "
            f"entities={state['extracted_entities']}, missing={state['missing_fields']}"
        )
        return state
```

---

### 7.2 Validator Node

**Purpose:** Determine if the extracted entities are sufficient to proceed, and detect unsupported queries.

**Model:** None (deterministic logic only)
**Input:** `state.query_type`, `state.extracted_entities`, `state.missing_fields`
**Output:** Sets `state.requires_clarification`, `state.clarification_question`

**Required fields per query type:**

| query_type | Required | Optional |
|-----------|---------|---------|
| recommend | industry, region | budget, priority |
| cheapest | industry, region | budget |
| compare | industry, region, compare_packages | budget |
| explain | — (previous context) | — |
| unsupported | — | — |

**Validation Logic:**

```python
def validator_node(state: AgentState) -> AgentState:
    qt = state["query_type"]
    
    if qt == QueryType.UNSUPPORTED:
        state["error"] = {
            "type": "UNSUPPORTED_QUERY",
            "message": "Request is outside the supported domain (insurance plan selection)."
        }
        state["execution_trace"].append("validator: query flagged UNSUPPORTED")
        return state
    
    required = REQUIRED_FIELDS[qt]
    missing = [f for f in required if not state["extracted_entities"].get(f)]
    state["missing_fields"] = missing
    
    if missing:
        state["requires_clarification"] = True
        state["clarification_question"] = build_clarification_question(missing)
        state["execution_trace"].append(
            f"validator: missing fields {missing}, will request clarification"
        )
    else:
        state["requires_clarification"] = False
        state["execution_trace"].append("validator: all required fields present, proceeding")
    
    return state
```

---

### 7.3 Planner Node

**Purpose:** Decide which tools to call and in what order. This is the true orchestrator brain.

**Model:** gemma4:31b-cloud via Ollama (lightweight planning prompt)
**Input:** `state.query_type`, `state.extracted_entities`
**Output:** `state.execution_plan` (ordered list of tool names)

**Planning Rules (deterministic overrides take precedence):**

| query_type | Plan |
|-----------|------|
| recommend | [retrieval_tool, scoring_tool, output_composer] |
| cheapest  | [retrieval_tool, scoring_tool, output_composer] |
| compare   | [retrieval_tool, scoring_tool, comparison_tool, output_composer] |
| explain   | [output_composer] (uses cached state from previous run) |

**Note:** The planner first checks deterministic rules. If the query_type maps to a known plan, it uses that plan without calling Gemma 4. Gemma 4 is only invoked for ambiguous or multi-step plans that don't match the static table. This keeps latency low for the happy path.

```python
async def planner_node(state: AgentState) -> AgentState:
    with langfuse.start_as_current_span("planner"):
        qt = state["query_type"]
        
        # Deterministic planning first
        if qt in STATIC_PLANS:
            plan = STATIC_PLANS[qt]
        else:
            # LLM-assisted planning for ambiguous cases
            plan = await llm_plan(state)
        
        state["execution_plan"] = plan
        state["execution_trace"].append(f"planner: selected plan={plan}")
        return state
```

---

### 7.4 Retrieval Tool Node

**Purpose:** Query the ChromaDB vector store with the user's request and return relevant packages, rules, and knowledge snippets.

**Model:** None (vector similarity search; embedding model: `nomic-embed-text` via Ollama)
**Input:** `state.extracted_entities`, `state.retrieval_query`
**Output:** `state.retrieval_results`

**ChromaDB Collection Design:**

| Collection | Contents | Documents |
|-----------|---------|----------|
| `packages` | Package catalog (Basic, Standard, Premium) | 3 |
| `benchmark_rules` | Industry risk, region cost, budget rules | 12 |
| `knowledge_snippets` | Network quality descriptions | 4 |
| `customer_profiles` | Sample customer profiles | 3 |

**Embedding Strategy:**
- Each document is embedded with `nomic-embed-text` (runs locally via Ollama)
- Each document includes structured metadata (industry, region, budget, network, etc.) for filtered retrieval
- At query time: embed the user query, retrieve top-5 by cosine similarity per collection, then apply metadata filters to narrow

**Retrieval Logic:**

```python
async def retrieval_tool_node(state: AgentState) -> AgentState:
    with langfuse.start_as_current_span("retrieval_tool"):
        state["tools_used"].append("retrieval_tool")
        
        entities = state["extracted_entities"]
        query = build_retrieval_query(entities)
        state["retrieval_query"] = query
        
        # Query all relevant collections
        packages = chroma_client.get_collection("packages").query(
            query_texts=[query],
            n_results=3,
            where=build_metadata_filter(entities)
        )
        rules = chroma_client.get_collection("benchmark_rules").query(
            query_texts=[query], n_results=5
        )
        snippets = chroma_client.get_collection("knowledge_snippets").query(
            query_texts=[query], n_results=3
        )
        
        results = merge_retrieval_results(packages, rules, snippets)
        state["retrieval_results"] = results
        
        if not results:
            state["execution_trace"].append(
                f"retrieval_tool: WARNING — no results found for query='{query}'"
            )
        else:
            state["execution_trace"].append(
                f"retrieval_tool: returned {len(results)} documents "
                f"(packages={len(packages['ids'][0])}, rules={len(rules['ids'][0])})"
            )
        
        return state
```

**Metadata Filter Builder:**

```python
def build_metadata_filter(entities: dict) -> dict:
    filters = {}
    if entities.get("budget"):
        filters["budget_tier"] = {"$in": get_compatible_budgets(entities["budget"])}
    if entities.get("industry"):
        filters["applicable_industry"] = {"$in": ["all", entities["industry"].lower()]}
    return filters if filters else {}
```

---

### 7.5 Scoring Tool Node

**Purpose:** Apply deterministic business rules to score each retrieved package. This is the single most important tool in the system — it must NOT use an LLM. All logic is explicit, auditable Python.

**Model:** None — pure deterministic rules engine
**Input:** `state.retrieval_results`, `state.extracted_entities`
**Output:** `state.scoring_results`, `state.scoring_breakdown`

**Scoring Rules (directly from PDF Section 5C):**

```python
INDUSTRY_RISK = {
    "healthcare":    "high",
    "construction":  "medium",
    "retail":        "medium-low",
}

REGION_COST = {
    "riyadh":  3,   # highest cost pressure
    "dammam":  2,
    "jeddah":  1,   # lowest
}

BUDGET_PACKAGE_COMPATIBILITY = {
    "low":    ["basic", "standard"],      # Premium not recommended unless justified
    "medium": ["basic", "standard", "premium"],
    "high":   ["standard", "premium"],
}

PRIORITY_PACKAGE_MAP = {
    "cheapest acceptable": ["basic", "standard"],
    "best coverage":       ["premium", "standard"],
    "balanced":            ["standard"],
    "stable service":      ["standard", "premium"],
}
```

**Scoring Function:**

```python
def scoring_tool_node(state: AgentState) -> AgentState:
    with langfuse.start_as_current_span("scoring_tool"):
        state["tools_used"].append("scoring_tool")
        entities = state["extracted_entities"]
        results = []
        breakdown = {}
        
        for pkg in state["retrieval_results"]:
            if pkg["type"] != "package":
                continue
            
            score = 100  # start at 100, deduct penalties
            pkg_name = pkg["name"].lower()
            reasons = []
            
            # Rule 1: Budget compatibility
            budget = (entities.get("budget") or "medium").lower()
            compatible = BUDGET_PACKAGE_COMPATIBILITY.get(budget, ["standard"])
            if pkg_name not in compatible:
                score -= 40
                reasons.append(f"PENALTY: {pkg_name} not compatible with {budget} budget")
            
            # Rule 2: Industry risk
            industry = (entities.get("industry") or "").lower()
            risk = INDUSTRY_RISK.get(industry, "medium")
            if risk == "high" and pkg_name == "basic":
                score -= 25
                reasons.append("PENALTY: Basic not suitable for high-risk industry (healthcare)")
            if risk == "medium-low" and pkg_name == "premium":
                score -= 10
                reasons.append("NOTE: Premium may be over-spec for medium-low risk industry")
            
            # Rule 3: Region cost pressure
            region = (entities.get("region") or "").lower()
            cost_pressure = REGION_COST.get(region, 1)
            if cost_pressure >= 3 and pkg_name == "premium" and budget == "medium":
                score -= 15
                reasons.append("PENALTY: High region cost pressure limits Premium viability on medium budget")
            
            # Rule 4: Priority alignment
            priority = (entities.get("priority") or "balanced").lower()
            preferred = []
            for k, v in PRIORITY_PACKAGE_MAP.items():
                if k in priority:
                    preferred = v
                    break
            if preferred and pkg_name not in preferred:
                score -= 20
                reasons.append(f"PENALTY: {pkg_name} does not align with priority '{priority}'")
            elif preferred and pkg_name in preferred:
                score += 10
                reasons.append(f"BONUS: {pkg_name} matches priority '{priority}'")
            
            # Rule 5: Dependents ratio (if provided)
            dep_ratio = entities.get("dependents_ratio")
            if dep_ratio and dep_ratio > 0.5 and pkg_name == "basic":
                score -= 15
                reasons.append("PENALTY: High dependents ratio increases benefits cost; Basic insufficient")
            
            score = max(0, min(100, score))  # clamp 0-100
            results.append({"package": pkg, "score": score, "reasons": reasons})
            breakdown[pkg_name] = {"score": score, "reasons": reasons}
        
        results.sort(key=lambda x: x["score"], reverse=True)
        state["scoring_results"] = results
        state["scoring_breakdown"] = breakdown
        
        if results:
            top = results[0]
            state["execution_trace"].append(
                f"scoring_tool: top package='{top['package']['name']}' "
                f"score={top['score']}, breakdown={breakdown}"
            )
        else:
            state["execution_trace"].append("scoring_tool: WARNING — no packages scored")
        
        # Determine confidence
        if results and results[0]["score"] >= 80:
            state["confidence"] = ConfidenceLevel.HIGH
        elif results and results[0]["score"] >= 60:
            state["confidence"] = ConfidenceLevel.MEDIUM_HIGH
        elif results and results[0]["score"] >= 40:
            state["confidence"] = ConfidenceLevel.MEDIUM
        else:
            state["confidence"] = ConfidenceLevel.LOW
        
        return state
```

---

### 7.6 Comparison Tool Node

**Purpose:** Build a structured comparison matrix between two or more packages. Only invoked when `query_type == COMPARE`.

**Model:** None (deterministic table construction) + minimal gemma4:31b-cloud for narrative pros/cons summary
**Input:** `state.scoring_results`, `state.extracted_entities`, `state.retrieved_results`
**Output:** `state.comparison_matrix`

**Comparison Matrix Schema:**

```python
comparison_matrix = {
    "packages": ["Standard", "Premium"],
    "dimensions": {
        "network":       {"Standard": "B", "Premium": "A"},
        "price_range":   {"Standard": "6000–7500", "Premium": "9000–12000"},
        "coverage":      {"Standard": "Medium", "Premium": "High"},
        "score":         {"Standard": 75, "Premium": 62},
        "budget_fit":    {"Standard": "Yes", "Premium": "Borderline"},
        "risk_fit":      {"Standard": "Yes", "Premium": "Yes"},
    },
    "recommendation": "Standard",
    "recommendation_reason": "Higher score due to budget alignment; Premium is viable but adds cost without proportional coverage gain for this risk profile."
}
```

---

### 7.7 Fallback Tool Node

**Purpose:** Handle all failure modes gracefully — unsupported queries, empty retrieval, parse errors, confidence below threshold.

**Model:** None
**Input:** `state.error`, `state.confidence`, `state.query_type`
**Output:** Sets `state.fallback_or_risk_note`, `state.recommendation = None`, `state.confidence = ConfidenceLevel.NONE`

**Fallback Response Templates:**

| Error Type | Fallback Message |
|-----------|----------------|
| UNSUPPORTED_QUERY | "This system is designed for insurance plan recommendations. Your query appears to be outside this scope. Please ask about plan selection, comparison, or cost optimization." |
| EMPTY_RETRIEVAL | "Unable to find relevant packages matching your criteria. Please verify the industry and region details, or contact an advisor for custom options." |
| LOW_CONFIDENCE | "The system was able to generate a recommendation but confidence is low due to conflicting constraints. Please review the reasoning carefully before proceeding." |
| PARSE_ERROR | "Unable to understand the request. Please rephrase your question with the industry type, region, and any budget or coverage preferences." |

---

### 7.8 Clarifier Node

**Purpose:** When required fields are missing, return a structured clarification request instead of guessing.

**Model:** Gemma 4 (natural language clarification generation)
**Input:** `state.missing_fields`, `state.user_request`
**Output:** `state.clarification_question`, routes to END

**Behavior:**
- Never infer missing fields
- Returns a conversational clarification question via the same response schema, with `recommendation = null` and `requires_clarification = true`
- The frontend displays this as a follow-up prompt in the chat

**Example clarification output:**

```json
{
  "user_request": "Recommend a plan for my company.",
  "requires_clarification": true,
  "clarification_question": "To give you an accurate recommendation, I need a few more details: (1) What industry is your company in? (e.g., healthcare, construction, retail) (2) Which region are you based in? (e.g., Riyadh, Jeddah, Dammam)",
  "recommendation": null,
  "confidence": "none",
  "execution_trace": ["intent_parser: query_type=recommend", "validator: missing fields [industry, region]", "clarifier: generated clarification question"]
}
```

---

### 7.9 Output Composer Node

**Purpose:** Synthesize all intermediate outputs into the final structured recommendation. Uses Gemma 4 only for the `reasoning` narrative — all structured fields are built deterministically.

**Model:** Gemma 4 (reasoning narrative only — 3-5 bullet points)
**Input:** `state.scoring_results`, `state.scoring_breakdown`, `state.extracted_entities`, `state.comparison_matrix`
**Output:** `state.recommendation`, `state.reasoning`, `state.fallback_or_risk_note`

**Composer Logic:**

```python
async def output_composer_node(state: AgentState) -> AgentState:
    with langfuse.start_as_current_span("output_composer"):
        state["tools_used"].append("output_composer")
        
        top_pkg = state["scoring_results"][0]["package"]
        
        # Deterministic fields — no LLM
        state["recommendation"] = {
            "plan_name":   top_pkg["name"],
            "network":     top_pkg["network"],
            "price_range": top_pkg["price_range"],
        }
        
        # LLM-generated reasoning narrative (grounded in scoring breakdown)
        reasoning_context = {
            "package":          top_pkg["name"],
            "entities":         state["extracted_entities"],
            "scoring_reasons":  state["scoring_breakdown"][top_pkg["name"].lower()]["reasons"],
            "all_scores":       {r["package"]["name"]: r["score"] for r in state["scoring_results"]}
        }
        
        raw_reasoning = await ollama_client.chat(
            model="gemma4:31b-cloud",
            messages=[
                {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(reasoning_context)}
            ]
        )
        
        state["reasoning"] = parse_reasoning_list(raw_reasoning)  # 3-5 bullets
        
        # Risk note: always generated deterministically
        state["fallback_or_risk_note"] = build_risk_note(
            top_pkg["name"],
            state["scoring_results"],
            state["extracted_entities"]
        )
        
        state["execution_trace"].append(
            f"output_composer: composed recommendation={state['recommendation']['plan_name']}, "
            f"confidence={state['confidence']}"
        )
        
        return state
```

**Reasoning System Prompt:**

```
You are a grounding engine. Based ONLY on the provided scoring data, generate 3-5 concise reasoning bullets
explaining why the selected package was chosen. 

Rules:
- Each bullet must reference at least one specific data point (score, rule, entity value)
- Do NOT introduce new information not present in the scoring data
- Do NOT express certainty beyond what the score supports
- Return ONLY a JSON list of strings: ["reason1", "reason2", ...]
```

---

### 7.10 Evaluator Node

**Purpose:** Measure the quality of the final recommendation. This is the evaluation layer, not part of the recommendation workflow. All scores are logged to Langfuse.

**Model:** Gemma 4 (hallucination check) + deterministic checks
**Input:** Full `state` (all fields)
**Output:** `state.eval_scores`, logs to Langfuse

**Evaluation Dimensions:**

| Dimension | Type | Description |
|-----------|------|-------------|
| `task_success` | Deterministic | Did the agent produce a recommendation (not null)? 0 or 1 |
| `plan_completeness` | Deterministic | Are all required fields in the output schema present? 0-1 score |
| `tool_coverage` | Deterministic | Were the planned tools actually called? 0-1 score |
| `grounding_score` | LLM (Gemma 4) | Does the reasoning reference actual retrieved data? 0-1 float |
| `hallucination_flag` | LLM (Gemma 4) | Did any reasoning claim contradict the scoring data? boolean |
| `confidence_calibration` | Deterministic | Is confidence consistent with the top score? 0-1 score |
| `trace_integrity` | Deterministic | Does execution_trace have ≥ 4 entries? 0-1 |

**Grounding / Hallucination Check Prompt:**

```
You are an evaluation judge. Given the scoring data (ground truth) and the generated reasoning (claims),
check if any claim in the reasoning is NOT supported by or contradicts the scoring data.

Ground truth: {scoring_breakdown}
Generated reasoning: {reasoning}

Return ONLY JSON:
{
  "grounding_score": float 0-1,
  "hallucination_detected": boolean,
  "hallucination_detail": string or null
}
```

**Langfuse Logging:**

```python
langfuse.score(
    trace_id=state["thread_id"],
    name="task_success",
    value=eval_scores["task_success"]
)
langfuse.score(
    trace_id=state["thread_id"],
    name="grounding_score",
    value=eval_scores["grounding_score"]
)
langfuse.score(
    trace_id=state["thread_id"],
    name="hallucination_flag",
    value=int(eval_scores["hallucination_detected"])
)
# ... repeat for all dimensions
```

---

## 8. Pydantic Data Models

### 8.1 Request / Response Models

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

# ─── Request ──────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    user_request: str = Field(..., min_length=5, max_length=2000)
    thread_id: Optional[str] = None  # for follow-up queries in same session

# ─── Recommendation ───────────────────────────────────────────────────
class Recommendation(BaseModel):
    plan_name: str
    network: str
    price_range: List[int]  # [min, max]

# ─── Comparison Matrix ────────────────────────────────────────────────
class ComparisonMatrix(BaseModel):
    packages: List[str]
    dimensions: dict
    recommendation: Optional[str]
    recommendation_reason: Optional[str]

# ─── Main Response ────────────────────────────────────────────────────
class AgentResponse(BaseModel):
    user_request: str
    thread_id: str
    plan: dict  # {steps: List[str]}
    tools_used: List[str]
    recommendation: Optional[Recommendation]
    comparison_matrix: Optional[ComparisonMatrix]
    reasoning: Optional[List[str]]
    confidence: str
    execution_trace: List[str]
    fallback_or_risk_note: Optional[str]
    requires_clarification: bool = False
    clarification_question: Optional[str] = None
    eval_scores: Optional[dict] = None
    latency_ms: Optional[int] = None

# ─── Trace Response ───────────────────────────────────────────────────
class TraceResponse(BaseModel):
    trace_id: str
    langfuse_url: str
    steps: List[dict]
    scores: dict
    total_latency_ms: int
    token_usage: dict

# ─── Health Response ──────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    services: dict  # {ollama: bool, chromadb: bool, langfuse: bool, redis: bool}
    version: str
```

---

## 9. MCP Server Design (Antigravity IDE)

### 9.1 What MCP Servers Do Here

MCP (Model Context Protocol) servers expose the agent's tools as callable functions to the **Antigravity IDE**'s Claude Sonnet 4.6 agent window. This means:

- During development, the Antigravity agent can directly invoke `retrieval_tool`, `scoring_tool`, etc. from the IDE
- During demo, the evaluator can run individual tools without going through the full FastAPI stack
- MCP servers are the bridge between the Antigravity AI assistant and the Python tool implementations

### 9.2 MCP Server Architecture

Each tool is a **separate MCP server** implemented as a **Python stdio MCP server** using the `mcp` SDK. They are registered in the Antigravity MCP config file.

```
project/
├── mcp_servers/
│   ├── retrieval_mcp_server.py       # exposes retrieval_tool
│   ├── scoring_mcp_server.py         # exposes scoring_tool
│   ├── comparison_mcp_server.py      # exposes comparison_tool
│   ├── validation_mcp_server.py      # exposes validator
│   └── agent_runner_mcp_server.py    # exposes full graph run
└── .antigravity/
    └── mcp_servers.json              # Antigravity MCP registration
```

### 9.3 `.antigravity/mcp_servers.json`

```json
{
  "mcpServers": {
    "retrieval_tool": {
      "command": "python",
      "args": ["mcp_servers/retrieval_mcp_server.py"],
      "env": {
        "CHROMADB_HOST": "localhost",
        "CHROMADB_PORT": "8001",
        "OLLAMA_HOST": "http://localhost:11434"
      }
    },
    "scoring_tool": {
      "command": "python",
      "args": ["mcp_servers/scoring_mcp_server.py"],
      "env": {}
    },
    "comparison_tool": {
      "command": "python",
      "args": ["mcp_servers/comparison_mcp_server.py"],
      "env": {}
    },
    "validation_tool": {
      "command": "python",
      "args": ["mcp_servers/validation_mcp_server.py"],
      "env": {}
    },
    "agent_runner": {
      "command": "python",
      "args": ["mcp_servers/agent_runner_mcp_server.py"],
      "env": {
        "FASTAPI_URL": "http://localhost:8000"
      }
    }
  }
}
```

### 9.4 Example MCP Server Implementation: `scoring_mcp_server.py`

```python
#!/usr/bin/env python3
"""
MCP Server: scoring_tool
Exposes the deterministic scoring engine to the Antigravity IDE agent.
"""
import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from agent.tools.scoring_tool import run_scoring

app = Server("scoring_tool")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="score_packages",
            description=(
                "Apply deterministic insurance package scoring rules. "
                "Given retrieved packages and customer entities, returns each package "
                "with a score (0-100) and reasoning breakdown."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "packages": {
                        "type": "array",
                        "description": "List of retrieved package objects",
                        "items": {"type": "object"}
                    },
                    "entities": {
                        "type": "object",
                        "description": "Extracted customer entities (industry, region, budget, priority, etc.)",
                        "properties": {
                            "industry":          {"type": "string"},
                            "region":            {"type": "string"},
                            "budget":            {"type": "string", "enum": ["low", "medium", "high"]},
                            "priority":          {"type": "string"},
                            "dependents_ratio":  {"type": "number"}
                        }
                    }
                },
                "required": ["packages", "entities"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "score_packages":
        results = run_scoring(
            packages=arguments["packages"],
            entities=arguments["entities"]
        )
        return [types.TextContent(type="text", text=json.dumps(results, indent=2))]
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

### 9.5 Agent Runner MCP Server

This is the most powerful MCP server — it exposes the entire LangGraph agent as a single callable tool. The Antigravity agent can run the full pipeline from the IDE with a single call.

```python
@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="run_insurance_agent",
            description=(
                "Run the full agentic insurance advisor pipeline. "
                "Accepts a user request and returns a complete structured recommendation "
                "with execution trace and confidence score."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "user_request": {
                        "type": "string",
                        "description": "The insurance plan query from the user"
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Optional session thread ID for conversation continuity"
                    }
                },
                "required": ["user_request"]
            }
        )
    ]
```

### 9.6 How to Use in Antigravity

Once MCP servers are registered, the Antigravity Claude agent can:

```
# In the Antigravity agent window:
User: "Test the scoring tool with a healthcare company in Riyadh with medium budget"

Claude: [calls score_packages MCP tool with appropriate args]
        [receives structured scoring results]
        [explains the scores and reasoning]
```

This enables the developer to debug individual tools without running the full system.

---

## 10. FastAPI Backend Specification

### 10.1 Application Structure

```
backend/
├── main.py                    # FastAPI app + lifespan
├── routers/
│   ├── query.py               # POST /api/v1/query
│   ├── trace.py               # GET /api/v1/trace/{trace_id}
│   ├── health.py              # GET /api/v1/health
│   └── eval.py                # POST /api/v1/eval/run (trigger eval harness)
├── agent/
│   ├── graph.py               # LangGraph compiled graph
│   ├── state.py               # AgentState TypedDict
│   ├── nodes/
│   │   ├── intent_parser.py
│   │   ├── validator.py
│   │   ├── planner.py
│   │   ├── output_composer.py
│   │   └── evaluator.py
│   ├── tools/
│   │   ├── retrieval_tool.py
│   │   ├── scoring_tool.py
│   │   ├── comparison_tool.py
│   │   ├── fallback_tool.py
│   │   └── clarifier.py
│   └── routing.py             # conditional edge functions
├── clients/
│   ├── ollama_client.py       # Ollama HTTP wrapper
│   ├── chroma_client.py       # ChromaDB client + seed data
│   ├── langfuse_client.py     # Langfuse tracer singleton
│   └── redis_client.py        # Redis wrapper
├── models/
│   └── schemas.py             # All Pydantic models
├── config.py                  # Settings (pydantic-settings)
└── tests/
    ├── test_scenarios.py      # 5 visible + edge case tests
    └── eval_harness.py        # Evaluation run runner
```

### 10.2 API Endpoints

#### `POST /api/v1/query`

```
Request:
  Content-Type: application/json
  Body: QueryRequest

Response:
  200 OK → AgentResponse
  400 Bad Request → {detail: "Invalid request"}
  422 Unprocessable Entity → Pydantic validation error
  503 Service Unavailable → {detail: "Ollama or ChromaDB not reachable"}

Processing:
  1. Validate with Pydantic
  2. Check Redis cache (key = hash(user_request))
  3. Generate thread_id (UUID4)
  4. Run compiled_graph.ainvoke(initial_state, config={"thread_id": thread_id})
  5. Map final AgentState → AgentResponse
  6. Cache result in Redis (TTL = 300s)
  7. Return AgentResponse
```

#### `GET /api/v1/trace/{trace_id}`

```
Response:
  200 OK → TraceResponse (fetched from Langfuse API)
  404 Not Found → {detail: "Trace not found"}
```

#### `GET /api/v1/health`

```
Response:
  200 OK → HealthResponse
  Checks: Ollama (/api/tags), ChromaDB (/api/v1/heartbeat), 
          Langfuse (/api/public/health), Redis (PING)
```

#### `POST /api/v1/eval/run`

```
Request:
  Body: {scenario_ids: List[str] or "all"}

Response:
  200 OK → {
    total_runs: int,
    passed: int,
    failed: int,
    results: List[{scenario_id, passed, scores, trace_id}]
  }

Behavior: Runs the eval harness against predefined scenarios, 
          logs all to Langfuse, returns summary
```

### 10.3 CORS & Middleware

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
)

# Request ID middleware (for tracing correlation)
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
```

### 10.4 Configuration (`config.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Ollama
    OLLAMA_HOST:          str = "http://ollama:11434"
    OLLAMA_MODEL:         str = "gemma4:31b-cloud"
    OLLAMA_EMBED_MODEL:   str = "nomic-embed-text"
    OLLAMA_TIMEOUT:       int = 60

    # ChromaDB
    CHROMA_HOST:          str = "chromadb"
    CHROMA_PORT:          int = 8001

    # Langfuse
    LANGFUSE_HOST:        str = "http://langfuse:3000"
    LANGFUSE_PUBLIC_KEY:  str
    LANGFUSE_SECRET_KEY:  str

    # Redis
    REDIS_URL:            str = "redis://redis:6379/0"
    CACHE_TTL_SECONDS:    int = 300

    # Agent
    MAX_RETRIES:          int = 2
    MIN_CONFIDENCE_THRESHOLD: float = 0.4

    class Config:
        env_file = ".env"
```

---

## 11. Next.js Frontend Specification

### 11.1 Tech Stack

- **Next.js 15** (App Router)
- **React 19**
- **Tailwind CSS v4**
- **shadcn/ui** for components
- **Zustand** for client state
- **React Query (TanStack)** for data fetching

### 11.2 Page Structure

```
frontend/
├── app/
│   ├── layout.tsx             # Root layout (dark theme, Inter font)
│   ├── page.tsx               # Main chat page
│   └── trace/[id]/page.tsx    # Trace detail viewer
├── components/
│   ├── ChatInterface.tsx       # Main chat area
│   ├── MessageBubble.tsx       # User / agent message bubbles
│   ├── RecommendationCard.tsx  # Structured recommendation display
│   ├── ExecutionTrace.tsx      # Collapsible trace panel
│   ├── ConfidenceBadge.tsx     # high/medium/low badge
│   ├── ComparisonTable.tsx     # Comparison matrix table
│   ├── ClarificationPrompt.tsx # Follow-up question display
│   └── HealthStatus.tsx        # System health indicator (top bar)
├── lib/
│   ├── api.ts                  # Typed API client (wraps fetch)
│   └── types.ts                # TypeScript types matching Pydantic models
└── store/
    └── chat.ts                 # Zustand store (messages, loading, thread_id)
```

### 11.3 Key Component: `RecommendationCard`

Renders the structured agent response in a clean card format with:
- **Plan name + network badge** (color-coded: A=green, B=blue, C=gray)
- **Price range** with currency (SAR)
- **Confidence badge** (color-coded)
- **Reasoning list** (3-5 bullets)
- **Risk note** (amber warning box, always visible)
- **Execution trace** (collapsible, monospace)
- **Tools used** (icon chips)

### 11.4 State Management

```typescript
// store/chat.ts
interface ChatStore {
  messages: Message[]
  threadId: string | null
  isLoading: boolean
  addMessage: (msg: Message) => void
  setThreadId: (id: string) => void
  setLoading: (loading: boolean) => void
  clearChat: () => void
}
```

### 11.5 API Client

```typescript
// lib/api.ts
export async function query(request: QueryRequest): Promise<AgentResponse> {
  const res = await fetch(`${API_BASE}/api/v1/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getTrace(traceId: string): Promise<TraceResponse> {
  const res = await fetch(`${API_BASE}/api/v1/trace/${traceId}`)
  if (!res.ok) throw new Error("Trace not found")
  return res.json()
}
```

---

## 12. Evaluation Strategy (Langfuse)

### 12.1 Langfuse Deployment

Langfuse runs as a **self-hosted Docker Compose service** with its own PostgreSQL database. No external cloud connection required.

Services in compose:
- `langfuse` — main application (port 3001, mapped from internal 3000)
- `langfuse-postgres` — dedicated PostgreSQL 16 instance for Langfuse

### 12.2 Tracing Structure

Every agent run produces a **Langfuse trace** with this hierarchy:

```
Trace: {thread_id}
  ├── Span: intent_parser        [latency, input=user_request, output=entities]
  ├── Span: validator            [latency, input=entities, output=missing_fields]
  ├── Span: planner              [latency, input=entities+qt, output=plan]
  ├── Span: retrieval_tool       [latency, input=query, output=results_count]
  │     └── Generation: embed   [model=nomic-embed-text, tokens=N]
  ├── Span: scoring_tool         [latency, input=packages+entities, output=scores]
  ├── Span: comparison_tool      [latency] (conditional)
  ├── Span: output_composer      [latency, input=scores, output=recommendation]
  │     └── Generation: gemma4:31b-cloud  [model=gemma4:31b-cloud, tokens=N, prompt, completion]
  ├── Span: evaluator            [latency]
  │     └── Generation: gemma4:31b-cloud  [model=gemma4:31b-cloud for hallucination check]
  └── Scores:
        task_success, plan_completeness, tool_coverage,
        grounding_score, hallucination_flag, confidence_calibration, trace_integrity
```

### 12.3 Evaluation Harness

**Location:** `backend/tests/eval_harness.py`

The harness runs all 5 visible scenarios plus additional edge cases, computes aggregate scores, and produces a JSON report.

**Test Scenarios Defined in Harness:**

```python
EVAL_SCENARIOS = [
    {
        "id": "S1_balanced",
        "user_request": "Recommend the best plan for a healthcare company in Riyadh.",
        "expected": {
            "query_type": "recommend",
            "recommendation_in": ["Standard", "Premium"],
            "must_mention": ["healthcare", "risk", "Riyadh"],
            "confidence_min": "medium",
            "task_success": 1,
        }
    },
    {
        "id": "S2_cheapest",
        "user_request": "Give me the cheapest acceptable option for a construction customer in Jeddah.",
        "expected": {
            "query_type": "cheapest",
            "recommendation_in": ["Basic", "Standard"],
            "confidence_min": "medium",
            "task_success": 1,
        }
    },
    {
        "id": "S3_compare",
        "user_request": "Compare Standard and Premium for a retail customer in Dammam.",
        "expected": {
            "query_type": "compare",
            "comparison_matrix_present": True,
            "task_success": 1,
        }
    },
    {
        "id": "S4_missing_info",
        "user_request": "Recommend a plan for my company.",
        "expected": {
            "requires_clarification": True,
            "recommendation": None,
            "task_success": 1,  # clarification IS task success for this scenario
        }
    },
    {
        "id": "S5_explain",
        "user_request": "Why did you choose that recommendation?",
        "expected": {
            "reasoning_present": True,
            "execution_trace_length_min": 3,
            "task_success": 1,
        }
    },
    # Edge cases
    {
        "id": "E1_conflicting_constraints",
        "user_request": "I want the best possible coverage but my budget is very low.",
        "expected": {
            "recommendation_in": ["Basic", "Standard"],
            "fallback_note_present": True,
            "confidence_max": "medium",
        }
    },
    {
        "id": "E2_unsupported",
        "user_request": "What is the weather in Riyadh today?",
        "expected": {
            "query_type": "unsupported",
            "recommendation": None,
            "fallback_note_present": True,
        }
    },
    {
        "id": "E3_empty_retrieval",
        "user_request": "Recommend a plan for a mining company in Tabuk.",
        "expected": {
            "fallback_note_present": True,
            "confidence_max": "medium",
        }
    },
]
```

### 12.4 Evaluation Report Format

```json
{
  "run_id": "eval_20260407_143022",
  "total_scenarios": 8,
  "passed": 7,
  "failed": 1,
  "pass_rate": 0.875,
  "avg_scores": {
    "task_success":           0.875,
    "grounding_score":        0.91,
    "hallucination_flag":     0.0,
    "plan_completeness":      0.94,
    "confidence_calibration": 0.88
  },
  "results": [
    {
      "scenario_id": "S1_balanced",
      "passed": true,
      "recommendation": "Standard",
      "confidence": "high",
      "scores": { "task_success": 1, "grounding_score": 0.95 },
      "latency_ms": 2340,
      "trace_id": "abc123",
      "langfuse_url": "http://localhost:3001/trace/abc123"
    }
  ]
}
```

---

## 13. Ollama + Gemma 4 (gemma4:31b-cloud) Integration

### 13.1 Model Selection Justification

| Criterion | gemma4:31b-cloud (Ollama) | Why It Wins |
|-----------|-----------------|-------------|
| Local execution | ✅ Fully local, no API key | No data egress; offline capable |
| Instruction following | ✅ Strong JSON output | Required for structured outputs |
| Context length | ✅ 128K tokens | More than enough for this workflow |
| Cost | ✅ $0 per inference | Critical for multiple eval runs |
| Reasoning quality | ✅ Sufficient for planning + narrative | Not frontier-model tasks; Gemma 4 is appropriate |
| Multilingual | Not required (English only) | Non-factor |

**Where gemma4:31b-cloud is used:**
- `intent_parser_node` — extract entities from free text
- `planner_node` — ambiguous plan selection (fallback from static table)
- `output_composer_node` — reasoning narrative generation
- `evaluator_node` — grounding / hallucination check

**Where gemma4:31b-cloud is NOT used (deterministic):**
- `validator_node` — pure Python logic
- `scoring_tool_node` — pure Python rules engine
- `comparison_tool_node` — table construction (Gemma 4 optional for narrative)
- `fallback_node` — template-based responses
- `clarifier_node` — template-based questions (optional Gemma 4 for phrasing)

### 13.2 Ollama Client Wrapper

```python
# clients/ollama_client.py
import httpx
import json
from typing import Optional

class OllamaClient:
    def __init__(self, host: str, model: str, timeout: int = 60):
        self.host = host
        self.model = model
        self.client = httpx.AsyncClient(timeout=timeout)

    async def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: float = 0.1,       # low temp for structured outputs
        max_tokens: int = 2048,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        r = await self.client.post(f"{self.host}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"]

    async def embed(self, text: str) -> list[float]:
        r = await self.client.post(
            f"{self.host}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text}
        )
        r.raise_for_status()
        return r.json()["embedding"]
```

### 13.3 Model Warm-Up

On FastAPI startup, pre-warm Ollama to avoid cold-start latency on first request:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up Ollama
    await ollama_client.chat([{"role": "user", "content": "ping"}])  # warms gemma4:31b-cloud
    # Seed ChromaDB if empty
    await seed_chromadb_if_empty()
    yield
    # Cleanup on shutdown
    await ollama_client.client.aclose()
```

---

## 14. Docker Compose Infrastructure

### 14.1 Service Map

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `ollama` | `ollama/ollama:latest` | 11434 | LLM + embedding inference |
| `ollama-init` | `curlimages/curl` | — | Pull gemma4 + nomic-embed-text on first run |
| `chromadb` | `chromadb/chroma:latest` | 8001 | Vector store (persistent) |
| `redis` | `redis:7-alpine` | 6379 | Caching + rate limiting |
| `langfuse` | `langfuse/langfuse:2` | 3001 | Observability UI + API |
| `langfuse-postgres` | `postgres:16-alpine` | 5433 | Langfuse DB |
| `backend` | `./backend` (custom) | 8000 | FastAPI agent backend |
| `frontend` | `./frontend` (custom) | 3000 | Next.js chat UI |

### 14.2 `docker-compose.yml`

```yaml
version: "3.9"

services:

  ollama:
    image: ollama/ollama:latest
    container_name: insurance-ollama
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 10s
      timeout: 5s
      retries: 10

  ollama-init:
    image: curlimages/curl:latest
    container_name: insurance-ollama-init
    depends_on:
      ollama:
        condition: service_healthy
    entrypoint: >
      sh -c "
        curl -X POST http://ollama:11434/api/pull -d '{\"name\": \"gemma4:31b-cloud\"}' &&
        curl -X POST http://ollama:11434/api/pull -d '{\"name\": \"nomic-embed-text\"}'
      "
    restart: "no"

  chromadb:
    image: chromadb/chroma:latest
    container_name: insurance-chromadb
    volumes:
      - chroma_data:/chroma/chroma
    ports:
      - "8001:8000"
    environment:
      CHROMA_SERVER_HOST: "0.0.0.0"
      CHROMA_SERVER_HTTP_PORT: "8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 5s
      timeout: 3s
      retries: 10

  redis:
    image: redis:7-alpine
    container_name: insurance-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  langfuse-postgres:
    image: postgres:16-alpine
    container_name: insurance-langfuse-db
    environment:
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse
      POSTGRES_DB: langfuse
    volumes:
      - langfuse_postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse"]
      interval: 5s
      timeout: 3s
      retries: 10

  langfuse:
    image: langfuse/langfuse:2
    container_name: insurance-langfuse
    depends_on:
      langfuse-postgres:
        condition: service_healthy
    ports:
      - "3001:3000"
    environment:
      DATABASE_URL: "postgresql://langfuse:langfuse@langfuse-postgres:5432/langfuse"
      NEXTAUTH_SECRET: "local-dev-secret-change-in-prod"
      NEXTAUTH_URL: "http://localhost:3001"
      TELEMETRY_ENABLED: "false"
      LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES: "false"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/api/public/health"]
      interval: 10s
      timeout: 5s
      retries: 15

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: insurance-backend
    depends_on:
      ollama:
        condition: service_healthy
      chromadb:
        condition: service_healthy
      redis:
        condition: service_healthy
      langfuse:
        condition: service_healthy
    ports:
      - "8000:8000"
    environment:
      OLLAMA_HOST: "http://ollama:11434"
      CHROMA_HOST: "chromadb"
      CHROMA_PORT: "8000"
      REDIS_URL: "redis://redis:6379/0"
      LANGFUSE_HOST: "http://langfuse:3000"
      LANGFUSE_PUBLIC_KEY: "${LANGFUSE_PUBLIC_KEY}"
      LANGFUSE_SECRET_KEY: "${LANGFUSE_SECRET_KEY}"
    volumes:
      - ./backend:/app
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 10s
      timeout: 5s
      retries: 10

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: insurance-frontend
    depends_on:
      backend:
        condition: service_healthy
    ports:
      - "3000:3000"
    environment:
      NEXT_PUBLIC_API_URL: "http://localhost:8000"

volumes:
  ollama_data:
  chroma_data:
  redis_data:
  langfuse_postgres_data:
```

### 14.3 Backend Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### 14.4 `requirements.txt`

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
langgraph==0.2.55
langchain-core==0.3.15
pydantic==2.9.2
pydantic-settings==2.6.1
chromadb==0.5.18
langfuse==2.57.0
httpx==0.27.2
redis==5.1.1
python-dotenv==1.0.1
mcp==1.1.0
```

---

## 15. Repository Structure

```
insurance-advisor-agent/
│
├── docker-compose.yml
├── .env.example                    # Template for required env vars
├── .env                            # NOT committed — local secrets
├── README.md                       # Setup instructions
├── ARCHITECTURE.md                 # Architecture narrative + diagram
│
├── .antigravity/
│   └── mcp_servers.json            # Antigravity MCP registration
│
├── mcp_servers/                    # Antigravity MCP servers
│   ├── retrieval_mcp_server.py
│   ├── scoring_mcp_server.py
│   ├── comparison_mcp_server.py
│   ├── validation_mcp_server.py
│   └── agent_runner_mcp_server.py
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── config.py
│   ├── routers/
│   │   ├── query.py
│   │   ├── trace.py
│   │   ├── health.py
│   │   └── eval.py
│   ├── agent/
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── routing.py
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── intent_parser.py
│   │   │   ├── validator.py
│   │   │   ├── planner.py
│   │   │   ├── clarifier.py
│   │   │   ├── fallback.py
│   │   │   ├── output_composer.py
│   │   │   └── evaluator.py
│   │   └── tools/
│   │       ├── __init__.py
│   │       ├── retrieval_tool.py
│   │       ├── scoring_tool.py
│   │       └── comparison_tool.py
│   ├── clients/
│   │   ├── ollama_client.py
│   │   ├── chroma_client.py
│   │   ├── langfuse_client.py
│   │   └── redis_client.py
│   ├── data/
│   │   └── seed_data.py            # ChromaDB seed (packages, rules, snippets)
│   ├── models/
│   │   └── schemas.py
│   └── tests/
│       ├── conftest.py
│       ├── test_scenarios.py       # 5 visible + 3 edge cases
│       ├── test_scoring_tool.py    # Unit tests for deterministic logic
│       ├── test_intent_parser.py
│       └── eval_harness.py
│
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── next.config.ts
│   ├── tailwind.config.ts
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── trace/[id]/page.tsx
│   ├── components/
│   │   ├── ChatInterface.tsx
│   │   ├── MessageBubble.tsx
│   │   ├── RecommendationCard.tsx
│   │   ├── ExecutionTrace.tsx
│   │   ├── ConfidenceBadge.tsx
│   │   ├── ComparisonTable.tsx
│   │   ├── ClarificationPrompt.tsx
│   │   └── HealthStatus.tsx
│   ├── lib/
│   │   ├── api.ts
│   │   └── types.ts
│   └── store/
│       └── chat.ts
│
└── docs/
    ├── architecture_diagram.png    # Export of architecture diagram
    ├── technical_report.md         # 5-page technical report (per deliverable req)
    └── evaluation_report.json      # ≥5 evaluated runs (per deliverable req)
```

---

## 16. Implementation Sequencing & Milestones

### Phase 0: Infrastructure Setup (Day 1 — ~4 hours)

**Goal:** All services running via Docker Compose

1. Write `docker-compose.yml` with all 8 services
2. Write `backend/Dockerfile` and `frontend/Dockerfile`
3. Verify `docker compose up` brings all services healthy
4. Verify Ollama responds to `/api/tags`
5. Verify ChromaDB responds to `/api/v1/heartbeat`
6. Verify Langfuse UI accessible at `localhost:3001`
7. Create initial Langfuse project, capture `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`, add to `.env`

**Checkpoint:** `curl localhost:8000/api/v1/health` returns all services `true`

---

### Phase 1: Data Layer (Day 1-2 — ~3 hours)

**Goal:** ChromaDB seeded with all knowledge base data

1. Write `backend/data/seed_data.py` with all package catalog, benchmark rules, knowledge snippets, customer profiles from PDF Section 5
2. Embed each document using `nomic-embed-text` via Ollama
3. Insert into 4 ChromaDB collections with metadata
4. Write `test_retrieval.py` to verify queries return expected packages

**Checkpoint:** Retrieval for "healthcare Riyadh" returns Standard and Premium packages

---

### Phase 2: Agent State & Graph Skeleton (Day 2 — ~4 hours)

**Goal:** LangGraph graph compiles and runs end-to-end with stub nodes

1. Define `AgentState` TypedDict in `state.py`
2. Write stub implementations for all 9 nodes (return state unchanged + append to trace)
3. Define all edges and conditional routing in `graph.py`
4. Write `routing.py` with `route_after_validation` and `route_after_scoring`
5. Compile graph with `MemorySaver`
6. Write smoke test: invoke graph with a simple query, verify it reaches END

**Checkpoint:** Graph executes all nodes in correct order for a simple query

---

### Phase 3: Core Nodes Implementation (Day 2-3 — ~8 hours)

**Goal:** All nodes implemented with full logic

Order of implementation:
1. `intent_parser_node` — Ollama + JSON parse with retry
2. `validator_node` — deterministic required fields check
3. `planner_node` — static plan table + Ollama fallback
4. `retrieval_tool_node` — ChromaDB queries with metadata filter
5. `scoring_tool_node` — full rules engine (most critical, test thoroughly)
6. `comparison_tool_node` — matrix builder
7. `output_composer_node` — deterministic struct + Ollama reasoning
8. `evaluator_node` — all 7 eval dimensions + Langfuse logging
9. `fallback_node` + `clarifier_node` — template responses

**Checkpoint:** All 5 visible test scenarios return correct structured output

---

### Phase 4: FastAPI Backend (Day 3 — ~3 hours)

**Goal:** REST API wrapping the agent graph

1. Write `main.py` with lifespan, middleware, router inclusion
2. Implement `POST /api/v1/query` with Redis caching
3. Implement `GET /api/v1/health`
4. Implement `GET /api/v1/trace/{trace_id}` via Langfuse API
5. Implement `POST /api/v1/eval/run`
6. Write Pydantic schemas for all request/response models

**Checkpoint:** `curl -X POST localhost:8000/api/v1/query -d '{"user_request":"..."}'` returns valid JSON

---

### Phase 5: MCP Servers (Day 3-4 — ~3 hours)

**Goal:** All tools accessible from Antigravity IDE

1. Install `mcp` Python SDK
2. Implement `scoring_mcp_server.py` first (simplest, no external deps)
3. Implement `retrieval_mcp_server.py`
4. Implement `comparison_mcp_server.py`
5. Implement `validation_mcp_server.py`
6. Implement `agent_runner_mcp_server.py`
7. Write `.antigravity/mcp_servers.json`
8. Test in Antigravity: each tool callable from agent window

**Checkpoint:** Antigravity Claude can call `score_packages` and receive structured results

---

### Phase 6: Next.js Frontend (Day 4 — ~4 hours)

**Goal:** Functional chat UI

1. Scaffold Next.js 15 app with Tailwind + shadcn/ui
2. Implement Zustand store
3. Build `ChatInterface` with input box and message history
4. Build `RecommendationCard` (most complex component)
5. Build `ExecutionTrace` (collapsible)
6. Build `ConfidenceBadge` and `ComparisonTable`
7. Implement `ClarificationPrompt` for incomplete queries
8. Connect to FastAPI via typed `api.ts` client

**Checkpoint:** Full end-to-end flow works through the UI

---

### Phase 7: Evaluation & Documentation (Day 4-5 — ~4 hours)

**Goal:** All deliverables complete

1. Run eval harness against all 8 scenarios, generate `evaluation_report.json`
2. Capture Langfuse screenshots of traces and scores
3. Write 5-page technical report (`docs/technical_report.md`)
4. Export architecture diagram (`docs/architecture_diagram.png`)
5. Write `README.md` with complete setup instructions
6. Record demo walkthrough video

**Checkpoint:** All 13 deliverables from PDF Section 13 are present

---

## 17. Edge Cases & Failure Handling

| Edge Case | Detection | Handling |
|-----------|----------|---------|
| **Conflicting constraints** (low budget + best coverage) | Scoring tool: budget penalty on Premium | Recommend Standard with fallback note explaining the conflict; confidence = medium |
| **Unsupported question** (weather, unrelated domain) | intent_parser: query_type = unsupported | Route to fallback immediately; never attempt retrieval |
| **Ambiguous wording** ("give me something good") | validator: industry + region missing | Route to clarifier; ask for industry and region |
| **Incomplete data fields** (no region provided) | validator: region in missing_fields | Route to clarifier with specific question |
| **Empty retrieval results** (industry not in KB) | retrieval_tool: empty results | Retry with expanded query; if still empty → fallback with note |
| **Tool failure / Ollama timeout** | try/except in each node | Set state.error, route to fallback; log error to Langfuse |
| **Model overconfidence** | evaluator: confidence_calibration check | If score < 60 but confidence = high, downgrade confidence; log discrepancy |
| **Partial retrieval** (only rules returned, no packages) | retrieval_tool: packages list empty | Retry retrieval; if still no packages → fallback |
| **Repeated tool calls without improvement** | planner: check retry_count | MAX_RETRIES = 2; after limit → fallback unconditionally |
| **Invalid JSON from Ollama** | safe_json_parse: retry 2x with stricter prompt | After 2 retries → route to fallback |
| **ChromaDB unavailable** | health check on startup | Fail fast at startup with clear error; do not start backend |
| **Ollama model not loaded** | health check on startup | Trigger model pull via API; retry 3x before fail |

---

## 18. Non-Functional Requirements

| Requirement | Target | Implementation |
|-------------|--------|---------------|
| **Latency (happy path)** | < 8 seconds end-to-end | Warm Ollama, Redis cache, async throughout |
| **Latency (cache hit)** | < 200ms | Redis TTL 5 min |
| **Reproducibility** | `docker compose up` from cold start in < 5 min | All services containerized; model pull automated |
| **Type safety** | 100% typed Python (mypy strict) | Pydantic v2 everywhere; TypeScript frontend |
| **Test coverage** | ≥ 80% on deterministic tools | pytest on scoring_tool, validator, routing |
| **Structured output compliance** | 100% of responses match schema | Pydantic validation on every response |
| **No hallucinated certainty** | confidence = "none" on all unsupported queries | Fallback node always sets confidence to "none" when recommendation is null |
| **Trace completeness** | Every response has ≥ 4 execution_trace entries | Validated in evaluator_node |
| **Langfuse availability** | Non-blocking (agent runs even if Langfuse is down) | All Langfuse calls wrapped in try/except with no-op fallback |

---

## 19. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Gemma 4 produces inconsistent JSON | Medium | High | Retry logic (2x) + strict prompt + schema validation |
| Ollama is slow on CPU-only machine | High | Medium | Use `gemma4:2b` quantized variant; pre-warm on startup; cache results |
| ChromaDB retrieval returns irrelevant results | Low | Medium | Metadata filters narrow the search; deterministic scoring re-ranks |
| Langfuse breaks traces under load | Low | Low | Langfuse calls are fire-and-forget; don't block agent response |
| Frontend/Backend CORS mismatch | Low | Medium | CORS configured in FastAPI; verified in Phase 4 |
| Docker volume data loss on rebuild | Medium | Medium | Named volumes persist; use `docker compose up` not `down -v` |
| MCP server stdio crashes | Low | Low | Each MCP server is independent; Antigravity restarts crashed servers |
| Model pull fails on first run | Medium | High | ollama-init service handles pull with retry; README documents manual pull command |

---

## 20. Appendix: Expected Outputs for All 5 Test Scenarios

### Scenario 1 — Balanced Recommendation

**Input:** `"Recommend the best plan for a healthcare company in Riyadh."`

```json
{
  "user_request": "Recommend the best plan for a healthcare company in Riyadh.",
  "plan": {
    "steps": ["retrieve relevant packages and rules", "apply scoring rules", "compose recommendation"]
  },
  "tools_used": ["retrieval_tool", "scoring_tool", "output_composer"],
  "recommendation": {
    "plan_name": "Standard",
    "network": "B",
    "price_range": [6000, 7500]
  },
  "reasoning": [
    "Healthcare is classified as high-risk industry, requiring at minimum Network B coverage.",
    "Riyadh has the highest regional medical cost pressure, making Premium a viable but not mandatory upgrade.",
    "No explicit budget or priority stated; Standard offers the best balance of risk coverage and cost control.",
    "Standard scored 80/100 vs Premium's 65/100 due to budget uncertainty favoring moderate spend."
  ],
  "confidence": "high",
  "execution_trace": [
    "intent_parser: query_type=recommend, entities={industry: healthcare, region: Riyadh}",
    "validator: all required fields present",
    "planner: selected plan=[retrieval_tool, scoring_tool, output_composer]",
    "retrieval_tool: returned 3 packages, 5 rules",
    "scoring_tool: Standard=80, Premium=65, Basic=35",
    "output_composer: selected Standard, composed 4 reasoning bullets"
  ],
  "fallback_or_risk_note": "Standard is recommended. Premium is available if the customer has high dependents ratio or specific network A requirements. Basic is not suitable for healthcare.",
  "requires_clarification": false
}
```

---

### Scenario 2 — Cost-Focused Recommendation

**Input:** `"Give me the cheapest acceptable option for a construction customer in Jeddah."`

```json
{
  "user_request": "Give me the cheapest acceptable option for a construction customer in Jeddah.",
  "plan": {
    "steps": ["retrieve relevant packages and cost rules", "apply scoring rules with cost priority", "compose recommendation"]
  },
  "tools_used": ["retrieval_tool", "scoring_tool", "output_composer"],
  "recommendation": {
    "plan_name": "Basic",
    "network": "C",
    "price_range": [4000, 5000]
  },
  "reasoning": [
    "Construction is medium-risk industry — does not require Premium network.",
    "User explicitly requested cheapest acceptable option, triggering cost-priority scoring.",
    "Jeddah has moderate (not high) regional cost pressure, making Basic viable.",
    "Basic scored 82/100 for this cost-priority profile vs Standard's 61/100."
  ],
  "confidence": "medium-high",
  "execution_trace": [
    "intent_parser: query_type=cheapest, entities={industry: construction, region: Jeddah}",
    "validator: all required fields present",
    "planner: selected plan=[retrieval_tool, scoring_tool, output_composer]",
    "retrieval_tool: returned Basic and Standard as top candidates",
    "scoring_tool: Basic=82 (cost priority), Standard=61, Premium=30 (over-budget penalty)",
    "output_composer: selected Basic with cost justification"
  ],
  "fallback_or_risk_note": "Basic is the lowest-cost option but provides Network C which may limit provider access. Standard is available at 6000-7500 SAR for broader network coverage.",
  "requires_clarification": false
}
```

---

### Scenario 3 — Comparison Workflow

**Input:** `"Compare Standard and Premium for a retail customer in Dammam."`

```json
{
  "user_request": "Compare Standard and Premium for a retail customer in Dammam.",
  "plan": {
    "steps": ["retrieve Standard and Premium packages", "apply scoring rules", "build comparison matrix", "compose recommendation"]
  },
  "tools_used": ["retrieval_tool", "scoring_tool", "comparison_tool", "output_composer"],
  "recommendation": {
    "plan_name": "Standard",
    "network": "B",
    "price_range": [6000, 7500]
  },
  "comparison_matrix": {
    "packages": ["Standard", "Premium"],
    "dimensions": {
      "network":       {"Standard": "B", "Premium": "A"},
      "price_range":   {"Standard": "6000–7500 SAR", "Premium": "9000–12000 SAR"},
      "coverage":      {"Standard": "Medium", "Premium": "High"},
      "score":         {"Standard": 78, "Premium": 58},
      "budget_fit":    {"Standard": "Yes (medium budget)", "Premium": "Borderline (medium budget)"},
      "risk_fit":      {"Standard": "Yes (medium-low risk)", "Premium": "Over-spec"}
    },
    "recommendation": "Standard",
    "recommendation_reason": "Retail is medium-low risk. Standard's coverage is appropriate and saves 3000-4500 SAR vs Premium. Premium is only justified if network A hospitals are a stated requirement."
  },
  "reasoning": [
    "Retail is classified medium-low risk — does not warrant Premium unless explicitly required.",
    "Dammam has moderate-to-high cost pressure, making the 3000-4500 SAR saving from Standard significant.",
    "Standard scored 78/100 vs Premium's 58/100; primary deduction on Premium was over-spec penalty."
  ],
  "confidence": "high",
  "execution_trace": [
    "intent_parser: query_type=compare, entities={industry: retail, region: Dammam, compare_packages: [Standard, Premium]}",
    "validator: all required fields present",
    "planner: selected plan=[retrieval_tool, scoring_tool, comparison_tool, output_composer]",
    "retrieval_tool: retrieved Standard and Premium package documents",
    "scoring_tool: Standard=78, Premium=58",
    "comparison_tool: built 6-dimension comparison matrix",
    "output_composer: composed recommendation with matrix"
  ],
  "fallback_or_risk_note": "If the retail company has specific premium hospital requirements or high staff seniority expectations, Premium may be worth the additional cost.",
  "requires_clarification": false
}
```

---

### Scenario 4 — Missing Information

**Input:** `"Recommend a plan for my company."`

```json
{
  "user_request": "Recommend a plan for my company.",
  "plan": {
    "steps": ["parse intent", "validate required fields", "request clarification"]
  },
  "tools_used": [],
  "recommendation": null,
  "reasoning": null,
  "confidence": "none",
  "execution_trace": [
    "intent_parser: query_type=recommend, entities={}, missing=[industry, region]",
    "validator: missing required fields [industry, region]",
    "clarifier: generated clarification question for missing fields"
  ],
  "requires_clarification": true,
  "clarification_question": "To give you an accurate recommendation, I need a few details: (1) What industry is your company in? (e.g., healthcare, construction, retail) (2) Which region are you based in? (e.g., Riyadh, Jeddah, Dammam). Optionally, you can also share your approximate budget level (low/medium/high) and any coverage priorities.",
  "fallback_or_risk_note": "Recommendation cannot be generated without industry and region information."
}
```

---

### Scenario 5 — Explanation Request

**Input:** `"Why did you choose that recommendation?"` (assumes prior context in thread)

```json
{
  "user_request": "Why did you choose that recommendation?",
  "plan": {
    "steps": ["retrieve prior recommendation from thread state", "compose explanation"]
  },
  "tools_used": ["output_composer"],
  "recommendation": null,
  "reasoning": [
    "Standard was selected because healthcare is high-risk, requiring at minimum Network B.",
    "Riyadh's high regional cost made Premium less cost-effective without a stated premium budget.",
    "Standard scored 80/100 vs Premium's 65/100 under the applied scoring rules.",
    "No dependents ratio was provided; if ratio exceeds 0.50, a re-evaluation toward Premium may be warranted."
  ],
  "confidence": "high",
  "execution_trace": [
    "intent_parser: query_type=explain",
    "validator: explain query type, no retrieval needed",
    "planner: selected plan=[output_composer] using cached thread state",
    "output_composer: generated explanation from prior scoring breakdown"
  ],
  "fallback_or_risk_note": "This explanation is based on the scoring rules applied to the prior query in this session. If you provide additional details, the recommendation may change.",
  "requires_clarification": false
}
```

---

*End of PRD — Version 1.0.1*
*Model confirmed: `gemma4:31b-cloud` via Ollama. All references updated.*
*All decisions confirmed via pre-PRD elicitation. No assumptions remain.*
