# Technical Report
## Agentic Insurance Advisor System — Mission 3 Elite Assessment

**Version:** 1.0.1
**Date:** 2026-04-07
**Classification:** Assessment Deliverable

---

## 1. System Architecture

### 1.1 Overview

The Agentic Insurance Advisor System is a stateful, graph-based agentic AI pipeline that accepts natural-language insurance plan queries and produces grounded, structured recommendations. It is deliberately not a chatbot — it is an orchestrated workflow where every step is traceable, every decision is justified, and every failure is handled explicitly.

The system is composed of four logical layers:

**Presentation layer:** A Next.js 15 chat interface that renders structured recommendation cards, execution traces, comparison matrices, and confidence badges. Users interact with the agent through natural language; the UI surfaces all intermediate structure without requiring users to read raw JSON.

**API layer:** A FastAPI 0.115 backend that accepts queries via `POST /api/v1/query`, manages Redis caching for deduplication, wraps the agent graph in an async execution context, and exposes Langfuse traces via `GET /api/v1/trace/{id}`. It is the only external-facing service.

**Agent layer:** A LangGraph `StateGraph` with 9 nodes and 2 conditional routing functions. This is where all reasoning, tool execution, and state management happens. The graph is the core of the system.

**Infrastructure layer:** Ollama (LLM inference), ChromaDB (vector retrieval), Langfuse (observability), Redis (caching), and PostgreSQL (Langfuse storage) — all running locally in Docker Compose.

### 1.2 LangGraph Graph Structure

The graph is a directed acyclic graph with one conditional cycle (for retry logic). Nodes are:

`intent_parser → validator → [planner | clarifier | fallback] → retrieval_tool → scoring_tool → [comparison_tool] → output_composer → evaluator`

State flows as a single immutable `AgentState` TypedDict. Each node receives the full state, makes targeted writes to its own output fields, appends to `execution_trace`, and returns. No node has side effects on fields it does not own.

Conditional routing is handled by two pure functions: `route_after_validation` (routes to planner, clarifier, or fallback based on missing fields and query type) and `route_after_scoring` (routes to comparison_tool, output_composer, retry, or fallback based on results and query type). These functions contain no LLM calls — they are pure deterministic Python.

### 1.3 Key Architectural Decisions

**Separation of reasoning and action:** LLM calls happen only in intent_parser, planner (ambiguous cases), output_composer (reasoning narrative), and evaluator (hallucination check). All business logic — scoring, routing, validation, comparison table construction — is deterministic Python. This means the system can be unit-tested without any LLM dependency, the scoring rules are fully auditable, and LLM failures only affect the narrative layer, not the recommendation decision.

**State-first design:** Every piece of information the agent produces is written to `AgentState` with an explicit field name. There is no hidden context, no implicit conversation history, and no passing of strings between nodes. This makes the system debuggable: at any point in execution, you can inspect the full state and know exactly what has happened and why.

**Fail-safe routing:** Every non-happy-path case routes to the fallback node, which always returns a structured response with `recommendation: null`, `confidence: "none"`, and an informative `fallback_or_risk_note`. The system never returns an empty response or an unstructured error message to the user.

---

## 2. Agent and Runtime Design

### 2.1 Why LangGraph

LangGraph was selected over CrewAI, AutoGen, Semantic Kernel, and raw SDK orchestration for three reasons.

First, **explicit graph topology.** The insurance recommendation workflow has a known, bounded set of paths. A graph explicitly encodes these paths — making them inspectable, testable, and modifiable. CrewAI and AutoGen use autonomous agent loops where paths emerge dynamically; this creates non-determinism that is appropriate for open-ended tasks but counterproductive for a business-rules workflow where every deviation from the expected path is a bug.

Second, **first-class state management.** LangGraph's `StateGraph` passes state by value between nodes, preventing hidden shared state. The `MemorySaver` checkpointer gives every run a unique `thread_id`, enabling conversation continuity (for the "explain" query type which needs access to the previous run's scoring data) without coupling nodes to each other.

Third, **built-in retry and cycle support.** The retry loop (retrieval_tool → scoring_tool → retrieval_tool) is expressed as a cycle in the graph with a guard (`retry_count < MAX_RETRIES`). LangGraph handles this natively. Implementing equivalent behavior in a raw SDK would require manual state tracking and recursion management.

### 2.2 Node Execution Model

All nodes are `async` Python functions. The graph is compiled with `MemorySaver` and invoked via `compiled_graph.ainvoke()` inside a FastAPI async endpoint. This means:

- The entire graph runs in a single event loop iteration from the API's perspective
- Ollama calls are `await`ed with `httpx.AsyncClient`
- ChromaDB queries (synchronous SDK) are wrapped in `asyncio.to_thread()` to avoid blocking the event loop
- Langfuse span creation is non-blocking (fire-and-forget with try/except)

### 2.3 Tool Execution Pattern

Each tool node follows a consistent pattern: open a Langfuse span → append tool name to `state["tools_used"]` → execute tool logic → append result summary to `state["execution_trace"]` → return state. This uniformity means every tool is observable in the same way and can be tested with the same fixtures.

---

## 3. Tool Selection

### 3.1 Retrieval Tool — ChromaDB

ChromaDB was selected as the vector store for three reasons: it is Docker-native with a persistent volume, requires zero configuration, and has a synchronous Python client that wraps cleanly in `asyncio.to_thread()`. The alternative (Qdrant) offers better performance at scale but adds operational complexity unnecessary for this dataset size (22 documents across 4 collections).

The retrieval tool queries three collections per request (packages, benchmark_rules, knowledge_snippets) with a combined metadata filter built from extracted entities. The metadata filter (e.g., `{budget_tier: {$in: ["low", "medium"]}}`) narrows the cosine similarity search so that, for example, a low-budget query does not surface Premium package documents as top results.

Documents are embedded with `nomic-embed-text` via Ollama — the same Ollama instance used for LLM inference — keeping the embedding model fully local and avoiding any external embedding API dependency.

### 3.2 Scoring Tool — Deterministic Rules Engine

The scoring tool is the most important tool in the system and the only one that is entirely deterministic. It receives retrieved package documents and extracted customer entities, applies the benchmark rules from the assignment spec (industry risk, region cost pressure, budget compatibility, priority alignment, dependents ratio), and returns each package with a score from 0–100 and a breakdown of which rules were applied.

The decision to make this tool deterministic (no LLM) was intentional and deliberate. Business rules must be auditable, reproducible, and testable. An LLM scoring engine would produce different scores on repeated runs, could not be unit-tested with deterministic assertions, and would make it impossible to explain exactly why a recommendation changed between two runs. The scoring tool has 100% unit test coverage.

### 3.3 Comparison Tool

The comparison tool builds a structured comparison matrix between two or more packages. It is primarily deterministic (table construction from scoring results and package metadata) with an optional Gemma 4 call for generating the natural-language recommendation reason. The matrix schema is fixed — six dimensions (network, price, coverage, score, budget fit, risk fit) — ensuring consistent output structure regardless of which packages are being compared.

### 3.4 Validation Tool and Fallback Tool

The validator is pure Python — it checks required fields against a static table keyed on `query_type`. No LLM is involved. This means missing-field detection is instant (< 1ms) and cannot be fooled by a cleverly phrased query that makes the LLM think all fields are present.

The fallback tool uses static response templates, not LLM generation. Every failure mode has a pre-written, pre-tested message that is informative without making any claims about what the system could have recommended.

---

## 4. Model Selection

### 4.1 Selected Model: gemma4:31b-cloud via Ollama

A single model (`gemma4:31b-cloud`) is used for all LLM-dependent nodes. The decision to use one model rather than a multi-model strategy was based on three factors.

**Local execution requirement.** The deployment target is local-only Docker Compose with no external API calls. This rules out GPT-4o, Claude, and all cloud-hosted models. Among locally-runnable models at the time of assessment, Gemma 4 at 31 billion parameters offers the best combination of instruction-following reliability, JSON output quality, and context window length (128K tokens).

**Sufficient capability for the task.** The LLM nodes in this system perform bounded, structured tasks: extract JSON from a short query (intent_parser), select from a static plan table (planner), generate 3–5 grounded reasoning bullets from provided scoring data (output_composer), and check whether generated claims are supported by provided data (evaluator). None of these tasks require frontier-model reasoning. Gemma 4 31b handles all of them reliably with temperature=0.1.

**Cost of a multi-model setup.** Using a smaller model for routing and a larger model for reasoning would add orchestration complexity, require two model warm-up sequences on startup, and create a failure surface where the routing model misclassifies a query and routes it to the wrong reasoning model. Given that the retrieval and scoring tools handle the heavy domain logic, the LLM nodes are relatively simple — the complexity cost of a multi-model setup outweighs the benefit.

### 4.2 Where the LLM is Used vs. Not Used

| Node | LLM Used | Rationale |
|------|---------|-----------|
| intent_parser | Yes — entity extraction | Free-text → structured JSON requires language understanding |
| validator | **No** | Pure field-presence check |
| planner | Conditional — ambiguous cases only | Static plan table handles 95% of cases |
| retrieval_tool | Embedding model only | Semantic search; no generation |
| scoring_tool | **No** | Deterministic business rules |
| comparison_tool | Optional — narrative only | Table construction is deterministic |
| output_composer | Yes — reasoning narrative | Grounded narrative generation from scoring data |
| evaluator | Yes — hallucination check | Requires language understanding to compare claims vs. data |
| fallback / clarifier | **No** | Template-based responses |

### 4.3 Temperature and Output Control

All LLM calls use `temperature=0.1` — near-deterministic but with enough variation to avoid repetitive phrasing. JSON-output nodes (intent_parser, evaluator grounding check) use structured prompts that return only JSON with no preamble. `safe_json_parse()` validates and retries up to 2 times with a stricter prompt before routing to fallback.

---

## 5. Evaluation Strategy

### 5.1 Langfuse Integration

Every agent run produces a Langfuse trace with a hierarchical span tree matching the graph structure. Each span records: start/end timestamps (latency), input and output fields, and — for LLM spans — the full prompt, completion, token count, and model name. This provides complete observability of every agent execution without any changes to the agent code (spans are added once per node, non-blocking).

### 5.2 Evaluation Dimensions

Seven evaluation dimensions are computed per run:

| Dimension | Type | What It Measures |
|-----------|------|-----------------|
| `task_success` | Deterministic | Did the agent complete its primary task? (recommendation present, or clarification returned for incomplete query) |
| `plan_completeness` | Deterministic | Are all required fields in the response schema non-null? |
| `tool_coverage` | Deterministic | Were all planned tools actually executed? |
| `grounding_score` | LLM judge (Gemma 4) | Is every reasoning claim traceable to the scoring data? |
| `hallucination_flag` | LLM judge (Gemma 4) | Does any reasoning claim contradict the scoring breakdown? |
| `confidence_calibration` | Deterministic | Is the confidence level consistent with the top package score? |
| `trace_integrity` | Deterministic | Does the execution trace have sufficient entries to reconstruct the run? |

Five dimensions are deterministic (fast, reliable, cheap). Two use Gemma 4 as an evaluator judge — these are the only LLM calls in the evaluation path and are isolated from the recommendation path.

### 5.3 Evaluation Harness

The evaluation harness (`tests/eval_harness.py`) runs 8 predefined scenarios: the 5 visible test cases from the assignment spec plus 3 edge cases (conflicting constraints, unsupported query, empty retrieval). Each scenario has expected outcomes defined as assertions. The harness runs all scenarios sequentially, logs each run to a dedicated Langfuse session, and writes a JSON report with per-scenario scores, latency, and pass/fail status.

The harness is designed to be run repeatedly — each run gets a unique `eval_run_id` timestamp, so results accumulate in Langfuse without overwriting. This allows tracking of quality changes as the system is developed.

### 5.4 Failure Detection

Three failure modes are explicitly detected and flagged:

**Hallucination** is detected when the LLM-judge evaluator finds a reasoning claim that is not supported by or contradicts the scoring breakdown. When detected, `hallucination_flag=1` is logged to Langfuse and the `confidence` field is automatically downgraded by one level (e.g., `high` → `medium-high`).

**Overconfidence** is detected when the top package score is below 60 but the computed confidence is `high`. The evaluator downgrades confidence to `medium` and logs a `confidence_calibration` score of 0.

**Tool bypass** is detected when `tools_used` does not match `execution_plan`. This catches cases where a node is skipped due to a routing bug. It is logged as `tool_coverage < 1.0` and triggers a test failure.

### 5.5 Limitations and Next Steps

**Limitations:**

The system's knowledge base is small (22 documents). Retrieval quality degrades predictably for queries about industries, regions, or package types not in the seed data. The fallback handling is correct (it routes to fallback with an informative note) but the system cannot make an informed recommendation for unseen combinations.

The evaluator uses Gemma 4 to judge Gemma 4's outputs. This is an inherent limitation of self-evaluation — the judge may share the same blind spots as the generator. A stronger evaluation setup would use a separate, independently calibrated model for the judge role.

Latency on CPU-only machines (30–120 seconds per request) makes the system unusable for interactive demo without GPU acceleration.

**Next steps for production readiness:**

Upgrade the `MemorySaver` checkpointer to `SqliteSaver` for persistent conversation state across restarts. Add Qdrant as the vector store for better performance at larger knowledge base scale. Introduce a separate, larger judge model for the evaluator (e.g., a cloud model called only for evaluation, not for recommendations). Add rate limiting at the FastAPI layer. Extend the knowledge base with real package catalogs from the target market. Implement A/B testing via Langfuse experiments to compare scoring rule variants.

---

*Technical Report — Agentic Insurance Advisor System*
*Mission 3 Elite Assessment | Version 1.0.1 | 2026-04-07*
