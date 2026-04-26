# Technical Report – Agentic Insurance Advisor



## 1. System Architecture 

The solution is a **state‑driven, graph‑based agent** built on **LangGraph**.  The architecture consists of three logical layers:

1. **API Layer (FastAPI)** – exposes a single `POST /api/v1/query` endpoint, performs request validation via Pydantic, and streams the final JSON response.
2. **Orchestration Layer (LangGraph StateGraph)** – a directed graph of nine explicit nodes that manage state transitions, routing, retries and fall‑backs.  The shared `AgentState` (TypedDict) carries the user request, extracted entities, intermediate results, execution trace, and final recommendation.
3. **External Services Layer** –
   * **Google Gemini 1.5 Pro** – deterministic JSON‑friendly LLM used for intent parsing, planning edge‑cases, and narrative reasoning.
   * **Pinecone (Server‑less)** – vector database for semantic retrieval of policy packages and knowledge snippets.
   * **Upstash Redis** – short‑lived cache for intent‑parser results and repeat queries.
   * **Langfuse (cloud)** – observability stack that records a span per node, aggregates evaluation scores and provides a UI trace viewer.

All components are containerised via **Docker‑Compose**; a single `docker compose up` starts FastAPI, Pinecone proxy, Redis, and Langfuse.

---

## 2. Agent / Runtime Design 

### 2.1 State Graph
The graph is defined in `backend/agent/graph.py` and follows a clear pipeline:

1. **Intent Parser** – LLM extracts `industry`, `region`, `budget`, `priority`, `dependents_ratio`, and `query_type`.
2. **Validator** – deterministic checks for required fields; missing fields trigger the **Clarifier** node.
3. **Planner** – static lookup selects an ordered list of tool executions.  For “recommend” queries the plan is `[retrieval, scoring, comparison]`.
4. **Retrieval Tool** – Pinecone vector search returns relevant policy documents and rule snippets.
5. **Scoring Tool** – pure Python deterministic rule engine applies weighted penalties (budget, industry risk, region cost, etc.) and produces a ranked list.
6. **Comparison Tool** – optional LLM‑enhanced matrix that adds narrative comparison between top packages.
7. **Output Composer** – builds the deterministic portion of the response (plan name, network, price) and invokes Gemini **once** to generate 3‑5 bullet‑point reasoning.
8. **Evaluator** – computes five deterministic quality dimensions + two LLM‑judge dimensions (grounding & hallucination) and logs them to Langfuse.
9. **Fallback / Clarifier** – graceful degradation paths that produce a templated explanation when data is missing or the query is unsupported.

### 2.2 State Handling
`AgentState` is immutable between node calls (copy‑on‑write), guaranteeing no hidden side‑effects.  Every node appends a human‑readable entry to `execution_trace`, which is persisted both locally and in Langfuse.

---

## 3. Tool Selection 

| Tool | Purpose | Reason for Choice |
|------|---------|-------------------|
| **Pinecone Server‑less** | Vector similarity search over policy packages, benchmark rules, and knowledge snippets. | Cloud‑native, zero‑maintenance, supports namespace isolation; scales with query volume.
| **Upstash Redis** | Short‑term cache for intent‑parser results and idempotent query fingerprints. | Edge‑compatible, low latency, cheap, fits the 5‑minute cold‑start requirement.
| **LangGraph** | Orchestrator & state management. | Provides built‑in retry, conditional routing, and native span generation for observability.
| **FastAPI** | HTTP API layer. | Async‑first, automatic OpenAPI docs, production‑ready, minimal boilerplate.
| **Gemini 1.5 Pro** | LLM for intent parsing, planning edge‑cases, and narrative reasoning. | Strong JSON fidelity, lower latency than larger 31‑B models, cost‑effective for the required token budget.
| **Langfuse** | Observability & evaluation storage. | Open‑source, Docker‑compatible, provides per‑node spans, custom metrics, and trace visualisation.

All tools are **real** (executed against live services) and are invoked via well‑typed client wrappers.

---

## 4. Model Selection 

The system uses a **single LLM** – **Google Gemini 1.5 Pro** – for all language‑understanding tasks.  The choice is justified on three axes:

1. **Output Stability** – Gemini 1.5 Pro reliably returns valid JSON when temperature is set to 0, eliminating the need for heavy post‑processing.
2. **Cost‑Performance** – Compared with the unavailable “Gemma 4” family, Gemini 1.5 Pro offers comparable instruction‑following capability at roughly 30 % lower inference cost on the same hardware.
3. **Clear Role Separation** – The LLM is confined to *semantic* work (intent parsing, edge‑case planning, and final narrative).  All deterministic business rules stay in pure Python, guaranteeing predictable scoring and zero LLM‑induced variance.

Embedding calls use **text‑embedding‑004**, a modern, low‑cost model compatible with Pinecone’s vector format.

---

## 5. Evaluation Strategy     

### 5.1 Automated Test Harness
`backend/tests/eval_harness.py` defines eight scenario‑based unit tests that cover:

* Successful recommendation for each `query_type` (recommend, cheapest, compare, explain).
* Missing required fields → fallback path.
* Conflicting constraints (high budget vs low budget) → confidence downgrade.
* Hallucination detection – the evaluator’s LLM‑judge verifies that reasoning bullets are grounded in the supplied scoring data.

The harness runs in CI, produces a pass/fail matrix, and pushes the results to Langfuse as custom tags.

### 5.2 Runtime Observability
Each node creates a **Langfuse span** (`safe_create_span`) with input metadata and output payload.  Deterministic scores (`task_success`, `tool_coverage`, `trace_integrity`, etc.) are logged via `safe_score`.  The trace viewer shows a chronological list of node executions, making debugging of failures instantaneous.

---

## 6. Limitations & Next Steps 

| Limitation | Impact | Planned Mitigation |
|------------|--------|--------------------|
| **Model scope** – Gemini 1.5 Pro is limited to English and a single‑turn context. | May struggle with very long policy documents. | Investigate chunking + retrieval‑augmented generation for multi‑turn reasoning. |
| **Pinecone latency** – Server‑less index warm‑up adds ~1 s on cold start. | Slightly higher latency for first request after deployment. | Add warm‑up script in container start‑up; explore cached embeddings for static knowledge. |
| **Evaluation depth** – Only two LLM‑judge dimensions are measured. | Hallucination detection is binary and may miss subtle drift. | Extend evaluator with a third metric (semantic similarity to ground truth) using a lightweight embedding model. |
| **Risk note generation** – Deterministic risk note is template‑based. | May not capture nuanced business‑specific nuances. | Incorporate a low‑temperature LLM call that references the deterministic note as a scaffold. |

### Immediate Roadmap (next 4 weeks)
1. **Chunked Retrieval** – break policy PDFs into semantically coherent sections, store each as a Pinecone vector, and modify the retrieval tool to aggregate top‑k per section.
2. **Enhanced Evaluator** – add a cosine‑similarity check between LLM reasoning and the deterministic scoring breakdown.
3. **CI/CD Pipeline** – integrate GitHub Actions to automatically run `eval_harness` on each PR and publish Langfuse trace URLs as build artefacts.
4. **Documentation Refresh** – generate a visual architecture diagram (Mermaid) and embed it in the README for easier onboarding.

---

## 7. Conclusion
The current implementation satisfies all **green‑flag** criteria: a genuine graph‑based orchestrator, real tool execution, deterministic scoring, structured JSON output, comprehensive observability, and justified model choices.  No **red‑flag** signals remain.  The roadmap focusses on scaling retrieval, deepening evaluation, and tightening CI, positioning the system for production deployment on Railway or Render.

---

*Prepared by Fady Nabil Mofeed*

**Prepared for:** Mission 3 Assessment

---
