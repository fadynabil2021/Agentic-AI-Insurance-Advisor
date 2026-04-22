# Technical Report: Agentic Insurance Advisor
**Prepared for:** Mission 3 Elite Assessment
**Document Constraints:** Maximum 5 pages

---

## 1. System Integration & High-Level Architecture

The Agentic Insurance Advisor has been engineered as a production-ready, cloud-native backend system. Moving past simple proof-of-concept scripts, the architecture leverages modern containerized deployment strategies and integrates decoupled external services. 

### Core Components
1. **Application Framework (FastAPI):** Serves as the high-throughput asynchronous entry point for all API requests. Provides native Pydantic validation for strict schema adherence (crucial for complying with PDF Section 10).
2. **Orchestration Engine (LangGraph):** Manages the stateful workflows of the agent. By treating multi-step reasoning as a directed, cyclic graph, LangGraph provides trace-level visibility into agent transitions (e.g., retrieving data, handling failures, composing output).
3. **Caching Layer (Upstash Redis):** An edge-compatible Redis cluster used to cache exact user intent parses and final outcomes, significantly reducing redundant LLM calls and overall latency.
4. **Vector Database (Pinecone):** Serves as the external memory for the system. Instead of loading JSON files into Context windows dynamically, all packages, benchmark rules, and knowledge snippets are embedded and stored in Pinecone namespaces.
5. **Observability (Langfuse Cloud):** Every agentic span (intent parsing, retrieval scores, evaluator outcomes) is pushed to Langfuse via non-blocking asynchronous calls. This is essential for debugging hallucination failures and optimizing constraints in production.

---

## 2. Agent and Runtime Design

The agent's logic flow dictates strict adherence to determinism. Large Language Models (LLMs) are notoriously unpredictable when asked to perform strict algorithmic sorting. Therefore, the runtime employs a "Hybrid Engine" approach.

### State Transitions (LangGraph Workflow)
1. **Intent Parser Node:** The query enters the Intent Parser, powered by Gemini. It extracts a predefined set of entities (`industry`, `region`, `budget`, `priority`, `dependents_ratio`, and `query_type`).
2. **Validator Node:** A pure Python node verifies that critical entities (e.g., `industry`) are present. If missing, it branches to a `Clarifier` node. If present, it passes execution forward.
3. **Planner Node:** A static lookup determines the order of tool execution. This minimizes the risk of LLM loop planning failures by defaulting to known-good deterministic step flows.
4. **Tool Nodes (Retrieval & Scoring):** The actual business logic execution phase (detailed in Section 3).
5. **Output Composer Node:** Fuses the rigid deterministic output of the Scoring Tool with a fluid conversational reasoning block synthesized by the LLM. 
6. **Evaluator Node (Post-Processing):** Runs inline to measure the quality of the generated response before committing the trace to Langfuse.

---

## 3. Tool Selection and Strategy

### A. Pinecone Retrieval Tool
Using Pinecone over in-memory stores like ChromaDB eliminates state-persistence issues across container re-deployments on Railway. It utilizes `gemini-embedding-001` vectors and queries across specialized namespaces.

### B. Deterministic Scoring Engine
The core of the logic happens *outside* the LLM. The scoring engine evaluates the retrieved rules against the extracted user state utilizing predefined weighted constants:
- `WEIGHT_BUDGET_INCOMPATIBLE = 35`
- `WEIGHT_INDUSTRY_HIGH_RISK = 25`
- `WEIGHT_REGION_COST_PRESSURE = 15`
- `WEIGHT_CONFLICTING_CONSTRAINTS = 10`

Packages begin with a score of 100 and are penalized. Confidence is deduced via dual-factor analysis: the absolute top score crossed against the gap score of the runner-up package. A newly implemented penalty catches unviable requests (e.g., "Best Coverage with Low Budget") scaling down confidence appropriately to reflect system uncertainty and edge-case conflict.

---

## 4. Model Selection & Key Design Decisions

### Gemini gemma-4-31b-it (Google AI Studio)
The decision to upgrade from `gemini-1.5-flash` to the `gemma-4-31b-it` reasoning model was deliberate for production deployment.

**Rationale:**
1. **JSON Output Stability:** `gemma-4-31b-it` has a substantially higher fidelity parameter constraint alignment, meaning it virtually never deviates from the JSON schema required by `IntentParser`.
2. **Abstract Intent Recognition:** The 31-billion parameter model is uniquely capable of parsing underlying tones and tradeoffs. For example, recognizing "I don't have much money to spend" accurately maps to `budget: "low"` rather than triggering fallback states.
3. **Tradeoffs:** While inference latency is slightly marginally higher than `flash`, the elimination of retries (due to broken schemas) results in faster overall P95 latency distributions. The configuration timeout has been hardened from 30s to 60s to accommodate peak initialization loads.

---

## 5. Evaluation Strategy (Eval Harness)

The project includes an embedded offline evaluation harness spanning 8 edge-case scenarios heavily derived from PDF specifications.
1. **Dimensional Analysis:** The harness leverages 5 deterministic metrics (e.g., task_success via entity mapping, tool coverage verification) and 2 LLM-Judge metrics (Reasoning Grounding and Hallucination Checks).
2. **Conflicting Constraints (E1 Scenario):** A major breakthrough involved testing edge-case queries. When the user requests mutually exclusive goals, the eval harness expects `confidence_max: "medium"`. The deterministic scoring engine detects this mathematically, successfully downgrading confidence and pushing a contextual risk note.
3. **Results Reporting:** Test runs automatically sync with Langfuse using custom evaluation tag injections, verifying end-to-end model performance across permutations.

---

## 6. Limitations & Future Extensions

1. **In-Context Learning Constraints:** While `gemma-4-31b-it` is powerful, complex "Compare" queries currently rely on predefined dimensional matching matrices. Future architectures shouldn't rely strictly on static column mapping.
2. **Context Window Expansion:** Integrating complete policy booklets (which can run up to hundreds of pages) would require an optimization of the RAG context threshold, potentially including Semantic chunking rather than the current flat-sentence ingestion logic.
3. **Human In The Loop (HITL):** Currently, ambiguous intents generate Clarification responses. Integrating true HITL capabilities within the LangGraph orchestrator would allow sales representatives to review `Medium-Low` confidence predictions manually before returning API responses to front-end clients. 
