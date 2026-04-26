# Design Choices Report: Agentic Insurance Advisor


**Live Demo:** [https://agentic-ai-insurance-advisor.vercel.app/](https://agentic-ai-insurance-advisor.vercel.app/)

## 1. Introduction
This report outlines the technical rationale behind the tool, framework, and model selections for the **Agentic Insurance Advisor**. The architecture is designed for production-grade reliability, observability, and scalability.

## 2. Framework Selection: LangGraph
We selected **LangGraph** over alternatives like CrewAI or AutoGen for several reasons:
- **State Management:** LangGraph provides a robust, typed state model (`AgentState`) that persists across node transitions.
- **Control Flow:** It allows for precise control over the agent's logic flow, including conditional branching and multi-step cycles.
- **Observability:** Native integration with Langfuse allows for granular tracing of every graph node.
- **Cycles and Retries:** Built-in support for cycles (e.g., retrieval retries) ensures the system can recover from transient failures or empty search results.

## 3. Model Selection: Gemini 1.5 Pro
The decision to use **Gemini 1.5 Pro** was driven by:
- **JSON Stability:** The model demonstrates high fidelity in adhering to strict JSON schemas, crucial for our `IntentParser` and `OutputComposer` nodes.
- **Context Handling:** Its ability to process complex constraints and map them to deterministic rules outperforms smaller models.
- **Production Efficiency:** By using a single powerful model for all semantic tasks, we reduce architectural complexity and ensure consistent reasoning across the graph.

## 4. Tool Selection
### 4.1 Pinecone (Vector Database)
- **Rationale:** Selected for cloud-native persistence and high-performance semantic search.
- **Implementation:** Replaced local ChromaDB to avoid state-persistence issues across container redeployments.

### 4.2 Upstash Redis (Caching)
- **Rationale:** Used to cache expensive LLM intent parses and final responses.
- **Benefit:** Significantly reduces P95 latency for repeat queries and minimizes API costs.

### 4.3 Deterministic Scoring Engine
- **Rationale:** Business rules (budget, industry risk, region pressure) are implemented in pure Python.
- **Benefit:** Guarantees 100% predictable outcomes for the core insurance logic, preventing LLM "drift" in critical calculations.

## 5. Deployment Strategy
- **Backend:** FastAPI on Railway for high-throughput async processing.
- **Frontend:** Next.js on Vercel for a seamless, responsive user interface.
- **Observability:** Langfuse Cloud for real-time monitoring of agent health and performance.

## 6. Conclusion
The selection of these technologies ensures a Green Flag architecture: modular, observable, and grounded in deterministic logic while leveraging SOTA LLM reasoning where it adds the most value.
