# Agentic Insurance Advisor — Mission 3 Elite Assessment

A production-grade, state-driven agentic system built with LangGraph, FastAPI, Google Gemini, and Pinecone.

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
- Python 3.10+
- Upstash Redis account (for caching)
- Pinecone account (for vector store)
- Google Gemini API key
- Langfuse account (for observability)

### 2. Setup
```bash
# Clone and install dependencies
git clone <repository_url>
cd Agent_Recommendation_Assignment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your actual API keys
```

### 3. Seed Vector Database
Before running the server, populate Pinecone with the package catalog, rules, and customer profiles:
```bash
cd backend
python data/seed_data.py --reset
```

### 4. Run the Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
The API will be available at `http://localhost:8000`

---

## 🏗 System Architecture

For a detailed visual mapping of nodes, state transitions, and component interactions, see the [Architecture Diagram](architecture_diagram.md).

Key characteristics of the architecture:
- **Stateful Orchestration**: Built on LangGraph `StateGraph`, maintaining a rigid schema across nodes.
- **Hybrid Logic Engine**: Combines deterministic Python heuristics for scoring (ensuring reliability) with Gemini's reasoning capabilities (for natural narrative synthesis).
- **Cloud-Native Integrations**: Upstash Redis (caching), Pinecone (retrieval), Langfuse (traceability). 

---

## 🔌 API Reference

### 1. Agent Query
`POST /api/v1/query`
Sends a natural language request to the agent and returns a structured decision.

**Request:**
```json
{
  "user_id": "user_123",
  "query": "Recommend the best plan for a healthcare company in Riyadh."
}
```

**Response (Extracted Sample):**
```json
{
  "plan": {"steps": ["intent_parsing", "retrieval", "scoring", "composition"]},
  "recommendation": {
    "plan_name": "Standard",
    "network": "B",
    "price_range": [6000, 7500]
  },
  "confidence": "medium-high",
  "execution_trace": ["..."]
}
```

### 2. Health Check
`GET /api/v1/health`
Verifies connectivity to all upstream providers (Gemini, Pinecone, Redis, Langfuse).

### 3. Evaluation
`POST /api/v1/eval/run`
Triggers the internal evaluation harness.

---

## 🌐 Deployment (Railway)

The application is deployed on Railway via Nixpacks.
- Uses `Procfile` mapping: `web: cd backend && /opt/venv/bin/uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}`
- Uses Nixpacks configurations in `nixpacks.toml` to install dependencies via `backend/requirements.txt` and expose library paths.

**Required Environment Variables (Railway Dashboard):**
- `GEMINI_API_KEY`
- `GEMINI_MODEL` (set to `gemma-4-31b-it`)
- `PINECONE_API_KEY`, `PINECONE_ENV`, `PINECONE_INDEX_NAME`
- `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN`
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

---

## 📊 Evaluation Harness

The system includes a rigorous evaluation harness (`backend/tests/eval_harness.py`) that tests 8 specific scenarios against 7 dimensions (5 deterministic + 2 LLM-judged).

To run the harness locally:
```bash
cd backend
python -m tests.eval_harness
```

Results are saved as a JSON report in `backend/docs/evaluation_report.json`.
