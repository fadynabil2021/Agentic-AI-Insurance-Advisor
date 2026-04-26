# Agentic Insurance Advisor 

**Live Demo:** [https://agentic-ai-insurance-advisor.vercel.app/](https://agentic-ai-insurance-advisor.vercel.app/)

A production-grade, state-driven agentic system built with **LangGraph**, **FastAPI**, **Google Gemini**, and **Pinecone**. This system provides structured insurance recommendations based on natural language queries, with built-in evaluation and observability.

---

##  Features

- **Stateful Orchestration**: LangGraph-based workflow with 9 specialized nodes.
- **Hybrid Scoring**: Combines deterministic business rules with LLM-powered reasoning.
- **RAG Integration**: Semantic retrieval of insurance packages and rules via Pinecone.
- **Production Observability**: Full trace logging and evaluation scores in Langfuse.
- **Evaluation Harness**: Automated testing across 8 edge-case scenarios.

---

##  Getting Started

### Prerequisites
- **Docker & Docker Compose** (Recommended)
- **Google Gemini API Key** (Gemini 1.5 Pro)
- **Pinecone Account** (API Key, Environment, Index Name)
- **Upstash Redis Account** (REST URL & Token for caching)
- **Langfuse Account** (Public Key, Secret Key, Host)

### 1. Environment Configuration
Copy the example environment file and fill in your credentials:
```bash
cp .env.example .env
```

### 2. Option A: Run with Docker (Recommended)
This brings up the Backend, Frontend, and a self-hosted Langfuse instance:
```bash
docker compose up --build
```
- **Frontend**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000`
- **Langfuse UI**: `http://localhost:3001`

### 3. Option B: Local Python Development
If you prefer running without Docker:

**Backend Setup:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Seed the Vector Database:**
```bash
# Populate Pinecone with packages, rules, and profiles
python data/seed_data.py --reset
```

**Run Server:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

##  Evaluation & Testing

The system includes a rigorous evaluation harness that measures task success, grounding, and hallucination.

**Run Evaluation Harness:**
```bash
cd backend
# Ensure venv is active
python -m tests.eval_harness
```
Results are pushed to Langfuse and saved locally in `backend/docs/evaluation_report.json`.

---

##  Documentation

- **[Technical Report](technical_report.md)**: Deep dive into design decisions, model selection, and evaluation strategy.
- **[Architecture Diagram](architecture_diagram.md)**: Visual mapping of the LangGraph workflow and data model.


---

##  API Quick Reference

### Agent Query
`POST /api/v1/query`
```json
{
  "user_id": "test_user",
  "query": "Recommend a construction insurance plan in Riyadh for a medium budget."
}
```

### Health Check
`GET /api/v1/health`
Verifies connectivity to Gemini, Pinecone, Redis, and Langfuse.

---

##  Deployment
This project is configured for **Railway** deployment using Nixpacks.
- See `railway.json` and `nixpacks.toml` for deployment settings.
- Ensure all environment variables in `.env.example` are mirrored in your Railway dashboard.


