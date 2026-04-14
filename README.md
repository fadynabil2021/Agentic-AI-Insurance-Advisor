# Agentic Insurance Advisor System
### Mission 3 – Elite Assessment | Senior/Elite Implementation

A production-grade agentic AI system built with LangGraph, Ollama (gemma4:31b-cloud), FastAPI, Next.js, ChromaDB, and Langfuse. Fully containerized — runs entirely locally via Docker Compose.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Architecture at a Glance](#2-architecture-at-a-glance)
3. [Prerequisites](#3-prerequisites)
4. [First-Time Setup (Read Carefully)](#4-first-time-setup-read-carefully)
5. [Running the Full Stack](#5-running-the-full-stack)
6. [Service URLs](#6-service-urls)
7. [Using the Chat UI](#7-using-the-chat-ui)
8. [Running the Evaluation Harness](#8-running-the-evaluation-harness)
9. [Antigravity IDE — MCP Server Setup](#9-antigravity-ide--mcp-server-setup)
10. [Development Workflow](#10-development-workflow)
11. [Environment Variables](#11-environment-variables)
12. [Testing](#12-testing)
13. [Troubleshooting](#13-troubleshooting)
14. [Deliverables Checklist](#14-deliverables-checklist)

---

## 1. What This System Does

This system accepts natural-language insurance plan queries, decomposes them into a structured execution plan, runs multiple specialized tools (retrieval, deterministic scoring, comparison), and returns a fully grounded, structured recommendation with execution trace and confidence score.

**It is not a chatbot.** It is a stateful, graph-based agentic workflow that:

- Plans before acting (LangGraph orchestrator)
- Uses real tools (ChromaDB retrieval, deterministic rules engine)
- Traces every step (Langfuse observability)
- Fails safely (graceful fallback on every edge case)
- Never hallucinates certainty (confidence is always evidence-based)

**Example input:**
```
"Recommend the best plan for a healthcare company in Riyadh."
```

**Example output:**
```json
{
  "recommendation": { "plan_name": "Standard", "network": "B", "price_range": [6000, 7500] },
  "confidence": "high",
  "reasoning": ["Healthcare is high-risk...", "Riyadh has highest cost pressure...", "..."],
  "execution_trace": ["intent_parser: ...", "scoring_tool: Standard=80 ...", "..."],
  "fallback_or_risk_note": "Premium available if dependents ratio > 0.50"
}
```

---

## 2. Architecture at a Glance

```
Antigravity IDE (Claude Sonnet 4.6 + MCP Servers)
        │
        ▼
Next.js Chat UI  ──►  FastAPI Backend  ──►  LangGraph Graph
                                               │
                              ┌────────────────┼────────────────┐
                              │                │                │
                           Ollama          ChromaDB         Langfuse
                       gemma4:31b-cloud   (vector store)  (observability)
                              │
                           Redis (cache)
```

**Services (all local, all Docker):**

| Service | Port | Purpose |
|---------|------|---------|
| Next.js Frontend | 3000 | Chat UI |
| FastAPI Backend | 8000 | Agent REST API |
| Ollama | 11434 | LLM + embedding inference |
| ChromaDB | 8001 | Vector store |
| Langfuse | 3001 | Observability & evaluation UI |
| Redis | 6379 | Cache & rate limiting |
| PostgreSQL (Langfuse) | 5433 | Langfuse data store |

---

## 3. Prerequisites

### Required Software

| Tool | Version | Install |
|------|---------|---------|
| Docker Desktop | ≥ 4.28 | https://docs.docker.com/get-docker/ |
| Docker Compose | v2 (bundled with Docker Desktop) | Included |
| Python | 3.12+ | https://python.org (for MCP servers + local dev) |
| Node.js | 20 LTS+ | https://nodejs.org (for frontend local dev only) |
| Antigravity IDE | Latest | Your internal distribution |

### Hardware Requirements

> ⚠️ **Critical:** `gemma4:31b-cloud` is a 31-billion parameter model.

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 16 GB | 32 GB |
| GPU VRAM | 16 GB (NVIDIA) | 24 GB |
| Disk space | 40 GB free | 60 GB free |
| CPU (CPU-only mode) | 16 cores | 32 cores |

**GPU acceleration (strongly recommended):**

- **NVIDIA:** Docker Desktop → Settings → Resources → Enable GPU. Ollama auto-detects CUDA.
- **Apple Silicon (M1/M2/M3):** Ollama runs natively on Metal. No special config needed — remove the `deploy` block from the `ollama` service in `docker-compose.yml`.
- **CPU-only fallback:** Works but inference will take 30–120 seconds per request. Acceptable for demo; not for development iteration.

---

## 4. First-Time Setup (Read Carefully)

The model pull is the most time-sensitive part of setup. Do **not** skip straight to `docker compose up` — pull the model first, then bring up the rest of the stack.

### Step 1 — Clone the repository

```bash
git clone <your-repo-url>
cd insurance-advisor-agent
```

### Step 2 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in:

```env
# Langfuse (generated in Step 5 below)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

All other values have working defaults for local Docker Compose. Do not change them unless you know what you're doing.

### Step 3 — Start Ollama only

```bash
docker compose up ollama -d
```

Wait until Ollama is healthy:

```bash
docker compose ps ollama
# Status should show: healthy
```

Verify:

```bash
curl http://localhost:11434/api/tags
# Should return: {"models":[]}  (empty list is fine — model not pulled yet)
```

### Step 4 — Pull the LLM model

> ⚠️ This downloads ~18–20 GB. Run this before anything else. Do it on a fast connection. It only needs to happen once — the model is stored in the `ollama_data` Docker volume and persists across restarts.

```bash
# Pull the main reasoning model
docker exec insurance-ollama ollama pull gemma4:31b-cloud

# Pull the embedding model (much smaller, ~274 MB)
docker exec insurance-ollama ollama pull nomic-embed-text
```

Monitor progress — you will see a download progress bar. Wait for both to complete:

```
pulling manifest
pulling <digest>... 100% ████████████ 18.5 GB
verifying sha256 digest
writing manifest
success
```

Verify both models are loaded:

```bash
docker exec insurance-ollama ollama list
# Expected output:
# NAME                    ID              SIZE    MODIFIED
# gemma4:31b-cloud        <hash>          18.5 GB  ...
# nomic-embed-text        <hash>          274 MB   ...
```

### Step 5 — Initialize Langfuse

Start Langfuse and its database:

```bash
docker compose up langfuse-postgres langfuse -d
```

Wait ~30 seconds, then open the Langfuse UI:

```
http://localhost:3001
```

Create a new account (local only, no external connection):
- Click **Sign Up**
- Use any email/password (e.g., `admin@local.dev` / `password123`)
- Create a new **Project** named `insurance-advisor`
- Go to **Settings → API Keys** → **Create new API key**
- Copy the **Public Key** and **Secret Key**
- Paste them into your `.env` file:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxxxxxx
```

### Step 6 — Seed the ChromaDB knowledge base

Start ChromaDB:

```bash
docker compose up chromadb -d
```

Wait for it to be healthy, then run the seed script:

```bash
# From the repo root
docker compose run --rm backend python data/seed_data.py
```

Expected output:

```
[seed] Connecting to ChromaDB at chromadb:8000...
[seed] Creating collection: packages (3 documents)
[seed] Creating collection: benchmark_rules (12 documents)
[seed] Creating collection: knowledge_snippets (4 documents)
[seed] Creating collection: customer_profiles (3 documents)
[seed] Embedding documents with nomic-embed-text...
[seed] Done. All collections seeded successfully.
```

Verify seeding:

```bash
curl http://localhost:8001/api/v1/collections
# Should list 4 collections with document counts
```

---

## 5. Running the Full Stack

Once Steps 1–6 above are complete, bring up the entire system in one command:

```bash
docker compose up -d
```

Wait for all services to reach `healthy` status:

```bash
docker compose ps
```

Expected output (all services `healthy`):

```
NAME                        STATUS
insurance-ollama            running (healthy)
insurance-chromadb          running (healthy)
insurance-redis             running (healthy)
insurance-langfuse-db       running (healthy)
insurance-langfuse          running (healthy)
insurance-backend           running (healthy)
insurance-frontend          running (healthy)
```

Verify the full system with a health check:

```bash
curl http://localhost:8000/api/v1/health | python -m json.tool
```

Expected:

```json
{
  "status": "healthy",
  "version": "1.0.1",
  "services": {
    "ollama": true,
    "chromadb": true,
    "langfuse": true,
    "redis": true
  }
}
```

If any service shows `false`, see [Troubleshooting](#13-troubleshooting).

### Stopping the stack

```bash
# Stop all services (preserves all data volumes)
docker compose down

# Stop AND wipe all data (forces re-seed and re-pull on next start)
docker compose down -v
```

> ⚠️ Never use `down -v` unless you want to re-download the 18 GB model.

---

## 6. Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Chat UI (Next.js) | http://localhost:3000 | None |
| FastAPI Docs (Swagger) | http://localhost:8000/docs | None |
| FastAPI Docs (ReDoc) | http://localhost:8000/redoc | None |
| Langfuse UI | http://localhost:3001 | Your email/password from Step 5 |
| ChromaDB API | http://localhost:8001/api/v1 | None |
| Ollama API | http://localhost:11434 | None |

---

## 7. Using the Chat UI

Open http://localhost:3000 in your browser.

### Try the 5 test scenarios:

```
1. "Recommend the best plan for a healthcare company in Riyadh."

2. "Give me the cheapest acceptable option for a construction customer in Jeddah."

3. "Compare Standard and Premium for a retail customer in Dammam."

4. "Recommend a plan for my company."
   (Tests clarification flow — system asks for missing details)

5. "Why did you choose that recommendation?"
   (Tests explanation trace — run after scenario 1, 2, or 3)
```

### UI Panels

- **Chat area** — Send queries, receive responses
- **Recommendation Card** — Structured plan name, network, price range, confidence badge
- **Reasoning panel** — 3–5 grounded reasoning bullets
- **Execution Trace** — Click to expand — shows every agent step with timestamps
- **Tools Used** — Icon chips showing which tools were invoked
- **Risk Note** — Always visible amber box with fallback/tradeoff note
- **Langfuse link** — Click "View full trace" to open the detailed trace in Langfuse

---

## 8. Running the Evaluation Harness

The evaluation harness runs all 8 defined scenarios (5 visible + 3 edge cases), logs every run to Langfuse, and produces a JSON report.

### Run via API

```bash
curl -X POST http://localhost:8000/api/v1/eval/run \
  -H "Content-Type: application/json" \
  -d '{"scenario_ids": "all"}' \
  | python -m json.tool
```

### Run directly (faster, more verbose output)

```bash
docker compose exec backend python tests/eval_harness.py
```

### Output

The harness prints a live summary and writes the full report to:

```
backend/docs/evaluation_report.json
```

Sample console output:

```
Running evaluation harness — 8 scenarios
─────────────────────────────────────────
[PASS] S1_balanced          confidence=high    grounding=0.95  latency=2340ms
[PASS] S2_cheapest          confidence=medium-high  grounding=0.91  latency=1980ms
[PASS] S3_compare           confidence=high    grounding=0.93  latency=3120ms
[PASS] S4_missing_info      clarification=true             latency=890ms
[PASS] S5_explain           reasoning_bullets=4            latency=1450ms
[PASS] E1_conflicting       confidence=medium  fallback=true   latency=2210ms
[PASS] E2_unsupported       query_type=unsupported          latency=450ms
[FAIL] E3_empty_retrieval   retry_count=2      fallback=true   latency=4800ms
         → Expected fallback_note, got empty string (bug: see issue #12)

─────────────────────────────────────────
Results:  7/8 passed  (87.5%)
Avg grounding score:  0.926
Hallucinations detected: 0
Avg latency: 2155ms
Full report: docs/evaluation_report.json
Langfuse session: http://localhost:3001/sessions/eval_20260407_143022
```

---

## 9. Antigravity IDE — MCP Server Setup

The MCP servers let the Antigravity Claude Sonnet 4.6 agent invoke each tool directly from the IDE agent window.

### Install Python dependencies for MCP servers

```bash
cd mcp_servers
pip install -r requirements.txt
```

`mcp_servers/requirements.txt`:

```
mcp==1.1.0
httpx==0.27.2
chromadb==0.5.18
pydantic==2.9.2
python-dotenv==1.0.1
```

### Register MCP servers in Antigravity

The file `.antigravity/mcp_servers.json` is already in the repository and pre-configured. Antigravity should auto-detect it when you open the project folder.

If it does not auto-detect, manually add via:

**Antigravity → Settings → MCP Servers → Add from file → `.antigravity/mcp_servers.json`**

### Verify MCP servers in Antigravity

Open the **Antigravity Agent Window** and type:

```
List available tools
```

You should see:

```
Available tools:
• score_packages          (scoring_tool MCP server)
• retrieve_packages       (retrieval_tool MCP server)
• compare_packages        (comparison_tool MCP server)
• validate_query          (validation_tool MCP server)
• run_insurance_agent     (agent_runner MCP server)
```

### Example MCP usage from Antigravity agent window

```
You: Test the scoring tool for a healthcare company in Riyadh with medium budget

Claude: [calls score_packages with {industry: "healthcare", region: "riyadh", budget: "medium"}]

Result:
{
  "Standard": { "score": 80, "reasons": ["Healthcare high-risk — Network B minimum", ...] },
  "Premium":  { "score": 65, "reasons": ["Region cost pressure reduces Premium viability", ...] },
  "Basic":    { "score": 30, "reasons": ["PENALTY: Basic not suitable for high-risk industry", ...] }
}

The scoring tool ranked Standard highest at 80/100. Basic was heavily penalized (–25)
due to healthcare's high-risk classification. Premium lost points due to Riyadh's
high cost pressure against a medium budget.
```

### MCP server environment requirements

MCP servers connect to **locally running Docker services**. Make sure the Docker stack is running before using MCP tools in Antigravity:

```bash
# Quick check before opening Antigravity
docker compose ps --format "table {{.Name}}\t{{.Status}}"
```

If ChromaDB or Ollama are not running, the retrieval and scoring MCP servers will fail with a connection error. The agent_runner MCP server requires the full backend to be healthy.

---

## 10. Development Workflow

### Backend (FastAPI + LangGraph)

Hot reload is enabled in the Docker container. Edit any file in `backend/` and changes are reflected immediately — no restart needed.

```bash
# Watch backend logs
docker compose logs -f backend

# Run a single test
docker compose exec backend pytest tests/test_scoring_tool.py -v

# Run all tests
docker compose exec backend pytest tests/ -v

# Open an interactive shell in the backend container
docker compose exec backend bash
```

### Frontend (Next.js)

For active frontend development, run Next.js locally (outside Docker) for faster HMR:

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:3000
# Set NEXT_PUBLIC_API_URL=http://localhost:8000 in frontend/.env.local
```

Or use the Docker container (slower HMR but identical to production):

```bash
docker compose logs -f frontend
```

### Adding / modifying scoring rules

All scoring logic lives in `backend/agent/tools/scoring_tool.py`. It is pure Python with no LLM dependency. Edit and save — the backend hot-reloads automatically.

To verify a rule change:

```bash
docker compose exec backend python -c "
from agent.tools.scoring_tool import run_scoring
from agent.state import QueryType

results = run_scoring(
    packages=[{'name': 'Standard', 'network': 'B', 'price_range': [6000, 7500], 'coverage': 'Medium'}],
    entities={'industry': 'healthcare', 'region': 'riyadh', 'budget': 'medium', 'priority': 'balanced'}
)
import json; print(json.dumps(results, indent=2))
"
```

### Updating ChromaDB seed data

If you modify `backend/data/seed_data.py`:

```bash
# Wipe and re-seed
docker compose exec backend python data/seed_data.py --reset
```

### Viewing Langfuse traces

Every agent run is automatically traced. Open http://localhost:3001, navigate to your project, and click **Traces**. You will see:

- Full span tree for every node
- Token usage per LLM call
- Latency breakdown per node
- Evaluation scores (grounding, hallucination, etc.)
- Input/output for every span

---

## 11. Environment Variables

All variables are in `.env` (copied from `.env.example`). The table below documents every variable.

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `OLLAMA_HOST` | `http://ollama:11434` | Yes | Ollama service URL (internal Docker network) |
| `OLLAMA_MODEL` | `gemma4:31b-cloud` | Yes | Main LLM model tag — must be pulled first |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Yes | Embedding model for ChromaDB |
| `OLLAMA_TIMEOUT` | `120` | No | Seconds before Ollama request times out (increase on CPU-only) |
| `CHROMA_HOST` | `chromadb` | Yes | ChromaDB hostname (internal Docker network) |
| `CHROMA_PORT` | `8000` | Yes | ChromaDB internal port |
| `REDIS_URL` | `redis://redis:6379/0` | Yes | Redis connection string |
| `CACHE_TTL_SECONDS` | `300` | No | Response cache TTL (5 minutes) |
| `LANGFUSE_HOST` | `http://langfuse:3000` | Yes | Langfuse internal URL |
| `LANGFUSE_PUBLIC_KEY` | _(none)_ | **Yes** | Generated in Langfuse UI (Step 5) |
| `LANGFUSE_SECRET_KEY` | _(none)_ | **Yes** | Generated in Langfuse UI (Step 5) |
| `MAX_RETRIES` | `2` | No | Max retrieval retry attempts before fallback |
| `MIN_CONFIDENCE_THRESHOLD` | `0.4` | No | Minimum score ratio before confidence downgrades |

> The backend will refuse to start if `LANGFUSE_PUBLIC_KEY` or `LANGFUSE_SECRET_KEY` are missing.

For **MCP servers** running locally (outside Docker), create `mcp_servers/.env`:

```env
CHROMADB_HOST=localhost
CHROMADB_PORT=8001
OLLAMA_HOST=http://localhost:11434
FASTAPI_URL=http://localhost:8000
```

---

## 12. Testing

### Unit tests (deterministic tools — fast, no LLM)

```bash
docker compose exec backend pytest tests/test_scoring_tool.py tests/test_validator.py tests/test_routing.py -v
```

These tests run purely on deterministic logic. They should complete in under 5 seconds and require no external services.

### Integration tests (full graph — requires all services running)

```bash
docker compose exec backend pytest tests/test_scenarios.py -v
```

These tests invoke the full LangGraph graph against all 5 visible scenarios and 3 edge cases. Each test takes 2–8 seconds depending on Ollama latency.

### Evaluation harness (generates the eval report)

```bash
docker compose exec backend python tests/eval_harness.py
```

Output written to `docs/evaluation_report.json`. Required for the assignment deliverable.

### Type checking

```bash
docker compose exec backend mypy agent/ routers/ models/ --strict
```

### Linting

```bash
docker compose exec backend ruff check .
```

---

## 13. Troubleshooting

### `ollama` container is unhealthy

```bash
# Check logs
docker compose logs ollama

# Common fix: increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory → set to at least 16 GB
```

### `gemma4:31b-cloud` not found after pull

```bash
# Verify the model is in the volume
docker exec insurance-ollama ollama list

# If not listed, re-pull
docker exec insurance-ollama ollama pull gemma4:31b-cloud

# If pull fails with "model not found", check the exact tag on ollama.com/library
```

### Backend fails with `LANGFUSE_PUBLIC_KEY not set`

```bash
# Verify .env file exists and has the keys
cat .env | grep LANGFUSE

# If missing, re-do Step 5 and paste the keys into .env
# Then restart the backend
docker compose restart backend
```

### ChromaDB returns empty results after restart

The ChromaDB data is persisted in the `chroma_data` Docker volume. If results are empty:

```bash
# Check collection document counts
curl http://localhost:8001/api/v1/collections | python -m json.tool

# If collections are empty, re-run the seed
docker compose exec backend python data/seed_data.py
```

### Ollama requests timing out

`gemma4:31b-cloud` can be slow on CPU-only machines. Increase the timeout:

```env
# In .env
OLLAMA_TIMEOUT=300
```

Then restart the backend:

```bash
docker compose restart backend
```

### MCP servers failing in Antigravity with "connection refused"

The MCP servers connect to `localhost:8001` (ChromaDB) and `localhost:11434` (Ollama). These are only available when the Docker stack is running.

```bash
# Verify ports are exposed
docker compose ps
curl http://localhost:8001/api/v1/heartbeat
curl http://localhost:11434/api/tags
```

### Langfuse shows no traces

```bash
# Verify keys in .env match the project in the Langfuse UI
# Go to http://localhost:3001 → Settings → API Keys
# Keys must match exactly (no trailing spaces)

# Test the connection from the backend
docker compose exec backend python -c "
import os
from langfuse import Langfuse
lf = Langfuse(
    public_key=os.environ['LANGFUSE_PUBLIC_KEY'],
    secret_key=os.environ['LANGFUSE_SECRET_KEY'],
    host=os.environ['LANGFUSE_HOST']
)
print('Langfuse connection:', lf.auth_check())
"
```

### Port conflicts

If any port is already in use:

```bash
# Find what's using port 8000 (example)
lsof -i :8000

# Change the host port in docker-compose.yml
# Example: change "8000:8000" to "8080:8000"
# Then update NEXT_PUBLIC_API_URL in frontend environment
```

---

## 14. Deliverables Checklist

Per Section 13 of the assignment PDF, the following must be submitted:

| # | Deliverable | Location | Status |
|---|------------|----------|--------|
| 1 | Working code | `backend/` + `frontend/` + `mcp_servers/` | ✅ |
| 2 | README with setup instructions | `README.md` (this file) | ✅ |
| 3 | Short technical report (max 5 pages) | `docs/technical_report.md` | ✅ |
| 4 | Architecture diagram | `docs/architecture_diagram.png` | ✅ |
| 5 | Tool/framework/model choice explanation | `docs/technical_report.md` §3 | ✅ |
| 6 | Evaluation report (≥ 5 runs) | `docs/evaluation_report.json` | ✅ (8 runs) |
| 7 | Demo walkthrough (video or live) | `docs/demo_walkthrough.mp4` | 🎬 Record after setup |

### Generating the architecture diagram

The diagram in `docs/architecture_diagram.png` can be regenerated:

```bash
docker compose exec backend python docs/generate_diagram.py
# Requires: pip install diagrams graphviz
```

Or export it directly from the Mermaid source in `docs/architecture.mmd` using any Mermaid renderer.

### Recording the demo walkthrough

Recommended sequence for the demo video:

1. Show `docker compose ps` — all services healthy
2. Open the Langfuse UI — show empty traces
3. Open the Next.js Chat UI at `localhost:3000`
4. Run Scenario 1 (healthcare/Riyadh) — narrate the Recommendation Card, Confidence Badge, Execution Trace
5. Switch to Langfuse — show the full span tree, scores, token usage
6. Run Scenario 4 (missing info) — show the clarification flow
7. Run Scenario 3 (comparison) — show the Comparison Table
8. Open Antigravity IDE — show the agent window calling `score_packages` via MCP
9. Run the eval harness — show the 8-scenario report
10. Show `docs/evaluation_report.json`

Recommended length: 8–12 minutes.

---

## Project Structure (Quick Reference)

```
insurance-advisor-agent/
├── .antigravity/mcp_servers.json    # Antigravity MCP registration
├── docker-compose.yml               # Full stack definition
├── .env.example                     # Environment variable template
├── README.md                        # This file
├── mcp_servers/                     # Antigravity MCP tool servers
│   ├── retrieval_mcp_server.py
│   ├── scoring_mcp_server.py
│   ├── comparison_mcp_server.py
│   ├── validation_mcp_server.py
│   └── agent_runner_mcp_server.py
├── backend/
│   ├── main.py                      # FastAPI app entrypoint
│   ├── config.py                    # Pydantic settings
│   ├── agent/
│   │   ├── graph.py                 # LangGraph compiled graph
│   │   ├── state.py                 # AgentState TypedDict
│   │   ├── routing.py               # Conditional edge functions
│   │   ├── nodes/                   # One file per graph node
│   │   └── tools/                   # retrieval, scoring, comparison
│   ├── clients/                     # Ollama, ChromaDB, Langfuse, Redis
│   ├── data/seed_data.py            # ChromaDB knowledge base seeding
│   ├── models/schemas.py            # Pydantic request/response models
│   ├── routers/                     # FastAPI route handlers
│   ├── tests/
│   │   ├── test_scenarios.py        # 5 visible + 3 edge case tests
│   │   ├── test_scoring_tool.py     # Unit tests for deterministic logic
│   │   └── eval_harness.py          # Full evaluation run generator
│   └── docs/
│       ├── technical_report.md      # 5-page technical report
│       ├── evaluation_report.json   # Generated by eval harness
│       └── architecture_diagram.png
└── frontend/
    ├── app/                         # Next.js App Router pages
    ├── components/                  # React UI components
    ├── lib/                         # API client + TypeScript types
    └── store/                       # Zustand state management
```

---

*Agentic Insurance Advisor System — Mission 3 Elite Assessment*
*Stack: LangGraph · Ollama (gemma4:31b-cloud) · FastAPI · Next.js · ChromaDB · Langfuse · Docker Compose*
*Model confirmed: `gemma4:31b-cloud` — must be pulled before starting the backend*
