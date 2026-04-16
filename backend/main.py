"""
FastAPI application entry point.
Handles lifespan (startup/shutdown), middleware, and router registration.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from clients.gemini_client import GeminiClient
from agent.graph import build_graph

# ─── Shared Application State ─────────────────────────────────────────────────

gemini_client: GeminiClient = None
compiled_graph = None
services_available = False


async def initialize_services(app: FastAPI):
    """Initialize external services and store handles on app.state."""
    print("[startup] Initializing Agentic Insurance Advisor...")

    # Build Gemini client
    gemini_client = GeminiClient(
        api_key=settings.GEMINI_API_KEY,
        model=settings.GEMINI_MODEL,
        timeout=settings.GEMINI_TIMEOUT,
    )

    # ALWAYS build and compile the LangGraph — don't block on pings
    app.state.compiled_graph = build_graph(gemini_client)
    app.state.gemini_client = gemini_client
    print("[startup] LangGraph compiled successfully.")

    # Attempt to pre-warm Gemini (non-fatal)
    try:
        print(f"[startup] Warming up Gemini model {settings.GEMINI_MODEL}...")
        await gemini_client.chat(
            [{"role": "user", "content": "ping"}],
            max_tokens=5,
            response_json=False,  # faster for warmup
        )
        print("[startup] Gemini warm-up complete.")
    except Exception as e:
        print(f"[startup] WARNING: Gemini warm-up failed — {e}. Graph will attempt again on first request.")

    app.state.services_available = True
    print("[startup] Services ready for requests.")


async def shutdown_services(app: FastAPI):
    """Shutdown external services."""
    print("[shutdown] Closing Gemini client...")
    try:
        client = getattr(app.state, "gemini_client", None)
        if client is not None:
            await client.aclose()
    except Exception:
        pass
    print("[shutdown] Done.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    await initialize_services(app)
    yield
    await shutdown_services(app)


# ─── Application ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic Insurance Advisor",
    description=(
        "A stateful, graph-based agentic AI that accepts natural-language insurance "
        "plan queries and returns grounded, structured recommendations."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://frontend:3000",
    "https://agentic-ai-insurance-advisor.vercel.app",
    # Railway internal
    "https://*.railway.app",
    # Vercel (set FRONTEND_URL env var to your deployed URL)
    os.environ.get("FRONTEND_URL", ""),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in ALLOWED_ORIGINS if o],
    allow_origin_regex=r"https://.*\.(railway\.app|vercel\.app)",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Request-ID"],
    allow_credentials=False,
)

# ─── Request ID Middleware ─────────────────────────────────────────────────────

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.time()
    response = await call_next(request)
    elapsed = int((time.time() - start_time) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    return response

# ─── Global Error Handler ─────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )

# ─── Routers ─────────────────────────────────────────────────────────────────

from routers import query, health, trace, eval as eval_router  # noqa: E402

app.include_router(query.router,       prefix="/api/v1")
app.include_router(health.router,      prefix="/api/v1")
app.include_router(trace.router,       prefix="/api/v1")
app.include_router(eval_router.router, prefix="/api/v1")
