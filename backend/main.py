"""
FastAPI application entry point.
Handles lifespan (startup/shutdown), middleware, and router registration.
"""
import os
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from clients.ollama_client import OllamaClient
from agent.graph import build_graph

# ─── Shared Application State ─────────────────────────────────────────────────

ollama_client: OllamaClient = None
compiled_graph = None
services_available = False


async def initialize_services():
    """Initialize external services (Ollama, ChromaDB, Redis, Langfuse)."""
    global ollama_client, compiled_graph, services_available

    try:
        print("[startup] Initializing Agentic Insurance Advisor...")

        # Build Ollama client
        ollama_client = OllamaClient(
            host=settings.OLLAMA_HOST,
            model=settings.OLLAMA_MODEL,
            embed_model=settings.OLLAMA_EMBED_MODEL,
            timeout=settings.OLLAMA_TIMEOUT,
        )

        # Build and compile the LangGraph
        compiled_graph = build_graph(ollama_client)
        print("[startup] LangGraph compiled successfully.")

        # Pre-warm Ollama (avoids cold-start latency on first user request)
        try:
            print(f"[startup] Warming up Ollama model {settings.OLLAMA_MODEL}...")
            await ollama_client.chat(
                [{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            print("[startup] Ollama warm-up complete.")
        except Exception as e:
            print(f"[startup] WARNING: Ollama warm-up failed — {e}. Continuing anyway.")

        services_available = True
        print("[startup] All services initialized successfully.")

    except Exception as e:
        print(f"[startup] WARNING: Service initialization failed — {e}")
        print("[startup] Running in limited mode. Some features may be unavailable.")
        services_available = False


async def shutdown_services():
    """Shutdown external services."""
    global ollama_client
    print("[shutdown] Closing Ollama client...")
    try:
        if ollama_client is not None:
            await ollama_client.aclose()
    except Exception:
        pass
    print("[shutdown] Done.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # Non-blocking startup - initialize services in background
    await initialize_services()
    yield
    await shutdown_services()


# ─── Application ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic Insurance Advisor",
    description=(
        "A stateful, graph-based agentic AI that accepts natural-language insurance "
        "plan queries and returns grounded, structured recommendations."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Request-ID"],
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
