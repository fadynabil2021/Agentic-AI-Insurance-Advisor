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
from clients.gemini_client import GeminiClient
from agent.graph import build_graph

# ─── Shared Application State ─────────────────────────────────────────────────

gemini_client: GeminiClient = None
compiled_graph = None
services_available = False


async def initialize_services():
    """Initialize external services (Gemini, Pinecone, Redis, Langfuse)."""
    global gemini_client, compiled_graph, services_available

    try:
        print("[startup] Initializing Agentic Insurance Advisor...")

        # Build Gemini client
        gemini_client = GeminiClient(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
            timeout=settings.GEMINI_TIMEOUT,
        )

        # Build and compile the LangGraph
        compiled_graph = build_graph(gemini_client)
        print("[startup] LangGraph compiled successfully.")

        # Pre-warm Gemini API (avoids cold-start latency on first user request)
        try:
            print(f"[startup] Warming up Gemini model {settings.GEMINI_MODEL}...")
            await gemini_client.chat(
                [{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            print("[startup] Gemini warm-up complete.")
        except Exception as e:
            print(f"[startup] WARNING: Gemini warm-up failed — {e}. Continuing anyway.")

        services_available = True
        print("[startup] All services initialized successfully.")

    except Exception as e:
        print(f"[startup] WARNING: Service initialization failed — {e}")
        print("[startup] Running in limited mode. Some features may be unavailable.")
        services_available = False
        # Set to None to avoid issues with uninitialized graph
        gemini_client = None
        compiled_graph = None


async def shutdown_services():
    """Shutdown external services."""
    global gemini_client
    print("[shutdown] Closing Gemini client...")
    try:
        if gemini_client is not None:
            await gemini_client.aclose()
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
    allow_origins=["*"],  # Allow all origins so Vercel can connect seamlessly
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Agentic Insurance Advisor API is running successfully!"}

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
