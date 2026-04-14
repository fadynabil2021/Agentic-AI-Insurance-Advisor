"""
Pydantic settings for the Agentic Insurance Advisor backend.
All values can be overridden via environment variables or .env file.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Ollama ───────────────────────────────────────────────────
    OLLAMA_HOST: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "gemma4:31b-cloud"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_TIMEOUT: int = 120

    # ── ChromaDB ─────────────────────────────────────────────────
    CHROMA_HOST: str = "chromadb"
    CHROMA_PORT: int = 8000

    # ── Langfuse ─────────────────────────────────────────────────
    LANGFUSE_HOST: str = "http://langfuse:3000"
    LANGFUSE_PUBLIC_KEY: str = "placeholder-public-key"
    LANGFUSE_SECRET_KEY: str = "placeholder-secret-key"

    # ── Redis ────────────────────────────────────────────────────
    REDIS_URL: str = "redis://redis:6379/0"
    CACHE_TTL_SECONDS: int = 300

    # ── Agent ────────────────────────────────────────────────────
    MAX_RETRIES: int = 2
    MIN_CONFIDENCE_THRESHOLD: float = 0.4

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
