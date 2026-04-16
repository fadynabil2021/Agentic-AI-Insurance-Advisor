"""
Pydantic settings for the Agentic Insurance Advisor backend.
Cloud-native configuration for deployment on Railway/Render/Fly.io.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Google Gemini API ─────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GEMINI_EMBED_MODEL: str = "text-embedding-004"
    GEMINI_TIMEOUT: int = 120

    # ── Pinecone Vector DB ─────────────────────────────────────────
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "insurance-advisor"
    PINECONE_ENVIRONMENT: str = "us-east-1"

    # ── Langfuse Cloud ─────────────────────────────────────────────
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""

    # ── Upstash Redis ──────────────────────────────────────────────
    UPSTASH_REDIS_REST_URL: str = ""
    UPSTASH_REDIS_REST_TOKEN: str = ""
    CACHE_TTL_SECONDS: int = 300

    # ── Agent ────────────────────────────────────────────────────
    MAX_RETRIES: int = 2
    MIN_CONFIDENCE_THRESHOLD: float = 0.4

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
