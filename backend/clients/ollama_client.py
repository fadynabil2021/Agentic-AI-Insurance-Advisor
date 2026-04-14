"""
Ollama HTTP client wrapper.
Uses httpx.AsyncClient for async-native operation across all node calls.
"""
import httpx
import json
from typing import Optional


class OllamaClient:
    def __init__(self, host: str, model: str, embed_model: str, timeout: int = 120):
        self.host = host
        self.model = model
        self.embed_model = embed_model
        self.client = httpx.AsyncClient(timeout=timeout)

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat completion request to Ollama. Returns the assistant content string."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            r = await self.client.post(f"{self.host}/api/chat", json=payload)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Ollama chat failed: {e}") from e

    async def embed(self, text: str) -> list[float]:
        """Get embeddings for a text string using nomic-embed-text."""
        try:
            r = await self.client.post(
                f"{self.host}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
            )
            r.raise_for_status()
            return r.json()["embedding"]
        except Exception as e:
            raise RuntimeError(f"Ollama embed failed: {e}") from e

    async def health_check(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            r = await self.client.get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    async def aclose(self):
        await self.client.aclose()


def safe_json_parse(raw: str) -> Optional[dict]:
    """
    Attempt to parse JSON from Ollama output, handling common model formatting issues.
    Strips markdown code fences if present.
    """
    text = raw.strip()
    # Strip markdown JSON code block if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        inner = [l for l in lines[1:] if l.strip() != "```"]
        text = "\n".join(inner).strip()

    # Try to find first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
