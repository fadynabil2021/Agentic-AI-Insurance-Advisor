"""
Google Gemini API client wrapper.
Direct API access to Gemma 4 and other Google models without Ollama.
"""
import google.generativeai as genai
from typing import Optional
import asyncio
import os


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", timeout: int = 120):
        self.api_key = api_key
        self.model_name = model
        self.timeout = timeout
        self._model: Optional[genai.GenerativeModel] = None
        self._configured = False

    def _configure(self):
        """Lazy initialization of the Gemini client."""
        if not self._configured:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
            self._configured = True

    @property
    def model(self) -> genai.GenerativeModel:
        """Get the configured model, initializing if needed."""
        if not self._configured:
            self._configure()
        return self._model

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat completion request to Gemini API."""
        if not self._configured:
            self._configure()

        try:
            # Convert messages to Gemini format
            # Gemini expects: system instruction + conversation history
            system_instruction = ""
            conversation = []

            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    system_instruction = content
                elif role == "assistant":
                    conversation.append({"role": "model", "parts": [content]})
                else:
                    conversation.append({"role": "user", "parts": [content]})

            # Configure model with system instruction
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            # Create model with system instruction if present
            if system_instruction:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_instruction,
                    generation_config=generation_config,
                )
            else:
                model = self.model
                model._generation_config = generation_config

            # Start chat and send message
            chat = model.start_chat(history=conversation[:-1] if len(conversation) > 1 else [])

            # Get the last message content
            last_message = conversation[-1]["parts"][0] if conversation else "Hello"

            response = await asyncio.to_thread(
                chat.send_message,
                last_message,
                generation_config=generation_config,
            )

            return response.text

        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}") from e

    async def embed(self, text: str) -> list[float]:
        """Get embeddings using Google's embedding model."""
        if not self._configured:
            self._configure()

        try:
            embedding_model = genai.GenerativeModel("models/text-embedding-004")
            result = await asyncio.to_thread(
                embedding_model.embed_content,
                text
            )
            return result["embedding"]
        except Exception as e:
            raise RuntimeError(f"Gemini embedding failed: {e}") from e

    async def health_check(self) -> bool:
        """Check if Gemini API is reachable."""
        try:
            # Simple test - try to generate a minimal response
            test_model = genai.GenerativeModel(self.model_name)
            response = await asyncio.to_thread(
                test_model.generate_content,
                "ping"
            )
            return response is not None
        except Exception:
            return False

    async def aclose(self):
        """Cleanup - no-op for Gemini since it's stateless HTTP."""
        pass


def safe_json_parse(raw: str) -> Optional[dict]:
    """
    Attempt to parse JSON from Gemini output, handling common model formatting issues.
    Strips markdown code fences if present.
    """
    import json
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
