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
        max_tokens: int = 512,  # Reduced for faster responses
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

            # FOR GEMMA 4: Single-message instruct-following works best with generate_content
            # rather than start_chat, as start_chat can confuse the context.
            if len(conversation) <= 1:
                prompt = ""
                if system_instruction:
                    prompt += f"CONTEXT/INSTRUCTIONS:\n{system_instruction}\n\n"
                
                user_msg = conversation[0]["parts"][0] if conversation else "No request"
                prompt += f"USER REQUEST: {user_msg}\n\nRESPONSE:"

                response = await asyncio.to_thread(
                    self.model.generate_content,
                    prompt,
                    generation_config=generation_config,
                )
                return response.text

            # FALLBACK for multi-turn history
            # Start chat and send message
            chat = self.model.start_chat(history=conversation[:-1])
            last_message = conversation[-1]["parts"][0]

            response = await asyncio.to_thread(
                chat.send_message,
                last_message,
                generation_config=generation_config,
            )

            return response.text

        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}") from e

    async def embed(self, text: str, task_type: str = "retrieval_query") -> list[float]:
        """Get embeddings using Google's embedding model."""
        if not self._configured:
            self._configure()

        try:
            from config import settings
            result = await asyncio.to_thread(
                genai.embed_content,
                model=f"models/{settings.GEMINI_EMBED_MODEL}",
                content=text,
                task_type=task_type,
            )
            return result["embedding"]
        except Exception as e:
            raise RuntimeError(f"Gemini embedding failed: {e}") from e

    async def health_check(self) -> bool:
        """Check if Gemini API is reachable."""
        try:
            if not self._configured:
                self._configure()
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
    import re
    text = raw.strip()
    
    # Try to extract content between <answer> tags first
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()

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
