"""
Google Gemini API client wrapper.
Direct API access to Gemini models without Ollama.
"""
import google.generativeai as genai
from typing import Optional, List, Dict, Any
import asyncio
import os

class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest", timeout: int = 120):
        self.api_key = api_key
        self.model_name = model
        self.timeout = timeout
        self._configured = False

    def _configure(self):
        """Lazy initialization of the Gemini client."""
        if not self._configured:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is missing. Please set it in your environment.")
            genai.configure(api_key=self.api_key)
            self._configured = True

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        response_json: bool = True,
    ) -> str:
        """Send a chat completion request to Gemini API."""
        self._configure()

        try:
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

            # Configure generation config
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if response_json:
                generation_config["response_mime_type"] = "application/json"

            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction if system_instruction else None,
                generation_config=generation_config,
            )

            # Single-turn or simple history mapping
            if len(conversation) <= 1:
                prompt = conversation[0]["parts"][0] if conversation else "Respond with {}"
                response = await asyncio.to_thread(model.generate_content, prompt)
            else:
                chat = model.start_chat(history=conversation[:-1])
                response = await asyncio.to_thread(chat.send_message, conversation[-1]["parts"][0])

            # Improved error feedback
            if not response or not response.candidates:
                raise RuntimeError("Gemini returned no candidates. Check safety filters or quota.")
            
            try:
                return response.text
            except ValueError:
                # If text is blocked by safety filters
                feedback = getattr(response, 'prompt_feedback', 'No feedback available')
                raise RuntimeError(f"Gemini response blocked or empty. Feedback: {feedback}")

        except Exception as e:
            # Try a fallback model if 404
            if "not found" in str(e).lower() and self.model_name != "gemini-1.5-flash":
                print(f"[gemini] {self.model_name} not found, falling back to gemini-1.5-flash...")
                self.model_name = "gemini-1.5-flash"
                return await self.chat(messages, temperature, max_tokens, response_json)
            raise RuntimeError(f"Gemini API call failed: {e}") from e

    async def embed(self, text: str, task_type: str = "retrieval_query") -> List[float]:
        """Get embeddings using Google's embedding model."""
        self._configure()

        try:
            result = await asyncio.to_thread(
                genai.embed_content,
                model="models/text-embedding-004",
                content=text,
                task_type=task_type,
            )
            return result["embedding"]
        except Exception as e:
            raise RuntimeError(f"Gemini embedding failed: {e}") from e

    async def health_check(self) -> bool:
        """Check if Gemini API is reachable."""
        try:
            self._configure()
            # Use a slightly more robust model identifier for the health check
            test_model = genai.GenerativeModel(self.model_name)
            response = await asyncio.to_thread(test_model.generate_content, "Say OK")
            # If we get any response back, consider it healthy
            return response is not None and (len(response.candidates) > 0)
        except Exception as e:
            print(f"[gemini] Health check detailed failure: {e}")
            return False

    async def aclose(self):
        """Cleanup - no-op for Gemini."""
        pass


def safe_json_parse(raw: str) -> Optional[dict]:
    """
    Attempt to parse JSON, stripping markdown if necessary.
    """
    import json
    if not raw:
        return None
        
    text = raw.strip()
    # Strip markdown if present
    if "```json" in text:
        text = text.split("```json")[-1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[-1].split("```")[0].strip()

    # Generic boundary finding
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        text = text[start : end + 1]

    try:
        return json.loads(text)
    except Exception:
        return None
