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
                    prompt += "--- SYSTEM INSTRUCTIONS ---\n"
                    prompt += f"{system_instruction}\n"
                    prompt += "--- END SYSTEM INSTRUCTIONS ---\n\n"
                
                user_msg = conversation[0]["parts"][0] if conversation else "No request"
                prompt += f"USER QUERY: {user_msg}\n\n"
                prompt += "YOUR FINAL RESPONSE (Exactly following instructions):"

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
    """
    import json
    import re
    text = raw.strip()

    # STRATEGY 1: Extract content between <answer> tags and find JSON there
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        answer_content = match.group(1).strip()
        # Find JSON object within the answer content
        start = answer_content.find("{")
        end = answer_content.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_text = answer_content[start : end + 1]
            try:
                result = json.loads(json_text)
                return result
            except json.JSONDecodeError:
                pass

    # STRATEGY 2: Find the LAST JSON object in the text (after all thinking)
    # This catches cases where LLM outputs thinking + JSON without proper tags
    # We look for the last { } pair which is most likely the actual JSON output
    last_start = -1
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "{":
            last_start = i
            break

    if last_start != -1:
        last_end = text.rfind("}")
        if last_end > last_start:
            json_text = text[last_start : last_end + 1]
            # Clean up common LLM formatting issues
            json_text = re.sub(r'`([^`]*)`', r'"\1"', json_text)  # Replace backticks with quotes
            try:
                result = json.loads(json_text)
                return result
            except json.JSONDecodeError:
                pass

    # STRATEGY 3: Try to find any valid JSON object in the text
    # Use regex to find potential JSON objects
    potential_jsons = re.findall(r'\{[^{}]*"[^"]*"[^{}]*\}', text)
    for candidate in potential_jsons:
        try:
            result = json.loads(candidate)
            if "query_type" in result:  # Validate it looks like our expected output
                return result
        except json.JSONDecodeError:
            continue

    return None
