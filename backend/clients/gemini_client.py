"""
Google Gemini API client wrapper.
Direct API access to Gemini 1.5 Pro and other Google models.
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

    def clean_json_text(j_text: str) -> str:
        # Remove any markdown JSON block syntax if present
        j_text = re.sub(r'^```(?:json)?|```$', '', j_text.strip(), flags=re.MULTILINE).strip()
        # Replace backticks with quotes for keys/values
        j_text = re.sub(r'`([^`]+)`', r'"\1"', j_text)
        return j_text

    # STRATEGY 1: Extract content between <answer> tags
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        answer_content = match.group(1).strip()
        answer_content = clean_json_text(answer_content)
        
        # Find JSON object bounds
        start = answer_content.find("{")
        end = answer_content.rfind("}")
        if start != -1 and end != -1 and end >= start:
            json_text = answer_content[start : end + 1]
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass
                
        # Try raw answer content (it might be a bare JSON string/object without braces)
        try:
            return json.loads(answer_content)
        except json.JSONDecodeError:
            pass

    # STRATEGY 2: Find any markdown code block
    match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if match:
        json_text = match.group(1).strip()
        json_text = clean_json_text(json_text)
        start = json_text.find("{")
        end = json_text.rfind("}")
        if start != -1 and end != -1 and end >= start:
            try:
                return json.loads(json_text[start : end + 1])
            except json.JSONDecodeError:
                pass

    # STRATEGY 3: Robust fallback searching for {...} starting from the end
    # This correctly parses cases where thinking content might contain '{'
    start_indices = [i for i, c in enumerate(text) if c == '{']
    end_indices = [i for i, c in enumerate(text) if c == '}']
    
    # Try combinations starting from the back to prioritize the actual output
    for start in reversed(start_indices):
        for end in reversed(end_indices):
            if end > start:
                json_text = text[start : end + 1]
                json_text = clean_json_text(json_text)
                try:
                    result = json.loads(json_text)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    pass

    return None
