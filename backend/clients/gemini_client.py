"""
Google Gemini API client wrapper.
Direct API access to Gemini models without Ollama.
"""
import google.generativeai as genai
from typing import Optional, List, Dict, Any
import asyncio
import os

class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", timeout: int = 120):
        self.api_key = api_key
        self.model_name = model
        self.timeout = timeout
        self._configured = False

    async def _configure(self):
        """Lazy initialization with automatic model discovery."""
        if not self._configured:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is missing. Please set it in your environment.")
            
            genai.configure(api_key=self.api_key)
            
            # --- Auto-Discovery Logic ---
            try:
                available_models = await asyncio.to_thread(genai.list_models)
                model_names = [m.name for m in available_models if "generateContent" in m.supported_generation_methods]
                print(f"[gemini] Available models for this key: {model_names}")
                
                # If current model isn't in the list, pick the best available Flash or Pro
                clean_current = self.model_name.split("/")[-1]
                exists = any(clean_current in m for m in model_names)
                
                if not exists:
                    # Pick Flash first, then Pro
                    flash_variants = [m for m in model_names if "gemini-1.5-flash" in m]
                    pro_variants = [m for m in model_names if "gemini-1.5-pro" in m]
                    legacy_pro = [m for m in model_names if "gemini-pro" in m and "1.5" not in m]
                    
                    if flash_variants:
                        self.model_name = flash_variants[0]
                    elif pro_variants:
                        self.model_name = pro_variants[0]
                    elif legacy_pro:
                        self.model_name = legacy_pro[0]
                    
                    print(f"[gemini] Redirected to compatible model: {self.model_name}")
            except Exception as e:
                print(f"[gemini] Discovery failed (continuing with default): {e}")
                
            self._configured = True

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        response_json: bool = True,
    ) -> str:
        """Send a chat completion request to Gemini API."""
        await self._configure()

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

            if len(conversation) <= 1:
                prompt = conversation[0]["parts"][0] if conversation else "Respond with {}"
                response = await asyncio.to_thread(model.generate_content, prompt)
            else:
                chat = model.start_chat(history=conversation[:-1])
                response = await asyncio.to_thread(chat.send_message, conversation[-1]["parts"][0])

            if not response or not response.candidates:
                raise RuntimeError("Gemini returned no candidates.")
            
            return response.text

        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}") from e

    async def embed(self, text: str, task_type: str = "retrieval_query") -> List[float]:
        """Get embeddings using Google's embedding model."""
        await self._configure()
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
            await self._configure()
            test_model = genai.GenerativeModel(self.model_name)
            response = await asyncio.to_thread(test_model.generate_content, "Say OK")
            return response is not None and len(response.candidates) > 0
        except Exception:
            return False

    async def aclose(self):
        pass


def safe_json_parse(raw: str) -> Optional[dict]:
    import json
    if not raw: return None
    text = raw.strip()
    if "```json" in text: text = text.split("```json")[-1].split("```")[0].strip()
    elif "```" in text: text = text.split("```")[-1].split("```")[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end >= start: text = text[start : end + 1]
    try: return json.loads(text)
    except Exception: return None
