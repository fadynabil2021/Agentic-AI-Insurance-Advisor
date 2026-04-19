import asyncio, json, os
from dotenv import load_dotenv

load_dotenv()

# We must adjust sys.path so it finds backend
import sys
sys.path.insert(0, os.path.abspath("backend"))

from clients.gemini_client import GeminiClient

REASONING_SYSTEM_PROMPT = 'You are a grounding engine. Return ONLY a JSON array of strings: ["reason1"]. No preamble.'
reasoning_context = {'package': 'Basic', 'entities': {'industry': 'construction'}, 'all_scores': {'Basic': 100}, 'query_type': 'cheapest'}

client = GeminiClient(api_key=os.environ['GEMINI_API_KEY'], model='gemma-4-31b-it')

async def test():
    print(repr(await client.chat([{'role': 'system', 'content': REASONING_SYSTEM_PROMPT}, {'role': 'user', 'content': json.dumps(reasoning_context)}])))

asyncio.run(test())
