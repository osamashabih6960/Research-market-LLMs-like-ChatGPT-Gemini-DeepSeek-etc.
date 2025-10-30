# src/api_wrappers.py
import os, time, requests
from typing import Dict, Any

class APIResult:
    def __init__(self, text, latency_s, tokens=None, cost=None, raw=None):
        self.text = text
        self.latency_s = latency_s
        self.tokens = tokens
        self.cost = cost
        self.raw = raw

# ---------- OpenAI ----------
def call_openai(prompt: str, model: str = "gpt-4o-mini") -> APIResult:
    import openai
    start = time.time()
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    latency = time.time() - start
    text = response['choices'][0]['message']['content']
    tokens = response.get('usage', {}).get('total_tokens')
    return APIResult(text=text, latency_s=latency, tokens=tokens, raw=response)

# ---------- Gemini (Google) ----------
def call_gemini(prompt: str, model_name: str = "gemini-1.5") -> APIResult:
    start = time.time()
    url = os.getenv("GEMINI_ENDPOINT")
    headers = {"Authorization": f"Bearer {os.getenv('GEMINI_API_KEY')}"}
    json_data = {"model": model_name, "prompt": prompt}
    response = requests.post(url, headers=headers, json=json_data, timeout=60)
    latency = time.time() - start
    data = response.json()
    text = data.get("candidates", [{}])[0].get("content", "")
    return APIResult(text=text, latency_s=latency, raw=data)

# ---------- DeepSeek ----------
def call_deepseek(prompt: str, model: str = "deepseek-v3.2") -> APIResult:
    start = time.time()
    url = os.getenv("DEEPSEEK_ENDPOINT")
    headers = {"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"}
    payload = {"prompt": prompt}
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    latency = time.time() - start
    data = response.json()
    return APIResult(text=data.get("output", ""), latency_s=latency, raw=data)
