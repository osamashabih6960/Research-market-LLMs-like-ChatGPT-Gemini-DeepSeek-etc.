# src/runner.py
import json
from src.api_wrappers import call_openai, call_gemini, call_deepseek

MODELS = [
    ("openai", lambda p: call_openai(p)),
    ("gemini", lambda p: call_gemini(p)),
    ("deepseek", lambda p: call_deepseek(p)),
]

def run_all(prompts_file: str, out_file: str):
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [json.loads(l) for l in f]

    results = []
    for p in prompts:
        for name, func in MODELS:
            print(f"Running {name} on {p['id']}")
            try:
                res = func(p['text'])
                results.append({
                    "id": p['id'],
                    "model": name,
                    "text": res.text,
                    "latency_s": res.latency_s
                })
            except Exception as e:
                results.append({"id": p['id'], "model": name, "text": f"ERROR: {e}"})
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("✅ Results saved to", out_file)

if __name__ == "__main__":
    run_all("data/prompts.jsonl", "results/raw_results.json")
