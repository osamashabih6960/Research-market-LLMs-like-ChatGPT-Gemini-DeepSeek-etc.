# src/evaluate.py
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_refs(ref_file):
    refs = {}
    with open(ref_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            refs[obj['id']] = obj['reference']
    return refs

def semantic_score(pred, ref):
    emb = embedder.encode([pred, ref])
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

if __name__ == "__main__":
    with open("results/raw_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    refs = load_refs("data/references.jsonl")

    rows = []
    for r in results:
        ref = refs.get(r["id"], "")
        score = semantic_score(r["text"], ref) if ref else None
        rows.append({**r, "semantic_score": score})

    df = pd.DataFrame(rows)
    df.to_csv("results/eval_results.csv", index=False)
    print(df.groupby("model")["semantic_score"].mean())
