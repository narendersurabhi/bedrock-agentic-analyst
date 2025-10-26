#!/usr/bin/env python3
"""
Minimal agent stub: loads synthetic evidence, retrieves top-k snippets by TF-IDF,
then writes an "investigator brief" with inline [CIT-#] markers.

No external LLM calls. Replace the `draft_brief()` function with Bedrock later.
"""
import argparse, json, pathlib, re
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DATA_DIR = pathlib.Path("data")
OUT_DIR = pathlib.Path("out")
OUT_DIR.mkdir(exist_ok=True)

def load_evidence() -> List[Dict]:
    # expected file written by data/synthetic_evidence.py
    f = DATA_DIR / "evidence.jsonl"
    docs = []
    with f.open() as r:
        for line in r:
            if line.strip():
                docs.append(json.loads(line))
    return docs

def retrieve(docs: List[Dict], query: str, k: int = 6) -> List[Dict]:
    corpus = [d["text"] for d in docs]
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(corpus + [query])
    sim = linear_kernel(tfidf[-1], tfidf[:-1]).flatten()
    top = sim.argsort()[::-1][:k]
    return [docs[i] | {"score": float(sim[i])} for i in top]

def draft_brief(query: str, hits: List[Dict]) -> str:
    lines = [f"# Investigator Brief", f"Query: {query}", ""]
    findings = []
    for idx, h in enumerate(hits, 1):
        cid = f"CIT-{idx}"
        snippet = re.sub(r"\s+", " ", h["text"]).strip()
        findings.append(f"- {h.get('title','evidence')} [{cid}]")
        lines.append(f"[{cid}] {h.get('title','evidence')} — src={h.get('source','synthetic')} score={h['score']:.3f}")
        lines.append(f"Snippet: {snippet[:300]}")
        lines.append("")
    lines.insert(3, "## Findings")
    lines[0:0] = ["---", "Disclaimer: synthetic data only.", "---", ""]
    lines.append("## Summary")
    lines.append("Pattern appears across multiple claims/providers based on retrieved evidence. Prioritize human review.")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help='e.g. "Pattern on provider 123?"')
    ap.add_argument("--topk", type=int, default=6)
    args = ap.parse_args()

    docs = load_evidence()
    hits = retrieve(docs, args.query, args.topk)
    brief = draft_brief(args.query, hits)
    out = OUT_DIR / "brief.md"
    out.write_text(brief, encoding="utf-8")
    print(f"Wrote {out} with {len(hits)} citations")

if __name__ == "__main__":
    main()
