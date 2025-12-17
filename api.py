
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re

import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "data" / "catalog.index"
META_PATH  = ROOT / "data" / "catalog.pkl"

# ---------- App ----------
app = FastAPI(title="SHL Assessment Recommendation Engine (RAG)")

class RecommendRequest(BaseModel):
    job_title: str = ""
    skills: list[str] = []
    job_description: str = ""
    top_k: int = 10

def normalize_text(s: str) -> str:
    s = s or ""
    return re.sub(r"\s+", " ", s.strip())

# ---------- Load catalog + embedder + FAISS ----------
catalog = pd.read_pickle(META_PATH).fillna("")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(str(INDEX_PATH))

def rule_boost(query: str, row: pd.Series) -> float:
    q = (query or "").lower()
    cat = (row.get("category", "") or "").lower()
    skills = (row.get("skills", "") or "").lower()
    lvl = (row.get("job_levels", "") or "").lower()

    boost = 0.0

    # ---------- Leadership / manager intent ----------
    if any(k in q for k in ["manager", "lead", "leadership", "stakeholder", "strategy"]):
        if any(k in cat for k in ["behavior", "sjt", "personality", "job focused"]):
            boost += 0.08
        if any(k in lvl for k in ["senior", "manager"]):
            boost += 0.06

    # ---------- AI / Tech intent ----------
    ai_intent = any(k in q for k in ["ai", "ml", "machine learning", "nlp", "deep learning", "data scientist", "research"])
    tech_intent = any(k in q for k in ["python", "sql", "coding", "programming", "developer", "engineer", "data", "analytics", "statistics"])

    if ai_intent or tech_intent:
        # Strong boosts: what we WANT on top for AI intern roles
        if any(k in skills for k in ["coding", "python", "algorithms", "machine learning", "data science", "data engineering", "problem solving"]):
            boost += 0.35

        if "cognitive" in cat:
            boost += 0.25

        if ("skills" in cat) or ("simulation" in cat):
            boost += 0.15

        if any(k in lvl for k in ["entry", "graduate", "intern", "junior"]):
            boost += 0.12

        # Strong penalties: what we DON'T want on top for AI intern roles
        if any(k in skills for k in ["business skills", "computer literacy", "workplace productivity"]):
            boost -= 0.45

        if any(k in cat for k in ["behavioral", "personality", "virtual assessment center"]):
            boost -= 0.30

        if any(k in lvl for k in ["senior", "manager"]):
            boost -= 0.60

        # Language tests: only if explicitly asked
        if ("language" in cat) or ("english" in skills):
            if not any(k in q for k in ["english", "communication", "writing", "grammar", "spoken"]):
                boost -= 0.35

    # ---------- Explicit language intent ----------
    if any(k in q for k in ["english", "communication", "writing", "grammar", "spoken"]):
        if ("language" in cat) or ("english" in skills):
            boost += 0.20

    # ---------- Entry-level intent ----------
    if any(k in q for k in ["fresher", "entry", "junior", "graduate", "intern"]):
        if "entry" in lvl:
            boost += 0.06
        if "graduate" in lvl:
            boost += 0.05

    return boost


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    query = " ".join([req.job_title, req.job_description, " ".join(req.skills)]).strip()
    if not query:
        return {"error": "Provide job_title or job_description or skills"}

    # Embed query (cosine similarity via normalized embeddings + inner product index)
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    k = max(1, int(req.top_k))
    scores, idxs = index.search(q_emb, k)  # scores: (1,k), idxs: (1,k)

    results = []
    for rank, i in enumerate(idxs[0], 1):
        row = catalog.iloc[int(i)].to_dict()

        boost = rule_boost(query, pd.Series(row))
        final_score = float(scores[0][rank - 1] + boost)

        desc = row.get("description", "") or ""
        evidence = desc[:220] + ("..." if len(desc) > 220 else "")

        results.append({
            "rank": rank,
            "assessment_id": row.get("assessment_id",""),
            "name": row.get("name",""),
            "url": row.get("url",""),
            "category": row.get("category",""),
            "job_levels": row.get("job_levels",""),
            "skills": row.get("skills",""),
            "score": round(final_score, 4),
            "evidence": evidence
        })
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        for j, r in enumerate(results, 1):
            r["rank"] = j

    return {"query": query, "results": results}

@app.post("/recommend/pretty")
def recommend_pretty(req: RecommendRequest):
    out = recommend(req)
    if "results" not in out:
        return out

    lines = []
    for r in out["results"]:
        lines.append(
            f"{r['rank']}. {r['name']} ({r['category']} | {r['job_levels']})\n"
            f"   Evidence: {r.get('evidence','')}\n"
            f"   Link: {r['url']}\n"
        )

    return {
        "query": out["query"],
        "summary": "\n".join(lines),
        "results": out["results"]
    }

    
