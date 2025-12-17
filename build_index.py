import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_CSV = ROOT / "data" / "catalog.csv"
INDEX_PATH = ROOT / "data" / "catalog.faiss"
META_PATH  = ROOT / "data" / "catalog_meta.pkl"

def normalize(s: str) -> str:
    s = s or ""
    return re.sub(r"\s+", " ", s.strip())

df = pd.read_csv(CATALOG_CSV).fillna("")
df["doc"] = (
    df["name"].astype(str) + " " +
    df["description"].astype(str) + " " +
    df["skills"].astype(str) + " " +
    df["category"].astype(str) + " " +
    df["job_levels"].astype(str)
).map(normalize)

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(df["doc"].tolist(), show_progress_bar=True, normalize_embeddings=True)
emb = np.asarray(emb, dtype="float32")

index = faiss.IndexFlatIP(emb.shape[1])  # cosine via inner product because normalized
index.add(emb)

faiss.write_index(index, str(INDEX_PATH))
df.to_pickle(META_PATH)

print("✅ Built:", INDEX_PATH)
print("✅ Saved:", META_PATH)
