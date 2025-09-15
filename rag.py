# rag.py — simple NumPy index (no Chroma)
import os, json
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
from rapidfuzz import fuzz

load_dotenv()
APP = Path(__file__).resolve().parent
OUTDIR = APP / "data" / "simple"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# lazy globals
APP = Path(__file__).resolve().parent
OUTDIR = APP / "data" / "simple"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_VEC = None
_META = None
_LAST_MTIME = 0.0

def _index_mtime():
    idx = OUTDIR / "index.npy"
    meta = OUTDIR / "meta.jsonl"
    if not idx.exists() or not meta.exists():
        return 0.0
    return max(idx.stat().st_mtime, meta.stat().st_mtime)

def _load_index(force: bool = False):
    global _VEC, _META, _LAST_MTIME
    idx_path = OUTDIR / "index.npy"
    meta_path = OUTDIR / "meta.jsonl"
    if not idx_path.exists() or not meta_path.exists():
        raise RuntimeError("Simple index not found. Run `python ingest.py` first.")

    mtime = _index_mtime()
    if force or _VEC is None or mtime > _LAST_MTIME:
        _VEC = np.load(idx_path).astype(np.float32)
        with open(meta_path, "r", encoding="utf-8") as f:
            _META = [json.loads(line) for line in f]
        _LAST_MTIME = mtime

def reload_index():
    _load_index(force=True)

def _embed_query(q: str):
    resp = client.embeddings.create(model=EMBED_MODEL, input=q)
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    v /= (np.linalg.norm(v) + 1e-12)
    return v

def retrieve(query: str, k: int = 8) -> List[Dict]:
    _load_index()
    qv = _embed_query(query)                # [D]
    sims = (_VEC @ qv)                      # cosine via dot (both normalized)
    topk_idx = np.argsort(-sims)[:max(k, 4)]
    hits = []
    for i in topk_idx:
        m = _META[int(i)]
        hits.append({
            "id": i.item(),
            "text": m["text"],
            "source": m.get("source"),
            "chunk": m.get("chunk"),
            "score": float(sims[int(i)]),
        })
    # light lexical rerank to bubble literal matches
    hits.sort(key=lambda h: (h["score"], fuzz.token_set_ratio(query, h["text"])), reverse=True)
    return hits[:k]

SYSTEM = (
    "You are a helpful subject tutor. "
    "Answer based ONLY on the provided context chunks. "
    "If the answer is not in the context, say you don't know. "
    "Keep answers concise, structured, and include inline citations like (source.pdf #chunk)."
)

def _format_context(ctx: List[Dict]) -> str:
    blocks = []
    for i, c in enumerate(ctx, 1):
        text = c["text"]
        if len(text) > 1200:
            text = text[:1200] + "…"
        src = c.get("source") or "unknown"
        blk = f"[{i}] ({src} #{c.get('chunk')}): {text}"
        blocks.append(blk)
    return "\n\n".join(blocks)

def answer(query: str):
    ctx = retrieve(query)
    context_block = _format_context(ctx) if ctx else "[no context found]"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context_block}"}
    ]
    # NOTE: the variable name is *client*, not clie...
    resp = client.chat.completions.create(
        model="gpt-4o",          # or "gpt-4o-mini" if you prefer cheaper
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content, ctx

