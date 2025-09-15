# ingest.py — simple NumPy index (no Chroma)
import os, re, glob, json
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from markdown_it import MarkdownIt
import numpy as np
from openai import OpenAI

load_dotenv()
APP = Path(__file__).resolve().parent
DATA = APP / "data"
SRC = DATA / "sources"
OUTDIR = DATA / "simple"
OUTDIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def log(x):
    try:
        print(x, flush=True)
    except UnicodeEncodeError:
        # Fallback: strip non-ASCII for legacy consoles
        if not isinstance(x, str):
            x = str(x)
        print(x.encode("ascii", "ignore").decode(), flush=True)

def read_pdf(p: Path) -> str:
    doc = PdfReader(str(p))
    return "\n".join((pg.extract_text() or "") for pg in doc.pages)

def read_md(p: Path) -> str:
    text = p.read_text(encoding="utf-8", errors="ignore")
    return MarkdownIt().render(text)

def read_txt(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def chunk_text(text: str, max_tokens=300, overlap=50):
    if len(text) > 2_000_000:
        text = text[:2_000_000]
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks, buf, tok = [], [], 0

    def flush():
        nonlocal buf
        if not buf: return
        joined = " ".join(buf).strip()
        max_chars = 350*4
        if len(joined) > max_chars: joined = joined[:max_chars]
        if joined: chunks.append(joined)
        buf = []

    for p in parts:
        t = (p or "").strip()
        if not t: continue
        est = max(1, len(t)//4)
        if tok + est > max_tokens and buf:
            flush()
            tail = (" ".join(buf))[-overlap*4:] if overlap else ""
            buf = [tail] if tail else []
            tok = sum(len(s)//4 for s in buf)
        buf.append(t); tok += est
    flush()
    return [c for c in chunks if c.strip()]

def embed_texts(texts):
    vecs = []
    B = 64
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([d.embedding for d in resp.data])
        log(f"   • Embedded {i+len(batch)}/{len(texts)}")
    arr = np.array(vecs, dtype=np.float32)
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
    return arr

def upsert_files():
    log("Ingest (simple) starting")
    SRC.mkdir(parents=True, exist_ok=True)
    files = [p for p in SRC.glob("*") if p.is_file()]
    if not files:
        log("No files in data/sources"); return

    docs, metas = [], []
    for p in files:
        try:
            if p.suffix.lower()==".pdf": text = read_pdf(p)
            elif p.suffix.lower() in [".md",".markdown"]: text = read_md(p)
            else: text = read_txt(p)
        except Exception as e:
            log(f"Read failed {p.name}: {e}"); continue
        chunks = chunk_text(text, max_tokens=300, overlap=50)
        log(f"   • {p.name}: {len(chunks)} chunks")
        for i,ch in enumerate(chunks):
            docs.append(ch); metas.append({"source": p.name, "chunk": i})

    if not docs:
        log("No extractable text found"); return

    log("Embedding …")
    vecs = embed_texts(docs)
    np.save(OUTDIR / "index.npy", vecs)
    with open(OUTDIR / "meta.jsonl","w",encoding="utf-8") as f:
        for m,t in zip(metas, docs):
            m = dict(m); m["text"] = t
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    log(f"Indexed {len(docs)} chunks from {len(set(m['source'] for m in metas))} file(s).")
    log(f"Saved: {OUTDIR/'index.npy'}, {OUTDIR/'meta.jsonl'}")

if __name__ == "__main__":
    upsert_files()
