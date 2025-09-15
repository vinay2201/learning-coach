import sys, os, traceback
print("=== START test_query_embed.py ===", flush=True)

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("ENV OK. Has OPENAI_API_KEY?", bool(os.getenv("OPENAI_API_KEY")) , flush=True)
except Exception as e:
    print("dotenv/load error:", e, flush=True)
    print(traceback.format_exc(), flush=True)

# 1) OpenAI sanity
print("\n[1] OpenAI chat sanity …", flush=True)
try:
    from openai import OpenAI
    oc = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    reply = oc.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":"Say OK if you can see this."}]
    )
    print("OpenAI reply:", reply.choices[0].message.content.strip(), flush=True)
except Exception as e:
    print("OpenAI chat failed:", e, flush=True)
    print(traceback.format_exc(), flush=True)

# 2) Manual query embedding + Chroma query
print("\n[2] Manual query embedding + Chroma …", flush=True)
try:
    from openai import OpenAI
    oc = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("Embedding…", flush=True)
    emb = oc.embeddings.create(model="text-embedding-3-large", input="gradient descent vs sgd").data[0].embedding
    print("Embedding len:", len(emb), flush=True)

    import chromadb
    print("Connecting to Chroma…", flush=True)
    col = chromadb.PersistentClient(path="data/chroma").get_collection("course")
    print("Querying with query_embeddings…", flush=True)
    res = col.query(query_embeddings=[emb], n_results=3)
    print("IDs:", res.get("ids"), flush=True)
    docs = res.get("documents", [[]])
    snippet = (docs[0][0] or "")[:200] if docs and docs[0] else ""
    print("First doc snippet:", snippet, flush=True)
except Exception as e:
    print("Chroma query failed:", e, flush=True)
    print(traceback.format_exc(), flush=True)

# 3) Full RAG answer()
print("\n[3] RAG answer() …", flush=True)
try:
    from rag import answer
    q = "Explain the difference between gradient descent and stochastic gradient descent."
    ans, ctx = answer(q)
    print("--- ANSWER ---\n", ans, flush=True)
    print("--- CONTEXT (up to 2) ---", flush=True)
    for c in (ctx or [])[:2]:
        print(f"{c.get('source')} #{c.get('chunk')} :: {(c.get('text') or '').replace(chr(10),' ')[:180]}", flush=True)
except Exception as e:
    print("answer() failed:", e, flush=True)
    print(traceback.format_exc(), flush=True)

print("\n=== END test_query_embed.py ===", flush=True)
