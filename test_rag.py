import os, traceback
from dotenv import load_dotenv

print("Step 0: loading .env …")
load_dotenv()
print("Has OPENAI_API_KEY?", bool(os.getenv("OPENAI_API_KEY")))

# 1) OpenAI chat sanity check
print("\nStep 1: OpenAI chat sanity check …")
try:
    from openai import OpenAI
    oc = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    mini_model = "gpt-4o-mini"  # cheaper; if unavailable, we'll try gpt-4o
    try:
        msg = oc.chat.completions.create(model=mini_model, messages=[{"role":"user","content":"Say OK if you can see this."}])
    except Exception:
        # fallback if account doesn't have mini model
        msg = oc.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":"Say OK if you can see this."}])
    print("OpenAI reply:", msg.choices[0].message.content.strip())
except Exception as e:
    print("OpenAI chat failed:", e)
    print(traceback.format_exc())

# 2) Chroma retrieval sanity check
print("\nStep 2: Chroma retrieval sanity check …")
try:
    import chromadb
    from chromadb.utils import embedding_functions
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-large",  # MUST match ingest.py
    )
    client = chromadb.PersistentClient(path="data/chroma")
    try:
        col = client.get_collection("course", embedding_function=ef)
        print("Got collection: course")
    except Exception as e:
        print("get_collection failed, creating:", e)
        col = client.get_or_create_collection("course", embedding_function=ef)
        print("Created collection: course (may be empty)")
    qr = col.query(query_texts=["gradient descent"], n_results=3)
    print("Query IDs:", qr.get("ids"))
    print("Query docs len:", len(qr.get("documents", [[]])[0]) if qr.get("documents") else 0)
except Exception as e:
    print("Chroma retrieval failed:", e)
    print(traceback.format_exc())

# 3) Full RAG answer() test
print("\nStep 3: RAG answer() test …")
try:
    from rag import answer
    q = "Explain the difference between gradient descent and stochastic gradient descent."
    ans, ctx = answer(q)
    print("\n--- ANSWER ---\n", ans)
    print("\n--- CONTEXT (first 2 chunks) ---")
    for c in (ctx or [])[:2]:
        t = (c["text"] or "").replace("\n"," ")
        print(f"{c['source']} #{c['chunk']} :: {t[:180]}")
except Exception as e:
    print("answer() failed:", e)
    print(traceback.format_exc())
