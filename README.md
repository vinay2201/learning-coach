# 🎓 Hyper-Personalized Learning Coach (RAG Tutor)

A fast, grounded learning assistant you can run locally. Upload your course materials
(PDF/Markdown/TXT), ask questions with citations, and generate quizzes.

## Features
- Document uploads (PDF/MD/TXT)
- Automatic chunking + embeddings (ChromaDB)
- Grounded chat answers with citations
- Quiz generator (MCQ/short) by topic & difficulty
- Local-only storage (./data/)

## Quickstart
```bash
# 1) Create and activate a venv (recommended)
python -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Set your OpenAI key
cp .env.example .env
# then edit .env with your key

# 4) Run the app
streamlit run app.py
```

### Usage
1. Upload PDFs/notes in the sidebar.
2. Click **Reindex** to build the vector index.
3. Ask a question in the main input.
4. Generate a quiz in the **Quiz Generator** section.

### Notes
- First run may take a moment to embed your documents.
- Everything is stored locally in `./data/`.
- For cheaper indexing, switch to `text-embedding-3-small` in `ingest.py`.
