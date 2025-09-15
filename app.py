import sys, subprocess, traceback
import streamlit as st
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from rag import answer, reload_index
from quiz import make_quiz

load_dotenv()
st.set_page_config(page_title="Learning Coach", page_icon="📚", layout="wide")

st.title("Personalized Learning Coach")
st.caption("Upload course materials → Chat with citations → Generate quizzes. Everything runs locally except the LLM.")

with st.sidebar:
    st.header("Upload")
    up = st.file_uploader("PDF/MD/TXT", type=["pdf","md","markdown","txt"], accept_multiple_files=True)
    if up:
        SRC = Path("data/sources"); SRC.mkdir(parents=True, exist_ok=True)
        for f in up:
            (SRC / f.name).write_bytes(f.read())
        st.success(f"Saved {len(up)} file(s). Click **Reindex** below.")
    if st.button("Reindex"):
        with st.spinner("Indexing your materials…"):
            proc = subprocess.run(
                [sys.executable, "ingest.py"],
                capture_output=True,
                text=True
            )
            # Show logs either way
            st.text((proc.stdout or "") + (proc.stderr or ""))

            if proc.returncode != 0:
                st.error(f"❌ Reindex failed (exit {proc.returncode}). See logs above.")
            else:
                try:
                    reload_index()   # hot-reload in-memory vectors
                    st.success("✅ Reindex complete. Index reloaded.")
                except Exception as e:
                    st.error(f"Index reload failed: {e}")
                    st.code(traceback.format_exc())
    st.markdown("---")
    st.caption("Tip: After adding or changing files, click **Reindex**.")

# --- Chat (form + state) ---
st.subheader("💬 Ask about your course")
if "chat_answer" not in st.session_state:
    st.session_state.chat_answer = None
    st.session_state.chat_ctx = []

with st.form("chat_form", clear_on_submit=False):
    q = st.text_input("Your question", placeholder="e.g., Explain backpropagation with a small numeric example.")
    ask = st.form_submit_button("Ask")

if ask and q:
    try:
        with st.spinner("Thinking…"):
            ans, ctx = answer(q)
        st.session_state.chat_answer = ans
        st.session_state.chat_ctx = ctx
    except Exception as e:
        import traceback
        st.error(f"❌ Error while answering: {e}")
        st.code(traceback.format_exc())

# Show result if present
if st.session_state.chat_answer:
    st.markdown(st.session_state.chat_answer)
    with st.expander("Show context chunks"):
        for c in st.session_state.chat_ctx:
            st.write(f"- **{c['source']}** (#{c['chunk']})")
            st.write((c['text'][:600] + '…') if len(c['text'])>600 else c['text'])

# --- 📝 Quiz Generator (MCQ only) ---
st.subheader("Quiz Generator")

# Keep quiz state across reruns
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "quiz_checked" not in st.session_state:
    st.session_state.quiz_checked = False
if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = {}
if "quiz_preview" not in st.session_state:
    st.session_state.quiz_preview = False  # Preview toggle state

with st.form("quiz_setup_form", clear_on_submit=False):
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        topic = st.text_input("Topic/Chapter", placeholder="Chapter 3: Linear Regression", key="topic")
    with c2:
        n = st.number_input("Questions", 1, 20, 5, key="n")
    with c3:
        diff = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], key="diff")
    with c4:
        shuffle = st.checkbox("Shuffle Answer Order", value=True)
    gen = st.form_submit_button("▸  Generate MCQ Quiz")  # subtle triangle

if gen:
    if not topic.strip():
        st.warning("Please enter a topic or chapter.")
    else:
        try:
            with st.spinner("Generating Quiz"):
                quiz = make_quiz(topic, n=int(n), difficulty=diff)
            # force MCQ only and optionally shuffle choices
            import random
            filtered = []
            for q in quiz.get("questions", []):
                if q.get("type") == "mcq":
                    if shuffle and isinstance(q.get("choices"), list):
                        random.shuffle(q["choices"])
                    filtered.append(q)
            st.session_state.quiz_data = {"questions": filtered}
            st.session_state.quiz_checked = False
            st.session_state.quiz_results = {}
            st.session_state.quiz_preview = False  # reset preview on fresh generation
        except Exception as e:
            st.error(f"Error generating quiz: {e}")
            st.code(traceback.format_exc())

quiz = st.session_state.quiz_data
if quiz and quiz.get("questions"):
    st.markdown("### Take the quiz")

    # Top row: Preview toggle only (no extra options)
    pc1, pc2, _ = st.columns([1,1,6])
    with pc1:
        if st.button(("▸ Preview" if not st.session_state.quiz_preview else "▾ Hide preview"), key="preview_btn"):
            st.session_state.quiz_preview = not st.session_state.quiz_preview

    # Optional preview: shows Q + correct answer so users know the style/content
    if st.session_state.quiz_preview:
        with st.expander("Preview (questions and correct answers)"):
            for i, q in enumerate(quiz["questions"], 1):
                st.write(f"**Q{i}. {q['q']}**")
                st.caption(f"Answer: {q.get('answer','')}")
                st.divider()

    # capture all answers in one form
    with st.form("quiz_take_form", clear_on_submit=False):
        for i, q in enumerate(quiz["questions"], 1):
            st.markdown(f"**Q{i}. {q['q']}**")
            st.radio(
                "Select an answer:",
                options=q.get("choices", []),
                index=None,
                key=f"q{i}_choice",
                label_visibility="collapsed",
            )
            st.divider()

        c1, c2 = st.columns([1, 1])
        check = c1.form_submit_button("✓ Check answers")  # clean checkmark
        reset = c2.form_submit_button("↻ Reset")          # clockwise arrow

    if reset:
        # clear selections & reset results
        for i in range(1, len(quiz["questions"]) + 1):
            st.session_state.pop(f"q{i}_choice", None)
        st.session_state.quiz_checked = False
        st.session_state.quiz_results = {}
        st.rerun()

    if check:
        results = {}
        score = 0
        total = len(quiz["questions"])

        def norm(s):
            return (s or "").strip().lower()

        for i, q in enumerate(quiz["questions"], 1):
            user = st.session_state.get(f"q{i}_choice", "")
            correct = q.get("answer", "")
            ok = norm(user) == norm(correct)
            results[i] = {
                "user": user,
                "correct": correct,
                "ok": ok,
                "explanation": q.get("explanation", ""),
            }
            if ok:
                score += 1

        st.session_state.quiz_results = results
        st.session_state.quiz_checked = True

    if st.session_state.quiz_checked:
        results = st.session_state.quiz_results
        score = sum(1 for r in results.values() if r["ok"])
        total = len(results)
        st.markdown(f"### Score: **{score} / {total}**")

        for i, q in enumerate(quiz["questions"], 1):
            r = results[i]
            if r["ok"]:
                st.success(f"Q{i}: Correct")
            else:
                st.error(f"Q{i}: Incorrect")
            st.write(f"**Your answer:** {r['user'] or '_no answer_'}")
            st.write(f"**Correct answer:** {r['correct']}")
            if r["explanation"]:
                with st.expander("Explanation"):
                    st.caption(r["explanation"])
            st.divider()

