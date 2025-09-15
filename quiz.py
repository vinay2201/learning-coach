# quiz.py
import os, json, math, re
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from rag import retrieve

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = (
    "You are a strict quiz author. You must return ONLY multiple choice questions in valid JSON. "
    "Each question must include: type='mcq', q, choices (3-5 strings), answer (one of the choices), explanation. "
    "Write explanations as a mini-lesson: define the concept, justify why the correct option is right, "
    "briefly refute each wrong option, and include a small example, rule, or equation if relevant. "
    "Explanations must be clear and self-contained, suitable for a learner."
)

def _ctx_to_text(ctx: List[Dict], limit_chars: int = 2500) -> str:
    blocks, total = [], 0
    for c in ctx:
        t = (c.get("text") or "").strip()
        if not t: continue
        if total + len(t) > limit_chars: break
        blocks.append(t); total += len(t)
    return "\n\n".join(blocks)


def _ask_for_mcqs_fast(topic: str, difficulty: str, k: int, ctx_text: str) -> List[Dict]:
    prompt = (
        "Create exactly {k} multiple-choice questions (MCQs) on the topic below.\n"
        "Difficulty: {difficulty}. Use context only if helpful.\n"
        "Return ONLY valid JSON with keys: questions -> list of objects each with:\n"
        "type='mcq', q, choices (3-5 strings), answer (must equal one of choices), explanation (120-180 words, mini-lesson with brief refutation of distractors).\n\n"
        "Topic: {topic}\n\nContext:\n{ctx}"
    ).format(k=k, difficulty=difficulty, topic=topic, ctx=ctx_text or "(no context)")

    resp = client.chat.completions.create(
        model="gpt-4o-mini",            # ⚡ faster/cheaper; switch to gpt-4o if you want max quality
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content":
             "You return strict JSON for MCQs only. Explanations are clear, ~120-180 words."},
            {"role": "user", "content": prompt},
        ],
        timeout=30,  # seconds
    )
    text = resp.choices[0].message.content.strip()
    try:
        data = json.loads(text)
        qs = data.get("questions", [])
        return [q for q in qs if isinstance(q, dict)]
    except Exception:
        return []



def _normalize_mcq(q: Dict) -> Dict | None:
    # ensure shape and validity
    if (q.get("type") or "").lower() != "mcq": return None
    qtext = (q.get("q") or "").strip()
    choices = q.get("choices") or []
    answer = (q.get("answer") or "").strip()
    expl = (q.get("explanation") or "").strip()
    if not qtext or not isinstance(choices, list) or len(choices) < 2: return None
    # remove empties and whitespace-only
    choices = [c.strip() for c in choices if isinstance(c, str) and c.strip()]
    if not choices or answer not in choices: return None
    return {"type":"mcq","q":qtext,"choices":choices,"answer":answer,"explanation":expl}

def _dedup(qs: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for q in qs:
        key = (q["q"].lower(), tuple(c.lower() for c in q["choices"]))
        if key not in seen:
            seen.add(key); out.append(q)
    return out

def _enrich_explanations(qs: List[Dict], ctx_text: str, min_words: int = 110) -> List[Dict]:
    """Ensure each MCQ has a detailed explanation; expand short ones using context."""
    enriched = []
    for q in qs:
        exp = (q.get("explanation") or "").strip()
        word_count = len(exp.split())
        if word_count >= min_words:
            enriched.append(q)
            continue
        prompt = (
            "Expand the explanation for this MCQ into a 120–220 word mini-lesson.\n"
            "Requirements:\n"
            "• Start with the concept in plain language.\n"
            "• Show a quick example or rule/equation if relevant.\n"
            "• Briefly refute each incorrect option.\n"
            "• Keep it factual and grounded in the given context.\n\n"
            f"Question: {q.get('q')}\n"
            f"Choices: {q.get('choices')}\n"
            f"Correct answer: {q.get('answer')}\n\n"
            f"Existing explanation (may be short): {exp}\n\n"
            f"Context:\n{ctx_text or '(no context)'}"
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.2,
                messages=[{"role":"system","content":"You improve explanations for MCQs, following instructions precisely."},
                          {"role":"user","content":prompt}]
            )
            new_exp = resp.choices[0].message.content.strip()
            q = dict(q); q["explanation"] = new_exp
        except Exception:
            # if enrichment fails, keep original
            pass
        enriched.append(q)
    return enriched

def make_quiz(topic: str, n: int = 5, difficulty: str = "easy", fast: bool = True) -> Dict:
    n = max(1, int(n))
    ctx = retrieve(topic, k=8)                        # smaller k -> faster
    ctx_text = _ctx_to_text(ctx, limit_chars=2500)    # tighter context -> faster

    # Single fast shot
    qs = _ask_for_mcqs_fast(topic, difficulty, n, ctx_text)
    normalized = [m for m in (_normalize_mcq(x) for x in qs) if m]
    normalized = _dedup(normalized)

    # If underfilled, do ONE tiny topic-only top-up (still fast)
    if len(normalized) < n:
        missing = n - len(normalized)
        qs2 = _ask_for_mcqs_fast(topic, difficulty, missing, ctx_text="")
        normalized.extend([m for m in (_normalize_mcq(x) for x in qs2) if m])
        normalized = _dedup(normalized)

    normalized = normalized[:n]
    return {"questions": normalized, "count": len(normalized), "requested": n, "type": "mcq"}
