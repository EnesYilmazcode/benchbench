"""The audit: blind solve, review, adjudicate.

Questions (mcq, short):
  1. solve   Claude and Codex each answer without seeing the key, and may flag a broken item.
  2. review  Items where either disagrees with the key or flags it go to both auditors again, now
             with the key, the other answers, tools and the web. Each returns a verdict.
  3. judge   When the two verdicts disagree, a third call reads both and decides.
Code, SWE and agent tasks have their own first steps (bb/code.py, bb/tasks.py) and share 2 and 3.

`python -X utf8 -m bb.audit <bench> [--limit N]` writes runs/results/<bench>.jsonl.
"""
import argparse
import json
import os
import re
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import data
from .agents import ask

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs" / "results"
TIMEOUT = int(os.environ.get("BB_TIMEOUT", "1200"))  # seconds per auditor call; heavy agent tasks need more
AUDITORS = ("claude", os.environ.get("BB_SECOND", "sonnet"))  # Claude Code on Opus, and a second auditor

NAMES = {
    "mmlu": "MMLU", "mmlu_redux_val": "MMLU", "mmlu_pro": "MMLU-Pro", "gpqa_diamond": "GPQA Diamond", "arc_challenge": "ARC-Challenge",
    "hellaswag": "HellaSwag", "truthfulqa": "TruthfulQA (MC1)", "commonsense_qa": "CommonsenseQA",
    "winogrande": "WinoGrande", "gsm8k": "GSM8K", "math500": "MATH-500", "aime2025": "AIME 2025",
    "hmmt2025": "HMMT February 2025", "simpleqa": "SimpleQA", "simpleqa_verified": "SimpleQA Verified",
    "hle": "Humanity's Last Exam", "frontierscience": "FrontierScience (Olympiad)", "humaneval": "HumanEval",
    "mbpp": "MBPP (sanitized)", "bigcodebench": "BigCodeBench", "swe_verified": "SWE-bench Verified",
    "swe_pro": "SWE-bench Pro", "terminal_bench2": "Terminal-Bench 2.0", "tau2": "tau2-bench",
}
FACTUAL = {"simpleqa", "simpleqa_verified", "truthfulqa"}  # review needs the web to check facts

ISSUES = ["wrong_key", "multiple_correct", "no_correct_option", "ambiguous_question", "missing_info",
          "outdated_fact", "garbled_text", "answer_leaked", "grader_too_strict", "grader_too_loose",
          "tests_wrong", "tests_weak", "underspecified_task", "spec_test_mismatch", "environment_broken", "other"]
VERDICTS = ["key_correct", "key_wrong", "ambiguous", "unanswerable", "grader_flawed", "tests_weak", "unsure"]
FLAWED = {"key_wrong", "ambiguous", "unanswerable", "grader_flawed", "tests_weak"}

SOLVE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["answer", "confidence", "flaw"],
    "properties": {
        "answer": {"type": "string", "description": "Your final answer only: one letter for multiple choice, else as short as possible."},
        "confidence": {"type": "number"},
        "flaw": {"type": "string", "description": "Empty if the item looks sound. Otherwise one sentence on what is broken."},
    },
}
REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["verdict", "correct_answer", "issues", "explanation", "evidence", "confidence"],
    "properties": {
        "verdict": {"type": "string", "enum": VERDICTS},
        "correct_answer": {"type": "string", "description": "What the answer should be, if not the official one; else empty."},
        "issues": {"type": "array", "items": {"type": "string", "enum": ISSUES}},
        "explanation": {"type": "string", "description": "At most 60 words, plain language."},
        "evidence": {"type": "string", "description": "Derivation, computation or primary sources (URLs). At most 100 words."},
        "confidence": {"type": "number"},
    },
}


# ---------- presenting an item ----------

def show_question(it):
    q = it["question"].strip()
    if it["kind"] == "mcq":
        q += "\n\nOptions:\n" + "\n".join(f"{data.LETTERS[i]}. {c}" for i, c in enumerate(it["choices"]))
    return q


def show_key(it):
    if it["kind"] == "mcq":
        return f"{it['answer']}. {it['choices'][data.LETTERS.index(it['answer'])]}"
    return it["answer"]


# ---------- step 1: blind solve ----------

def solve_prompt(it):
    if it["kind"] == "mcq":
        how = "Answer this multiple-choice question. Reply with the letter of the single best option."
    else:
        how = ("Answer this question. Give only the final answer, as short as possible "
               "(a number, expression, name or short phrase), no working.")
    return (f"{how}\nIt comes from the benchmark {NAMES[it['bench']]}. Then judge the item itself, and fill `flaw` "
            f"(one sentence) if any of these hold: your answer is only the least-bad choice rather than actually "
            f"correct; another option is about as defensible; the wording is unclear enough that careful experts "
            f"would read it differently; it needs context it does not give; the text is garbled. Leave `flaw` empty "
            f"for a sound item, and do not invent problems.\n\n---\n{show_question(it)}\n---")


def _norm(s):
    s = str(s).strip().lower()
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)
    s = re.sub(r"\\(text|mathrm|textbf)\{([^}]*)\}", r"\2", s)
    s = s.replace("$", "").replace("\\!", "").replace("\\,", "").replace("dfrac", "frac").replace("tfrac", "frac")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = re.sub(r"[\s\.]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def matches(it, ans):
    if it["kind"] == "mcq":
        m = re.match(r"\s*\(?([A-P])\b", str(ans).strip().upper())
        return bool(m) and m.group(1) == it["answer"]
    a, k = _norm(ans), _norm(it["answer"])
    if a == k:
        return True
    try:
        return abs(float(a) - float(k)) <= 1e-6 * max(1, abs(float(k)))
    except ValueError:
        return False


def blind(it):
    out = {}
    for who in AUDITORS:
        out[who] = ask(who, solve_prompt(it), SOLVE_SCHEMA, effort="medium", tag=f"solve:{it['id']}")
        out[who]["match"] = matches(it, out[who]["answer"])
    return out


# ---------- step 2: review ----------

QUESTION_VERDICTS = (
    "Verdicts:\n"
    "- key_correct: the official answer is right and the item is fine. Solvers who disagreed made a mistake, or "
    "gave an answer the grader should accept as equivalent.\n"
    "- key_wrong: the official answer is wrong and a different answer is clearly right.\n"
    "- ambiguous: more than one answer is defensible, or the wording is unclear enough that experts would split.\n"
    "- unanswerable: no option is right, needed information is missing, or the text is garbled.\n"
    "- unsure: you cannot decide with confidence.\n"
    "Work it out properly: derive, compute (you may run code), and check facts against primary sources. Do not "
    "flag an item over a pedantic quibble that no reasonable expert would raise; do flag it if a careful expert "
    "would mark the official answer wrong.")


def review_prompt(it, context, verdicts=QUESTION_VERDICTS, body=None):
    if body is None:  # questions: keep this wording fixed, the cache keys on it
        return (f"An item from the benchmark {NAMES[it['bench']]} needs checking. Decide whether its official answer "
                f"is right and the item is sound.\n\n=== QUESTION ===\n{show_question(it)}\n\n=== OFFICIAL ANSWER ===\n"
                f"{show_key(it)}\n\n{context}\n\n{verdicts}")
    return (f"An item from the benchmark {NAMES[it['bench']]} needs checking. Decide whether it is sound.\n\n"
            f"{body}\n\n{context}\n\n{verdicts}")


def review(it, context, tools_claude, tools_codex, effort="high", files=None, trees=None, **kw):
    p = review_prompt(it, context, **kw)
    return {who: ask(who, p, REVIEW_SCHEMA, effort=effort, tools=(tools_claude if who in ("claude", "sonnet") else tools_codex),
                     timeout=TIMEOUT, files=files, trees=trees, tag=f"review:{it['id']}") for who in AUDITORS}


def _sound(v):
    return v["verdict"] == "key_correct"


def judge(it, context, reviews, files=None, trees=None, **kw):
    """Settle a split between the two reviews with a third, stronger call."""
    a, b = (reviews[w] for w in AUDITORS)
    if _sound(a) and _sound(b):
        return {"verdict": "key_correct", "by": "both"}
    if a["verdict"] in FLAWED and b["verdict"] in FLAWED:
        return {"verdict": a["verdict"], "by": "both", "issues": sorted(set(a["issues"]) | set(b["issues"]))}
    ctx = (context + "\n\n=== TWO INDEPENDENT REVIEWS (they disagree) ===\n"
           + "\n".join(f"Reviewer {n}: verdict={r['verdict']}; correct_answer={r['correct_answer']!r}; "
                       f"issues={r['issues']}; explanation={r['explanation']} evidence={r['evidence']}"
                       for n, r in (("1", a), ("2", b))))
    v = ask("claude", review_prompt(it, ctx, **kw) + "\nWeigh both reviews, check their claims yourself, and give the final verdict.",
            REVIEW_SCHEMA, effort="xhigh", tools=("Bash", "Read", "Grep", "Glob", "WebSearch", "WebFetch"), timeout=TIMEOUT,
            files=files, trees=trees, tag=f"judge:{it['id']}")
    return {**v, "by": "judge"}


# ---------- one item, end to end ----------

def audit_question(it):
    rec = {"id": it["id"], "bench": it["bench"], "kind": it["kind"], "key": it["answer"]}
    rec["solve"] = blind(it)
    suspect = [w for w in AUDITORS if not rec["solve"][w]["match"] or rec["solve"][w]["flaw"].strip()]
    if not suspect:
        rec["final"] = {"verdict": "key_correct", "by": "solve"}
        return rec
    context = "=== WHAT TWO INDEPENDENT SOLVERS ANSWERED WITHOUT SEEING THE KEY ===\n" + "\n".join(
        f"Solver {i + 1}: answer={rec['solve'][w]['answer']!r}; flagged={rec['solve'][w]['flaw'] or 'nothing'}"
        for i, w in enumerate(AUDITORS))
    web = it["bench"] in FACTUAL or it["kind"] == "short" and it["bench"] in {"hle", "frontierscience"}
    tc = ("Bash", "Read", "WebSearch", "WebFetch")
    rec["review"] = review(it, context, tc, ("web",) if web or it["kind"] == "mcq" else ())
    rec["final"] = judge(it, context, rec["review"])
    return rec


def audit_item(it):
    if it["kind"] in ("mcq", "short"):
        return audit_question(it)
    if it["kind"] == "code":
        from .code import audit_code
        return audit_code(it)
    from .tasks import audit_task
    return audit_task(it)


def run(bench, limit=None, workers=10):
    items = data.sample(bench)[:limit] if limit else data.sample(bench)
    out = RESULTS / f"{bench}.jsonl"
    done = {}
    if out.exists():
        for line in out.open(encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r
    todo = [it for it in items if it["id"] not in done]
    print(f"{bench}: {len(items)} items, {len(done)} already done, {len(todo)} to go", flush=True)
    lock = threading.Lock()
    failed = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(audit_item, it): it for it in todo}
        for n, f in enumerate(as_completed(futs), 1):
            it = futs[f]
            try:
                rec = f.result()
            except Exception as e:
                failed.append(it["id"])
                print(f"  FAIL {it['id']}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                traceback.print_exc(limit=2, file=sys.stderr)
                continue
            with lock, out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            v = rec["final"]["verdict"]
            if v != "key_correct" or n % 10 == 0:
                print(f"  [{n}/{len(todo)}] {it['id']}: {v} ({rec['final'].get('by')})", flush=True)
    if failed:
        raise SystemExit(f"{bench}: {len(failed)} items failed and are not in the results: {failed[:10]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("benches", nargs="+")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    for b in a.benches:
        run(b, a.limit, a.workers)
