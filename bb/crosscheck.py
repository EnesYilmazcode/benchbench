"""A third vote from another model family on every item the audit flagged.

The audit's two auditors are both Claude models. Each flagged item (except executed weak-test
findings, which are measured) goes to Codex (GPT-5.6), which sees the item and the same evidence
the reviewers saw but not their verdicts. Its vote is stored next to the result; the report
counts how many flags it confirms.

`python -X utf8 -m bb.crosscheck [bench ...]` updates runs/crosscheck.jsonl. BB_CROSS=gw uses
DeepSeek V4 Pro on the gateway instead, and stops when its balance drops under BB_GW_FLOOR dollars.
"""
import json
import os
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import data
from .agents import ask
from .audit import FLAWED, REVIEW_SCHEMA, RESULTS, ROOT, review_prompt
from .report import ORDER, verdict

OUT = ROOT / "runs" / "crosscheck.jsonl"
FLOOR = float(os.environ.get("BB_GW_FLOOR", "0.3"))
WHO = os.environ.get("BB_CROSS", "codex")


def balance():
    req = urllib.request.Request("https://ai-gateway.vercel.sh/v1/credits",
                                 headers={"Authorization": f"Bearer {os.environ['AI_GATEWAY_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return float(json.loads(r.read())["balance"])


def prompt_for(it, r):
    """Rebuild what the reviewers were shown, minus anything the reviewers themselves wrote."""
    if it["kind"] in ("mcq", "short"):
        ctx = "=== WHAT TWO INDEPENDENT SOLVERS ANSWERED WITHOUT SEEING THE KEY ===\n" + "\n".join(
            f"Solver {i + 1}: answer={s['answer']!r}; flagged={s['flaw'] or 'nothing'}" for i, s in enumerate(r["solve"].values()))
        return review_prompt(it, ctx), None, None
    if it["kind"] == "code":
        from . import code
        ref = code.reference(it)
        body = (f"=== TASK (what a solver sees) ===\n{code.task_text(it)}\n\n=== REFERENCE SOLUTION ===\n```python\n{ref}\n```\n\n"
                f"=== OFFICIAL TESTS ===\n```python\n{it['extra']['test'][:8000]}\n```")
        ex = r["exec"]
        ctx = "=== RUN RESULTS AGAINST THE OFFICIAL TESTS ===\n" + "\n".join(
            f"{k}: {'PASSED' if v['ok'] else 'FAILED'} (exit {v['rc']}): {v['out'][-500:]}" for k, v in ex.items())
        return review_prompt(it, ctx, verdicts=CODE_VERDICTS, body=body), None, None
    from . import tasks
    files, trees, body, verdicts = {"swe": tasks._swe, "agent": tasks._tb if it["bench"] == "terminal_bench2" else tasks._tau}[it["kind"]](it)
    return review_prompt(it, "", verdicts=verdicts, body=body), files, trees


CODE_VERDICTS = (
    "Verdicts:\n- key_correct: task, reference solution and tests are sound.\n- key_wrong: the reference solution or a test's "
    "expected value is wrong.\n- ambiguous: the task text is underspecified, so a reasonable reading fails the tests.\n"
    "- grader_flawed: the tests reject correct solutions for reasons the task does not state.\n- tests_weak: the tests accept "
    "a plausible solution that breaks a stated requirement.\n- unanswerable: the environment or task is broken.\n- unsure.")


def main(benches):
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in OUT.open(encoding="utf-8")}
    todo = []
    for b in benches:
        p = RESULTS / f"{b}.jsonl"
        if not p.exists():
            continue
        sample = {it["id"]: it for it in data.sample(b)}
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            v = verdict(r)
            if v in FLAWED and v != "tests_weak" and r["id"] not in done:
                todo.append((sample[r["id"]], r))
    print(f"{len(todo)} flagged items to cross-check with {WHO}" + (f"; balance ${balance():.2f}" if WHO == "gw" else ""), flush=True)
    lock = threading.Lock()
    stop = threading.Event()

    def one(pair):
        it, r = pair
        if stop.is_set():
            return None
        p, files, trees = prompt_for(it, r)
        return it["id"], ask(WHO, p, REVIEW_SCHEMA, effort="high", files=files, trees=trees, timeout=900, tag=f"cross:{it['id']}")

    with ThreadPoolExecutor(8) as pool:
        futs = [pool.submit(one, x) for x in todo]
        for n, f in enumerate(as_completed(futs), 1):
            try:
                res = f.result()
            except Exception as e:
                print("  FAIL", str(e)[:200], flush=True)
                continue
            if res is None:
                continue
            with lock, OUT.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"id": res[0], "vote": res[1]}, ensure_ascii=False) + "\n")
            if n % 10 == 0:
                print(f"  [{n}/{len(todo)}]", flush=True)
            if WHO == "gw" and n % 10 == 0:
                b = balance()
                print(f"  [{n}/{len(todo)}] balance ${b:.2f}", flush=True)
                if b < FLOOR:
                    stop.set()
                    print("  gateway balance under the floor; stopping", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ORDER)
