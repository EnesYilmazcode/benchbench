"""Memorization probe: do models reproduce answer keys that are known to be wrong?

MMLU-Redux experts found 101 MMLU questions whose official key is wrong and wrote down the right
answer. A model that reasons should give the expert answer. A model that gives the wrong official
key, on a question where that key is wrong, has most likely seen the answer key in training.

Each question is asked twice: in the original option order, and with the options shuffled. A model
that memorized the key's letter still says that letter after the options move.

`python -X utf8 -m bb.memo run claude sonnet haiku` writes runs/memo/<model>.jsonl (resumable);
`python -X utf8 -m bb.memo score` prints the table and writes results/memo.json.
"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from .agents import ask
from .audit import ROOT

OUT = ROOT / "runs" / "memo"
LETTERS = "ABCD"
SYSTEM = "You answer multiple-choice questions. Your final message must be only the JSON object the schema asks for."
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer"],
          "properties": {"answer": {"type": "string", "enum": list(LETTERS)}}}
MODELS = {"claude": ("claude", "opus"), "sonnet": ("sonnet", "sonnet"), "haiku": ("sonnet", "haiku"),
          "codex": ("codex", None)}


def items():
    """The wrong-key MMLU-Redux questions that have a single expert answer."""
    cache = ROOT / "data" / "memo_items.jsonl"
    if cache.exists():
        return [json.loads(l) for l in cache.open(encoding="utf-8")]
    from datasets import get_dataset_config_names, load_dataset
    out = []
    for s in get_dataset_config_names("edinburgh-dawg/mmlu-redux-2.0"):
        for i, r in enumerate(load_dataset("edinburgh-dawg/mmlu-redux-2.0", s, split="test")):
            fix = str(r["correct_answer"]).strip()
            fix = LETTERS.index(fix) if fix in LETTERS else int(fix) if fix in "0123" and fix else None
            if r["error_type"] != "wrong_groundtruth" or fix is None or fix == r["answer"] or len(r["choices"]) != 4:
                continue
            out.append({"id": f"{s}/{i}", "question": r["question"], "choices": r["choices"], "key": r["answer"], "fix": fix})
    cache.parent.mkdir(exist_ok=True)
    cache.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    return out


def variants(it):
    """Original order, and one fixed shuffle in which neither the key nor the fix keeps its letter."""
    yield "orig", list(range(4))
    rng = random.Random(it["id"])
    while True:
        perm = list(range(4))
        rng.shuffle(perm)
        if perm.index(it["key"]) != it["key"] and perm.index(it["fix"]) != it["fix"]:
            yield "shuf", perm
            return


def prompt(it, perm):
    opts = "\n".join(f"{LETTERS[j]}. {it['choices'][k]}" for j, k in enumerate(perm))
    return f"Answer this multiple-choice question with the letter of the correct option.\n\n{it['question']}\n\n{opts}"


def run(models):
    OUT.mkdir(parents=True, exist_ok=True)
    qs = items()
    for m in models:
        who, model = MODELS[m]
        path = OUT / f"{m}.jsonl"
        done = {(r["id"], r["v"]) for r in map(json.loads, path.open(encoding="utf-8"))} if path.exists() else set()
        jobs = [(it, v, perm) for it in qs for v, perm in variants(it) if (it["id"], v) not in done]
        print(f"{m}: {len(qs)} questions, {len(jobs)} calls to go", flush=True)

        def one(job):
            it, v, perm = job
            a = ask(who, prompt(it, perm), SCHEMA, effort="low", model=model, system=SYSTEM, tag=f"memo:{m}:{it['id']}:{v}")
            return {"id": it["id"], "v": v, "perm": perm, "answer": a["answer"]}

        with ThreadPoolExecutor(6) as pool, path.open("a", encoding="utf-8") as fh:
            for n, f in enumerate(as_completed([pool.submit(one, j) for j in jobs]), 1):
                try:
                    fh.write(json.dumps(f.result()) + "\n")
                    fh.flush()
                except Exception as e:  # one bad call must not lose the rest
                    print(f"  FAIL {e}"[:300], flush=True)
                if n % 25 == 0:
                    print(f"  [{n}/{len(jobs)}]", flush=True)


def score():
    qs = {it["id"]: it for it in items()}
    table = {}
    for path in sorted(OUT.glob("*.jsonl")):
        rows = [json.loads(l) for l in path.open(encoding="utf-8")]
        t = {"orig": {"fix": 0, "key": 0, "other": 0}, "shuf": {"fix": 0, "key": 0, "key_letter": 0, "other": 0}}
        for r in rows:
            it, perm = qs[r["id"]], r["perm"]
            picked = perm[LETTERS.index(r["answer"])]  # index into the original choices
            c = t[r["v"]]
            if picked == it["fix"]:
                c["fix"] += 1
            elif picked == it["key"]:
                c["key"] += 1
            elif r["v"] == "shuf" and LETTERS.index(r["answer"]) == it["key"]:
                c["key_letter"] += 1  # the letter the wrong key had before the shuffle
            else:
                c["other"] += 1
        table[path.stem] = t
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "memo.json").write_text(json.dumps({"questions": len(qs), "models": table}, indent=1), encoding="utf-8")
    print(f"{len(qs)} MMLU questions whose official key is wrong (MMLU-Redux experts)")
    print(f"{'model':8s} {'expert answer':>14s} {'wrong key':>10s} {'other':>6s} | shuffled: {'expert':>6s} {'wrong key':>10s} {'old letter':>10s}")
    for m, t in table.items():
        o, s = t["orig"], t["shuf"]
        print(f"{m:8s} {o['fix']:14d} {o['key']:10d} {o['other']:6d} |           {s['fix']:6d} {s['key']:10d} {s['key_letter']:10d}")


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2:]), "score": score}[sys.argv[1]]()
