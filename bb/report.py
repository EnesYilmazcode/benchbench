"""Turn the audit results into one table and a findings file.

`python -X utf8 -m bb.report` writes results/summary.json and results/findings.jsonl (every
flagged item with its verdict and the auditors' reasons) and prints the table.
"""
import json
import math
import os
import time
from collections import Counter

from . import data
from .audit import AUDITORS, FLAWED, NAMES, RESULTS, ROOT, show_key, show_question

OUT = ROOT / "results"
HARD = {"key_wrong", "ambiguous", "unanswerable", "grader_flawed"}
ORDER = ["mmlu", "mmlu_pro", "gpqa_diamond", "arc_challenge", "hellaswag", "truthfulqa", "commonsense_qa", "winogrande",
         "gsm8k", "math500", "aime2025", "hmmt2025", "simpleqa", "simpleqa_verified", "hle", "frontierscience",
         "humaneval", "mbpp", "bigcodebench", "swe_verified", "swe_pro", "terminal_bench2", "tau2"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


LOOSE = {"tests_weak", "grader_too_loose"}


def verdict(r):
    """The final verdict, with one correction: reviewers sometimes file 'tests too weak' under grader_flawed,
    while their own issue tags say only that. Those count as weak tests."""
    v = r["final"]["verdict"]
    if v == "grader_flawed":
        tags = set(r["final"].get("issues") or [])
        for rv in r.get("review", {}).values():
            if rv["verdict"] in FLAWED:
                tags |= set(rv["issues"])
        if tags and tags <= LOOSE:
            return "tests_weak"
    return v


def reasons(r):
    """The clearest one-paragraph reason for a flag: the judge's, else the first flagging reviewer's."""
    f = r["final"]
    if f.get("explanation"):
        return f["explanation"], f.get("correct_answer", ""), f.get("evidence", "")
    for w in AUDITORS:
        v = r.get("review", {}).get(w)
        if v and v["verdict"] in FLAWED:
            return v["explanation"], v.get("correct_answer", ""), v.get("evidence", "")
    return "", "", ""


def main():
    static = json.loads((ROOT / "runs" / "static.json").read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    table, findings = [], []
    for b in ORDER:
        p = RESULTS / f"{b}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.open(encoding="utf-8")]
        sample = {it["id"]: it for it in data.sample(b)}
        if len(rows) < len(sample) and not os.environ.get("BB_PARTIAL"):
            print(f"{b}: only {len(rows)} of {len(sample)} audited, left out of the table")
            continue
        v = Counter(verdict(r) for r in rows)
        n = len(rows) - v["unsure"]
        hard = sum(v[k] for k in HARD)
        weak = v["tests_weak"]
        lo, hi = wilson(hard, n)
        table.append({"bench": b, "name": NAMES[b], "kind": rows[0]["kind"], "audited": len(rows), "decided": n,
                      "population": static.get(b, {}).get("items"), "flawed": hard, "rate": hard / n if n else 0,
                      "ci95": [lo, hi], "tests_weak": weak, "tests_weak_rate": weak / n if n else 0,
                      "unsure": v["unsure"], "verdicts": dict(v), "static": static.get(b, {})})
        for r in rows:
            if verdict(r) in FLAWED:
                it = sample[r["id"]]
                why, fix, ev = reasons(r)
                gated = it["meta"].get("gated", False)
                findings.append({"id": r["id"], "bench": b, "verdict": verdict(r), "by": r["final"].get("by"),
                                 "question": None if gated else show_question(it)[:1500] if it["kind"] in ("mcq", "short") else it["question"][:600],
                                 "official": None if gated else (show_key(it) if it["kind"] in ("mcq", "short") else None),
                                 "should_be": fix, "why": why, "evidence": ev})
    (OUT / "meta.json").write_text(json.dumps({"date": time.strftime("%Y-%m-%d"), "partial": bool(os.environ.get("BB_PARTIAL"))}),
                                   encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(table, indent=1, ensure_ascii=False), encoding="utf-8")
    with (OUT / "findings.jsonl").open("w", encoding="utf-8") as f:
        for x in findings:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{'benchmark':28s} {'audited':>7s} {'flawed':>7s} {'rate':>6s} {'95% CI':>13s} {'weak tests':>10s}")
    for t in sorted(table, key=lambda t: -t["rate"]):
        print(f"{t['name']:28s} {t['audited']:7d} {t['flawed']:7d} {t['rate']:6.1%} "
              f"{t['ci95'][0]:5.1%}-{t['ci95'][1]:5.1%} {t['tests_weak_rate']:9.1%}")


if __name__ == "__main__":
    main()
