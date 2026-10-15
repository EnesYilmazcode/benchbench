"""Build the report page, docs/index.html, from the results.

Run after bb.report (and bb.crosscheck if it ran): `python -X utf8 -m bb.site`.
"""
import json
import statistics
from collections import Counter

from . import data, validate
from .audit import ROOT

CODE = {"humaneval", "mbpp", "bigcodebench"}


def _validation():
    v = validate.main(show=False)
    labels = {it["id"]: it["extra"]["redux"]["error_type"] for it in data.sample("mmlu_redux_val")}
    rows = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "runs/results/mmlu_redux_val.jsonl").open(encoding="utf-8")}
    from .audit import FLAWED
    caught = Counter()
    total = Counter()
    for i, t in labels.items():
        total[t] += 1
        caught[t] += rows[i]["final"]["verdict"] in FLAWED
    return {**{k: v[k] for k in ("tp", "fp", "fn", "tn", "precision", "recall")}, "items": len(labels),
            "broken": sum(n for t, n in total.items() if t != "ok"),
            "wrong_caught": caught["wrong_groundtruth"], "wrong_total": total["wrong_groundtruth"],
            "clarity_caught": caught["bad_question_clarity"] + caught["bad_options_clarity"],
            "clarity_total": total["bad_question_clarity"] + total["bad_options_clarity"]}


def _timed_out(summary):
    gone = [i.split("/", 1)[1] for r in summary for i in r.get("timed_out", [])]
    if not gone:
        return ""
    return (f"{len(gone)} Terminal-Bench tasks ({', '.join(gone)}) are left out: the auditors tried to solve them "
            "outright and ran past 40 minutes twice.")


def _cross_coverage(findings):
    need = [f for f in findings if f["verdict"] != "tests_weak"]
    ran = sum(1 for f in need if f.get("cross"))
    return f"({ran} of {len(need)})" if ran < len(need) else f"(all {len(need)})"


def main():
    summary = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))
    findings = [json.loads(l) for l in (ROOT / "results/findings.jsonl").open(encoding="utf-8")]
    cross = {}
    cp = ROOT / "runs/crosscheck.jsonl"
    if cp.exists():
        for l in cp.open(encoding="utf-8"):
            r = json.loads(l)
            cross[r["id"]] = r["vote"]["verdict"]
    for f in findings:
        v = cross.get(f["id"])
        if v:
            f["cross"] = "Codex agrees" if v != "key_correct" and v != "unsure" else (
                "Codex disagrees" if v == "key_correct" else "Codex unsure")
    # Hard findings first, then weak tests; inside each, benchmarks in leaderboard order.
    rank = {r["bench"]: i for i, r in enumerate(sorted(summary, key=lambda r: -r["rate"]))}
    findings.sort(key=lambda f: (f["verdict"] == "tests_weak", rank.get(f["bench"], 99)))
    static = json.loads((ROOT / "runs/static.json").read_text(encoding="utf-8"))
    audited = sum(r["audited"] for r in summary)
    flagged = sum(r["flawed"] for r in summary)
    bias = []
    if "truthfulqa" in static:
        s = static["truthfulqa"]
        bias.append({"title": "TruthfulQA (MC1): where the right answer sits", "flag": True, "share": s["position_share"],
                     "note": f"In the released file the correct option is always listed first ({s['n_with_k']} items with four options shown). Evaluations that do not shuffle reward a model that always picks A."})
    if "mmlu" in static:
        s = static["mmlu"]
        bias.append({"title": "MMLU: where the right answer sits", "flag": True, "share": s["position_share"],
                     "note": f"Across all {s['items']:,} questions D is right {s['position_share'][3]:.1%} of the time and A {s['position_share'][0]:.1%}. A model that always says D beats chance (p = {s['position_p']:.0e})."})
    notes = []
    if "mmlu_pro" in static:
        s = static["mmlu_pro"]
        notes.append({"title": "MMLU-Pro: duplicates and long answers",
                      "text": f"{s['duplicate_questions']} of its {s['items']:,} questions repeat another question word for word (after normalising case and punctuation). The longest option is the right one {s['longest_is_answer']:.1%} of the time against {s['longest_chance']:.0%} by chance."})
    if "mmlu" in static:
        s = static["mmlu"]
        notes.append({"title": "MMLU: 'all of the above'",
                      "text": f"{s['all_none_of_above_share']:.1%} of questions offer an 'all / none / both of the above' option, and when they do it is the right answer {s['all_none_of_above_is_answer']:.0%} of the time, against 25% by chance."})
    if "tau2" in static:
        s = static["tau2"]
        notes.append({"title": "tau2-bench: repeated scenarios",
                      "text": f"{s['duplicate_questions']} of its {s['items']} leaderboard tasks give the simulated customer exactly the same instructions as another task."})
    payload = {
        "date": json.loads((ROOT / "results/meta.json").read_text(encoding="utf-8"))["date"],
        "summary": summary, "findings": findings, "validation": _validation(), "bias": bias, "notes": notes,
        "totals": {"audited": audited, "flagged": flagged, "median_rate": statistics.median(r["rate"] for r in summary)},
        "method": [
            "<b>Sample.</b> A fixed random sample of each benchmark: 150 items (100 for the SWE-bench sets), or all of it when it is smaller.",
            "<b>Solve blind.</b> Claude Opus 5.5 and Claude Sonnet 5.5, each run as Claude Code from the shell, answer every question without seeing the key, and say whether the item itself looks broken.",
            "<b>Review.</b> Any item where a solver disagrees with the key or flags it goes back to both, now with the key, a shell to compute in and the web to check facts. Each gives a verdict with its reasoning and sources.",
            "<b>Judge.</b> When the two verdicts split, a third call (Opus at its highest effort) reads both, checks their claims and decides. An item counts as broken only if both reviewers or the judge say so.",
            "<b>Run the code.</b> For HumanEval and MBPP the official tests run in BigCodeBench's own evaluation image on Modal against the reference, two independent solutions, and two deliberately buggy ones. A bug that passes the tests and prints a different result from the reference on a valid input is a measured weak test.",
            "<b>Read the tasks.</b> SWE-bench, Terminal-Bench and tau2 tasks are reviewed with their real files in the auditor's folder: issue, gold patch and tests; the task folder; or the domain policy, database and tool code. " + _timed_out(summary),
            "<b>Cross-check.</b> Every broken verdict on a question or task also went to Codex (GPT-5.6), a model from another lab, which saw the evidence but not the verdicts. Its agreement is shown on each finding where it ran " + _cross_coverage(findings) + ".",
            "<b>Check the checker.</b> The same pipeline ran on MMLU-Redux items with human labels, above.",
        ],
        "footer": [
            'Code, prompts and every per-item result: <a href="https://github.com/EnesYilmazcode/benchbench">github.com/EnesYilmazcode/benchbench</a>.',
            "Rates are for samples, with 95% Wilson intervals. A flag means two AI reviewers (or a judge) agreed an item is broken; spot-check any of them yourself.",
        ],
    }
    tpl = (ROOT / "site/template.html").read_text(encoding="utf-8")
    out = tpl.replace("/*DATA*/null", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/index.html").write_text(out, encoding="utf-8")
    print(f"docs/index.html: {len(summary)} benchmarks, {len(findings)} findings, {len(out) // 1024} KB")


if __name__ == "__main__":
    main()
