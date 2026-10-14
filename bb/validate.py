"""Check the checker: score the audit of the MMLU-Redux validation set against its human labels.

Prints a confusion table, precision and recall, and every disagreement so it can be read by hand.
"""
import json
import sys
from collections import Counter

from . import data
from .audit import FLAWED, RESULTS


def main(show=True):
    labels = {it["id"]: it["extra"]["redux"] for it in data.sample("mmlu_redux_val")}
    rows = [json.loads(l) for l in (RESULTS / "mmlu_redux_val.jsonl").open(encoding="utf-8")]
    if len(rows) < len(labels):
        sys.exit(f"only {len(rows)} of {len(labels)} items audited: finish the run first")
    tp = fp = fn = tn = unsure = 0
    by_type = Counter()
    misses = []
    for r in rows:
        human_bad = labels[r["id"]]["error_type"] != "ok"
        v = r["final"]["verdict"]
        if v == "unsure":
            unsure += 1
            continue
        ours = v in FLAWED
        tp += human_bad and ours
        fp += ours and not human_bad
        fn += human_bad and not ours
        tn += not human_bad and not ours
        if human_bad:
            by_type[(labels[r["id"]]["error_type"], ours)] += 1
        if human_bad != ours:
            misses.append((r["id"], labels[r["id"]]["error_type"], v, r["final"].get("by")))
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    print(f"items {len(rows)}  unsure {unsure}")
    print(f"             human: broken   human: fine")
    print(f"we: flawed   {tp:13d}   {fp:11d}")
    print(f"we: sound    {fn:13d}   {tn:11d}")
    print(f"precision {prec:.2f}  recall {rec:.2f}  agreement {(tp + tn) / max(1, tp + tn + fp + fn):.2f}")
    for (t, caught), n in sorted(by_type.items()):
        print(f"  {t:28s} {'caught' if caught else 'missed'} {n}")
    if show:
        for m in misses:
            print("  DISAGREE", *m)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "unsure": unsure, "precision": prec, "recall": rec, "misses": misses}


if __name__ == "__main__":
    main()
