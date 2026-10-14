"""Whole-benchmark checks that need no model: answer position, length bias, duplicates, leaks.

Runs on every item of a benchmark, not the sample. `python -X utf8 -m bb.static` writes
runs/static.json.
"""
import json
import math
import re
from collections import Counter, defaultdict

from . import data
from .audit import ROOT


def _chi2_p(counts):
    """Uniform-distribution chi-square test, p-value via the regularized gamma function."""
    n, k = sum(counts), len(counts)
    if k < 2 or n == 0:
        return 1.0
    e = n / k
    x = sum((c - e) ** 2 / e for c in counts)
    return _gammaincc((k - 1) / 2, x / 2)


def _gammaincc(a, x):
    if x <= 0:
        return 1.0
    if x < a + 1:  # series for P, return 1 - P
        s = term = 1 / a
        n = a
        for _ in range(500):
            n += 1
            term *= x / n
            s += term
            if abs(term) < abs(s) * 1e-12:
                break
        return max(0.0, 1 - s * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    b, c, d = x + 1 - a, 1e300, 1 / (x + 1 - a)
    h = d
    for i in range(1, 500):  # continued fraction for Q
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1 / d if abs(d) > 1e-300 else 1e300
        c = b + an / c if abs(c) > 1e-300 else b
        dl = d * c
        h *= dl
        if abs(dl - 1) < 1e-12:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - math.lgamma(a)) * h)


def _norm(s):
    return re.sub(r"\W+", " ", str(s).lower()).strip()


def mcq(items):
    by_k = defaultdict(list)
    for it in items:
        by_k[len(it["choices"])].append(it)
    k, group = max(by_k.items(), key=lambda kv: len(kv[1]))  # the usual option count
    pos = Counter(data.LETTERS.index(it.get("meta", {}).get("raw_answer_pos") is not None and
                                     data.LETTERS[it["meta"]["raw_answer_pos"]] or it["answer"]) for it in group)
    counts = [pos.get(i, 0) for i in range(k)]
    longest = sum(1 for it in group if _longest(it)) / len(group)
    nota = sum(1 for it in items if any(re.search(r"\b(all|none|both) of the (above|options)\b", str(c), re.I)
                                        for c in it["choices"])) / len(items)
    nota_right = [it for it in items if re.search(r"\b(all|none|both) of the (above|options)\b",
                                                   str(it["choices"][data.LETTERS.index(it["answer"])]), re.I)]
    dup_opts = sum(1 for it in items if len({_norm(c) for c in it["choices"]}) < len(it["choices"])) / len(items)
    return {"options": k, "n_with_k": len(group), "answer_position": counts,
            "position_share": [round(c / len(group), 3) for c in counts], "position_p": _chi2_p(counts),
            "longest_is_answer": round(longest, 3), "longest_chance": round(1 / k, 3),
            "all_none_of_above_share": round(nota, 3),
            "all_none_of_above_is_answer": round(len(nota_right) / max(1, sum(1 for it in items if any(
                re.search(r"\b(all|none|both) of the (above|options)\b", str(c), re.I) for c in it["choices"]))), 3),
            "duplicate_options_share": round(dup_opts, 4)}


def _longest(it):
    lens = [len(str(c)) for c in it["choices"]]
    m = max(lens)
    return lens.count(m) == 1 and lens.index(m) == data.LETTERS.index(it["answer"])


def common(items):
    qs = Counter(_norm(it["question"]) for it in items)
    dups = sum(c - 1 for c in qs.values() if c > 1)
    leak = 0
    for it in items:
        if it["kind"] == "short":
            a = _norm(it["answer"])
            if len(a) >= 4 and not a.isdigit() and re.search(rf"\b{re.escape(a)}\b", _norm(it["question"])):
                leak += 1
    return {"items": len(items), "duplicate_questions": dups, "duplicate_share": round(dups / len(items), 4),
            "answer_in_question": leak}


def main():
    out = {}
    for b in data.BENCHES:
        try:
            items = data.items(b)
        except FileNotFoundError:
            continue
        r = common(items)
        if items[0]["kind"] == "mcq":
            r.update(mcq(items))
        out[b] = r
        print(b, json.dumps(r))
    p = ROOT / "runs" / "static.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
