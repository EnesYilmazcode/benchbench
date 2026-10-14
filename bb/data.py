"""Load each benchmark into one item shape and draw a fixed sample.

An item is a dict:
  id, bench, kind ("mcq" | "short" | "code" | "swe" | "agent"), question, choices (list or None),
  answer (letter for mcq, text otherwise), meta (subject etc.), extra (tests, patches, paths).

`python -X utf8 -m bb.data` writes data/items/<bench>.jsonl (all items) and
data/sample/<bench>.jsonl (the audited sample).
"""
import ast
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC = DATA / "src"
SEED = 20261014
LETTERS = "ABCDEFGHIJKLMNOP"


def _env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _hf(repo, path):
    from huggingface_hub import hf_hub_download
    _env()
    return hf_hub_download(repo, path, repo_type="dataset")


def _read(repo, path):
    p = _hf(repo, path)
    if path.endswith(".parquet"):
        return pd.read_parquet(p)
    if path.endswith(".csv"):
        return pd.read_csv(p)
    return pd.read_json(p, lines=path.endswith(".jsonl"))


def _mcq(bench, i, q, choices, ans_idx, **meta):
    return {"id": f"{bench}/{i}", "bench": bench, "kind": "mcq", "question": q, "choices": list(choices),
            "answer": LETTERS[int(ans_idx)], "meta": meta, "extra": {}}


def _short(bench, i, q, a, **meta):
    return {"id": f"{bench}/{i}", "bench": bench, "kind": "short", "question": q, "choices": None,
            "answer": str(a), "meta": meta, "extra": {}}


# ---------- multiple choice ----------

def mmlu():
    # MMLU-Redux 2.0 re-annotated 100 questions per subject by hand; we keep those labels so the
    # audit of MMLU can be scored against human experts (see bb/validate.py).
    from datasets import load_dataset
    _env()
    subjects = sorted({p.split("/")[0] for p in _siblings("edinburgh-dawg/mmlu-redux-2.0") if "/" in p})
    out = []
    for s in subjects:
        ds = load_dataset("edinburgh-dawg/mmlu-redux-2.0", s, split="test")
        for j, r in enumerate(ds):
            it = _mcq("mmlu", f"{s}/{j}", r["question"], r["choices"], r["answer"], subject=s)
            it["extra"]["redux"] = {"error_type": r["error_type"], "correct_answer": r["correct_answer"]}
            out.append(it)
    return out


def _siblings(repo):
    from huggingface_hub import HfApi
    _env()
    return [s.rfilename for s in HfApi().dataset_info(repo).siblings]


def mmlu_pro():
    df = _read("TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet")
    return [_mcq("mmlu_pro", r.question_id, r.question, list(r.options), r.answer_index, subject=r.category, src=r.src)
            for r in df.itertuples()]


def gpqa_diamond():
    df = _read("Idavidrein/gpqa", "gpqa_diamond.csv")
    out = []
    for i, r in df.iterrows():
        opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        order = list(range(4))
        random.Random(f"{SEED}-{r['Record ID']}").shuffle(order)
        out.append(_mcq("gpqa_diamond", r["Record ID"], r["Question"], [str(opts[k]).strip() for k in order],
                        order.index(0), subject=r["High-level domain"], gated=True))
    return out


def arc_challenge():
    df = _read("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet")
    return [_mcq("arc_challenge", r.id, r.question, list(r.choices["text"]), list(r.choices["label"]).index(r.answerKey))
            for r in df.itertuples()]


def hellaswag():
    df = _read("Rowan/hellaswag", "data/validation-00000-of-00001.parquet")
    return [_mcq("hellaswag", r.ind, f"Pick the most likely continuation.\n\n{r.activity_label}: {r.ctx}",
                 list(r.endings), int(r.label), subject=r.activity_label) for r in df.itertuples()]


def truthfulqa():
    # The released file always lists the correct choice first; we shuffle for the auditors and
    # keep the original position in meta so the bias check still sees the raw layout.
    df = _read("truthfulqa/truthful_qa", "multiple_choice/validation-00000-of-00001.parquet")
    out = []
    for i, r in df.iterrows():
        ch, lab = list(r.mc1_targets["choices"]), list(r.mc1_targets["labels"])
        order = list(range(len(ch)))
        random.Random(f"{SEED}-tqa-{i}").shuffle(order)
        it = _mcq("truthfulqa", i, r.question, [ch[k] for k in order], order.index(lab.index(1)))
        it["meta"]["raw_answer_pos"] = lab.index(1)
        out.append(it)
    return out


def commonsense_qa():
    df = _read("tau/commonsense_qa", "data/validation-00000-of-00001.parquet")
    return [_mcq("commonsense_qa", r.id, r.question, list(r.choices["text"]), list(r.choices["label"]).index(r.answerKey))
            for r in df.itertuples()]


def winogrande():
    df = _read("allenai/winogrande", "winogrande_xl/validation-00000-of-00001.parquet")
    return [_mcq("winogrande", i, f"Which option fills the blank (_)?\n\n{r.sentence}", [r.option1, r.option2], int(r.answer) - 1)
            for i, r in df.iterrows()]


# ---------- short answer ----------

def gsm8k():
    df = _read("openai/gsm8k", "main/test-00000-of-00001.parquet")
    return [_short("gsm8k", i, r.question, r.answer.split("####")[-1].strip(), solution=r.answer) for i, r in df.iterrows()]


def math500():
    df = _read("HuggingFaceH4/MATH-500", "test.jsonl")
    return [_short("math500", r.unique_id.replace("/", "_"), r.problem, r.answer, subject=r.subject, level=int(r.level))
            for r in df.itertuples()]


def aime2025():
    df = _read("MathArena/aime_2025", "data/train-00000-of-00001.parquet")
    return [_short("aime2025", r.problem_idx, r.problem, r.answer) for r in df.itertuples()]


def hmmt2025():
    df = _read("MathArena/hmmt_feb_2025", "data/train-00000-of-00001.parquet")
    return [_short("hmmt2025", r.problem_idx, r.problem, r.answer) for r in df.itertuples()]


def simpleqa():
    df = _read("basicv8vc/SimpleQA", "simple_qa_test_set.csv")
    out = []
    for i, r in df.iterrows():
        m = ast.literal_eval(r.metadata)
        out.append(_short("simpleqa", i, r.problem, r.answer, subject=m.get("topic"), urls=m.get("urls")))
    return out


def simpleqa_verified():
    df = _read("google/simpleqa-verified", "simpleqa_verified.csv")
    return [_short("simpleqa_verified", r.original_index, r.problem, r.answer, subject=r.topic, urls=str(r.urls).split(","))
            for r in df.itertuples()]


def hle():
    df = _read("cais/hle", "data/test-00000-of-00001.parquet")
    out = []
    for r in df.itertuples():
        if isinstance(r.image, str) and r.image:
            continue  # text-only: auditors here do not see images
        if r.answer_type == "multipleChoice":
            it = _short("hle", r.id, r.question, r.answer, subject=r.category, answer_type="multipleChoice", gated=True)
        else:
            it = _short("hle", r.id, r.question, r.answer, subject=r.category, answer_type=r.answer_type, gated=True)
        it["extra"]["rationale"] = r.rationale
        out.append(it)
    return out


def frontierscience():
    df = _read("openai/frontierscience", "olympiad/test.jsonl")
    return [_short("frontierscience", i, r.problem, r.answer, subject=r.subject) for i, r in df.iterrows()]


# ---------- code with tests ----------

def humaneval():
    df = _read("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet")
    return [{"id": f"humaneval/{r.task_id.split('/')[-1]}", "bench": "humaneval", "kind": "code", "question": r.prompt,
             "choices": None, "answer": r.canonical_solution, "meta": {},
             "extra": {"style": "humaneval", "prompt": r.prompt, "test": r.test, "entry_point": r.entry_point}}
            for r in df.itertuples()]


def mbpp():
    df = _read("google-research-datasets/mbpp", "sanitized/test-00000-of-00001.parquet")
    out = []
    for r in df.itertuples():
        tests = "\n".join(list(r.test_imports) + list(r.test_list))
        q = f"{r.prompt}\nYour code should pass tests like this one:\n{r.test_list[0]}"
        out.append({"id": f"mbpp/{r.task_id}", "bench": "mbpp", "kind": "code", "question": q, "choices": None,
                    "answer": r.code, "meta": {}, "extra": {"style": "mbpp", "test": tests, "visible_test": r.test_list[0]}})
    return out


def bigcodebench():
    df = _read("bigcode/bigcodebench", "data/v0.1.4-00000-of-00001.parquet")
    return [{"id": f"bigcodebench/{r.task_id.split('/')[-1]}", "bench": "bigcodebench", "kind": "code",
             "question": r.complete_prompt, "choices": None, "answer": r.code_prompt + r.canonical_solution,
             "meta": {"libs": r.libs},
             "extra": {"style": "bigcodebench", "prompt": r.complete_prompt, "test": r.test, "entry_point": r.entry_point}}
            for r in df.itertuples()]


# ---------- software engineering ----------

def _swe(bench, df, f2p, p2p):
    out = []
    for r in df.itertuples():
        out.append({"id": f"{bench}/{r.instance_id}", "bench": bench, "kind": "swe", "question": r.problem_statement,
                    "choices": None, "answer": r.patch, "meta": {"repo": r.repo},
                    "extra": {"test_patch": r.test_patch, "fail_to_pass": getattr(r, f2p), "pass_to_pass": getattr(r, p2p),
                              "base_commit": r.base_commit}})
    return out


def swe_verified():
    df = _read("SWE-bench/SWE-bench_Verified", "data/test-00000-of-00001.parquet")
    out = _swe("swe_verified", df, "FAIL_TO_PASS", "PASS_TO_PASS")
    for it, r in zip(out, df.itertuples()):
        it["meta"]["difficulty"] = r.difficulty
        it["extra"]["hints"] = r.hints_text
    return out


def swe_pro():
    df = _read("ScaleAI/SWE-bench_Pro", "data/default/test-00000-of-00001.parquet")
    out = _swe("swe_pro", df, "fail_to_pass", "pass_to_pass")
    for it, r in zip(out, df.itertuples()):
        it["meta"]["language"] = r.repo_language
        it["extra"]["requirements"] = r.requirements
        it["extra"]["interface"] = r.interface
    return out


# ---------- agent tasks ----------

def terminal_bench2():
    out = []
    for d in sorted((SRC / "tb2").iterdir()):
        if (d / "instruction.md").exists():
            out.append({"id": f"terminal_bench2/{d.name}", "bench": "terminal_bench2", "kind": "agent",
                        "question": (d / "instruction.md").read_text(encoding="utf-8"), "choices": None, "answer": None,
                        "meta": {}, "extra": {"dir": str(d.relative_to(ROOT)).replace("\\", "/")}})
    return out


def tau2():
    out = []
    for dom in ("airline", "retail", "telecom"):
        ddir = SRC / "tau2/data/tau2/domains" / dom
        base = set(json.loads((ddir / "split_tasks.json").read_text(encoding="utf-8"))["base"])
        tasks = json.loads((ddir / "tasks.json").read_text(encoding="utf-8"))
        for t in (t for t in tasks if t["id"] in base):  # the "base" split is what leaderboards run
            out.append({"id": f"tau2/{dom}/{t['id']}", "bench": "tau2", "kind": "agent",
                        "question": json.dumps(t["user_scenario"], ensure_ascii=False), "choices": None,
                        "answer": json.dumps(t.get("evaluation_criteria"), ensure_ascii=False), "meta": {"domain": dom},
                        "extra": {"dir": f"data/src/tau2/data/tau2/domains/{dom}", "task": t}})
    return out


BENCHES = {
    "mmlu": mmlu, "mmlu_pro": mmlu_pro, "gpqa_diamond": gpqa_diamond, "arc_challenge": arc_challenge,
    "hellaswag": hellaswag, "truthfulqa": truthfulqa, "commonsense_qa": commonsense_qa, "winogrande": winogrande,
    "gsm8k": gsm8k, "math500": math500, "aime2025": aime2025, "hmmt2025": hmmt2025, "simpleqa": simpleqa,
    "simpleqa_verified": simpleqa_verified, "hle": hle, "frontierscience": frontierscience,
    "humaneval": humaneval, "mbpp": mbpp, "bigcodebench": bigcodebench,
    "swe_verified": swe_verified, "swe_pro": swe_pro, "terminal_bench2": terminal_bench2, "tau2": tau2,
}
SAMPLE = {"swe_verified": 100, "swe_pro": 100, "humaneval": 164}  # default 150; smaller sets are taken whole
DEFAULT_N = 150


def items(bench):
    p = DATA / "items" / f"{bench}.jsonl"
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def sample(bench):
    p = DATA / "sample" / f"{bench}.jsonl"
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def build_validation(n_each=60):
    """A labelled check of the checker: MMLU items the MMLU-Redux annotators marked broken, plus as many marked fine."""
    rows = items("mmlu")
    bad = [r for r in rows if r["extra"]["redux"]["error_type"] not in ("ok", "expert")]
    ok = [r for r in rows if r["extra"]["redux"]["error_type"] == "ok"]
    rng = random.Random(f"{SEED}-validation")
    pick = rng.sample(bad, n_each) + rng.sample(ok, n_each)
    rng.shuffle(pick)
    pick = [{**r, "id": r["id"].replace("mmlu/", "mmlu_redux_val/", 1), "bench": "mmlu_redux_val"} for r in pick]
    _write(DATA / "sample" / "mmlu_redux_val.jsonl", pick)
    print(f"mmlu_redux_val     {len(bad)} labelled broken, sample {len(pick)}", flush=True)


def build(names):
    if names == ["mmlu_redux_val"]:
        return build_validation()
    for b in names:
        try:
            rows = BENCHES[b]()
        except Exception as e:  # a gated set we cannot read yet; say so and move on
            print(f"{b:18s} FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)
            continue
        _write(DATA / "items" / f"{b}.jsonl", rows)
        n = SAMPLE.get(b, DEFAULT_N)
        pick = rows if len(rows) <= n else random.Random(f"{SEED}-{b}").sample(rows, n)
        _write(DATA / "sample" / f"{b}.jsonl", pick)
        print(f"{b:18s} {len(rows):6d} items, sample {len(pick)}", flush=True)


if __name__ == "__main__":
    build(sys.argv[1:] or list(BENCHES))
