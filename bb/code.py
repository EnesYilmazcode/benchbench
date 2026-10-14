"""Audit a coding item by running things against its own tests.

  reference  The official solution must pass the official tests.
  solve      Claude and Codex each write a solution from the prompt alone; both are run.
  mutant     Each also writes a subtly wrong solution meant to slip past the tests, plus a probe
             that prints the function's behaviour on one input. If the mutant passes the tests and
             the probe prints something different for the mutant than for the reference, the tests
             accept wrong code: that is measured, not opined.
Anything odd goes to the shared review and judge steps with the run logs attached.
"""
from .agents import ask
from .audit import AUDITORS, FLAWED, judge, review
from .sandbox import run_jobs

TIMEOUT = {"humaneval": 30, "mbpp": 30, "bigcodebench": 120}

CODE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["code", "flaw"],
    "properties": {"code": {"type": "string", "description": "A complete, self-contained Python program defining the function, imports included."},
                   "flaw": {"type": "string", "description": "Empty if the task is clear. Otherwise one sentence on what is unclear or broken."}},
}
MUTANT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["code", "probe", "why_wrong"],
    "properties": {
        "code": {"type": "string", "description": "Complete program defining the function with a subtle bug that the given tests do not catch."},
        "probe": {"type": "string", "description": "Python statements that call the function on a valid input from the task's spec and print the result. Must be deterministic."},
        "why_wrong": {"type": "string", "description": "One sentence: which requirement of the spec the mutant breaks on the probe input."},
    },
}


def task_text(it):
    x = it["extra"]
    if x["style"] == "mbpp":
        return it["question"]
    return f"Complete this Python function so it meets its docstring.\n\n```python\n{x['prompt']}\n```"


def reference(it):
    x = it["extra"]
    return x["prompt"] + it["answer"] if x["style"] == "humaneval" else it["answer"]


def with_tests(it, program):
    x = it["extra"]
    if x["style"] == "humaneval":
        return f"{program}\n\n{x['test']}\n\ncheck({x['entry_point']})\n"
    if x["style"] == "mbpp":
        return f"{program}\n\n{x['test']}\n"
    return (f"{program}\n\n{x['test']}\n\nimport unittest, sys\n"
            "r = unittest.main(argv=['t'], exit=False, verbosity=1).result\n"
            "sys.exit(0 if r.wasSuccessful() and r.testsRun else 1)\n")


def with_probe(program, probe):
    return f"{program}\n\n# --- probe ---\n{probe}\n"


def _fence(s):
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        s = s.rsplit("```", 1)[0]
    return s


def audit_code(it):
    t = TIMEOUT[it["bench"]]
    rec = {"id": it["id"], "bench": it["bench"], "kind": "code"}
    ref = reference(it)
    sol = {w: ask(w, f"{task_text(it)}\n\nWrite the solution. Do not run it against hidden tests; there are none here. "
                     "If the task is unclear or broken, say so in `flaw`.",
                  CODE_SCHEMA, effort="medium", tag=f"code:{it['id']}") for w in AUDITORS}
    mut_prompt = (f"{task_text(it)}\n\nThis is the reference solution:\n```python\n{ref}\n```\n\nThese are the benchmark's "
                  f"official tests:\n```python\n{it['extra']['test']}\n```\n\nWrite a plausible but WRONG implementation: "
                  "it must break a requirement stated in the task, on a valid input, while still passing every official "
                  "test. Make it a bug a real programmer could plausibly write (an off-by-one, a wrong comparison, a "
                  "missed edge case the task mentions, the wrong library call), never code that special-cases particular "
                  "inputs or detects the tests. Then write a probe that prints the function's result on such an input, so running the probe "
                  "with your code and with the reference prints different things. If the tests are thorough and you "
                  "cannot find such a gap, return your best attempt anyway.")
    mut = {w: ask(w, mut_prompt, MUTANT_SCHEMA, effort="medium", tag=f"mutant:{it['id']}") for w in AUDITORS}

    jobs = [{"program": with_tests(it, ref), "timeout": t}]
    jobs += [{"program": with_tests(it, _fence(sol[w]["code"])), "timeout": t} for w in AUDITORS]
    for w in AUDITORS:
        mc = _fence(mut[w]["code"])
        jobs += [{"program": with_tests(it, mc), "timeout": t},
                 {"program": with_probe(mc, mut[w]["probe"]), "timeout": t},
                 {"program": with_probe(ref, mut[w]["probe"]), "timeout": t}]
    res = run_jobs(jobs)
    rec["exec"] = {"reference": res[0], **{f"solve_{w}": res[1 + i] for i, w in enumerate(AUDITORS)}}
    rec["mutants"] = {}
    for i, w in enumerate(AUDITORS):
        passes, pm, pr = res[3 + 3 * i: 6 + 3 * i]
        slipped = passes["ok"] and pm["ok"] and pr["ok"] and pm["out"].strip() != pr["out"].strip()
        rec["mutants"][w] = {"passes_tests": passes["ok"], "probe_mutant": pm["out"][-300:], "probe_reference": pr["out"][-300:],
                             "slipped": slipped, "why_wrong": mut[w]["why_wrong"], "code": _fence(mut[w]["code"]),
                             "probe": mut[w]["probe"]}
    rec["solve"] = {w: {"passes": rec["exec"][f"solve_{w}"]["ok"], "flaw": sol[w]["flaw"], "code": _fence(sol[w]["code"])}
                    for w in AUDITORS}

    odd = (not res[0]["ok"] or any(not rec["solve"][w]["passes"] or rec["solve"][w]["flaw"].strip() for w in AUDITORS)
           or any(m["slipped"] for m in rec["mutants"].values()))
    if not odd:
        rec["final"] = {"verdict": "key_correct", "by": "exec"}
        return rec

    def show(r):
        return f"{'PASSED' if r['ok'] else 'FAILED'} (exit {r['rc']}). Output tail:\n{r['out'][-700:]}"
    ctx = [f"=== RUN RESULTS AGAINST THE OFFICIAL TESTS ===\nReference solution: {show(res[0])}"]
    for i, w in enumerate(AUDITORS):
        ctx.append(f"Independent solution {i + 1} (written from the task alone): {show(res[1 + i])}\n"
                   f"Its author flagged: {sol[w]['flaw'] or 'nothing'}\n```python\n{rec['solve'][w]['code'][:3000]}\n```")
    for i, w in enumerate(AUDITORS):
        m = rec["mutants"][w]
        if m["slipped"]:
            ctx.append(f"Deliberately wrong solution {i + 1} PASSED the official tests, and its probe output differs from "
                       f"the reference's. Claimed bug: {m['why_wrong']}\n```python\n{m['code'][:3000]}\n```\nProbe:\n"
                       f"```python\n{m['probe']}\n```\nprobe with this code -> {m['probe_mutant']!r}\nprobe with reference -> "
                       f"{m['probe_reference']!r}")
    verdicts = (
        "Verdicts:\n"
        "- key_correct: task, reference solution and tests are sound. Failing independent solutions are simply wrong, "
        "and a 'wrong' solution that passed only differs on inputs the task does not define or in ways it allows.\n"
        "- key_wrong: the reference solution is wrong, or a test asserts a wrong expected value.\n"
        "- ambiguous: the task text is underspecified, so a reasonable reading fails the tests.\n"
        "- grader_flawed: the tests accept a solution that breaks a stated requirement on a valid input (tests too weak), "
        "or reject correct solutions for reasons the task does not state (tests too strict).\n"
        "- unanswerable: the task cannot be done as written, or the environment is broken (reference fails for reasons "
        "outside the code).\n"
        "- unsure: you cannot decide.\n"
        "You may write and run Python to check claims. Judge against the task text: it is all a solver sees.")
    body = (f"=== TASK (what a solver sees) ===\n{task_text(it)}\n\n=== REFERENCE SOLUTION ===\n```python\n{ref}\n```\n\n"
            f"=== OFFICIAL TESTS ===\n```python\n{it['extra']['test'][:8000]}\n```")
    context = "\n\n".join(ctx)
    rec["review"] = review(it, context, ("Bash", "Read"), (), verdicts=verdicts, body=body)
    rec["final"] = judge(it, context, rec["review"], verdicts=verdicts, body=body)
    return rec
