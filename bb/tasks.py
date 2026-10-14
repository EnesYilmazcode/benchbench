"""Audit software-engineering and agent tasks by reading them with their files at hand.

These tasks cannot be blind-solved cheaply, so both auditors go straight to review, each in a
folder holding the task's real files (issue, gold patch, test patch; or the Terminal-Bench task
folder; or the tau2 domain's policy, database and tool code). The judge settles splits.
"""
import json
from pathlib import Path

from .audit import ROOT, judge, review

SWE_VERDICTS = (
    "Verdicts (judge as a careful maintainer would, using only what a solver is given):\n"
    "- key_correct: the issue says clearly enough what a fix must do, the gold patch does it, and the FAIL_TO_PASS "
    "tests check that behaviour without demanding details the issue does not state.\n"
    "- ambiguous: the issue is underspecified: a competent engineer could not tell what a successful fix looks like, "
    "or reasonable fixes would differ in ways the tests punish.\n"
    "- grader_flawed: the tests reject valid fixes: they check names, messages, signatures or behaviour the issue never "
    "asks for, or test something unrelated to the issue.\n"
    "- tests_weak: the tests would pass for a fix that does not really solve the issue.\n"
    "- key_wrong: the gold patch is wrong or incomplete for the issue.\n"
    "- unanswerable: the task is broken in some other way.\n"
    "- unsure: you cannot decide.\n"
    "Do not flag a task only because it is hard. The files are in this folder: read them all before deciding.")

TB_VERDICTS = (
    "Verdicts:\n"
    "- key_correct: instruction.md tells the agent everything the tests check, the tests really verify the task, and "
    "the reference solution is consistent with both.\n"
    "- ambiguous: the tests check requirements (file names, formats, exact values, paths) that instruction.md does not "
    "state or implies differently, so a reasonable agent following the instruction would fail.\n"
    "- tests_weak: the tests can be passed without doing the task (for example by writing the expected file directly, "
    "or because a check is missing).\n"
    "- grader_flawed: the tests are wrong in another way: flaky, timing-dependent, depend on the network or unpinned "
    "versions that will drift, or reject correct work.\n"
    "- key_wrong: the reference solution would not pass, or is wrong.\n"
    "- unanswerable: the environment cannot support the task as written.\n"
    "- unsure: you cannot decide.\n"
    "Read every file in the task folder (instruction.md, task.toml, tests/, solution/, environment/) before deciding.")

TAU_VERDICTS = (
    "Verdicts:\n"
    "- key_correct: following the policy, a correct agent dealing with this user ends with exactly the expected "
    "actions / database state / communicated info, and the user instructions are consistent with the database.\n"
    "- key_wrong: the expected outcome breaks the policy or contradicts the database (wrong ids, prices, dates, "
    "rules applied wrongly), so a policy-following agent would be marked wrong.\n"
    "- ambiguous: the policy or the user instructions allow more than one correct outcome, and the grader accepts only one.\n"
    "- unanswerable: the scenario contradicts the database or cannot be completed.\n"
    "- grader_flawed: the evaluation criteria check something the policy does not require, or miss what it does.\n"
    "- unsure: you cannot decide.\n"
    "The domain files are in this folder: policy, db.json (large: search it with code, do not read it whole), and the "
    "tools' source. Check every expected action and value against the database and policy yourself.")


def _swe(it):
    x = it["extra"]
    files = {"issue.md": it["question"], "gold.patch": it["answer"], "test.patch": x["test_patch"],
             "fail_to_pass.txt": str(x["fail_to_pass"]), "pass_to_pass.txt": str(x["pass_to_pass"])[:20000]}
    if x.get("requirements"):
        files["requirements.md"] = x["requirements"]
        files["interface.md"] = x.get("interface") or ""
    if x.get("hints"):
        files["hints.md"] = x["hints"]
    given = "issue.md" + (", requirements.md and interface.md" if x.get("requirements") else "")
    body = (f"Task {it['id']} from repository {it['meta']['repo']}. A solver is given {given} and the repository at the "
            f"base commit, writes a patch, and is graded by the tests in fail_to_pass.txt (added by test.patch) plus "
            f"pass_to_pass.txt. The reference fix is gold.patch. All of these are files in your current folder.\n\n"
            f"=== ISSUE (excerpt) ===\n{it['question'][:2500]}")
    return files, None, body, SWE_VERDICTS


def _tb(it):
    d = ROOT / it["extra"]["dir"]
    body = (f"Task {it['id']} from Terminal-Bench 2.0. An agent gets instruction.md inside the Docker environment "
            f"built from environment/, works in a shell, and is graded by tests/ (run by tests/test.sh). The reference "
            f"solution is solution/. The whole task folder is in your current folder under task/.\n\n"
            f"=== instruction.md ===\n{it['question'][:4000]}")
    return None, {"task": str(d)}, body, TB_VERDICTS


def _tau(it):
    dom = it["meta"]["domain"]
    d = ROOT / it["extra"]["dir"]
    src = ROOT / "data/src/tau2/src/tau2/domains" / dom
    files = {"task.json": json.dumps(it["extra"]["task"], indent=1, ensure_ascii=False)}
    body = (f"Task {it['id']} from tau2-bench ({dom} domain). A user simulator follows the user scenario in task.json; "
            f"the agent follows the policy in domain/ and uses the tools in tools_src/. The run is graded by "
            f"task.json's evaluation_criteria (expected actions, database end state, info to communicate, assertions).\n\n"
            f"=== task.json ===\n{files['task.json'][:6000]}")
    return files, {"domain": str(d), "tools_src": str(src)}, body, TAU_VERDICTS


def audit_task(it):
    files, trees, body, verdicts = {"swe": _swe, "agent": _tb if it["bench"] == "terminal_bench2" else _tau}[it["kind"]](it)
    rec = {"id": it["id"], "bench": it["bench"], "kind": it["kind"]}
    ctx = ""
    rec["review"] = review(it, ctx, ("Bash", "Read", "Grep", "Glob"), (), verdicts=verdicts, body=body,
                           files=files, trees=trees)
    rec["final"] = judge(it, ctx, rec["review"], verdicts=verdicts, body=body, files=files, trees=trees)
    return rec
