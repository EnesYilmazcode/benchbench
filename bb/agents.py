"""Run Claude Code and Codex as auditors from the shell, with structured output.

Each call gets a fresh working folder (optionally seeded with task files), a JSON schema for its
answer, and a tool set. Results are cached by a hash of everything that shapes the call, so a run
can stop and resume without paying twice.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    if "=" in _line and not _line.startswith("#"):
        os.environ.setdefault(*(x.strip() for x in _line.split("=", 1)))
CACHE = ROOT / "runs" / "cache"
CLAUDE = shutil.which("claude") or "claude"
CODEX = shutil.which("codex") or "codex"

SYSTEM = (
    "You are a careful expert auditing a published AI benchmark for errors. You work alone, in an empty "
    "folder, with no human to ask. Never look up copies of the benchmark itself (Hugging Face pages, GitHub "
    "copies, papers' appendices, leaderboards, answer dumps): judge the item on its own merits and on primary "
    "sources. Be skeptical in both directions: benchmarks do contain wrong answer keys, and you can also be wrong. "
    "Your final message must be only the JSON object the schema asks for."
)

# Claude and Codex each get their own cap on concurrent calls: their rate limits are separate.
_SLOTS = {"claude": threading.BoundedSemaphore(int(os.environ.get("BB_CLAUDE_JOBS", "6"))),
          "codex": threading.BoundedSemaphore(int(os.environ.get("BB_CODEX_JOBS", "6")))}
_LOG_LOCK = threading.Lock()


class AgentError(RuntimeError):
    pass


def _key(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:24]


def _seed(workdir, files):
    for rel, content in (files or {}).items():
        p = Path(workdir) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def _copytree(workdir, trees):
    for rel, src in (trees or {}).items():
        shutil.copytree(src, Path(workdir) / rel, dirs_exist_ok=True)


def _claude(prompt, schema, model, effort, tools, workdir, timeout, system=SYSTEM):
    cmd = [CLAUDE, "-p", "--model", model, "--effort", effort, "--output-format", "json",
           "--json-schema", json.dumps(schema), "--tools", ",".join(tools) if tools else "",
           "--system-prompt", system, "--no-session-persistence", "--strict-mcp-config",
           "--setting-sources", "", "--disable-slash-commands", "--permission-mode", "bypassPermissions"]
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=workdir, timeout=timeout)
    try:
        out = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise AgentError(f"claude exit {r.returncode}: {(r.stderr or r.stdout)[-400:]}")
    if out.get("is_error") or out.get("structured_output") is None:
        raise AgentError(f"claude error: {str(out.get('result') or out.get('api_error_status'))[:400]}")
    usage = out.get("usage") or {}
    return out["structured_output"], {"cost_usd": out.get("total_cost_usd"), "turns": out.get("num_turns"),
                                      "ms": out.get("duration_ms"), "out_tokens": usage.get("output_tokens")}


def _codex(prompt, schema, model, effort, tools, workdir, timeout, system=SYSTEM):
    sfile = Path(workdir) / ".schema.json"
    ofile = Path(workdir) / ".answer.json"
    sfile.write_text(json.dumps(schema), encoding="utf-8")
    web = "live" if "web" in tools else "disabled"
    sandbox = "workspace-write" if "write" in tools else "read-only"
    cmd = [CODEX, "exec", "--skip-git-repo-check", "--ignore-user-config", "-C", str(workdir), "-s", sandbox,
           "-m", model, "-c", f"model_reasoning_effort={effort}", "-c", f"web_search={web}",
           "-c", "windows.sandbox=unelevated",  # without it the Windows sandbox blocks every shell command
           "--output-schema", str(sfile), "-o", str(ofile), "-"]
    t0 = time.time()
    r = subprocess.run(cmd, input=system + "\n\n" + prompt, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=workdir, timeout=timeout)
    if not ofile.exists():
        raise AgentError(f"codex exit {r.returncode}: {(r.stderr or r.stdout)[-400:]}")
    try:
        ans = json.loads(ofile.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise AgentError(f"codex bad json: {ofile.read_text(encoding='utf-8')[:300]}")
    tok = None
    tail = (r.stderr or "") + (r.stdout or "")
    if "tokens used" in tail:
        try:
            tok = int(tail.rsplit("tokens used", 1)[1].split()[0].replace(",", ""))
        except (ValueError, IndexError):
            pass
    return ans, {"ms": int((time.time() - t0) * 1000), "tokens": tok}


GATEWAY = "https://ai-gateway.vercel.sh/v1/chat/completions"
TEXT_SUFFIXES = {".md", ".txt", ".py", ".sh", ".toml", ".json", ".patch", ".yaml", ".yml", ".cfg", ".r", ".R", ".c", ".js",
                 ".ts", ".rs", ".go", ".java", ".cpp", ".h", ".csv", ".html", ".sql", ".ini", ""}


def _inline(workdir, budget=160_000):
    """A model without a shell gets the folder's text files pasted in, smallest first, up to a budget."""
    files = sorted((p for p in Path(workdir).rglob("*") if p.is_file() and not p.name.startswith(".")),
                   key=lambda p: p.stat().st_size)
    parts, used, skipped = [], 0, []
    for p in files:
        rel = p.relative_to(workdir).as_posix()
        if p.suffix not in TEXT_SUFFIXES or p.stat().st_size > 60_000 or used + p.stat().st_size > budget:
            skipped.append(f"{rel} ({p.stat().st_size} bytes)")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        parts.append(f"----- FILE {rel} -----\n{text}")
        used += len(text)
    out = "\n\n".join(parts)
    if skipped:
        out += "\n\n(Not shown, too large or binary: " + ", ".join(skipped[:40]) + ")"
    return out


def _gw(prompt, schema, model, effort, tools, workdir, timeout):
    """A plain model behind Vercel's AI Gateway: no tools, the folder's files pasted into the prompt."""
    import urllib.request
    key = os.environ.get("AI_GATEWAY_API_KEY")
    if not key:
        raise AgentError("AI_GATEWAY_API_KEY is not set")
    files = _inline(workdir) if any(Path(workdir).iterdir()) else ""
    body = (prompt + ("\n\n=== THE FILES IN YOUR FOLDER ===\n" + files if files else "")
            + "\n\nYou have no tools in this session: reason it through yourself. Reply with only a JSON object matching "
              "this schema, no prose and no code fence:\n" + json.dumps(schema))
    req = {"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}],
           "response_format": {"type": "json_object"}, "max_tokens": 16000,
           "reasoning": {"effort": {"low": "low", "medium": "medium"}.get(effort, "high")}}
    t0 = time.time()
    r = urllib.request.Request(GATEWAY, data=json.dumps(req).encode(), method="POST",
                               headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            out = json.loads(resp.read())
    except Exception as e:
        detail = getattr(e, "read", lambda: b"")()[:300]
        raise AgentError(f"gateway: {e} {detail!r}")
    text = (out["choices"][0]["message"].get("content") or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        ans = json.loads(text)
    except json.JSONDecodeError:
        raise AgentError(f"gateway bad json: {text[:300]}")
    missing = [k for k in schema.get("required", []) if k not in ans]
    if missing:
        raise AgentError(f"gateway answer missing {missing}")
    enum = schema["properties"].get("verdict", {}).get("enum")
    if enum and ans.get("verdict") not in enum:
        raise AgentError(f"gateway verdict {ans.get('verdict')!r} not in schema")
    u = out.get("usage") or {}
    return ans, {"ms": int((time.time() - t0) * 1000), "in_tokens": u.get("prompt_tokens"),
                 "out_tokens": u.get("completion_tokens"), "cost_usd": (out.get("usage") or {}).get("cost")}


RUNNERS = {"claude": _claude, "sonnet": _claude, "codex": _codex, "gw": _gw}
MODELS = {"claude": os.environ.get("BB_CLAUDE_MODEL", "opus"), "codex": os.environ.get("BB_CODEX_MODEL", "gpt-5.6-sol"),
          "gw": os.environ.get("BB_GW_MODEL", "deepseek/deepseek-v4-pro"), "sonnet": os.environ.get("BB_SONNET_MODEL", "sonnet")}
_SLOTS["gw"] = threading.BoundedSemaphore(int(os.environ.get("BB_GW_JOBS", "12")))
_SLOTS["sonnet"] = threading.BoundedSemaphore(int(os.environ.get("BB_SONNET_JOBS", "6")))


def ask(who, prompt, schema, *, effort="medium", tools=(), files=None, trees=None, timeout=900, tag="", retries=2, model=None, system=None):
    """One auditor call. `system` replaces the auditor system prompt (claude and codex only). `tools` for claude are tool names; for codex, "web" and/or "write" switch those on."""
    model = model or MODELS[who]
    key = _key(who, model, effort, sorted(tools), prompt, schema, sorted((files or {}).items()),
               sorted((trees or {}).keys()), *([system] if system else []), *(["shell-v2"] if who == "codex" and (files or trees) else []))
    cp = CACHE / key[:2] / f"{key}.json"
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))["answer"]
    last = None
    for attempt in range(retries + 1):
        with _SLOTS[who]:
            wd = tempfile.mkdtemp(prefix="bb-")
            try:
                _seed(wd, files)
                _copytree(wd, trees)
                ans, stats = RUNNERS[who](prompt, schema, model, effort, list(tools), wd, timeout, **({"system": system} if system else {}))
                break
            except (AgentError, subprocess.TimeoutExpired) as e:
                last = e
                _log({"tag": tag, "who": who, "error": str(e)[:300], "attempt": attempt})
                time.sleep(20 * (attempt + 1))
            finally:
                shutil.rmtree(wd, ignore_errors=True)
    else:
        raise AgentError(f"{who} failed {retries + 1} times on {tag}: {last}")
    cp.parent.mkdir(parents=True, exist_ok=True)
    tmp = cp.with_suffix(".tmp")
    tmp.write_text(json.dumps({"who": who, "model": model, "effort": effort, "tag": tag, "answer": ans, "stats": stats},
                              ensure_ascii=False), encoding="utf-8")
    tmp.replace(cp)
    _log({"tag": tag, "who": who, **stats})
    return ans


def _log(row):
    p = ROOT / "runs" / "calls.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK, p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.strftime("%H:%M:%S"), **row}, ensure_ascii=False) + "\n")
