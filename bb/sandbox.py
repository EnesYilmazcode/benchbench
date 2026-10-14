"""Run candidate programs against a benchmark's own tests, on Modal, inside BigCodeBench's official
evaluation image (it has every library BigCodeBench needs; HumanEval and MBPP need none).

    modal deploy bb/sandbox.py          # once
    from bb.sandbox import run_jobs     # then call from anywhere

A job is {"program": str, "timeout": seconds}. A result is {"ok": bool, "rc": int, "out": last 1500 chars}.
`ok` means the process exited 0 within the time limit.
"""
import modal

app = modal.App("benchbench-sandbox")
image = modal.Image.from_registry("bigcodebench/bigcodebench-evaluate:latest", add_python=None).entrypoint([])


@app.function(image=image, cpu=8, memory=16384, timeout=1800, max_containers=20)
def run_batch(jobs):
    import os
    import subprocess
    import sys
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    def one(job):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "prog.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write(job["program"])
        env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"}
        try:
            r = subprocess.run([sys.executable, p], cwd=d, capture_output=True, text=True, timeout=job.get("timeout", 60), env=env)
            return {"ok": r.returncode == 0, "rc": r.returncode, "out": (r.stdout + r.stderr)[-1500:]}
        except subprocess.TimeoutExpired as e:
            return {"ok": False, "rc": -9, "out": f"TIMEOUT after {job.get('timeout', 60)} s"}

    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(one, jobs))


@app.local_entrypoint()
def smoke():
    print(run_batch.remote([{"program": "import pandas, numpy, matplotlib; print('ok', pandas.__version__)"},
                            {"program": "raise SystemExit(3)"}]))


_fn = None


def run_jobs(jobs):
    """Call the deployed function from a normal Python process."""
    global _fn
    if _fn is None:
        _fn = modal.Function.from_name("benchbench-sandbox", "run_batch")
    return _fn.remote(jobs)
