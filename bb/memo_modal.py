"""The memorization probe for open-weight models, on a Modal GPU.

Same 100 questions and the same two option orders as bb.memo. The model sees the question in its
chat template and we read its next-token probabilities for A, B, C and D, so there is no sampling.

`modal run bb/memo_modal.py` writes runs/memo/<model>.jsonl for every model in OPEN.
"""
import json
import os
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
OPEN = {  # name in the table -> Hugging Face repo
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2-9b": "google/gemma-2-9b-it",
    "olmo2-7b": "allenai/OLMo-2-1124-7B-Instruct",  # training data is public: the control
}

app = modal.App("benchbench-memo")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.51.3", "accelerate==1.2.1")
hf_cache = modal.Volume.from_name("benchbench-hf", create_if_missing=True)


@app.function(image=image, gpu="A100-40GB", timeout=3600, volumes={"/hf": hf_cache},
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def probe(repo, prompts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.environ["HF_HOME"] = "/hf"
    tok = AutoTokenizer.from_pretrained(repo, cache_dir="/hf")
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="cuda", cache_dir="/hf")
    hf_cache.commit()
    ids = [tok.encode(l, add_special_tokens=False)[-1] for l in "ABCD"]
    out = []
    for p in prompts:
        msgs = [{"role": "user", "content": p + "\n\nReply with only the letter."}]
        kw = {"enable_thinking": False} if "Qwen3" in repo else {}
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        x = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            logits = model(**x).logits[0, -1]
        probs = torch.softmax(logits[ids].float(), -1).tolist()
        out.append(probs)
    return out


@app.local_entrypoint()
def main():
    sys.path.insert(0, str(ROOT))
    from bb import memo
    qs = memo.items()
    jobs = [(it, v, perm) for it in qs for v, perm in memo.variants(it)]
    prompts = [memo.prompt(it, perm) for it, v, perm in jobs]
    memo.OUT.mkdir(parents=True, exist_ok=True)
    names = list(OPEN)
    for name, res in zip(names, probe.starmap([(OPEN[n], prompts) for n in names], return_exceptions=True)):
        if isinstance(res, Exception):
            print(f"{name}: FAILED {res}"[:400])
            continue
        with (memo.OUT / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for (it, v, perm), probs in zip(jobs, res):
                fh.write(json.dumps({"id": it["id"], "v": v, "perm": perm, "answer": "ABCD"[probs.index(max(probs))],
                                     "probs": probs}) + "\n")
        print(f"{name}: {len(res)} answers")
