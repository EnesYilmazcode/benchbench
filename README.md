# benchbench

A benchmark for benchmarks. It audits the questions, answer keys and tests of popular AI benchmarks and measures how many of them are broken.

Benchmark scores get quoted to a tenth of a point. If 10% of a benchmark's answer keys are wrong, that tenth means nothing. benchbench takes a random sample of each benchmark, has two Claude models solve and review every item without seeing each other's work, runs the code where there is code, and asks a model from another lab to check every flag.

## Results

| Benchmark | Audited | Broken | Rate | 95% CI | Weak tests |
|---|---:|---:|---:|---|---:|
| SWE-bench Verified | 100 | 15 | 15.0% | 9.3% to 23.3% | |
| MMLU-Pro | 150 | 17 | 11.3% | 7.2% to 17.4% | |
| HumanEval | 164 | 14 | 8.5% | 5.2% to 13.8% | 77.4% |
| SimpleQA | 150 | 7 | 4.7% | 2.3% to 9.3% | |
| tau2-bench | 150 | 6 | 4.0% | 1.8% to 8.5% | |
| MMLU | 150 | 3 | 2.0% | 0.7% to 5.7% | |
| GSM8K | 150 | 3 | 2.0% | 0.7% to 5.7% | |
| HellaSwag | 150 | 1 | 0.7% | 0.1% to 3.7% | |
| MATH-500 | 150 | 0 | 0.0% | 0.0% to 2.5% | |
| AIME 2025 | 30 | 0 | 0.0% | 0.0% to 11.4% | |
| HMMT February 2025 | 30 | 0 | 0.0% | 0.0% to 11.4% | |

Still running: Terminal-Bench 2, SWE-bench Pro and MBPP. They join the table when every sampled item is audited. A partial benchmark is never reported.

"Broken" means a wrong key, an ambiguous or unanswerable item, or a grader that rejects correct answers. "Weak tests" is measured, not judged: the share of deliberately buggy solutions that pass the official tests and still print a different result from the reference on a valid input.

The full report, with every finding and the reasons behind it, is `docs/index.html`. Every flagged item is in `results/findings.jsonl`.

## Examples

**GSM8K 952.** "Vince can staple 30 reports every 15 minutes. If he was stapling reports from 8:00 AM until 11:00 PM, how many reports did he staple?" The key says 360, which is three hours. 8 AM to 11 PM is fifteen hours, so the answer is 1800.

**GSM8K 1309.** Kim raises $750, Alexandra $430, Maryam $700, Sarah $300. The key says 2280. The total is 2180; the key added Maryam's $400 difference in place of Sarah's $300.

**HumanEval 97.** `multiply(a, b)` should return the product of the unit digits. The reference takes Python's modulo before `abs`, so `multiply(14, -12)` returns 32 instead of 8. The one negative test ends in 5, which hides the bug.

**MMLU electrical engineering 17.** The feedback factor of a Wien bridge oscillator is 1/3. The key says 1/2.

## How it works

1. **Sample.** A fixed random sample per benchmark: 150 items, 100 for the SWE-bench sets, and every item of the small ones (AIME, HMMT, HumanEval, Terminal-Bench). Static checks also run on every item: duplicates, empty answers, keys that are not among the options.
2. **Solve blind.** Claude Opus and Claude Sonnet each answer the item without seeing the key, then review it with the key.
3. **Judge.** When the two disagree, Opus at its highest effort reads both reviews, checks their claims and decides. An item counts as broken only if both reviewers or the judge say so.
4. **Run the code.** For HumanEval and MBPP the official tests run on Modal against the reference, two independent solutions and two deliberately buggy ones.
5. **Read the tasks.** SWE-bench, Terminal-Bench and tau2 are reviewed with the real files in the auditor's folder: the issue, gold patch and tests; the task folder; or the domain policy, database and tool code.
6. **Cross-check.** Every broken verdict on a question or task also goes to Codex (GPT-5.6), which sees the evidence but not the verdicts. So far it agrees with 56 of 66 flags, calls 5 of them correct, and is unsure on 5.

## Check the checker

The same pipeline ran on 120 MMLU-Redux items that humans have already labeled. It reaches 93% precision and 42% recall: when it says an item is broken it is almost always right, and it finds a bit under half of the broken items. Read the rates above as a lower bound.

## Run it

```
pip install -r requirements.txt
python -X utf8 -m bb.static              # checks on every item
python -X utf8 -m bb.audit gsm8k mmlu    # resumable, results in runs/results/
python -X utf8 -m bb.validate            # precision and recall against MMLU-Redux
python -X utf8 -m bb.crosscheck gsm8k    # Codex vote on every flag
python -X utf8 -m bb.report              # results/summary.json and findings.jsonl
python -X utf8 -m bb.site                # docs/index.html
```

The auditors are the Claude Code and Codex CLIs, so it runs on their logins. Code execution needs a Modal account. GPQA Diamond and Humanity's Last Exam are gated and their terms forbid publishing items, so they are not included.
