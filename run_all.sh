#!/usr/bin/env bash
# The full audit, in two lanes so a slow benchmark does not stall the rest. Resumable: rerun to continue.
cd "$(dirname "$0")"
export MODAL_PROFILE=galaxyenocan BB_CLAUDE_JOBS=4 BB_SONNET_JOBS=4
python -X utf8 -m bb.audit mmlu_pro gsm8k math500 aime2025 hmmt2025 simpleqa mmlu hellaswag arc_challenge \
  truthfulqa commonsense_qa winogrande simpleqa_verified frontierscience --workers 8 > runs/lane1.log 2>&1 &
python -X utf8 -m bb.audit humaneval swe_verified terminal_bench2 tau2 mbpp bigcodebench swe_pro --workers 6 > runs/lane2.log 2>&1 &
wait
