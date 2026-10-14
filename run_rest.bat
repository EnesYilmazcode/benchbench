@echo off
rem The trimmed audit: benchmarks people still quote, plus MMLU and HellaSwag. Resumable: rerun to continue.
cd /d "%~dp0"
set MODAL_PROFILE=galaxyenocan
set BB_CLAUDE_JOBS=3
set BB_SONNET_JOBS=3
start "bb lane3" /min cmd /c "python -X utf8 -m bb.audit mmlu hellaswag --workers 6 > runs\rest1.log 2>&1"
start "bb lane4" /min cmd /c "python -X utf8 -m bb.audit tau2 swe_pro mbpp --workers 6 > runs\rest2.log 2>&1"
