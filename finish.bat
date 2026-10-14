@echo off
rem Finish the half-done benchmarks. Resumable: rerun to continue.
cd /d "%~dp0"
set MODAL_PROFILE=galaxyenocan
set BB_CLAUDE_JOBS=4
set BB_SONNET_JOBS=4
start "bb lane1" /min cmd /c "python -X utf8 -m bb.audit simpleqa --workers 8 > runs\finish1.log 2>&1"
start "bb lane2" /min cmd /c "python -X utf8 -m bb.audit terminal_bench2 --workers 6 > runs\finish2.log 2>&1"
