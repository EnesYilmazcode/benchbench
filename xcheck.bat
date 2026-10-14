@echo off
rem Codex third vote on every flagged item. Resumable: rerun to continue.
cd /d "%~dp0"
set BB_CODEX_JOBS=6
start "bb cross" /min cmd /c "python -X utf8 -m bb.crosscheck %* > runs\cross.log 2>&1"
