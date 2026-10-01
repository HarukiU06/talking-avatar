#!/usr/bin/env bash
# Phase 2 test: render S6 first (short; the closing clip crashed before), then
# S1 (has the English line). Resumable - re-run to continue after a failure.
# Run from a normal terminal with other apps closed, not from inside Claude Code.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python
"$PY" event/render_sections.py S6 S1 2>&1 | tee -a output/event/video_render_test.log
