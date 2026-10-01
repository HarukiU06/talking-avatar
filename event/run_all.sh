#!/usr/bin/env bash
# Phase 3: render S2-S5 with the settings approved in the S6/S1 test (lightx2v
# 4 steps, 704x576 render input, seeds 1000+n, same scene prompt). Resumable:
# finished sections are skipped, so re-run the same command after any failure.
# Run from a normal terminal with other apps closed, not from inside Claude Code.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python
"$PY" event/render_sections.py S2 S3 S4 S5 2>&1 | tee -a output/event/video_render_all.log
