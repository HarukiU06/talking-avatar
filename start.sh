#!/usr/bin/env bash
# Talking Avatar: start the app (macOS, Linux, or Git Bash on Windows).
cd "$(dirname "$0")" || exit 1
for python in .venv/bin/python .venv/Scripts/python.exe; do
  if [ -x "$python" ]; then
    exec "$python" app.py "$@"
  fi
done
echo "Talking Avatar is not installed yet. Run ./setup.sh first: see README section 3." >&2
exit 1
