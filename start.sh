#!/usr/bin/env bash
# Talking Avatar: the one command. Installs whatever is missing, then opens the app.
#
#   ./start.sh                 first run: install (about 15-20 GB, 30-60 min), then open the app
#                              after that: open the app straight away
#   ./start.sh --skip-setup    open the app without checking the install
#
# Any other arguments go to app.py: --browser, --port N, --listen, --auth USER:PASSWORD.
# Works on macOS, Linux, and Git Bash on Windows (start.bat runs this for you).
cd "$(dirname "$0")" || exit 1

venv_python() {  # prints the Python inside a virtual environment folder, if there is one
  local exe
  for exe in "$1/bin/python" "$1/Scripts/python.exe"; do
    if [ -x "$exe" ]; then echo "$exe"; return 0; fi
  done
  return 1
}

# A part counts as installed once its setup script has finished here (the
# .setup-complete marker), or, for installs made by hand before this script
# did it, when the files that part needs are already in place.
is_installed() {
  local py
  case "$1" in
    setup.sh)
      [ -f .venv/.setup-complete ] && return 0
      py=$(venv_python .venv) && "$py" -c "import chatterbox, gradio" >/dev/null 2>&1 ;;
    setup_latentsync.sh)
      [ -f .venv-latentsync/.setup-complete ] && return 0
      venv_python .venv-latentsync >/dev/null && [ -f LatentSync/checkpoints/latentsync_unet.pt ] \
        && [ -f LatentSync/checkpoints/whisper/tiny.pt ] ;;
    setup_liveportrait.sh)
      [ -f .venv-liveportrait/.setup-complete ] && return 0
      venv_python .venv-liveportrait >/dev/null && [ -d LivePortrait/pretrained_weights/liveportrait ] ;;
  esac
}

venv_of() {
  case "$1" in
    setup.sh) echo .venv ;;
    setup_latentsync.sh) echo .venv-latentsync ;;
    setup_liveportrait.sh) echo .venv-liveportrait ;;
  esac
}

skip_setup=0
app_args=()
for arg in "$@"; do
  if [ "$arg" = "--skip-setup" ]; then skip_setup=1; else app_args+=("$arg"); fi
done

failed=()
if [ "$skip_setup" = 0 ]; then
  # The voice cloning plus the default (Standard) video engine. The other
  # engines are optional; the app's Setup tab shows how to add them.
  missing=()
  for script in setup.sh setup_latentsync.sh setup_liveportrait.sh; do
    is_installed "$script" || missing+=("$script")
  done

  if [ "${#missing[@]}" -gt 0 ]; then
    echo "== Talking Avatar: installing (${missing[*]}) =="
    echo "This downloads about 15-20 GB and can take 30-60 minutes. It only happens once;"
    echo "after that ./start.sh opens the app straight away."
    if ! command -v ffmpeg > /dev/null 2>&1; then
      echo ""
      echo "WARNING: ffmpeg isn't installed (or isn't on PATH). The app needs it for every video:"
      echo "  Windows: winget install ffmpeg   macOS: brew install ffmpeg   Linux: sudo apt install ffmpeg"
    fi
    for script in "${missing[@]}"; do
      echo ""
      echo "== Running $script =="
      if bash "./$script"; then
        touch "$(venv_of "$script")/.setup-complete"
      elif [ "$script" = "setup.sh" ]; then
        echo ""
        echo "Setup stopped: setup.sh failed (see the messages above, and README section 8)." >&2
        echo "Fix the problem and run ./start.sh again; it carries on from where it stopped." >&2
        exit 1
      else
        failed+=("$script")
      fi
    done
  fi
fi

if ! python=$(venv_python .venv); then
  echo "Talking Avatar isn't installed yet: run ./start.sh (without --skip-setup)." >&2
  exit 1
fi
if [ "${#failed[@]}" -gt 0 ]; then
  echo ""
  echo "NOTE: ${failed[*]} didn't finish, so the Standard engine isn't ready yet."
  echo "The app opens anyway: its Setup tab shows what's missing. ./start.sh tries again next"
  echo "time; to open the app without retrying, use ./start.sh --skip-setup."
fi
exec "$python" app.py ${app_args[@]+"${app_args[@]}"}
