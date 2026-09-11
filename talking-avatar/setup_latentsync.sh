#!/usr/bin/env bash
set -euo pipefail

echo "== LatentSync setup =="

# LatentSync gets its own venv, separate from .venv (chatterbox-tts +
# SadTalker). Its pinned numpy==1.26.4/diffusers==0.32.2/transformers==4.48.0
# would very likely re-open the exact numpy 1.x/2.x conflicts already fought
# through for the main pipeline. Since it's only ever invoked as a
# subprocess (same pattern as SadTalker's inference.py), isolating it in its
# own venv sidesteps that entirely.

if [ ! -d "LatentSync" ]; then
  git clone https://github.com/bytedance/LatentSync.git
fi

# 1. Python virtual environment (same detection logic as setup.sh)
PYTHON=""
for candidate in "py -3.11" "py -3.10" python3.11 python3.10 python3 python; do
  cmd=($candidate)
  if command -v "${cmd[0]}" &> /dev/null && "${cmd[@]}" --version &> /dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo "Python 3.10+ is required but wasn't found (or only the Microsoft"
  echo "Store alias stub is on PATH). Install Python from python.org, or"
  echo "disable the stub under Settings > Apps > Advanced app settings >"
  echo "App execution aliases, then re-run this script."
  exit 1
fi

echo "Using $($PYTHON --version) ($PYTHON)"
PYTHON_CMD=($PYTHON)
"${PYTHON_CMD[@]}" -m venv .venv-latentsync

if [ ! -f ".venv-latentsync/Scripts/activate" ] && [ ! -f ".venv-latentsync/bin/activate" ]; then
  echo "venv creation failed."
  exit 1
fi

if [ -f ".venv-latentsync/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv-latentsync/bin/activate
else
  # shellcheck disable=SC1091
  source .venv-latentsync/Scripts/activate
fi
python -m pip install --upgrade pip

cd LatentSync

# 2. PyTorch with CUDA. LatentSync pins torch==2.5.1+cu121, which predates
# Blackwell (RTX 50-series) support entirely. Same fix as the main venv's
# chatterbox-tts: strip the torch/torchvision pins and install a modern
# Blackwell-compatible build instead.
grep -vE '^(torch|torchvision)==|^--extra-index-url' requirements.txt > requirements.filtered.txt
python -m pip install "torch>=2.7.0" torchvision --index-url https://download.pytorch.org/whl/cu128

# stringzilla (pulled in transitively) has a gap in its newest release's
# wheel matrix — no cp311-win_amd64 wheel for 5.1.2 — which makes pip fall
# back to building it from source, where it then hits an unrelated MSVC
# compiler error on this locale. Pin to the last 3.x release, which does
# ship that wheel, before it gets pulled in transitively.
python -m pip install "stringzilla<5"

python -m pip install -r requirements.filtered.txt

echo ""
echo "NOTE: pip may warn that some package requires torch==2.5.1 but you have"
echo "a newer torch+cu128 build — that's expected, same reasoning as the main"
echo "setup.sh (Blackwell GPU support requires the newer build)."
echo ""

# 3. insightface has no Windows wheel on PyPI and must build its Cython
# extensions from source, which needs a C++ compiler (Visual Studio Build
# Tools' "Desktop development with C++" workload). Install it last so a
# missing compiler doesn't abort everything else above it.
if ! python -m pip install insightface==0.7.3; then
  echo ""
  echo "ERROR: insightface failed to build. It needs a C++ compiler on"
  echo "Windows (no prebuilt wheel exists on PyPI). Install it with:"
  echo "  winget install Microsoft.VisualStudio.2022.BuildTools --override \"--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended\""
  echo "then open a NEW terminal and re-run this script."
  exit 1
fi

# 4. Download LatentSync 1.5 checkpoints (NOT the repo's default 1.6 —
# 1.6 needs 18GB VRAM for inference; 1.5 needs 8GB, which fits a 12GB GPU).
python -m pip install "huggingface_hub[cli]"
huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir checkpoints
huggingface-cli download ByteDance/LatentSync-1.5 whisper/tiny.pt --local-dir checkpoints

cd ..

echo ""
echo "LatentSync setup complete."
if [ -f ".venv-latentsync/bin/activate" ]; then
  echo "Its venv: source .venv-latentsync/bin/activate"
else
  echo "Its venv: source .venv-latentsync/Scripts/activate"
fi
