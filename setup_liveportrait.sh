#!/usr/bin/env bash
set -euo pipefail

echo "== LivePortrait setup =="

# LivePortrait gets its own venv, separate from .venv and .venv-latentsync.
# Its pinned numpy==1.26.4 happens to match LatentSync's, but onnxruntime-gpu
# is pinned to a different version in each (1.18.0 here vs 1.21.0 there) —
# every time this project has shared a venv between two pinned dependency
# sets it has caused a conflict, so keep the per-tool isolation consistent.

if [ ! -d "LivePortrait" ]; then
  git clone https://github.com/KwaiVGI/LivePortrait.git
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
"${PYTHON_CMD[@]}" -m venv .venv-liveportrait

if [ ! -f ".venv-liveportrait/Scripts/activate" ] && [ ! -f ".venv-liveportrait/bin/activate" ]; then
  echo "venv creation failed."
  exit 1
fi

if [ -f ".venv-liveportrait/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv-liveportrait/bin/activate
else
  # shellcheck disable=SC1091
  source .venv-liveportrait/Scripts/activate
fi
python -m pip install --upgrade pip

cd LivePortrait

# 2. PyTorch with CUDA. LivePortrait's README pins old torch/CUDA
# combinations (e.g. torch==2.3.0+cu121), none of which support Blackwell
# (RTX 50-series). Unlike LatentSync/chatterbox-tts, torch is NOT pinned in
# requirements.txt/requirements_base.txt itself (only the README's install
# instructions pin it) — so a plain install just works with our standard
# Blackwell-compatible build, no --no-deps override needed.
python -m pip install "torch>=2.7.0" torchvision --index-url https://download.pytorch.org/whl/cu128

# albumentations pulls in albucore, which floors stringzilla>=5.1.2 — and
# that specific version has a genuine upstream bug in its Windows/MSVC build
# (undeclared identifiers even in its portable/non-SIMD fallback code, not
# fixable via compiler flags or disabling SIMD codepaths; verified by
# testing both). Unlike setup_latentsync.sh's stringzilla situation, there's
# no version below the floor with a working wheel to fall back to here.
# albumentations is declared in requirements_base.txt but never actually
# imported anywhere in this repo (checked: `grep -rln "import
# albumentations" .` finds nothing) — it's inference-safe to drop entirely.
#
# requirements.txt itself contains `-r requirements_base.txt` internally, so
# passing it to pip re-includes the unfiltered file regardless of what else
# is on the command line — install its two other explicit lines directly
# instead of `-r requirements.txt`.
grep -v '^albumentations==' requirements_base.txt > requirements_base.filtered.txt
python -m pip install -r requirements_base.filtered.txt

# requirements.txt's pinned onnxruntime-gpu==1.18.0 fails at runtime on this
# setup with "LoadLibrary failed... onnxruntime_providers_cuda.dll" (its
# bundled CUDA provider doesn't match the CUDA/cuDNN actually available
# here). 1.21.0 — the same version LatentSync already uses successfully in
# this same environment — works.
python -m pip install "onnxruntime-gpu==1.21.0" "transformers==4.38.0"

# 3. Download pretrained weights. Note: hosted under the KlingTeam org on
# HuggingFace, not KwaiVGI (the code repo's org) — easy to get wrong.
# Use `hf download` (not the older `huggingface-cli download`): this
# version of huggingface_hub prints a deprecation warning containing an
# emoji when using the old command, which crashes on a non-UTF-8 console
# locale (e.g. Windows set to Japanese/cp932).
python -m pip install "huggingface_hub[cli]"
hf download KlingTeam/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"

# LivePortrait's own progress-bar text includes a rocket emoji
# ('🚀Animating...'), which crashes the same way on a non-UTF-8 console
# locale — this time via `rich`'s low-level Windows legacy-console writer,
# which isn't affected by redirecting the subprocess's output (tried;
# didn't help) since its "legacy terminal" detection isn't pipe-vs-console
# based. Patch the emoji out directly.
python - <<'PYEOF'
import pathlib
for rel in ["src/live_portrait_pipeline.py", "src/live_portrait_pipeline_animal.py"]:
    f = pathlib.Path(rel)
    if not f.exists():
        continue
    text = f.read_text(encoding="utf-8")
    patched = text.replace("'\U0001F680Animating...'", "'Animating...'")
    if patched != text:
        f.write_text(patched, encoding="utf-8")
        print(f"Patched {rel} for console-encoding compatibility.")
PYEOF

cd ..

echo ""
echo "LivePortrait setup complete."
if [ -f ".venv-liveportrait/bin/activate" ]; then
  echo "Its venv: source .venv-liveportrait/bin/activate"
else
  echo "Its venv: source .venv-liveportrait/Scripts/activate"
fi
