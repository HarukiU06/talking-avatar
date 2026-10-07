#!/usr/bin/env bash
set -euo pipefail

echo "== XTTS-v2 setup (optional --tts xtts) =="

# XTTS gets its own venv for the same reason as every other engine here: the
# maintained coqui-tts fork pins transformers/numpy versions that conflict with
# chatterbox-tts in .venv. It's only ever invoked as a subprocess (xtts_synth.py),
# so its pins can't affect anything else.
#
# XTTS-v2 is the A/B alternative to Chatterbox for voice-cloning fidelity. It
# covers 17 languages (en/es/fr/de/it/pt/pl/tr/ru/nl/cs/ar/zh/ja/hu/ko/hi), which
# is why it was picked over F5-TTS and IndexTTS-2 — both clone at least as well
# but are effectively English/Chinese only, and this project promises multilingual.
#
# Note: the XTTS-v2 *weights* are under the Coqui Public Model License
# (non-commercial). The code is MPL-2.0. Fine for personal use; see README.

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
"${PYTHON_CMD[@]}" -m venv .venv-xtts

if [ -f ".venv-xtts/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv-xtts/bin/activate
else
  # shellcheck disable=SC1091
  source .venv-xtts/Scripts/activate
fi
python -m pip install --upgrade pip

# 2. PyTorch with CUDA 12.8 (Blackwell / RTX 50-series), installed before
# coqui-tts so pip doesn't pull a CPU-only build as a transitive dependency.
#
# Pinned below 2.9 deliberately. From torch 2.9, coqui-tts switches its audio
# IO to torchcodec, and torchcodec needs FFmpeg's *shared* libraries
# (avutil-*.dll etc.) at runtime. The usual Windows ffmpeg builds - including
# the gyan.dev "full" build that scoop and winget install - are static, so
# ffmpeg.exe being on PATH is not enough and torchcodec fails to load its core
# library with a WinError 126/127. torch 2.8 keeps the soundfile-backed path
# and needs no extra binaries. 2.8+cu128 still ships sm_120 kernels, so
# Blackwell GPUs are fully supported.
#
# torch and torchaudio must be the same minor version or torchaudio's C
# extension fails to load ("WinError 127: The specified procedure could not
# be found") - hence pinning both rather than letting torchaudio float.
python -m pip install "torch>=2.7,<2.9" "torchaudio>=2.7,<2.9" --index-url https://download.pytorch.org/whl/cu128

# 3. coqui-tts is the maintained fork of the archived coqui-ai/TTS package.
# Install "TTS" and you get the dead original.
#
# transformers must be pinned below 5.0: coqui-tts's tortoise/XTTS layers
# import transformers.pytorch_utils.isin_mps_friendly, which 5.x removed, so
# an unpinned install resolves to 5.x and then fails at import time with
# "cannot import name 'isin_mps_friendly'". Pinning here rather than after
# the fact avoids downloading a transformers/tokenizers pair twice.
# The [ja] extra brings cutlet/fugashi/unidic-lite, which XTTS needs to read
# Japanese text; without it --tts xtts --lang ja dies with
# "No module named 'cutlet'".
python -m pip install "coqui-tts[ja]" soundfile "transformers>=4.43,<5"

# 4. Pre-download the XTTS-v2 checkpoint (~2GB) so the first real run doesn't
# stall on it. COQUI_TOS_AGREED is the non-interactive equivalent of the
# license prompt the downloader shows on a TTY — without it this hangs
# forever waiting on stdin when run from a script.
echo ""
echo "Downloading the XTTS-v2 checkpoint (~2GB). By continuing you accept the"
echo "Coqui Public Model License (non-commercial use only):"
echo "  https://huggingface.co/coqui/XTTS-v2 (see its LICENSE.txt)"
COQUI_TOS_AGREED=1 python -c "
from TTS.utils.manage import ModelManager
ModelManager().download_model('tts_models/multilingual/multi-dataset/xtts_v2')
print('XTTS-v2 checkpoint ready.')
"


echo ""
echo "XTTS setup complete. Use it with:  python make_avatar.py --tts xtts ..."
echo "Compare it against Chatterbox with: python make_avatar.py --voice-compare ..."
