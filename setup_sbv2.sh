#!/usr/bin/env bash
set -euo pipefail

echo "== Style-Bert-VITS2 setup (Japanese with correct pitch accent, --tts sbv2) =="

# Chatterbox and XTTS-v2 are multilingual models with no explicit model of
# Japanese pitch accent (高低アクセント); both were rejected by ear for Japanese
# intonation even with every reading spelled out in kana. Style-Bert-VITS2
# JP-Extra predicts accent from pyopenjtalk and accepts a per-mora override, so
# a wrong accent can also be fixed by hand ({雨|あ'め} in the script).
#
# It has no zero-shot cloning: train_sbv2.py trains it once on ~10 minutes of
# you speaking Japanese. After that, --lang ja uses it automatically.

if [ ! -d "Style-Bert-VITS2" ]; then
  git clone https://github.com/litagin02/Style-Bert-VITS2.git
fi

# 1. Python virtual environment. 3.11 is what the rest of this project uses,
# and pyopenjtalk-dict ships a prebuilt cp311 Windows wheel (no compiler needed).
PYTHON=""
for candidate in "py -3.11" python3.11 "py -3.10" python3.10 python3 python; do
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
"${PYTHON_CMD[@]}" -m venv .venv-sbv2

if [ -f ".venv-sbv2/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv-sbv2/bin/activate
else
  # shellcheck disable=SC1091
  source .venv-sbv2/Scripts/activate
fi
python -m pip install --upgrade pip

# SBV2's configs/paths.yml needs a path Python understands; Git Bash's /c/...
# form would be read as a relative path on Windows.
ROOT_NATIVE="$(python -c 'import os; print(os.getcwd())')"

cd Style-Bert-VITS2
# requirements.txt pins torch<2.4 / torchaudio<2.4. Those builds have no sm_120
# kernels, so on an RTX 50-series card torch.cuda.is_available() is simply
# False. Filter the torch family out (same pattern as setup_seedvc.sh) and
# install a CUDA 12.8 build afterwards.
#
# faster-whisper goes too. Its 0.10.1 pin needs av==10.*, which has no Python
# 3.11 wheel and no longer compiles with current Cython, and its CTranslate2
# 3.x build predates sm_120 anyway. transcribe.py only imports it on its
# default path; train_sbv2.py always passes --use_hf_whisper, which runs
# Whisper on this venv's torch instead.
#
# Two upstream ranges are open-ended and now resolve to releases Style-Bert-VITS2
# was never written against, so hold them on their last compatible major:
#   transformers 5.x
#   pyannote.audio 4.x, which pulls in opentelemetry and with it a protobuf
#   that contradicts SBV2's protobuf==4.25 pin. pip then backtracks through
#   every numba release down to one that only builds on Python < 3.11.
# and two that SBV2's old librosa==0.9.2 pin needs from around it:
#   setuptools<81: tensorboard upgrades setuptools, and 81+ no longer ships
#   pkg_resources, which librosa 0.9.2 imports.
#   soxr: transformers 4.5x imports it whenever librosa is installed, assuming
#   librosa 0.10+ (which depends on it); 0.9.2 doesn't, so DeBERTa - and with
#   it SBV2's BERT - fails to import.
grep -viE '^[[:space:]]*(torch(vision|audio)?|faster-whisper)([[:space:]]|[=<>!~]|$)' requirements.txt > requirements.filtered.txt
printf '%s\n' "transformers<5" "pyannote.audio<4" "setuptools<81" "soxr" >> requirements.filtered.txt
python -m pip install -r requirements.filtered.txt
cd ..

# Installed last so nothing in the requirements can pull torch back down.
# Pinned below 2.9 like setup_seedvc.sh/setup_xtts.sh: 2.9's torchaudio needs
# torchcodec and FFmpeg's shared libraries, which static Windows ffmpeg builds
# don't provide. torchvision only if something above pulled it in, pinned to
# match, or it drags torch back with it.
TORCH_PKGS=("torch==2.8.0" "torchaudio==2.8.0")
if python -m pip show torchvision > /dev/null 2>&1; then
  TORCH_PKGS+=("torchvision==0.23.0")
fi
python -m pip install "${TORCH_PKGS[@]}" --index-url https://download.pytorch.org/whl/cu128

# 2. Models: the Japanese BERT, the JP-Extra pretrained base that training
# starts from, and the stock JVNV voices (used to smoke-test the install before
# spending an hour on training). Trained voices go to models/sbv2/.
cd Style-Bert-VITS2
python initialize.py --dataset_root "$ROOT_NATIVE/models/sbv2-data" --assets_root "$ROOT_NATIVE/models/sbv2"

# The Japanese BERT ships only as pytorch_model.bin, one 1.3GB storage.
# transformers 4.5x first reads it with torch.load(map_location="meta"), which
# on this torch build tries to allocate that storage in one piece and fails
# ("DefaultCPUAllocator: not enough memory") even with 20+GB free, while a
# memory-mapped load of the same file works. transformers prefers a
# model.safetensors next to it, so write one once. Each tensor is cloned so
# the tied embedding/decoder weights don't share memory (safetensors refuses
# shared tensors; transformers re-ties them on load).
python - <<'PY'
from pathlib import Path
import torch
from safetensors.torch import save_file

bert = Path("bert/deberta-v2-large-japanese-char-wwm")
target = bert / "model.safetensors"
if not target.exists():
    state = torch.load(bert / "pytorch_model.bin", map_location="cpu", mmap=True, weights_only=True)
    save_file({k: v.contiguous().clone() for k, v in state.items()}, str(target), metadata={"format": "pt"})
    print(f"Wrote {target}")
PY
cd ..

echo ""
echo "Style-Bert-VITS2 setup complete. Next, train it on a recording of you"
echo "speaking Japanese (about 10 minutes is plenty):"
echo "  python train_sbv2.py --voice voice_samples/me.wav"
