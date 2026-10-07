#!/usr/bin/env bash
set -euo pipefail

echo "== Seed-VC setup (voice conversion, --voice-convert) =="

# Seed-VC converts the *timbre* of already-generated speech to a target
# speaker, which is a different job from the TTS engines in setup.sh /
# setup_xtts.sh. The two stages together beat zero-shot TTS cloning alone:
# the TTS supplies natural prosody without straining to imitate anyone, and
# Seed-VC supplies the identity from a reference clip. Zero-shot TTS cloning
# on its own has to do both at once, and audibly compromises on both.
#
# Chosen over RVC deliberately: RVC needs a per-speaker training run, and on
# RTX 50-series (Blackwell) it additionally needs CUDA 12.8 + nightly-PyTorch
# workarounds that are still a moving target. Seed-VC works zero-shot, and
# ships train.py if fine-tuning on a longer recording is wanted later.

if [ ! -d "seed-vc" ]; then
  git clone https://github.com/Plachtaa/seed-vc.git
fi

# 1. Python virtual environment (same detection logic as the other scripts).
# Upstream suggests 3.10; 3.11 is what the rest of this project uses and
# works, so prefer whichever is present rather than forcing an extra install.
PYTHON=""
for candidate in "py -3.10" "py -3.11" python3.10 python3.11 python3 python; do
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
"${PYTHON_CMD[@]}" -m venv .venv-seedvc

if [ -f ".venv-seedvc/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv-seedvc/bin/activate
else
  # shellcheck disable=SC1091
  source .venv-seedvc/Scripts/activate
fi
python -m pip install --upgrade pip

cd seed-vc
# Drop every torch-family line from the pinned requirements before installing
# them, then install the CUDA 12.8 builds afterwards.
#
# seed-vc's requirements.txt lists the torch family twice: once as
# "torch --pre --index-url .../nightly/cu126" and once as "torch==2.4.0".
# Two traps here, both of which silently produce a CPU-only torch 2.4.0 that
# cannot run on Blackwell (sm_120 is absent from its arch list, so
# torch.cuda.is_available() is simply False):
#   - the filter has to match the space-separated form as well as "torch==",
#     or the nightly line survives and drags in its own index
#   - it has to include torchvision, or torchvision==0.19.0 survives and
#     pins torch back down to 2.4.0 as a dependency
grep -viE '^[[:space:]]*torch(vision|audio)?([[:space:]]|[=<>!~]|$)'   requirements.txt > requirements.filtered.txt
python -m pip install -r requirements.filtered.txt
cd ..

# PyTorch with CUDA 12.8 (Blackwell / RTX 50-series), installed last so that
# nothing in the requirements can pull it back to a CPU or older build.
#
# Pinned below 2.9 for the same reason as setup_xtts.sh: from torch 2.9,
# torchaudio.save() delegates to torchcodec, and torchcodec needs FFmpeg's
# *shared* libraries at runtime. The common Windows ffmpeg builds (gyan.dev
# via scoop/winget) are static, so ffmpeg.exe on PATH does not satisfy it.
# seed-vc's inference.py saves its result with torchaudio.save(), so on 2.9+
# the conversion runs to completion on the GPU and then dies at the very last
# line with "TorchCodec is required for save_with_torchcodec". 2.8+cu128 still
# ships sm_120 kernels, so Blackwell is fully supported.
#
# All three must be pinned together: an unpinned torchvision resolves to one
# built against a different torch and silently drags torch back with it.
python -m pip install "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0"   --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "Seed-VC setup complete. Checkpoints download automatically on first run."
echo "Use it with:  python make_avatar.py --voice-convert ..."
