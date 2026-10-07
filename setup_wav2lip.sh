#!/usr/bin/env bash
set -euo pipefail

echo "== Wav2Lip setup (optional --refine-lipsync pass) =="

# Wav2Lip gets its own venv, separate from .venv/.venv-latentsync/.venv-liveportrait.
# It pins torch==1.1.0/torchvision==0.3.0/numpy==1.17.1/opencv-python==4.1.0.25 —
# ancient versions with no Blackwell (RTX 50-series) support and no modern wheels
# for Python 3.10/3.11 either. Same isolation reasoning as the other three engines:
# it's only ever invoked as a subprocess (its own inference.py), so a broken/ancient
# pin in its venv can't affect anything else.

if [ ! -d "Wav2Lip" ]; then
  git clone https://github.com/Rudrabha/Wav2Lip.git
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
"${PYTHON_CMD[@]}" -m venv .venv-wav2lip

if [ ! -f ".venv-wav2lip/Scripts/activate" ] && [ ! -f ".venv-wav2lip/bin/activate" ]; then
  echo "venv creation failed."
  exit 1
fi

if [ -f ".venv-wav2lip/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv-wav2lip/bin/activate
else
  # shellcheck disable=SC1091
  source .venv-wav2lip/Scripts/activate
fi
python -m pip install --upgrade pip

cd Wav2Lip

# 2. PyTorch with CUDA. Wav2Lip's requirements.txt pins torch==1.1.0, which
# predates Blackwell (RTX 50-series) support by years. Strip the torch/
# torchvision/numpy/opencv pins and install current, Blackwell-compatible
# versions instead — Wav2Lip's inference code (a plain conv-net forward pass,
# no exotic ops) runs fine against them.
grep -vE '^(torch|torchvision|numpy|opencv-python)(==|>=|$)' requirements.txt > requirements.filtered.txt
python -m pip install "torch>=2.7.0" torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install "numpy>=1.24,<2.0" opencv-python
python -m pip install -r requirements.filtered.txt

echo ""
echo "NOTE: pip may warn about torch/numpy/opencv version mismatches against"
echo "Wav2Lip's original pins — expected, same reasoning as setup.sh (Blackwell"
echo "GPU support requires newer builds than this project originally pinned)."
echo ""

# 3. Face detection weights (S3FD). Wav2Lip's own face_detection/ submodule
# downloads this on first run via a URL that has repeatedly gone dark
# (redirects/404s depending on the day); pre-fetch it from a mirror that's
# stayed up so first inference doesn't fail mid-run.
S3FD_DIR="face_detection/detection/sfd"
S3FD_PATH="$S3FD_DIR/s3fd.pth"
mkdir -p "$S3FD_DIR"
# A .pth file is a pickle, and loading a pickle runs code, so the download is
# checked before it's kept: the "619a316812" in the upstream filename is the
# start of its SHA-256 (torch.hub's naming convention).
S3FD_SHA256_PREFIX="619a316812"
if [ ! -f "$S3FD_PATH" ]; then
  if curl -L --fail -o "$S3FD_PATH.part" \
      "https://www.adrianbulat.com/downloads/python-fan/s3fd-${S3FD_SHA256_PREFIX}.pth"; then
    if python -c "import hashlib, sys; sys.exit(0 if hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest().startswith(sys.argv[2]) else 1)" \
        "$S3FD_PATH.part" "$S3FD_SHA256_PREFIX"; then
      mv "$S3FD_PATH.part" "$S3FD_PATH"
    else
      rm -f "$S3FD_PATH.part"
      echo "WARNING: the downloaded s3fd.pth failed its checksum and was deleted — see README's Wav2Lip troubleshooting entry."
    fi
  else
    rm -f "$S3FD_PATH.part"
    echo "WARNING: s3fd.pth download failed — see README's Wav2Lip troubleshooting entry for a manual link."
  fi
fi

# 4. wav2lip_gan.pth checkpoint. Hosted on the authors' OneDrive (linked from
# Wav2Lip's README) — no stable direct-download URL, and it changes over
# time, so this can't be scripted reliably with curl/wget.
mkdir -p checkpoints
if [ ! -f "checkpoints/wav2lip_gan.pth" ]; then
  echo ""
  echo "ACTION NEEDED: download wav2lip_gan.pth manually."
  echo "  1. Open the link in the 'Model' table of https://github.com/Rudrabha/Wav2Lip#getting-the-weights"
  echo "  2. Save it as: $(pwd)/checkpoints/wav2lip_gan.pth"
  echo "Re-run this script (or just make_avatar.py --refine-lipsync) once it's in place —"
  echo "everything else above is already done."
  echo ""
fi

cd ..

echo ""
echo "Wav2Lip setup complete."
if [ -f ".venv-wav2lip/bin/activate" ]; then
  echo "Its venv: source .venv-wav2lip/bin/activate"
else
  echo "Its venv: source .venv-wav2lip/Scripts/activate"
fi
