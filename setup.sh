#!/usr/bin/env bash
set -euo pipefail

echo "== Talking Avatar setup =="

# 1. Python virtual environment
# On Windows (Git Bash), the launcher is `python`, not `python3`, and the
# venv's activate script lives under Scripts/, not bin/. Also, on Windows,
# `python3`/`python` may resolve to the Microsoft Store alias stub, which is
# present on PATH but does not run real Python (it just prints a message
# telling you to install from the Store) — command -v alone can't detect
# that, so we actually run `--version` and check it worked.
PYTHON=""
# Prefer 3.11/3.10 (what this project is tested against) if present, via
# either the `py` launcher or a versioned binary, before falling back to
# whatever plain `python3`/`python` resolves to.
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

PY_VER=$($PYTHON --version 2>&1)
echo "Using $PY_VER ($PYTHON)"
case "$PY_VER" in
  *" 3.10."*|*" 3.11."*) ;;
  *)
    echo ""
    echo "WARNING: this project is tested against Python 3.10/3.11; you have"
    echo "$PY_VER. Pinned packages (dlib-bin, kornia, transformers, etc.) have"
    echo "3.13 wheels as of writing, so this will likely work, but if install"
    echo "or import errors show up below, install Python 3.11 from python.org"
    echo "and re-run this script (it'll be picked up automatically)."
    echo ""
    ;;
esac
PYTHON_CMD=($PYTHON)
"${PYTHON_CMD[@]}" -m venv .venv

if [ ! -f ".venv/Scripts/activate" ] && [ ! -f ".venv/bin/activate" ]; then
  echo "venv creation failed — no .venv/Scripts/activate or .venv/bin/activate was created."
  exit 1
fi

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
fi
python -m pip install --upgrade pip

# 2. PyTorch with CUDA.
# Check your CUDA version with `nvidia-smi` (top-right corner) and match the
# index-url below to it. RTX 50-series (Blackwell, e.g. 5070 Ti) needs
# torch>=2.7 built against CUDA 12.6+ (cu121/cu124 wheels lack the sm_120
# kernels and will fail or silently fall back to CPU) — cu128 is the safest
# match for Blackwell. No NVIDIA GPU (e.g. Apple Silicon)? Just run:
#   pip install torch torchvision torchaudio
python -m pip install "torch>=2.7.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Chatterbox (voice cloning TTS) + config parsing.
# chatterbox-tts hard-pins torch==2.6.0/torchaudio==2.6.0 in its own metadata,
# which predates Blackwell support entirely — installing it normally would
# silently downgrade the GPU-compatible torch we just installed. Install it
# with --no-deps and bring in its other (non-torch) dependencies ourselves.
python -m pip install --no-deps chatterbox-tts
# numpy: chatterbox-tts itself requires <2.0 on Python <3.13 but >=2.0 on
# Python >=3.13 (numpy dropped 1.x wheels for 3.13) — pick the right pin
# instead of hardcoding one, or pip will try to compile 1.26.x from source
# and fail without a C compiler installed.
if python -c "import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)"; then
  NUMPY_SPEC="numpy>=2.0.0"
else
  NUMPY_SPEC="numpy>=1.24,<2.0"
fi
python -m pip install pyyaml "$NUMPY_SPEC" "librosa==0.11.0" s3tokenizer \
  "transformers==5.2.0" "diffusers==0.29.0" resemble-perth "conformer==0.3.2" \
  "safetensors==0.5.3" spacy-pkuseg "pykakasi==2.3.0" "gradio==6.8.0" \
  pyloudnorm omegaconf
echo ""
echo "NOTE: pip may warn above that 'chatterbox-tts requires torch==2.6.0 but"
echo "you have torch 2.x+cu128, incompatible' — that's expected. We deliberately"
echo "kept the newer torch for Blackwell (5070 Ti) GPU support instead of the"
echo "old pin; chatterbox-tts's own code doesn't actually require exactly 2.6.0."
echo ""

# 4. SadTalker (photo -> talking head video)
if [ ! -d "SadTalker" ]; then
  git clone https://github.com/Winfredy/SadTalker.git
fi
cd SadTalker
# SadTalker's requirements.txt pins old numpy/librosa/scipy that would
# conflict with the newer ones Chatterbox needs in this shared venv, and pins
# scikit-image==0.19.3, which has no Python 3.13 wheel and can't build from
# source (its old setuptools-based build touches pkgutil.ImpImporter, removed
# in 3.12+). Strip those four pins and let pip pick current versions —
# SadTalker's inference code works fine with newer ones.
grep -vE '^(numpy|librosa|scipy|scikit-image)==' requirements.txt > requirements.filtered.txt
python -m pip install -r requirements.filtered.txt
# Plain "scikit-image" can resolve to an old version (e.g. 0.20.0) via pip's
# backtracking against other pins in requirements.filtered.txt — old
# scikit-image wheels have C extensions built against numpy's pre-2.0 ABI
# and crash ("numpy.dtype size changed") against the numpy 2.x we install
# above. Force a version built against numpy 2.x.
python -m pip install "scikit-image>=0.24"
python -m pip install dlib-bin
python -m pip install "git+https://github.com/TencentARC/GFPGAN"

# download_models.sh calls `wget`, which Git Bash on Windows doesn't ship
# (only curl). Shim a minimal wget -> curl translation for the flags it
# actually uses (-nc, -O) instead of requiring a separate wget install.
if ! command -v wget &> /dev/null; then
  wget() {
    local url="" out=""
    while [ $# -gt 0 ]; do
      case "$1" in
        -nc) shift ;;
        -O) out="$2"; shift 2 ;;
        http*://*) url="$1"; shift ;;
        *) shift ;;
      esac
    done
    if [ -f "$out" ]; then
      echo "Already exists, skipping: $out"
      return 0
    fi
    curl -L --create-dirs -o "$out" "$url"
  }
  export -f wget
fi

bash scripts/download_models.sh
cd ..

# SadTalker's source uses several numpy aliases (np.int, np.float,
# np.VisibleDeprecationWarning, etc.) that numpy removed entirely in 2.0,
# which we need for Python 3.11 wheel availability elsewhere in this stack.
# Patch them to their real equivalents.
python - <<'PYEOF'
import pathlib, re

root = pathlib.Path("SadTalker/src")
replacements = [
    (re.compile(r"\bnp\.VisibleDeprecationWarning\b"), "np.exceptions.VisibleDeprecationWarning"),
    (re.compile(r"\bnp\.float\b(?!\d)"), "float"),
    (re.compile(r"\bnp\.int\b(?!\d)"), "int"),
    (re.compile(r"\bnp\.bool\b(?!\d)"), "bool"),
    (re.compile(r"\bnp\.object\b"), "object"),
    (re.compile(r"\bnp\.str\b"), "str"),
    (re.compile(r"\bnp\.complex\b(?!\d)"), "complex"),
    (re.compile(r"\bnp\.long\b"), "int"),
    (re.compile(r"\bnp\.unicode\b"), "str"),
]

changed = 0
if root.exists():
    for f in root.rglob("*.py"):
        text = f.read_text(encoding="utf-8", errors="ignore")
        new_text = text
        for pat, repl in replacements:
            new_text = pat.sub(repl, new_text)
        if new_text != text:
            f.write_text(new_text, encoding="utf-8")
            changed += 1
if changed:
    print(f"Patched {changed} SadTalker source file(s) for numpy 2.0 compatibility.")
PYEOF

# src/face3d/util/preprocess.py's resize_n_crop_img/align_img compute a
# translation vector from np.linalg.lstsq() on a 2D column vector, so t[0]/
# t[1] come out as 1-element arrays rather than scalars. Numpy <1.25 let
# float() silently convert those with a deprecation warning; numpy 2.x makes
# it a hard TypeError ("only 0-dimensional arrays can be converted to Python
# scalars"). Patch the three spots that call float()/np.array() on them.
python - <<'PYEOF'
import pathlib

f = pathlib.Path("SadTalker/src/face3d/util/preprocess.py")
if f.exists():
    text = f.read_text(encoding="utf-8")
    replacements = [
        (
            "left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)",
            "left = (w/2 - target_size/2 + float(np.ravel((t[0] - w0/2)*s)[0])).astype(np.int32)",
        ),
        (
            "up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)",
            "up = (h/2 - target_size/2 + float(np.ravel((h0/2 - t[1])*s)[0])).astype(np.int32)",
        ),
        (
            "trans_params = np.array([w0, h0, s, t[0], t[1]])",
            "trans_params = np.array([w0, h0, s, float(np.ravel(t[0])[0]), float(np.ravel(t[1])[0])])",
        ),
    ]
    new_text = text
    for old, new in replacements:
        new_text = new_text.replace(old, new)
    if new_text != text:
        f.write_text(new_text, encoding="utf-8")
        print("Patched preprocess.py for numpy 2.0 scalar-conversion compatibility.")
PYEOF

# The float()-on-size-1-array issue just patched above shows up in one more
# spot in this same file: np.hsplit() on trans_params (a length-5 1D array)
# yields five shape-(1,) arrays, and the list comprehension below calls
# float() on each directly.
python - <<'PYEOF'
import pathlib

f = pathlib.Path("SadTalker/src/utils/preprocess.py")
if f.exists():
    text = f.read_text(encoding="utf-8")
    old = "trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)"
    new = "trans_params = np.array([float(np.ravel(item)[0]) for item in np.hsplit(trans_params, 5)]).astype(np.float32)"
    if old in text:
        f.write_text(text.replace(old, new), encoding="utf-8")
        print("Patched preprocess.py's hsplit conversion for numpy 2.0 compatibility.")
PYEOF

# basicsr/gfpgan (pulled in above) import
# `torchvision.transforms.functional_tensor`, which was removed in
# torchvision>=0.17 (bundled with the modern torch we need for the 5070 Ti).
# Patch the installed basicsr package to the current import path. Locate the
# file via find_spec rather than `import basicsr` — actually importing it
# fails with the exact ModuleNotFoundError we're trying to patch, since
# basicsr's own __init__.py chain hits the broken import before we get a
# chance to fix it.
python - <<'PYEOF'
import importlib.util, pathlib

spec = importlib.util.find_spec("basicsr")
if spec is not None and spec.submodule_search_locations:
    pkg_dir = pathlib.Path(list(spec.submodule_search_locations)[0])
    f = pkg_dir / "data" / "degradations.py"
    if f.exists():
        text = f.read_text()
        patched = text.replace(
            "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
            "from torchvision.transforms.functional import rgb_to_grayscale",
        )
        if patched != text:
            f.write_text(patched)
            print("Patched basicsr for torchvision>=0.17 compatibility.")
PYEOF

# 5. ffmpeg check
if ! command -v ffmpeg &> /dev/null; then
  echo ""
  echo "WARNING: ffmpeg not found on PATH."
  echo "  Linux:   sudo apt install ffmpeg"
  echo "  macOS:   brew install ffmpeg"
  echo "  Windows: https://ffmpeg.org/download.html"
fi

echo ""
echo "Setup complete."
if [ -f ".venv/bin/activate" ]; then
  echo "Next time, activate the environment with: source .venv/bin/activate"
else
  echo "Next time, activate the environment with: source .venv/Scripts/activate"
fi
