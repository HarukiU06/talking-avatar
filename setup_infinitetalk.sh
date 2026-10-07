#!/usr/bin/env bash
set -euo pipefail

echo "== InfiniteTalk setup (optional --engine infinitetalk) =="
echo "NOTE: this downloads a 14B-parameter base model plus adapters - budget ~35-40GB"
echo "of additional disk space on top of what setup.sh/setup_latentsync.sh/"
echo "setup_liveportrait.sh already use."
echo ""

# InfiniteTalk gets its own venv, separate from the other three. Same isolation
# reasoning as LatentSync/LivePortrait/Wav2Lip: it's only ever invoked as a
# subprocess (generate_infinitetalk.py), so its specific, heavyweight pins
# (xformers, flash_attn) can't conflict with anything else in this project.

if [ ! -d "InfiniteTalk" ]; then
  git clone https://github.com/MeiGen-AI/InfiniteTalk.git
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
"${PYTHON_CMD[@]}" -m venv .venv-infinitetalk

if [ ! -f ".venv-infinitetalk/Scripts/activate" ] && [ ! -f ".venv-infinitetalk/bin/activate" ]; then
  echo "venv creation failed."
  exit 1
fi

if [ -f ".venv-infinitetalk/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv-infinitetalk/bin/activate
else
  # shellcheck disable=SC1091
  source .venv-infinitetalk/Scripts/activate
fi
python -m pip install --upgrade pip

cd InfiniteTalk

# 2. PyTorch with CUDA. InfiniteTalk's own instructions pin torch==2.4.1+cu121,
# which predates Blackwell (RTX 50-series) support. Same fix as the other three
# setup scripts: strip the torch/torchvision pins and install a modern
# Blackwell-compatible build instead.
grep -vE '^(torch|torchvision|xformers)(==|>=|$)' requirements.txt > requirements.filtered.txt
python -m pip install "torch>=2.7.0" torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.filtered.txt

echo ""
echo "NOTE: pip may warn about torch version mismatches against InfiniteTalk's"
echo "original pins - expected, same reasoning as the other setup scripts"
echo "(Blackwell GPU support requires newer builds than originally pinned)."
echo ""

# 3. xformers - install separately since the repo's pin is tied to its old
# torch pin; let pip resolve a build matching the torch we just installed.
python -m pip install -U xformers

# 4. flash_attn - REQUIRED by InfiniteTalk, with no documented fallback and no
# documented Windows support. This is the step most likely to need manual
# troubleshooting on Windows: no prebuilt wheel may exist for this exact
# torch/CUDA/Python combination, and building from source needs the full CUDA
# Toolkit (not just the driver) plus MSVC Build Tools, and can take a long time.
# SKIP_FLASH_ATTN=1 (used by the Colab notebook) skips it: infinitetalk_run.py
# falls back to PyTorch's scaled_dot_product_attention when flash_attn is
# missing, and flash_attn has no build for Turing GPUs like the Colab T4.
FLASH_ATTN_OK=0
if [ "${SKIP_FLASH_ATTN:-0}" = "1" ]; then
  echo "SKIP_FLASH_ATTN=1: not installing flash_attn (SDPA attention fallback will be used)."
  FLASH_ATTN_OK=1
elif python -m pip install "flash_attn==2.7.4.post1"; then
  FLASH_ATTN_OK=1
elif [[ "${OSTYPE:-}" == msys* || "${OSTYPE:-}" == cygwin* ]]; then
  # flash-attn's source tree has very deeply nested paths (its
  # composable_kernel vendor tree especially); pip's default per-user TEMP
  # location on Windows is often already ~50-70 chars deep, and combined
  # with that nesting the full path can exceed Windows' 260-char MAX_PATH,
  # which surfaces as a plain "No such file or directory" error during
  # extraction - not a missing-compiler problem, so retrying with a short,
  # top-level TEMP dir is worth trying before assuming the worse case.
  echo ""
  echo "flash_attn failed to install normally - retrying with a shorter TEMP"
  echo "path in case this was a Windows MAX_PATH issue during extraction"
  echo "(a known problem with flash-attn's deeply-nested source tree)..."
  mkdir -p /c/pipbuild-flashattn
  if TMPDIR="/c/pipbuild-flashattn" TEMP="C:\\pipbuild-flashattn" TMP="C:\\pipbuild-flashattn" \
     python -m pip install --no-cache-dir "flash_attn==2.7.4.post1"; then
    FLASH_ATTN_OK=1
  fi
  rm -rf /c/pipbuild-flashattn
fi

if [ "$FLASH_ATTN_OK" -eq 0 ]; then
  echo ""
  echo "ERROR: flash_attn failed to install. InfiniteTalk requires it with no"
  echo "documented fallback attention path, so --engine infinitetalk will not"
  echo "work until this is resolved. Options:"
  echo "  1. Enable Windows long-path support (fixes MAX_PATH issues like the"
  echo "     one above, if that was the cause) - as Administrator in PowerShell:"
  echo "       New-ItemProperty -Path \"HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem\" -Name \"LongPathsEnabled\" -Value 1 -PropertyType DWORD -Force"
  echo "     then reboot and retry 'pip install flash_attn==2.7.4.post1'."
  echo "  2. Install the CUDA Toolkit (not just the GPU driver) matching the"
  echo "     torch build above (cu128), and MSVC Build Tools (same package as"
  echo "     setup_latentsync.sh's insightface step needs):"
  echo "       winget install Microsoft.VisualStudio.2022.BuildTools --override \"--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended\""
  echo "     then open a NEW terminal, re-activate this venv, and retry."
  echo "  3. Search for a prebuilt Windows wheel matching your exact torch/CUDA/"
  echo "     Python versions (community-built wheels circulate for this package"
  echo "     but availability varies - check what's current before relying on it)."
  echo "  4. Run this engine from WSL2 (Linux) instead, where flash_attn wheel"
  echo "     availability is much better - InfiniteTalk's own docs assume Linux."
  echo ""
  echo "Continuing setup so the checkpoint downloads below still happen; fix"
  echo "flash_attn and re-run 'pip install flash_attn==2.7.4.post1' afterward."
  echo ""
fi

# 5. Download checkpoints (~85GB total, scoped as below - the full
# MeiGen-AI/InfiniteTalk repo is ~110GB+ and includes multi-person,
# non-quantized, and ComfyUI-specific variants this project never uses;
# downloading it unscoped both wastes tens of GB and can exhaust disk space
# without ever fetching the two files actually needed - found by actually
# doing this once and having to clean up 90GB+ afterward. --include scopes
# the InfiniteTalk repo to just the single-person model and its FP8
# quantized variant (what render_infinitetalk() in make_avatar.py uses).
# A few Windows-specific gotchas here, found by actually running this:
#
# - Use `hf download` (not the older `huggingface-cli download`): current
#   huggingface_hub versions have dropped huggingface-cli entirely (it now
#   just prints a deprecation notice and does nothing) - same reasoning
#   already noted in setup_liveportrait.sh.
# - The `hf` console-script .exe launcher itself crashes silently on some
#   Windows setups (exits 1, zero output, even for `hf --version`) for
#   reasons unrelated to this project - invoking the same CLI through
#   `python -c "from huggingface_hub.cli.hf import app; app()"` instead
#   sidesteps that launcher entirely and is what this script actually uses.
# - The resumable-download cache is nested under --local-dir itself
#   (weights/<repo>/.cache/huggingface/download/<org>/<file>/<long
#   hash>.incomplete) regardless of HF_HOME - it is NOT redirected by
#   setting HF_HOME, that was tried and confirmed not to help. If this
#   project's own path is deeply nested (e.g. under a synced Desktop
#   folder), the combined path can exceed Windows' 260-char MAX_PATH and
#   fail with a plain "No such file or directory" on download - move the
#   whole project to a short path (e.g. C:\avatar\) if you hit this,
#   which is the only fix confirmed to work for this specific failure.
python -m pip install "huggingface_hub[cli]"
hf_download() {
  # $1=repo $2=local_dir; any further args are --include patterns
  repo="$1"; local_dir="$2"; shift 2
  include_args=()
  for pattern in "$@"; do
    include_args+=("$pattern")
  done
  python -c "
from huggingface_hub.cli.hf import app
import sys
sys.argv = ['hf', 'download', '$repo', '--local-dir', '$local_dir']
for p in [$(printf "'%s'," "${include_args[@]}")]:
    sys.argv += ['--include', p]
app()
"
}
if [ "${SKIP_DOWNLOADS:-0}" = "1" ]; then
  echo "SKIP_DOWNLOADS=1: leaving the checkpoint downloads to the caller."
  exit 0
fi
hf_download Wan-AI/Wan2.1-I2V-14B-480P weights/Wan2.1-I2V-14B-480P
hf_download TencentGameMate/chinese-wav2vec2-base weights/chinese-wav2vec2-base
# The quantized DiT needs the matching quantized T5 encoder alongside it
# (t5_fp8.safetensors + t5_map_fp8.json, ~6.7GB) - wan/modules/t5.py loads
# it from the same quant_models directory and fails without it. Upstream only
# publishes the fp8 T5, which is why make_avatar.py offers no int8 option.
hf_download MeiGen-AI/InfiniteTalk weights/InfiniteTalk "single/*" "quant_models/infinitetalk_single_fp8*" "quant_models/t5_fp8*" "quant_models/t5_map_fp8*"
# lightx2v step-distillation LoRA (~300MB) for --infinitetalk-accel lightx2v:
# 4 sampling steps at text CFG 1 instead of 40 at full CFG, the pairing
# InfiniteTalk's README recommends. infinitetalk_run.py applies it to the
# quantized model (upstream only applies LoRAs to the unquantized one).
hf_download Kijai/WanVideo_comfy weights/lora "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"

cd ..

echo ""
echo "InfiniteTalk setup complete."
if [ -f ".venv-infinitetalk/bin/activate" ]; then
  echo "Its venv: source .venv-infinitetalk/bin/activate"
else
  echo "Its venv: source .venv-infinitetalk/Scripts/activate"
fi
echo "If flash_attn failed above, resolve that before using --engine infinitetalk."
echo "make_avatar.py drives this checkout through infinitetalk_run.py, which"
echo "applies several runtime patches (Windows commit-limit loader, transformers 5,"
echo "RTX 50-series kernels, GPU memory cap) - nothing inside InfiniteTalk/ itself"
echo "is modified, so re-cloning it is always safe."
