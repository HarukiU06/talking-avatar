#!/usr/bin/env bash
set -euo pipefail

echo "== 2D puppet avatar setup =="

# No new venv: the stylizer and rig builder run in .venv-liveportrait (torch,
# cv2, LivePortrait), the emotion classifier and animator in .venv. This only
# pre-fetches the two models so the first real run doesn't stall on downloads.
# Both land in gitignored places: avatar2d/weights/ and the HF cache.

for venv in .venv .venv-liveportrait; do
  if [ ! -d "$venv" ]; then
    echo "$venv is missing - run setup.sh and setup_liveportrait.sh first."
    exit 1
  fi
done

py() {  # py <venv> <args...>
  local exe="$1/Scripts/python.exe"
  [ -x "$exe" ] || exe="$1/bin/python"
  shift
  "$exe" "$@"
}

mkdir -p avatar2d/weights
# AnimeGANv2 (illustration style) via torch.hub; the hub dir is kept inside the repo.
py .venv-liveportrait -c "
import torch
torch.hub.set_dir('avatar2d/weights/hub')
for w in ('face_paint_512_v2', 'paprika'):
    torch.hub.load('bryandlee/animegan2-pytorch:main', 'generator', pretrained=w, trust_repo=True)
print('AnimeGANv2 ok')
"

# Disney/Pixar-style 3D cartoon look (avatar2d/cartoonize.py, ~10 GB): an SDXL
# 3D-cartoon checkpoint + IP-Adapter plus-face. Runs in .venv-infinitetalk, which
# already has CUDA torch + diffusers; skipped if that venv isn't set up.
if [ -d .venv-infinitetalk ]; then
  py .venv-infinitetalk -c "
from huggingface_hub import snapshot_download as d
d('GHArt/Samaritan_3d_Cartoon_V4.0_xl_fp16')
d('h94/IP-Adapter', allow_patterns=['sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors', 'models/image_encoder/*'])
print('3D cartoon models ok')
"
else
  echo "(.venv-infinitetalk missing: skipping the 3D cartoon models; run setup_infinitetalk.sh to enable cartoonize.py)"
fi

# Multilingual emotion classifier (joy / anger / fear / sadness).
py .venv -c "
from transformers import pipeline
pipeline('text-classification', model='MilaNLProc/xlm-emo-t', top_k=None)('test')
print('xlm-emo-t ok')
"

echo "Done. See README section '2D puppet avatar' for the commands."
