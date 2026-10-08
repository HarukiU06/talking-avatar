"""What's installed on this computer, for the Setup tab and for refusing a job
up front instead of letting it fail minutes in.

Each check looks for the same files make_avatar.py needs before it starts that
engine, so "ready" here means the run won't stop at "couldn't find ...".
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import make_avatar as ma

ROOT = Path(ma.__file__).resolve().parent


@dataclass
class Part:
    key: str
    name: str
    purpose: str
    setup: str     # what to run to install it
    ready: bool


def _has_venv(venv: str) -> bool:
    return any((Path(venv) / exe).exists() for exe in ("Scripts/python.exe", "bin/python"))


def _exists(*parts: str) -> bool:
    return Path(*parts).exists()


def check_parts() -> dict:
    parts = [
        Part("chatterbox", "Chatterbox", "Speech in your voice. Always needed.", "./setup.sh",
             importlib.util.find_spec("chatterbox") is not None),
        Part("ffmpeg", "ffmpeg", "Reads and writes audio and video. Always needed.",
             "Install ffmpeg and put it on PATH (README section 1)",
             shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None),
        Part("latentsync", "LatentSync", "Lip sync for the Standard engine.", "./setup_latentsync.sh",
             _has_venv(ma.LATENTSYNC_VENV)
             and _exists(ma.LATENTSYNC_DIR, "scripts", "inference.py")
             and _exists(ma.LATENTSYNC_DIR, "checkpoints", "latentsync_unet.pt")),
        Part("liveportrait", "LivePortrait", "Head motion when the Standard engine starts from a photo.",
             "./setup_liveportrait.sh",
             _has_venv(ma.LIVEPORTRAIT_VENV) and _exists(ma.LIVEPORTRAIT_DIR, "inference.py")),
        Part("infinitetalk", "InfiniteTalk",
             "The Expressive engine. Needs about 70 GB of disk and 32 GB of RAM.", "./setup_infinitetalk.sh",
             _has_venv(ma.INFINITETALK_VENV)
             and _exists(ma.INFINITETALK_DIR, "generate_infinitetalk.py")
             and _exists(ma.INFINITETALK_DIR, "weights", "InfiniteTalk", "single", "infinitetalk.safetensors")),
        Part("sadtalker", "SadTalker", "The Lightweight engine.", "./setup.sh",
             _exists(ma.SADTALKER_DIR, "inference.py") and _exists(ma.SADTALKER_DIR, "checkpoints")),
        Part("seedvc", "Seed-VC", "The \"match my voice more closely\" option.", "./setup_seedvc.sh",
             _has_venv(ma.SEEDVC_VENV) and _exists(ma.SEEDVC_DIR, "inference.py")),
        Part("xtts", "XTTS-v2", "Alternative voice model (non-commercial licence).", "./setup_xtts.sh",
             _has_venv(ma.XTTS_VENV)),
        Part("wav2lip", "Wav2Lip", "Optional lip refinement for the Lightweight engine.", "./setup_wav2lip.sh",
             _has_venv(ma.WAV2LIP_VENV) and _exists(ma.WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth")),
    ]
    return {p.key: p for p in parts}


def lightx2v_ready() -> bool:
    return _exists(ma.INFINITETALK_DIR, "weights", "lora", ma.INFINITETALK_LIGHTX2V_LORA)


def gpu_summary() -> tuple:
    """(has an NVIDIA GPU, name and size, memory in use right now)."""
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            out = []
        if out:
            name, total, used = (field.strip() for field in out[0].split(","))
            try:
                return True, f"{name}, {float(total) / 1024:.0f} GB", f"{float(used) / 1024:.1f} GB in use"
            except ValueError:
                return True, name, ""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return False, "Apple Silicon: no CUDA, so rendering runs much slower", ""
    return False, "No NVIDIA GPU found: rendering will be very slow", ""


def torch_version() -> str:
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def free_disk_gb() -> float:
    return shutil.disk_usage(ROOT).free / 1024 ** 3
