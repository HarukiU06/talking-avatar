#!/usr/bin/env python3
"""Train a Style-Bert-VITS2 JP-Extra voice on one recording of you.

    python train_sbv2.py --voice voice_samples/me.wav [--name me] [--epochs 100]

Afterwards `--lang ja` uses it automatically (make_avatar.py --tts auto, the
default) whenever --voice is this same recording - matched by content, so the
app's saved copy of it counts too.

Resumable: rerun the same command after an interruption and it continues at the
first step whose output is missing; training itself resumes from its last
checkpoint. Whisper's mistakes become training labels, so
`--stop-after transcribe` lets you correct models/sbv2-data/<name>/esd.list
before training on it.

Runs in .venv; every Style-Bert-VITS2 step runs in .venv-sbv2 as a subprocess
(see CLAUDE.md's cross-venv rule).
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import soundfile as sf

import make_avatar as ma

ROOT = Path(__file__).resolve().parent
STEPS = ("stage", "slice", "transcribe", "preprocess", "train", "styles")
_JAPANESE = re.compile(r"[぀-ヿ一-鿿]")
MIN_MINUTES = 5.0


def run_sbv2(args: list) -> None:
    cmd = [ma.get_venv_python(ma.SBV2_VENV), *args]
    print("Running Style-Bert-VITS2:", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    subprocess.run(cmd, cwd=ma.SBV2_DIR, check=True, env=env)


def record_source(model_dir: Path, voice: str) -> None:
    """Remember which recording this model is of; refuse to mix two recordings under one name."""
    model_dir.mkdir(parents=True, exist_ok=True)
    source_path = model_dir / "source.json"
    digest = ma.file_sha256(voice)
    if source_path.exists():
        recorded = json.loads(source_path.read_text(encoding="utf-8")).get("sha256")
        if recorded != digest:
            raise SystemExit(f"{model_dir} was trained from a different recording. "
                             "Pick another --name (or delete that folder to start over).")
        return
    source_path.write_text(json.dumps({"sha256": digest, "voice": os.path.abspath(voice)}, indent=2),
                           encoding="utf-8")


def drop_non_japanese(esd_list: Path) -> list:
    """Remove transcript lines with no Japanese in them; JP-Extra trains on Japanese only."""
    kept, dropped = [], []
    for line in esd_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        text = line.split("|", 3)[-1]
        (kept if _JAPANESE.search(text) else dropped).append(line)
    esd_list.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return [line.split("|", 1)[0] for line in dropped]


def stage(voice: str, staged: Path, denoise: float) -> None:
    """Mono 44.1 kHz wav, the rate JP-Extra trains at; optionally denoised first."""
    import librosa

    staged.parent.mkdir(parents=True, exist_ok=True)
    audio, _ = librosa.load(voice, sr=44100, mono=True)
    if denoise > 0:
        # The model learns the room along with the voice, exactly as a TTS
        # reference does (see make_avatar.denoise).
        audio = ma.denoise(audio, 44100, strength=denoise)
    sf.write(staged, audio, 44100)


def transcribe(name: str, data_dir: Path, device: str) -> None:
    # --use_hf_whisper: runs on the venv's torch 2.8. The default path needs
    # faster-whisper, which setup_sbv2.sh leaves out (see the comment there).
    run_sbv2(["transcribe.py", "--model_name", name, "--language", "ja", "--model", "large-v3",
              "--use_hf_whisper", "--device", device])
    esd = data_dir / "esd.list"
    dropped = drop_non_japanese(esd)
    if dropped:
        print(f"Dropped {len(dropped)} clip(s) with no Japanese in the transcript: {', '.join(dropped)}")
    minutes = sum(sf.info(data_dir / "raw" / line.split("|", 1)[0]).duration
                  for line in esd.read_text(encoding="utf-8").splitlines() if line.strip()) / 60
    print(f"Transcript: {esd} ({minutes:.1f} min of speech). Check it for mistakes before training.")
    if minutes < MIN_MINUTES:
        print(f"WARNING: under {MIN_MINUTES:.0f} minutes of usable speech; the voice will be less stable.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--voice", required=True, help="A recording of you speaking Japanese (~10 min is plenty)")
    ap.add_argument("--name", help="Model name (default: the voice file's name)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2, help="Raise if you have VRAM to spare (12 GB: 2-4)")
    ap.add_argument("--clusters", type=int, default=4, help="How many emotion styles to find in the recording")
    ap.add_argument("--denoise", type=float, default=0.0, help="noisereduce strength on the recording (0 = off)")
    ap.add_argument("--whisper-device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--stop-after", choices=STEPS)
    args = ap.parse_args()

    if not ma.sbv2_installed():
        raise SystemExit("Style-Bert-VITS2 isn't set up. Run ./setup_sbv2.sh first.")
    if not os.path.isfile(args.voice):
        raise SystemExit(f"No such file: {args.voice}")
    name = args.name or Path(args.voice).stem
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise SystemExit(f"--name {name!r}: use letters, digits, - and _ only (it becomes a folder name).")

    model_dir = Path(ma.SBV2_MODELS) / name
    data_dir = Path(ma.SBV2_DATA) / name
    staged = data_dir / "input" / f"{name}.wav"
    record_source(model_dir, args.voice)

    steps = {
        "stage": (lambda: staged.exists(),
                  lambda: stage(args.voice, staged, args.denoise)),
        "slice": (lambda: any((data_dir / "raw").glob("*.wav")),
                  lambda: run_sbv2(["slice.py", "--model_name", name, "-i", str(staged.parent),
                                    "-m", "2", "-M", "12"])),
        "transcribe": (lambda: (data_dir / "esd.list").exists(),
                       lambda: transcribe(name, data_dir, args.whisper_device)),
        "preprocess": (lambda: (data_dir / "train.list").exists() and (data_dir / "config.json").exists(),
                       lambda: run_sbv2(["preprocess_all.py", "-m", name, "--use_jp_extra",
                                         "-e", str(args.epochs), "-b", str(args.batch_size),
                                         "--yomi_error", "skip"])),
        "train": (lambda: any(model_dir.glob(f"{name}_e{args.epochs}_s*.safetensors")),
                  lambda: run_sbv2(["train_ms_jp_extra.py", "-m", str(data_dir),
                                    "-c", str(data_dir / "config.json"),
                                    "--assets_root", str(Path(ma.SBV2_MODELS))])),
        "styles": (lambda: (model_dir / "styles.yaml").exists(),
                   lambda: run_sbv2([str(ROOT / "sbv2_styles.py"), "--data", str(data_dir),
                                     "--model", str(model_dir), "--clusters", str(args.clusters)])),
    }
    for step in STEPS:
        done, run = steps[step]
        if done():
            print(f"[{step}] already done")
        else:
            print(f"[{step}] ...", flush=True)
            run()
        if step == args.stop_after:
            print(f"Stopped after {step}. Rerun the same command to continue.")
            return
    print(f"\nTrained: {model_dir}")
    print(f"Listen to {model_dir / 'style_samples'} and name the styles in {model_dir / 'styles.yaml'}.")


if __name__ == "__main__":
    sys.exit(main())
