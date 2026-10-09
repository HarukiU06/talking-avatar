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
import shutil
import subprocess
import sys
from pathlib import Path

import soundfile as sf

import make_avatar as ma

ROOT = Path(__file__).resolve().parent
STEPS = ("stage", "slice", "transcribe", "preprocess", "vectors", "train", "styles")
_JAPANESE = re.compile(r"[぀-ヿ一-鿿]")
MIN_MINUTES = 5.0


def run_sbv2(args: list) -> None:
    """Run a Style-Bert-VITS2 script through sbv2_run.py (its runtime fixes are explained there)."""
    cmd = [ma.get_venv_python(ma.SBV2_VENV), str(ROOT / "sbv2_run.py"), *args]
    print("Running Style-Bert-VITS2:", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    subprocess.run(cmd, cwd=ma.SBV2_DIR, check=True, env=env)


def model_name(voice: str, name: str = None) -> str:
    """--name, or the recording's file name. The app saves every avatar's voice
    as avatars/<avatar>/voice.<ext>, so that stem names nobody in particular."""
    if name is None:
        name = Path(voice).stem
        if name.lower() == "voice":
            raise SystemExit(f"{voice}: give this voice a name with --name (a file called 'voice' "
                             "could be anyone's, e.g. any avatar saved by the app).")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise SystemExit(f"--name {name!r}: use letters, digits, - and _ only (it becomes a folder name).")
    return name


def record_source(model_dir: Path, data_dir: Path, voice: str) -> None:
    """Remember which recording a model is of; refuse to mix two recordings under one name.

    Checked in both folders: the data folder holds the transcript, features
    and resume checkpoints, so reusing it for another recording would quietly
    train the old voice and label it with the new recording's hash.
    """
    digest = ma.file_sha256(voice)
    source = json.dumps({"sha256": digest, "voice": os.path.abspath(voice)}, indent=2)
    for folder in (model_dir, data_dir):
        source_path = folder / "source.json"
        if source_path.exists():
            if json.loads(source_path.read_text(encoding="utf-8")).get("sha256") != digest:
                raise SystemExit(f"This name was already used for a different recording ({folder}). "
                                 f"Pick another --name, or delete both {model_dir} and {data_dir} to "
                                 "start over.")
    for folder in (model_dir, data_dir):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "source.json").write_text(source, encoding="utf-8")


def finished(data_dir: Path, step: str) -> bool:
    return (data_dir / ".done" / step).exists()


def run_step(data_dir: Path, step: str, done, run) -> None:
    """Run a step unless it finished before; record it as finished only once its output is there.

    A marker, not the output's mere existence, decides: an interrupted run
    leaves partial output (transcribe.py appends to esd.list clip by clip), and
    resuming on that would quietly train on part of the recording.
    """
    if finished(data_dir, step):
        print(f"[{step}] already done")
        return
    print(f"[{step}] ...", flush=True)
    run()
    if not done():
        raise SystemExit(f"[{step}] ran but didn't produce its output; see the log above.")
    (data_dir / ".done").mkdir(parents=True, exist_ok=True)
    (data_dir / ".done" / step).write_text("", encoding="utf-8")


def check_epochs(data_dir: Path, epochs: int) -> None:
    """--epochs is fixed when the dataset is prepared; a different value later would never finish."""
    config = data_dir / "config.json"
    if config.exists():
        prepared = json.loads(config.read_text(encoding="utf-8")).get("train", {}).get("epochs")
        if prepared is not None and prepared != epochs:
            raise SystemExit(f"{data_dir} was prepared for {prepared} epochs. Run with --epochs {prepared}, "
                             "or delete that folder to prepare it again.")


def listed_clips(data_dir: Path) -> list:
    clips = []
    for name in ("train.list", "val.list"):
        listing = data_dir / name
        if listing.exists():
            clips += [Path(line.split("|", 1)[0])
                      for line in listing.read_text(encoding="utf-8").splitlines() if line.strip()]
    return clips


# preprocess_all.py exits 0 even when one of its steps fails (its style-vector
# step did, under torch 2.8), so each step is judged by what it leaves behind.
def preprocess_done(data_dir: Path) -> bool:
    clips = listed_clips(data_dir)
    return (data_dir / "config.json").exists() and bool(clips) and all(
        c.with_suffix(".bert.pt").exists() for c in clips)


def vectors_done(data_dir: Path) -> bool:
    clips = listed_clips(data_dir)
    return bool(clips) and all(Path(f"{c}.npy").exists() for c in clips)


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
    (data_dir / "esd.list").unlink(missing_ok=True)  # transcribe.py appends; drop a partial list
    # --use_hf_whisper: runs on the venv's torch 2.8. The default path needs
    # faster-whisper, which setup_sbv2.sh leaves out (see the comment there).
    # That path takes its model from --hf_repo_id (empty by default), not --model.
    run_sbv2(["transcribe.py", "--model_name", name, "--language", "ja",
              "--use_hf_whisper", "--hf_repo_id", "openai/whisper-large-v3", "--device", device])
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
    name = model_name(args.voice, args.name)

    model_dir = Path(ma.SBV2_MODELS) / name
    data_dir = Path(ma.SBV2_DATA) / name
    staged = data_dir / "input" / f"{name}.wav"
    record_source(model_dir, data_dir, args.voice)

    steps = {
        "stage": (lambda: staged.exists(),
                  lambda: stage(args.voice, staged, args.denoise)),
        "slice": (lambda: any((data_dir / "raw").glob("*.wav")),
                  lambda: (shutil.rmtree(data_dir / "raw", ignore_errors=True),
                           run_sbv2(["slice.py", "--model_name", name, "-i", str(staged.parent),
                                     "-m", "2", "-M", "12"]))),
        "transcribe": (lambda: (data_dir / "esd.list").exists(),
                       lambda: transcribe(name, data_dir, args.whisper_device)),
        "preprocess": (lambda: preprocess_done(data_dir),
                       lambda: run_sbv2(["preprocess_all.py", "-m", name, "--use_jp_extra",
                                         "-e", str(args.epochs), "-b", str(args.batch_size),
                                         "--yomi_error", "skip"])),
        # Also the last step of preprocess_all.py, but there it runs as its own
        # subprocess, out of sbv2_run.py's reach, and fails; run it here.
        "vectors": (lambda: vectors_done(data_dir),
                    lambda: run_sbv2(["style_gen.py", "--config", str(data_dir / "config.json")])),
        "train": (lambda: any(model_dir.glob(f"{name}_e{args.epochs}_s*.safetensors")),
                  lambda: run_sbv2(["train_ms_jp_extra.py", "-m", str(data_dir),
                                    "-c", str(data_dir / "config.json"),
                                    "--assets_root", str(Path(ma.SBV2_MODELS))])),
        "styles": (lambda: (model_dir / "styles.yaml").exists(),
                   lambda: run_sbv2([str(ROOT / "sbv2_styles.py"), "--data", str(data_dir),
                                     "--model", str(model_dir), "--clusters", str(args.clusters)])),
    }
    check_epochs(data_dir, args.epochs)
    for step in STEPS:
        done, run = steps[step]
        run_step(data_dir, step, done, run)
        if step == args.stop_after:
            print(f"Stopped after {step}. Rerun the same command to continue.")
            return
    print(f"\nTrained: {model_dir}")
    print(f"Listen to {model_dir / 'style_samples'} and name the styles in {model_dir / 'styles.yaml'}.")
    print(f"--lang ja now uses this voice whenever --voice or --vc-target is {args.voice} (or a copy). "
          "In the app: add it to the avatar as the voice sample or the voice-match recording, and "
          "set the voice model to Auto (a saved Chatterbox choice stays Chatterbox).")


if __name__ == "__main__":
    sys.exit(main())
