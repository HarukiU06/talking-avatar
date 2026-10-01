#!/usr/bin/env python3
"""Generate TTS takes for one script line, to replace it in the recorded audio.

Used for line 1, the English opening, whose recorded reading didn't sound
native. The voice is cloned from the reader's own recording, so the timbre
matches the recorded lines around it. Takes come from Chatterbox English at
cfg_weight 0 and 0.3 (lower = less accent carried over from the Japanese
reference) and from XTTS-v2 English, each with fixed seeds:

    .venv/Scripts/python.exe event/gen_line.py 1 --reference voice_samples/Finalaudio.m4a

Writes output/event/line_takes/line01/<engine>_seed<N>.wav (noise-cleaned
reference, trimmed, 24 kHz). Use one with
    build_audio.py --route recording ... --replace-line 1=<take>.wav
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import make_avatar as ma  # noqa: E402


def script_line(n: int):
    for raw in (ROOT / "event" / "script.txt").read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"(\d+)\s+\[(ja|en)\]\s+(.*?)\s*\(([\d.]+)\)", raw.strip())
        if m and int(m[1]) == n:
            return m[2], m[3]
    raise SystemExit(f"line {n} not in script.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("line", type=int)
    ap.add_argument("--reference", default=str(ROOT / "voice_samples" / "Finalaudio.m4a"))
    ap.add_argument("--seeds", default="1,2,3")
    a = ap.parse_args()

    lang, text = script_line(a.line)
    out = ROOT / "output" / "event" / "line_takes" / f"line{a.line:02d}"
    out.mkdir(parents=True, exist_ok=True)
    reference = Path(a.reference)
    if reference.suffix.lower() not in (".wav", ".flac"):  # soundfile can't read m4a/mp3
        wav = out.parent / f"{reference.stem}_reference.wav"
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(reference), "-ac", "1", str(wav)], check=True)
        reference = wav
    ref = ma.get_prepared_voice(str(reference))
    seeds = [int(s) for s in a.seeds.split(",")]
    log = {}
    print(f"line {a.line} [{lang}]: {text}\nreference: {a.reference}")

    import torch

    model = ma.get_tts_model()
    for cfg in (0.0, 0.3):
        for seed in seeds:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            audio = model.generate(text, language_id=lang, audio_prompt_path=ref,
                                   cfg_weight=cfg, exaggeration=0.5).squeeze().cpu().numpy()
            name = f"chatterbox_cfg{cfg}_seed{seed}.wav"
            sf.write(out / name, audio, model.sr)
            log[name] = {"engine": "chatterbox", "cfg_weight": cfg, "seed": seed, "text": text}
            print("wrote", name, flush=True)
    ma.release_tts_model()

    for seed in seeds:
        # XTTS has no seed flag in make_avatar's helper; run one job per seed through xtts_synth.
        import subprocess, tempfile, shutil, os
        work = Path(tempfile.mkdtemp(prefix="xtts_"))
        (work / "job.json").write_text(json.dumps({
            "sentences": [text], "seeds": [seed], "lang": lang,
            "speaker_wav": os.path.abspath(ref), "out_dir": str(work)}), encoding="utf-8")
        subprocess.run([ma.get_venv_python(ma.XTTS_VENV), str(ROOT / "xtts_synth.py"), str(work / "job.json")],
                       check=True)
        produced = json.loads((work / "job.json.result").read_text(encoding="utf-8"))["files"][0]
        name = f"xtts_seed{seed}.wav"
        shutil.move(produced, out / name)
        shutil.rmtree(work, ignore_errors=True)
        log[name] = {"engine": "xtts", "seed": seed, "text": text}
        print("wrote", name, flush=True)

    (out / "takes.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    ma.cleanup_prepared_voices()
    print(f"\n{len(log)} takes in {out}")


if __name__ == "__main__":
    main()
