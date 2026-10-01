#!/usr/bin/env python3
"""Phase 1 of the event-host video: the audio.

Route (chosen by ear from voice_ab_kana.py): every Japanese line is spelled
out in hiragana (readings_ja.txt), spoken by XTTS-v2 for its intonation, then
passed through Seed-VC to move the timbre toward the reference voice. The
English line is XTTS English plus the same Seed-VC pass. Pauses are digital
silence, and the per-section WAVs and the full WAV share one gain
(-16 LUFS over the full track, with a soft limiter for the loudest peaks).

    .venv/Scripts/python.exe event/build_audio.py --voice voice_samples/newtest.wav

Every clip is cached in output/event/audio/raw/ under a name that includes
its seed, its text and the reference voice, so re-running (or rebuilding with
another line-18 take via --line18 SEED:ENDING) generates only what is
missing. XTTS is seeded per clip (seeds are in tts_log.json); Seed-VC has no
seed option, so a clip that is deleted and regenerated can differ slightly -
the cache is what keeps a take fixed.
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import make_avatar as ma  # noqa: E402

SCRIPT = ROOT / "event" / "script.txt"
READINGS = ROOT / "event" / "readings_ja.txt"
OUT = ROOT / "output" / "event" / "audio"
RAW = OUT / "raw"
STAGE = RAW / "xtts_stage"   # XTTS output before Seed-VC

SR = 24000              # everything is written at this rate
FADE_S = 0.010          # click protection on every clip edge
TARGET_LUFS = -16.0
PEAK_CEILING = 0.891    # -1 dBFS
KNEE = 0.5              # -6 dBFS: samples below this are untouched by the limiter
LINE18_ENDINGS = {"dash": "——", "bare": "", "comma": "、", "ellipsis": "…"}
LINE18_SEEDS = (181, 182, 183)
# ~2.5 GPU-minutes per second of audio, measured as ~7.5 min per streaming chunk
# (81 frames first, then 72 new frames each) plus ~1.5 min of model loading.
MIN_PER_CHUNK, LOAD_MIN, FPS = 7.5, 1.5, 25

VOICE_KEY = ""  # set in main(): the reference sample's name and size


def parse_script(path: Path):
    sections, current = {}, None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if re.fullmatch(r"S\d+", line):
            current = line
            sections[current] = []
            continue
        m = re.fullmatch(r"(\d+)\s+\[(ja|en)\]\s+(.*?)\s*\(([\d.]+)\)", line)
        if not m or current is None:
            raise ValueError(f"cannot parse script line: {raw!r}")
        sections[current].append({"n": int(m[1]), "lang": m[2], "text": m[3], "pause": float(m[4])})
    return sections


def parse_readings(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\d+)\s+(.+)", line)
        if m and not line.startswith("#"):
            out[int(m[1])] = m[2].strip()
    return out


def text_hash(text: str) -> str:
    """Part of every raw-clip name, so a changed reading or reference voice can
    never reuse a stale clip."""
    return hashlib.md5((VOICE_KEY + "|" + text).encode("utf-8")).hexdigest()[:6]


def soft_limit(x: np.ndarray, ceiling: float = PEAK_CEILING) -> np.ndarray:
    """Bend only the peaks above KNEE toward `ceiling` with a tanh curve."""
    over = np.abs(x) > KNEE
    y = x.copy()
    y[over] = np.sign(x[over]) * (KNEE + (ceiling - KNEE) * np.tanh((np.abs(x[over]) - KNEE) / (ceiling - KNEE)))
    return y


def make_job(label: str, lang: str, text: str, seed: int) -> dict:
    return {"tag": f"{label}_seed{seed}_{text_hash(text)}", "lang": lang, "text": text, "seed": seed}


def xtts_stage(jobs, ref: str, log: dict) -> None:
    """XTTS-v2: one subprocess per language, every missing clip in one batch."""
    STAGE.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (RAW / f"{j['tag']}.wav").exists() and not (STAGE / f"{j['tag']}.wav").exists()]
    for lang in sorted({j["lang"] for j in todo}):
        batch = [j for j in todo if j["lang"] == lang]
        work = Path(tempfile.mkdtemp(prefix="xtts_"))
        job_path = work / "job.json"
        job_path.write_text(json.dumps({
            "sentences": [j["text"] for j in batch], "seeds": [j["seed"] for j in batch],
            "lang": lang, "speaker_wav": os.path.abspath(ref), "out_dir": str(work)}), encoding="utf-8")
        t = time.time()
        subprocess.run([ma.get_venv_python(ma.XTTS_VENV), str(ROOT / "xtts_synth.py"), str(job_path)], check=True)
        result = json.loads(Path(str(job_path) + ".result").read_text(encoding="utf-8"))
        for j, f in zip(batch, result["files"]):
            shutil.move(f, STAGE / f"{j['tag']}.wav")
            log.setdefault(j["tag"], {}).update({
                "text_tts": j["text"], "lang": lang, "xtts_seed": j["seed"]})
        shutil.rmtree(work, ignore_errors=True)
        print(f"XTTS [{lang}]: {len(batch)} clips in {time.time() - t:.0f}s", flush=True)


def seedvc_stage(jobs, ref: str, log: dict) -> None:
    """Seed-VC: timbre conversion toward the reference, one launch per clip."""
    for i, j in enumerate(jobs, 1):
        final = RAW / f"{j['tag']}.wav"
        if final.exists():
            continue
        tmp = RAW / f"{j['tag']}.vc.wav"
        t = time.time()
        for attempt in (1, 2):
            try:
                ma.convert_voice(str(STAGE / f"{j['tag']}.wav"), ref, str(tmp))
                break
            except Exception as exc:
                print(f"Seed-VC failed on {j['tag']} (attempt {attempt}): {exc}", flush=True)
                if attempt == 2:
                    raise
        audio, rate = sf.read(tmp, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(1)
        if rate != SR:
            g = math.gcd(SR, rate)
            audio = resample_poly(audio, SR // g, rate // g).astype(np.float32)
        sf.write(final, audio, SR)
        tmp.unlink(missing_ok=True)
        log.setdefault(j["tag"], {})["seedvc_seconds"] = round(time.time() - t, 1)
        print(f"Seed-VC {i}/{len(jobs)} {j['tag']} {time.time() - t:.0f}s", flush=True)


def tidy(clip: np.ndarray, sr: int) -> np.ndarray:
    """Trim the model's own leading/trailing silence (pauses are ours to set),
    then 10 ms fades so a splice can never click."""
    import librosa

    _, (a, b) = librosa.effects.trim(clip, top_db=45)
    guard = int(0.02 * sr)
    clip = clip[max(a - guard, 0): min(b + guard, len(clip))].copy()
    n = int(FADE_S * sr)
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    clip[:n] *= ramp
    clip[-n:] *= ramp[::-1]
    return clip


def build_section(lines, clips, sr):
    parts = []
    for ln in lines:
        parts.append(clips[ln["n"]])
        parts.append(np.zeros(int(round(ln["pause"] * sr)), dtype=np.float32))
    return np.concatenate(parts)


def chunks_for(duration_s: float) -> int:
    frames = int(np.ceil(duration_s * FPS))
    return 1 + max(0, int(np.ceil((frames - 81) / 72)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=str(ROOT / "voice_samples" / "newtest.wav"),
                    help="reference sample for both languages")
    ap.add_argument("--line18", default="181:dash",
                    help="SEED:ENDING used in S6 (ending: dash/bare/comma/ellipsis)")
    args = ap.parse_args()

    global VOICE_KEY
    VOICE_KEY = f"{Path(args.voice).name}:{Path(args.voice).stat().st_size}|xtts+seedvc"
    seed18, ending18 = args.line18.split(":")
    seed18 = int(seed18)
    sections = parse_script(SCRIPT)
    kana = parse_readings(READINGS)
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(exist_ok=True)
    log_path = OUT / "tts_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}

    # Every clip that may be needed: the script lines, plus the line-18 candidates.
    main_jobs, take_jobs = {}, {}
    for lines in sections.values():
        for ln in lines:
            n = ln["n"]
            text = ln["text"] if ln["lang"] == "en" else kana[n]
            seed = 100 + n
            if n == 18:
                text, seed = kana[18] + LINE18_ENDINGS[ending18], seed18
            main_jobs[n] = make_job(f"line_{n:02d}", ln["lang"], text, seed)
    for seed in LINE18_SEEDS:
        for ending, suffix in LINE18_ENDINGS.items():
            take_jobs[(seed, ending)] = make_job("line_18", "ja", kana[18] + suffix, seed)
    jobs = list({j["tag"]: j for j in [*main_jobs.values(), *take_jobs.values()]}.values())

    ref = ma.get_prepared_voice(args.voice)
    xtts_stage(jobs, ref, log)
    seedvc_stage(jobs, ref, log)
    ma.cleanup_prepared_voices()
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")

    clips, tts_lines = {}, []
    for name, lines in sections.items():
        for ln in lines:
            j = main_jobs[ln["n"]]
            audio, _ = sf.read(RAW / f"{j['tag']}.wav", dtype="float32")
            clips[ln["n"]] = tidy(audio, SR)
            tts_lines.append(f"{ln['n']} [{ln['lang']}] {j['text']}")
            print(f"line {ln['n']:2d} [{ln['lang']}] seed {j['seed']} {len(clips[ln['n']]) / SR:5.2f}s  {j['text'][:40]}",
                  flush=True)

    # Line-18 alternatives, so the cut-off can be chosen by ear.
    takes_dir = OUT / "line18_takes"
    shutil.rmtree(takes_dir, ignore_errors=True)
    takes_dir.mkdir()
    pause18 = next(l for l in sections["S6"] if l["n"] == 18)["pause"]
    tail = np.zeros(int(round(pause18 * SR)), dtype=np.float32)
    for (seed, ending), j in take_jobs.items():
        audio, _ = sf.read(RAW / f"{j['tag']}.wav", dtype="float32")
        sf.write(takes_dir / f"take_seed{seed}_{ending}.wav", np.concatenate([tidy(audio, SR), tail]), SR)

    # Sections and full track, one common gain.
    built = {name: build_section(lines, clips, SR) for name, lines in sections.items()}
    full = np.concatenate(list(built.values()))
    import pyloudnorm as pyln

    # Speech like this has a high crest factor, so plain gain to -16 LUFS could push
    # its few loudest peaks far past full scale. Solve the gain with the soft limiter
    # in the loop, so the loudness is measured on what is actually written.
    meter = pyln.Meter(SR)
    measured = meter.integrated_loudness(full)
    gain = 10 ** ((TARGET_LUFS - measured) / 20)
    for _ in range(8):
        final_lufs = meter.integrated_loudness(soft_limit(full * gain))
        if abs(final_lufs - TARGET_LUFS) < 0.1:
            break
        gain *= 10 ** ((TARGET_LUFS - final_lufs) / 20)
    limited = soft_limit(full * gain)
    final_peak_db = 20 * np.log10(float(np.abs(limited).max()))
    touched = float((np.abs(full * gain) > KNEE).mean()) * 100
    for name, audio in built.items():
        sf.write(OUT / f"{name}.wav", soft_limit(audio * gain), SR, subtype="PCM_24")
    sf.write(OUT / "full.wav", limited, SR, subtype="PCM_24")
    (OUT / "script_tts.txt").write_text("\n".join(tts_lines) + "\n", encoding="utf-8")

    report = [f"voice: {args.voice}  route: XTTS-v2 (kana readings) -> Seed-VC (25 steps, cfg 0.7)",
              f"sample rate {SR} Hz, loudness {final_lufs:.1f} LUFS (target {TARGET_LUFS}), "
              f"peak {final_peak_db:.1f} dBFS, {touched:.2f}% of samples soft-limited",
              "", "section  duration  chunks  est. render"]
    total_min = 0.0
    for name, audio in built.items():
        d = len(audio) / SR
        c = chunks_for(d)
        est = LOAD_MIN + c * MIN_PER_CHUNK
        total_min += est
        report.append(f"{name:7s} {d:7.1f}s  {c:5d}  {est:5.0f} min")
    report.append(f"total   {len(full) / SR:7.1f}s  (line 18 = seed {seed18}, ending {ending18})  ~{total_min / 60:.1f} h")
    text = "\n".join(report)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
