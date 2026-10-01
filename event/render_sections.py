#!/usr/bin/env python3
"""Render event sections with InfiniteTalk (lightx2v), resumably.

    .venv/Scripts/python.exe event/render_sections.py S6 S1

For each section: one output file (output/event/video/S<n>.mp4), skipped if
it already exists and was rendered from the current section WAV (a hash is
stored beside it; a stale video is set aside and the section re-rendered); a fixed seed (1000 + n unless overridden with
--seed S6=2006), recorded with the start time, duration and outcome in
render_log.jsonl. Each section runs in its own child process with its output
in S<n>.log, so a crash can't take the runner down. On an out-of-memory
failure the partial output is removed and the section retried once after a
pause; any other failure, or a second OOM, is recorded and the runner moves
on. Failures are listed at the end. A running ETA is printed, corrected by
the measured minutes per chunk as sections finish.

Inputs: the section WAVs from build_audio.py (output/event/audio/S<n>.wav)
and the approved framed render input (output/event/frame/render_input.png,
704x576 = InfiniteTalk's 480p bucket for this aspect, so it renders uncropped).
"""
import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "output" / "event" / "audio"
VIDEO = ROOT / "output" / "event" / "video"
PHOTO = ROOT / "output" / "event" / "frame" / "render_input.png"
LOG = VIDEO / "render_log.jsonl"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = ROOT / ".venv" / "bin" / "python"

SCENE_PROMPT = (
    "The person faces the camera as a poised event host: calm friendly expression, steady eye contact, "
    "small nods at phrase ends, natural blinking, a slight smile on jokes, mouth relaxed and closed in pauses. "
    "Static camera, chest-up, soft even lighting, no hand gestures."
)
STEPS = 4                 # lightx2v
LOAD_MIN = 1.5            # model load per section (measured on the earlier test)
DEFAULT_MIN_PER_CHUNK = 7.5
OOM_SIGNS = ("OutOfMemoryError", "CUDA out of memory", "out of memory", "3221225477", "0xC0000005")


def audio_hash(section: str) -> str:
    import hashlib

    return hashlib.md5((AUDIO / f"{section}.wav").read_bytes()).hexdigest()[:12]


def is_current(section: str) -> bool:
    """A finished section counts only if it was rendered from today's audio:
    its .mp4 exists and the hash stored next to it matches the section WAV."""
    out, stamp = VIDEO / f"{section}.mp4", VIDEO / f"{section}.audio_md5"
    if not (out.exists() and out.stat().st_size > 0):
        return False
    if stamp.exists() and stamp.read_text().strip() == audio_hash(section):
        return True
    stale = VIDEO / f"{section}.stale-{datetime.now():%m%d-%H%M%S}.mp4"
    out.replace(stale)
    print(f"[{section}] its audio changed since it was rendered - kept the old video as {stale.name}, re-rendering")
    return False


def chunks_for(seconds: float) -> int:
    frames = math.ceil(seconds * 25)
    return 1 + max(0, math.ceil((frames - 81) / 72))


def render_one(section: str, seed: int) -> None:
    """Child-process body: one section, straight through make_avatar."""
    sys.path.insert(0, str(ROOT))
    import make_avatar as ma

    out = VIDEO / f"{section}.partial.mp4"
    ma.render_infinitetalk(str(PHOTO), str(AUDIO / f"{section}.wav"), str(out), scene_prompt=SCENE_PROMPT,
                           size="480", sample_steps=STEPS, accel="lightx2v", seed=seed)
    out.replace(VIDEO / f"{section}.mp4")
    (VIDEO / f"{section}.audio_md5").write_text(audio_hash(section))


def log_event(**fields) -> None:
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(fields, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="+", help="e.g. S6 S1")
    ap.add_argument("--seed", action="append", default=[], help="override, e.g. --seed S6=2006 (a deliberate re-roll)")
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    seeds = {s: 1000 + int(s[1:]) for s in args.sections}
    for item in args.seed:
        name, value = item.split("=")
        seeds[name] = int(value)

    if args.one:
        render_one(args.sections[0], seeds[args.sections[0]])
        return 0

    VIDEO.mkdir(parents=True, exist_ok=True)
    for path in [PHOTO, *(AUDIO / f"{s}.wav" for s in args.sections)]:
        if not path.exists():
            print(f"missing input: {path}")
            return 2
    durations = {s: sf.info(str(AUDIO / f"{s}.wav")).duration for s in args.sections}
    chunks = {s: chunks_for(d) for s, d in durations.items()}
    min_per_chunk = DEFAULT_MIN_PER_CHUNK
    failures = []

    print(f"Rendering {' '.join(args.sections)} at 704x576 (InfiniteTalk 480p bucket), lightx2v {STEPS} steps")
    for i, section in enumerate(args.sections):
        out = VIDEO / f"{section}.mp4"
        if is_current(section):
            print(f"[{section}] already done, skipping")
            continue
        remaining = [s for s in args.sections[i:] if not (VIDEO / f"{s}.mp4").exists()]
        eta = sum(LOAD_MIN + chunks[s] * min_per_chunk for s in remaining)
        print(f"\n[{section}] {durations[section]:.1f}s audio, {chunks[section]} chunks, seed {seeds[section]} "
              f"- ETA for what's left: {eta:.0f} min (done around {datetime.fromtimestamp(time.time() + eta * 60):%H:%M})",
              flush=True)
        for attempt in (1, 2):
            partial = VIDEO / f"{section}.partial.mp4"
            partial.unlink(missing_ok=True)
            section_log = VIDEO / f"{section}.log"
            start = time.time()
            with section_log.open("w", encoding="utf-8", errors="replace") as fh:
                rc = subprocess.run([str(PY), str(Path(__file__).resolve()), "--one", section,
                                     "--seed", f"{section}={seeds[section]}"],
                                    cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                                    env={**__import__("os").environ, "PYTHONUNBUFFERED": "1",
                                         "PYTHONIOENCODING": "utf-8"}).returncode
            minutes = (time.time() - start) / 60
            text = section_log.read_text(encoding="utf-8", errors="replace")
            # 3221225477 / -1073741819 is Windows' access violation, how a commit-limit failure exits.
            oom = rc != 0 and (rc in (3221225477, -1073741819) or any(sign in text for sign in OOM_SIGNS))
            status = "ok" if rc == 0 and out.exists() else ("oom" if oom else f"failed (exit {rc})")
            log_event(section=section, seed=seeds[section], attempt=attempt, status=status,
                      started=datetime.fromtimestamp(start).isoformat(timespec="seconds"),
                      minutes=round(minutes, 1), audio_s=round(durations[section], 2), chunks=chunks[section],
                      steps=STEPS, size="704x576", audio_md5=audio_hash(section), scene_prompt=SCENE_PROMPT)
            print(f"[{section}] attempt {attempt}: {status} after {minutes:.1f} min (log: {section_log})", flush=True)
            if status == "ok":
                measured = max(minutes - LOAD_MIN, 0.5) / chunks[section]
                min_per_chunk = measured if min_per_chunk == DEFAULT_MIN_PER_CHUNK else (min_per_chunk + measured) / 2
                print(f"[{section}] {measured:.1f} min per chunk measured", flush=True)
                break
            partial.unlink(missing_ok=True)
            if status == "oom" and attempt == 1:
                print(f"[{section}] out of memory - cleaning up and retrying once in 30 s", flush=True)
                time.sleep(30)
                continue
            failures.append((section, status))
            break

    print("\n=== summary")
    for section in args.sections:
        out = VIDEO / f"{section}.mp4"
        print(f"  {section}: {'done' if out.exists() else 'NOT DONE'}  seed {seeds[section]}")
    if failures:
        print("Failures: " + ", ".join(f"{s} ({why})" for s, why in failures))
        print("Re-run the same command to retry only those sections.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
