#!/usr/bin/env python3
"""Animate the 2D puppet rig to the event reading.

    .venv/Scripts/python.exe avatar2d/animate.py [--seconds 15] [--out output/avatar2d/event_avatar2d.mp4]

Inputs: the rig from build_rig.py, output/event/audio/full.wav (soundtrack),
event/script.txt + event/readings_ja.txt (text), and expressions.json from
expressions.py. Each frame is a blend of rig sprites plus a 2D head sway:

- Lines: the script's pauses are written into full.wav as exact digital
  silence (event/build_audio.py build_section), so runs of zeros >= 0.3 s
  split it into one voiced span per line.
- Mouth: Japanese is mora-timed, so each line's kana reading becomes a vowel
  per mora (small ゃゅょ/ぁぃぅぇぉ recolour the previous mora, ー repeats it,
  っ/ん close the lips), spread evenly over the line's voiced frames; quiet
  frames close it. Mouth shapes snap rather than crossfade (blending two lip
  shapes double-exposes into a smear) and each holds >= MOUTH_HOLD frames.
  Line 1 (English) gets a rough vowel guess from its spelling.
- Expression: per line from expressions.json, held through the pause after
  it, crossfaded over EXPR_FADE frames that end as the line starts: a fade
  during speech blends two open mouths into a double exposure, and every
  script pause (>= 0.4 s) is longer than the fade.
- Eyes: seeded random blinks every 2.5-5 s, never during an expression fade.
- Motion: slow seeded sway (rotation, shift, breathing scale) pivoting at the
  bottom centre of the face box, plus a small nod after every line.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from expressions import script_lines  # noqa: E402

FPS = 25
EXPR_FADE = 9          # frames, odd
MOUTH_HOLD = 2         # frames
OPEN_AT = 0.35         # fraction of the line's loud level that opens the mouth
SILENCE_RUN_S = 0.3
SWAY_DEG, SWAY_PX, BREATH, NOD_PX, ZOOM = 1.2, 6.0, 0.004, 7.0, 1.04

ROWS = {"a": "あかさたなはまやらわがざだばぱ", "i": "いきしちにひみりぎじぢびぴ",
        "u": "うくすつぬふむゆるぐずづぶぷゔ", "e": "えけせてねへめれげぜでべぺ",
        "o": "おこそとのほもよろをごぞどぼぽ"}
VOWEL = {k: v for v, ks in ROWS.items() for k in ks}
SMALL = {"ゃ": "a", "ゅ": "u", "ょ": "o", "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o", "ゎ": "a"}


def kana_vowels(text: str) -> list:
    out = []
    for ch in text:
        if "ァ" <= ch <= "ヶ":  # katakana -> hiragana
            ch = chr(ord(ch) - 0x60)
        if ch in SMALL and out:
            out[-1] = SMALL[ch]
        elif ch == "ー" and out:
            out.append(out[-1])
        elif ch in "っん":
            out.append("closed")
        elif ch in VOWEL:
            out.append(VOWEL[ch])
    return out


def english_vowels(text: str) -> list:
    groups = re.findall(r"[aeiouy]+", text.lower())
    return [{"a": "a", "e": "e", "i": "i", "o": "o", "u": "u", "y": "i"}[g[0]] if g != "oo" else "u"
            for g in groups]


def readings(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\d+)\s+(.+)", line)
        if m and not line.startswith("#"):
            out[int(m[1])] = m[2]
    return out


def voiced_spans(audio: np.ndarray, sr: int, expect: int) -> list:
    z = (audio == 0).astype(np.int8)
    d = np.diff(np.r_[0, z, 0])
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    gaps = [(s, e) for s, e in zip(starts, ends) if (e - s) / sr >= SILENCE_RUN_S]
    spans, prev = [], 0
    for s, e in gaps:
        if s > prev:
            spans.append((prev / sr, s / sr))
        prev = e
    if prev < len(audio):
        spans.append((prev / sr, len(audio) / sr))
    if len(spans) != expect:
        raise SystemExit(f"found {len(spans)} voiced spans in the audio but the script has {expect} lines; "
                         "is this the full.wav that build_audio.py wrote for this script?")
    return spans


def smooth(x: np.ndarray, width: int) -> np.ndarray:
    """Box-smooth each column over time (x: frames x channels)."""
    k = np.ones(width) / width
    pad = np.pad(x, ((width // 2, width // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(x.shape[1])], 1)


def timeline(audio, sr, lines, kana, exprs, rig, n_frames, seed):
    mouths, expr_names = list(rig["mouths"]), list(rig["expressions"])
    spans = voiced_spans(audio, sr, len(lines))
    hop = sr // FPS
    rms = np.array([np.sqrt(np.mean(audio[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n_frames)])

    mouth = np.full(n_frames, mouths.index("closed"))
    expr_idx = np.zeros(n_frames, int)
    nods = []
    for k, (ln, (t0, t1)) in enumerate(zip(lines, spans)):
        f0, f1 = int(t0 * FPS), min(int(np.ceil(t1 * FPS)), n_frames)
        next_f = int(spans[k + 1][0] * FPS) if k + 1 < len(spans) else n_frames
        expr_idx[max(f0 - EXPR_FADE // 2 - 1, 0):] = expr_names.index(exprs[ln["n"]])  # later lines overwrite
        if k == 0:
            expr_idx[:f0] = expr_idx[f0]
        nods.append(f1)
        if f0 >= n_frames:
            continue
        seq = english_vowels(ln["text"]) if ln["lang"] == "en" else kana_vowels(kana.get(ln["n"], ""))
        if not seq:
            continue
        seg = rms[f0:f1]
        loud = np.percentile(seg, 90)
        voiced = np.nonzero(seg > 0.12 * loud)[0]
        for j, fi in enumerate(voiced):
            v = seq[min(int(j * len(seq) / len(voiced)), len(seq) - 1)]
            if seg[fi] >= OPEN_AT * loud:
                mouth[f0 + fi] = mouths.index(v)
    for i in range(1, n_frames):  # no shape shorter than MOUTH_HOLD frames
        if mouth[i] != mouth[i - 1] and (mouth[i:i + MOUTH_HOLD] != mouth[i]).any():
            mouth[i] = mouth[i - 1]
    mouth_w = np.eye(len(mouths))[mouth]
    expr_w = smooth(np.eye(len(expr_names))[expr_idx], EXPR_FADE)

    rng = np.random.default_rng(seed)
    eyes = np.array(["open"] * n_frames, dtype=object)
    changing = np.abs(np.diff(expr_idx, prepend=expr_idx[0])) > 0
    busy = np.convolve(changing, np.ones(EXPR_FADE + 4), mode="same") > 0
    f = int(rng.uniform(1.0, 3.0) * FPS)
    while f < n_frames - 4:
        if not busy[f:f + 4].any():
            eyes[f:f + 4] = ["half", "closed", "closed", "half"]
            f += int(rng.uniform(2.5, 5.0) * FPS)
        else:
            f += 3

    # Sway: sums of slow sines with seeded phases; nod: a smooth bump after each line.
    t = np.arange(n_frames) / FPS
    def wobble(freqs):
        ph = rng.uniform(0, 2 * np.pi, len(freqs))
        return sum(np.sin(2 * np.pi * fr * t + p) for fr, p in zip(freqs, ph)) / len(freqs)
    angle = SWAY_DEG * wobble([0.07, 0.13, 0.21])
    dx, dy = SWAY_PX * wobble([0.05, 0.11, 0.17]), 0.6 * SWAY_PX * wobble([0.06, 0.15])
    scale = 1 + BREATH * np.sin(2 * np.pi * t / 4.2)
    bump = np.sin(np.linspace(0, np.pi, 14)) ** 2
    for nf in nods:
        seg = slice(nf, min(nf + len(bump), n_frames))
        dy[seg] += NOD_PX * bump[:seg.stop - seg.start]
    return mouth_w, expr_w, eyes, angle, dx, dy, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", default=str(ROOT / "output" / "avatar2d" / "rig"))
    ap.add_argument("--audio", default=str(ROOT / "output" / "event" / "audio" / "full.wav"))
    ap.add_argument("--script", default=str(ROOT / "event" / "script.txt"))
    ap.add_argument("--readings", default=str(ROOT / "event" / "readings_ja.txt"))
    ap.add_argument("--expressions", default=str(ROOT / "output" / "avatar2d" / "expressions.json"))
    ap.add_argument("--out", default=str(ROOT / "output" / "avatar2d" / "event_avatar2d.mp4"))
    ap.add_argument("--seconds", type=float, help="render only the first N seconds (preview)")
    ap.add_argument("--seed", type=int, default=7, help="blink and sway randomness")
    a = ap.parse_args()

    rig_dir = Path(a.rig)
    rig = json.loads((rig_dir / "rig.json").read_text(encoding="utf-8"))
    base = cv2.imread(str(rig_dir / "base.png"))
    x0, y0, x1, y1 = rig["box"]
    H, W = base.shape[:2]
    sprites = {}
    for p in (rig_dir / "sprites").glob("*.png"):
        sprites[tuple(p.stem.split("_"))] = cv2.imread(str(p))

    audio, sr = sf.read(a.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    duration = len(audio) / sr
    n_frames = int(round(duration * FPS))
    if a.seconds:
        n_frames = min(n_frames, int(round(a.seconds * FPS)))
    lines = script_lines(Path(a.script))
    exprs = {e["n"]: e["expression"] for e in json.loads(Path(a.expressions).read_text(encoding="utf-8"))}
    mouth_w, expr_w, eyes, angle, dx, dy, scale = timeline(
        audio, sr, lines, readings(Path(a.readings)), exprs, rig, n_frames, a.seed)
    mouths, expr_names = list(rig["mouths"]), list(rig["expressions"])

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-i", a.audio, "-t", f"{n_frames / FPS:.3f}",
         "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE)
    pivot = ((x0 + x1) / 2, y1)
    frame = base.copy()
    for i in range(n_frames):
        face = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        for ei in np.nonzero(expr_w[i] > 1e-3)[0]:
            for mi in np.nonzero(mouth_w[i] > 1e-3)[0]:
                face += expr_w[i, ei] * mouth_w[i, mi] * sprites[expr_names[ei], mouths[mi], eyes[i]]
        frame[y0:y1, x0:x1] = np.clip(face, 0, 255).astype(np.uint8)
        M = cv2.getRotationMatrix2D(pivot, float(angle[i]), float(ZOOM * scale[i]))
        M[:, 2] += (dx[i], dy[i])
        enc.stdin.write(cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE).tobytes())
        if i % 250 == 0:
            print(f"frame {i}/{n_frames}", flush=True)
    enc.stdin.close()
    if enc.wait():
        raise SystemExit("ffmpeg failed")
    print(f"wrote {out} ({n_frames} frames, {n_frames / FPS:.2f} s)")


if __name__ == "__main__":
    main()
