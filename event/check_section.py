#!/usr/bin/env python3
"""Numbers and a contact sheet for a rendered section (Phase 2/3 checks).

    .venv/Scripts/python.exe event/check_section.py S6 [S1 ...] [--device cpu]

Per section:
- frames vs audio: frame count against audio length x 25 fps
- background drift: mean colour of two wall patches (top corners), first vs
  last frame and the worst frame, as CIE76 delta-E (about 2 is the smallest
  difference people notice side by side, above ~5 it is obvious)
- colour shift: the same for a patch on the cheek, tracked with landmarks
- lip sync: SyncNet (LatentSync's eval, in .venv-latentsync) confidence and
  audio-video offset in frames. The landmark/loudness correlation also
  reported is only a rough cross-check: in S6 its curve had no clear peak.
- tail: mouth opening and its spread in the section's trailing silence,
  against the speech frames
Writes output/event/check/<S>_sheet.jpg (one frame per second) and
<S>_metrics.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "output" / "event" / "video"
AUDIO = ROOT / "output" / "event" / "audio"
CHECK = ROOT / "output" / "event" / "check"
FPS = 25


def read_frames(path: Path):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    w, h = map(int, probe.stdout.strip().split(","))
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-255, ...x3) -> CIE Lab (D65)."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def delta_e(a, b) -> float:
    return float(np.linalg.norm(to_lab(np.asarray(a)) - to_lab(np.asarray(b))))


def syncnet(video: Path) -> dict:
    """LatentSync's SyncNet scorer (subprocess: it lives in .venv-latentsync)."""
    import re

    ls = ROOT / "LatentSync"
    py = ROOT / ".venv-latentsync" / "Scripts" / "python.exe"
    tmp = CHECK / "syncnet_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    out = subprocess.run([str(py), "-m", "eval.eval_sync_conf", "--video_path", str(video.resolve()),
                          "--temp_dir", str(tmp.resolve())], cwd=str(ls), capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    conf = re.search(r"SyncNet confidence: ([\d.-]+)", out)
    off = re.search(r"AV offset: (-?\d+)", out)
    return {"syncnet_confidence": float(conf[1]) if conf else None, "syncnet_offset_frames": int(off[1]) if off else None}


def check(section: str, device: str) -> dict:
    import face_alignment

    video, audio_path = VIDEO / f"{section}.mp4", AUDIO / f"{section}.wav"
    frames = read_frames(video)
    n, h, w, _ = frames.shape
    audio, sr = sf.read(audio_path, dtype="float32")
    expected = len(audio) / sr * FPS

    # wall patches: the two top corners, clear of the hair
    pw, ph = w // 9, h // 6
    wall = np.concatenate([frames[:, :ph, :pw].reshape(n, -1, 3), frames[:, :ph, -pw:].reshape(n, -1, 3)], 1).mean(1)
    bg_first_last = delta_e(wall[0], wall[-1])
    bg_worst = max(delta_e(wall[0], c) for c in wall)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device, flip_input=False)
    opening, cheek, missing = np.full(n, np.nan), np.full((n, 3), np.nan), 0
    for i, frame in enumerate(frames):
        found = fa.get_landmarks_from_image(frame)
        if not found:
            missing += 1
            continue
        lm = found[0]
        face_h = np.linalg.norm(lm[8] - lm[27])                       # chin to nose bridge
        opening[i] = np.mean([np.linalg.norm(lm[61 + k] - lm[67 - k]) for k in range(3)]) / face_h
        cx, cy = (lm[2] * 0.6 + lm[31] * 0.4).astype(int)           # left cheek, between jaw and nose
        r = max(4, int(face_h * 0.05))
        cheek[i] = frame[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r].reshape(-1, 3).mean(0)
    ok = ~np.isnan(opening)
    face_shift = delta_e(cheek[ok][0], cheek[ok][-1]) if ok.sum() > 1 else float("nan")

    # loudness envelope per video frame
    hop = sr // FPS
    env = np.array([np.sqrt(np.mean(audio[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
    env_db = 20 * np.log10(env)
    speaking = env_db > env_db.max() - 35
    best = (0, -1.0)
    for lag in range(-10, 11):
        a = opening[max(lag, 0):n + min(lag, 0)]
        b = env_db[max(-lag, 0):n + min(-lag, 0)]
        m = ~np.isnan(a)
        r = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else float("nan")
        if r > best[1]:
            best = (lag, r)

    tail_start = n
    while tail_start > 0 and not speaking[tail_start - 1]:
        tail_start -= 1
    tail = opening[tail_start:]
    speech_open = float(np.nanmedian(opening[speaking]))
    closed_ref = float(np.nanpercentile(opening[ok], 10))
    metrics = {
        "section": section, "frames": n, "expected_frames": round(expected, 1), "size": f"{w}x{h}",
        "background_dE_first_last": round(bg_first_last, 2), "background_dE_worst": round(bg_worst, 2),
        "face_colour_dE_first_last": round(face_shift, 2),
        "lipsync_corr": round(best[1], 3), "lipsync_lag_frames": best[0],
        "mouth_open_speech_median": round(speech_open, 3), "mouth_open_closed_ref": round(closed_ref, 3),
        "tail_s": round((n - tail_start) / FPS, 2),
        "tail_mouth_open_mean": round(float(np.nanmean(tail)), 3) if len(tail) else None,
        "tail_mouth_open_last": round(float(tail[-1]), 3) if len(tail) and not np.isnan(tail[-1]) else None,
        "frames_without_face": missing,
        **syncnet(video),
    }
    CHECK.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf",
                    f"fps=1,scale=352:-1,tile=5x{int(np.ceil(n / FPS / 5))}", "-frames:v", "1",
                    str(CHECK / f"{section}_sheet.jpg")], check=True)
    (CHECK / f"{section}_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="+")
    ap.add_argument("--device", default="cpu", help="cpu keeps the GPU free for a render in progress")
    a = ap.parse_args()
    for s in a.sections:
        m = check(s, a.device)
        print(json.dumps(m, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
