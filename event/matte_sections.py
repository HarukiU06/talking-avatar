#!/usr/bin/env python3
"""Cut the person out of each section render, for assemble.py --background matrix.

    .venv-latentsync/Scripts/python.exe event/matte_sections.py [S1 S2 ...]

Runs in .venv-latentsync for its bundled mediapipe selfie-segmentation model
(no download). Writes output/event/matte/S<n>.mkv: FFV1 BGRA, same frame count
as output/event/video/S<n>.mp4, alpha in A and the colour un-mixed from the wall.

Per frame:
- mediapipe gives a coarse person mask (smoothed over time).
- The wall is a smooth light-grey gradient, so it is fitted with a low-order
  polynomial over the pixels well outside that mask: a per-pixel clean plate.
- Alpha is a colour key against that plate, limited to near the mediapipe mask.
  Lightness differences count 0.3x, so the wall darkened by the person's shadow
  (between hair and shoulders) stays background; dark hair and the black top
  are far enough off in lightness anyway, skin in chroma. The deep interior of
  the mediapipe mask is forced opaque (teeth, eye whites, neckline are close to
  the wall colour).
- Edge colour is un-mixed from the plate, (I - (1-a)·plate) / a, so hair edges
  carry no light-grey halo onto a dark background. Where a ~ 0 the original
  pixel is kept, so assemble.py can still read the wall colour in the corners.
"""
import subprocess
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "output" / "event"
VIDEO, MATTE = EV / "video", EV / "matte"
W, H = 704, 576

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
xn, yn = xx / W - 0.5, yy / H - 0.5
BASIS = np.stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn, yn ** 3, xn * xn * yn], -1).reshape(-1, 8)
CORE_K = np.ones((61, 61), np.uint8)
REACH_K = np.ones((31, 31), np.uint8)
BG_K = np.ones((41, 41), np.uint8)


def wall_plate(f: np.ndarray, bgmask: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(bgmask.ravel())[::7]
    coef, *_ = np.linalg.lstsq(BASIS[idx], f.reshape(-1, 3)[idx], rcond=None)
    return np.clip((BASIS @ coef).reshape(H, W, 3), 0, 1).astype(np.float32)


def matte(bgr: np.ndarray, m: np.ndarray) -> np.ndarray:
    f = bgr.astype(np.float32) / 255
    plate = wall_plate(f, cv2.dilate((m > 0.1).astype(np.uint8), BG_K) == 0)
    dl = cv2.cvtColor(f, cv2.COLOR_BGR2LAB) - cv2.cvtColor(plate, cv2.COLOR_BGR2LAB)
    dl[..., 0] *= 0.3
    key = np.clip((np.linalg.norm(dl, axis=2) - 6) / (16 - 6), 0, 1)
    core = cv2.erode((m > 0.5).astype(np.uint8), CORE_K).astype(np.float32)
    reach = cv2.GaussianBlur(cv2.dilate((m > 0.2).astype(np.uint8), REACH_K).astype(np.float32), (21, 21), 0)
    a = cv2.GaussianBlur(np.maximum(key * reach, core), (3, 3), 0)[..., None]
    fg = np.clip((f - (1 - a) * plate) / np.maximum(a, 0.05), 0, 1)
    fg = np.where(a < 0.02, f, fg)
    return np.concatenate([fg, a], 2).__mul__(255).round().astype(np.uint8)


def run(section: str) -> None:
    src = VIDEO / f"{section}.mp4"
    dst = MATTE / f"{section}.mkv"
    dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(src), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}",
                            "-r", "25", "-i", "-", "-c:v", "ffv1", "-pix_fmt", "bgra", str(dst)], stdin=subprocess.PIPE)
    seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
    smooth, n = None, 0
    while True:
        buf = dec.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        bgr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        m = seg.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).segmentation_mask.astype(np.float32)
        smooth = m if smooth is None else 0.6 * m + 0.4 * smooth
        enc.stdin.write(matte(bgr, smooth).tobytes())
        n += 1
    dec.wait()
    enc.stdin.close()
    enc.wait()
    print(f"{section}: {n} frames -> {dst.relative_to(ROOT)}", flush=True)


def main():
    MATTE.mkdir(parents=True, exist_ok=True)
    for s in sys.argv[1:] or ["S1", "S2", "S3", "S4", "S5", "S6"]:
        run(s)


if __name__ == "__main__":
    main()
