#!/usr/bin/env python3
"""Read the photoreal event performance as a few smooth animation curves.

    .venv-liveportrait/Scripts/python.exe avatar2d/drive_signals.py

For characters too flat for LivePortrait to warp (a chibi's mouth is a single
line: there is nothing to open), the motion is redrawn instead of warped, and
this supplies it. Per frame of output/event/final/event_clean.mp4, with a fixed
crop from frame 0 (so head translation counts) and LivePortrait's own models:

- pitch / yaw / roll (degrees, relative to the clip's median pose)
- mouth: lip-open ratio, 0 = closed, ~1 = wide (scaled by its 98th percentile)
- eyes:  eye-open ratio, 1 = normally open, 0 = shut (relative to its median)

Writes output/avatar2d/drive.npz, lightly smoothed.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_rig  # noqa: E402
from retarget import frames, gauss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driving", default=str(ROOT / "output" / "event" / "final" / "event_clean.mp4"))
    ap.add_argument("--out", default=str(ROOT / "output" / "avatar2d" / "drive.npz"))
    a = ap.parse_args()
    driving, out = str(Path(a.driving).resolve()), str(Path(a.out).resolve())
    import subprocess
    W, H, n = (int(v) for v in subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames",
         "-of", "csv=p=0", driving], capture_output=True, text=True, check=True).stdout.strip().split(","))

    pipe = build_rig.load_pipeline()   # chdirs into LivePortrait
    w = pipe.live_portrait_wrapper
    from src.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio

    M = None
    pose, lip, eye = [], [], []
    for i, fr in enumerate(frames(driving, W, H, n)):
        if M is None:
            M = pipe.cropper.crop_source_image(fr, pipe.cropper.crop_cfg)["M_o2c"][:2]
        c512 = cv2.warpAffine(fr, M, (512, 512), flags=cv2.INTER_AREA)
        with torch.no_grad():
            info = w.get_kp_info(w.prepare_source(cv2.resize(c512, (256, 256), interpolation=cv2.INTER_AREA)))
        pose.append([float(info["pitch"]), float(info["yaw"]), float(info["roll"])])
        lmk = pipe.cropper.calc_lmk_from_cropped_image(c512)
        if lmk is None:
            lip.append(np.nan)
            eye.append(np.nan)
        else:
            lip.append(float(calc_lip_close_ratio(lmk[None])[0][0]))
            eye.append(float(calc_eye_close_ratio(lmk[None]).mean()))
        if i % 500 == 0:
            print(f"frame {i}/{n}", flush=True)

    fill = lambda x: np.where(np.isnan(x), np.nanmedian(x), x)
    pose = np.array(pose)
    pose = gauss(pose - np.median(pose, 0), 1.5)
    lip = fill(np.array(lip))
    lip = gauss(np.clip((lip - np.percentile(lip, 5)) / (np.percentile(lip, 98) - np.percentile(lip, 5)), 0, 1.2), 0.7)
    eye = fill(np.array(eye))
    eye = gauss(np.clip(eye / np.median(eye), 0, 1.3), 0.7)
    np.savez(out, pitch=pose[:, 0], yaw=pose[:, 1], roll=pose[:, 2], mouth=lip, eyes=eye)
    print(f"wrote {out} ({len(lip)} frames); pose range pitch {np.ptp(pose[:, 0]):.1f} yaw {np.ptp(pose[:, 1]):.1f} "
          f"roll {np.ptp(pose[:, 2]):.1f} deg; mouth open >0.3 on {np.mean(lip > 0.3):.0%} of frames; "
          f"eyes <0.4 (blink) on {np.mean(eye < 0.4):.1%}")


if __name__ == "__main__":
    main()
