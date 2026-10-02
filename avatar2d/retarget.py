#!/usr/bin/env python3
"""Animate the cartoon avatar with the motion of the finished photoreal event video.

    .venv/Scripts/python.exe avatar2d/animate.py --rig output/avatar2d/rig_pixar --timeline-out output/avatar2d/timeline.npz
    .venv-liveportrait/Scripts/python.exe avatar2d/retarget.py [--seconds 15]

The sprite puppet (animate.py) can only cut between fixed faces and sway the whole
picture. This instead carries over continuous motion: InfiniteTalk already animated
the event host (output/event/final/event_clean.mp4) with natural head turns, nods,
blinks, lip sync and expression for the presentation scene. LivePortrait reads
that motion frame by frame and re-poses the cartoon face with it.

- Driving crop: fixed, from frame 0. LivePortrait's own driving crop follows the
  face every frame, which throws away head translation; with a fixed crop the
  head's movement in the frame is part of the motion.
- Relative motion, as LivePortrait's "expression-friendly" mode: the cartoon gets
  the driving face's change since frame 0 (rotation, expression, scale,
  translation), not its absolute pose, so her own proportions stay.
- Light temporal smoothing (Gaussian, more on pose than on expression) removes
  keypoint jitter without softening the lip sync.
- On top: the rig preset's eyebrow raise (the cartoon model draws angry-looking
  inner brows) and each line's expression from expressions.json (smile/brow
  offsets, crossfaded by the timeline animate.py exports).
- Streams both passes through ffmpeg; the 1080p driving video never sits in RAM.

Writes output/avatar2d/event_avatar_live.mp4 with output/event/audio/full.wav.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_rig  # noqa: E402  (LivePortrait loading + source preparation)

FPS = 25
SIGMA_POSE, SIGMA_EXP = 1.5, 0.7   # frames


def frames(path: str, w: int, h: int, limit: int):
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", path, "-frames:v", str(limit),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    p.wait()


def gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth along axis 0 (time), edges padded."""
    if sigma <= 0:
        return x
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    flat = pad.reshape(len(pad), -1)
    out = np.stack([np.convolve(flat[:, j], k, mode="valid") for j in range(flat.shape[1])], 1)
    return out.reshape(x.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "output" / "avatar2d" / "style" / "pixar_s0.70_2.png"))
    ap.add_argument("--driving", default=str(ROOT / "output" / "event" / "final" / "event_clean.mp4"))
    ap.add_argument("--audio", default=str(ROOT / "output" / "event" / "audio" / "full.wav"))
    ap.add_argument("--timeline", default=str(ROOT / "output" / "avatar2d" / "timeline.npz"))
    ap.add_argument("--preset", default="cartoon3d", choices=list(build_rig.PRESETS))
    ap.add_argument("--out", default=str(ROOT / "output" / "avatar2d" / "event_avatar_live.mp4"))
    ap.add_argument("--motion-scale", type=float, default=1.0, help="scale the transferred motion")
    ap.add_argument("--smile-scale", type=float, default=0.6,
                    help="how much of each line's preset smile to add (the driving face already smiles on jokes)")
    ap.add_argument("--seconds", type=float, help="render only the first N seconds (preview)")
    a = ap.parse_args()

    src_path, driving, audio, out_path = (str(Path(p).resolve()) for p in (a.src, a.driving, a.audio, a.out))
    tl = np.load(a.timeline)
    expr_w, expr_names = tl["expr_w"], [str(n) for n in tl["expr_names"]]
    presets = build_rig.PRESETS[a.preset]["expressions"]
    smile = a.smile_scale * expr_w @ np.array([presets[n].get("smile", 0) for n in expr_names], np.float32)
    brow = expr_w @ np.array([presets[n].get("eyebrow", 0) for n in expr_names], np.float32)

    W, H = 1920, 1080
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,nb_frames", "-of", "csv=p=0", driving],
                           capture_output=True, text=True, check=True).stdout.strip().split(",")
    W, H, total = int(probe[0]), int(probe[1]), int(probe[2])
    n = min(total, len(expr_w), int(round(a.seconds * FPS)) if a.seconds else total)

    img = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)
    rig = build_rig.Rigger(build_rig.load_pipeline(), img)   # chdirs into LivePortrait
    p, w, dev = rig.p, rig.w, rig.w.device
    from src.utils.camera import get_rotation_matrix
    from src.utils.crop import paste_back

    # Pass 1: driving motion, fixed crop from frame 0.
    M_o2c = None
    ang, exp, sc, tr = [], [], [], []
    for i, fr in enumerate(frames(driving, W, H, n)):
        if M_o2c is None:
            crop = p.cropper.crop_source_image(fr, p.cropper.crop_cfg)
            if crop is None:
                raise SystemExit("no face in the driving video's first frame")
            M_o2c = crop["M_o2c"][:2]
        c = cv2.resize(cv2.warpAffine(fr, M_o2c, (512, 512), flags=cv2.INTER_AREA), (256, 256),
                       interpolation=cv2.INTER_AREA)
        with torch.no_grad():
            info = w.get_kp_info(w.prepare_source(c))
        ang.append([float(info["pitch"]), float(info["yaw"]), float(info["roll"])])
        exp.append(info["exp"].cpu().numpy()[0])
        sc.append(float(info["scale"]))
        tr.append(info["t"].cpu().numpy()[0])
        if i % 500 == 0:
            print(f"motion {i}/{n}", flush=True)
    n = len(ang)
    ang, sc, tr = gauss(np.array(ang), SIGMA_POSE), gauss(np.array(sc), SIGMA_POSE), gauss(np.array(tr), SIGMA_POSE)
    exp = gauss(np.array(exp), SIGMA_EXP)

    # Pass 2: re-pose the cartoon.
    T = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=dev)
    info_s = rig.info
    R_s = rig.R_s.to(dev)
    kp_s, exp_s, scale_s, t_s = (info_s[k].to(dev) for k in ("kp", "exp", "scale", "t"))
    x_s, f_s = rig.x_s.to(dev), rig.f_s.to(dev)
    R_d0_T = get_rotation_matrix(T(ang[0, 0:1]), T(ang[0, 1:2]), T(ang[0, 2:3])).to(dev).permute(0, 2, 1)
    t_s0 = t_s.clone()
    t_s0[..., 2] = 0
    x_d0 = scale_s * (kp_s @ R_s + exp_s) + t_s0

    enc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{img.shape[1]}x{img.shape[0]}", "-r", str(FPS), "-i", "-",
         "-i", audio, "-t", f"{n / FPS:.3f}",
         "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", out_path],
        stdin=subprocess.PIPE)
    with torch.no_grad():
        for i in range(n):
            R_d = get_rotation_matrix(T(ang[i, 0:1]), T(ang[i, 1:2]), T(ang[i, 2:3])).to(dev)
            R_new = (R_d @ R_d0_T) @ R_s
            delta = exp_s + T(exp[i] - exp[0])[None]
            if smile[i]:
                delta = p.update_delta_new_smile(T(smile[i]), delta)
            if brow[i]:
                delta = p.update_delta_new_eyebrow(T(brow[i]), delta)
            scale_new = scale_s * float(sc[i] / sc[0])
            t_new = t_s + T(tr[i] - tr[0])[None]
            t_new[..., 2] = 0
            x_d = scale_new * (kp_s @ R_new + delta) + t_new
            x_d = x_s + (x_d - x_d0) * a.motion_scale
            x_d = w.stitching(x_s, x_d)
            face = w.parse_output(w.warp_decode(f_s, x_s, x_d)["out"])[0]
            enc.stdin.write(paste_back(face, rig.M_c2o, img, rig.mask).tobytes())
            if i % 250 == 0:
                print(f"frame {i}/{n}", flush=True)
    enc.stdin.close()
    if enc.wait():
        raise SystemExit("ffmpeg failed")
    print(f"wrote {out_path} ({n} frames, {n / FPS:.2f} s)")


if __name__ == "__main__":
    main()
