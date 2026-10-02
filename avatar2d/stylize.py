#!/usr/bin/env python3
"""Turn the framed photo into 2D illustration candidates for the puppet rig.

    .venv-liveportrait/Scripts/python.exe avatar2d/stylize.py [--src output/event/frame/final_c74.png]

Writes output/avatar2d/style/<name>.png (same size as the source) and sheet.jpg
(source + candidates side by side). Pick one and pass it to build_rig.py.

- face_paint: AnimeGANv2 face_paint_512_v2, a painterly illustration that keeps
  the likeness closest.
- paprika: AnimeGANv2 paprika, a stronger anime palette.
- cel: face_paint flattened into cel shading (edge-preserving smoothing, a
  small colour palette, dark ink lines), the most "puppet"-looking of the three.

AnimeGANv2 is fully convolutional, so it runs on the whole frame at once; it is
run at STYLE_HEIGHT because it was trained on ~512px faces and gets noisy at full
1080p detail, then scaled back up. The source is bilateral-smoothed first
(PRESMOOTH passes): straight from the photo, AnimeGAN turns fine skin texture
into dark painted wrinkle lines, which made every candidate look harsh and aged.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "avatar2d" / "style"
STYLE_HEIGHT = 768
PRESMOOTH = 5


def presmooth(rgb: np.ndarray, passes: int = PRESMOOTH) -> np.ndarray:
    for _ in range(passes):
        rgb = cv2.bilateralFilter(rgb, 15, 30, 15)
    return rgb


def animegan(rgb: np.ndarray, weights: str) -> np.ndarray:
    torch.hub.set_dir(str(ROOT / "avatar2d" / "weights" / "hub"))
    net = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator",
                         pretrained=weights, trust_repo=True).cuda().eval()
    h, w = rgb.shape[:2]
    sw = int(round(w * STYLE_HEIGHT / h / 8)) * 8
    small = cv2.resize(rgb, (sw, STYLE_HEIGHT), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(small).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
    with torch.no_grad():
        y = net(x)[0].clamp(-1, 1)
    out = ((y.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)


def cel(rgb: np.ndarray, colours: int = 14) -> np.ndarray:
    smooth = rgb
    for _ in range(3):
        smooth = cv2.bilateralFilter(smooth, 9, 40, 9)
    # Palette from k-means on a subsample; every pixel snaps to its nearest colour.
    px = smooth.reshape(-1, 3).astype(np.float32)
    sample = px[np.random.default_rng(0).choice(len(px), 60000, replace=False)]
    _, _, centres = cv2.kmeans(sample, colours, None,
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 4,
                               cv2.KMEANS_PP_CENTERS)
    d = ((px[:, None, :] - centres[None]) ** 2).sum(-1)
    flat = centres[d.argmin(1)].reshape(rgb.shape).astype(np.uint8)
    flat = cv2.medianBlur(flat, 5)
    # Ink only on strong contours (hairline, eyes, lips, jaw): Canny on the smoothed
    # image, so skin shading doesn't turn into lines.
    edges = cv2.Canny(cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY), 40, 110)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    ink = 1 - cv2.GaussianBlur(edges, (3, 3), 0).astype(np.float32)[..., None] / 255
    return (flat * (0.3 + 0.7 * ink)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "output" / "event" / "frame" / "final_c74.png"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(cv2.imread(a.src), cv2.COLOR_BGR2RGB)

    soft = presmooth(rgb)
    fp = animegan(soft, "face_paint_512_v2")
    styles = {"face_paint": fp, "paprika": animegan(soft, "paprika"), "cel": cel(fp)}
    for name, img in styles.items():
        cv2.imwrite(str(OUT / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print("wrote", OUT / f"{name}.png")

    # Contact sheet: the person region only (the frame is mostly wall), labelled.
    y0, y1, x0, x1 = 120, 1080, 460, 1460
    tiles = []
    for name, img in [("source", rgb)] + list(styles.items()):
        t = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2BGR).copy()
        cv2.putText(t, name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
        tiles.append(cv2.resize(t, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(OUT / "sheet.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", OUT / "sheet.jpg")


if __name__ == "__main__":
    main()
