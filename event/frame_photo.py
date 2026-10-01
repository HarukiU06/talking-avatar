#!/usr/bin/env python3
"""Put a tight head-and-shoulders photo on a clean 16:9 canvas.

For a plain, light wall behind the person this needs no segmentation model:

1. Fit a smooth plane to the wall colour from wall-only pixels near the
   photo's border, and evaluate it over the whole canvas (plus fine grain).
2. Key the person out of the photo by colour distance to that wall model
   (soft alpha, so hair strands and the soft shadow by the neck survive).
3. Place the person centred, flush with the bottom edge. Anything that
   touches the photo's side edges (a shirt, say) is continued past them by
   shearing the edge column down and away, so shoulders round off and fall
   out of frame instead of ending in a vertical cut. With a portrait whose
   body is narrower than the photo at the bottom edge (use --crop-bottom),
   nothing touches the sides and only wall is extended.

    .venv/Scripts/python.exe event/frame_photo.py photos/TestImage.jpg out.png [--height 720]
"""
import argparse

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter(a.astype(np.float32), sigma=(sigma, sigma, 0) if a.ndim == 3 else sigma)


def frame(photo_path: str, out_path: str, height: int, canvas=(1920, 1080),
          slope: float = 0.12, curve: float = 0.006, crop_bottom: float = 1.0) -> None:
    cw, ch = canvas
    im = Image.open(photo_path).convert("RGB")
    if crop_bottom < 1.0:  # keep only the top part, e.g. to stop above where the arms reach the edges
        im = im.crop((0, 0, im.width, round(im.height * crop_bottom)))
        im.save(out_path.replace(".png", "_crop.png"))
    w = round(im.width * height / im.height)
    photo = np.asarray(im.resize((w, height), Image.LANCZOS), dtype=np.float32)
    x0, y0 = (cw - w) // 2, ch - height
    if y0 < 0 or x0 < 0:
        raise ValueError("photo is larger than the canvas; lower --height")

    # 1. wall model: plane fitted on light, low-saturation pixels in a border ring
    yy, xx = np.mgrid[0:height, 0:w].astype(np.float32)
    lum = photo.mean(-1)
    sat = photo.max(-1) - photo.min(-1)
    ring = (xx < 0.15 * w) | (xx > 0.85 * w) | (yy < 0.10 * height)
    wall = ring & (lum > 150) & (sat < 55)
    if wall.sum() < 500:
        raise RuntimeError("could not find enough plain wall around the subject to model it")
    A = np.stack([np.ones(wall.sum()), xx[wall] + x0, yy[wall] + y0], axis=1)
    coef = np.linalg.lstsq(A, photo[wall], rcond=None)[0]           # (3, 3)
    cy, cx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    model = np.stack([np.ones(cw * ch), cx.ravel(), cy.ravel()], axis=1) @ coef
    model = model.reshape(ch, cw, 3)
    grain = float(np.std((photo - blur(photo, 3))[wall]))
    rng = np.random.default_rng(0)
    backdrop = model + blur(rng.normal(0, grain, model.shape).astype(np.float32), 0.6) * 1.6

    # 2. soft key of the person against the wall model
    dist = np.sqrt(((photo - model[y0:, x0:x0 + w]) ** 2).sum(-1))
    alpha = blur(np.clip((dist - 22.0) / (60.0 - 22.0), 0, 1), 1.0)
    alpha[wall & (dist < 22.0)] = 0.0

    # 3. composite; shirt continued sideways with a falling, rounding shoulder line
    fg = np.zeros((ch, cw, 3), np.float32)
    a = np.zeros((ch, cw), np.float32)
    fg[y0:, x0:x0 + w] = photo
    a[y0:, x0:x0 + w] = alpha
    rows = np.arange(height, dtype=np.float32)
    for side in (0, 1):
        col_a = alpha[:, 0] if side == 0 else alpha[:, -1]
        col_c = photo[:, 0] if side == 0 else photo[:, -1]
        xs = np.arange(0, x0) if side == 0 else np.arange(x0 + w, cw)
        for x in xs:
            d = float((x0 - x) if side == 0 else (x - (x0 + w - 1)))
            src = np.clip(rows - (slope * d + curve * d * d), 0, height - 1)
            lo = np.floor(src).astype(int)
            hi = np.minimum(lo + 1, height - 1)
            t = (src - lo)
            a[y0:, x] = col_a[lo] * (1 - t) + col_a[hi] * t
            fg[y0:, x] = col_c[lo] * (1 - t)[:, None] + col_c[hi] * t[:, None]
    a = blur(a, 1.2)[..., None]
    out = backdrop * (1 - a) + fg * a
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(out_path)
    Image.fromarray(np.clip(backdrop, 0, 255).astype(np.uint8)).save(out_path.replace(".png", "_backdrop.png"))
    print(f"{out_path}: person {w}x{height} at ({x0},{y0}) on {cw}x{ch}; wall grain {grain:.2f}, "
          f"wall model at centre {model[ch // 2, cw // 2].round(0)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("out")
    ap.add_argument("--height", type=int, default=720, help="height the photo is scaled to on the 1080 canvas")
    ap.add_argument("--crop-bottom", type=float, default=1.0,
                    help="keep only this top fraction of the photo (0.68 stops above the arms)")
    ap.add_argument("--slope", type=float, default=0.12, help="shoulder line fall per pixel outward")
    ap.add_argument("--curve", type=float, default=0.006, help="extra fall that rounds off the shoulder tip")
    a = ap.parse_args()
    frame(a.photo, a.out, a.height, slope=a.slope, curve=a.curve, crop_bottom=a.crop_bottom)
