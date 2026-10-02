#!/usr/bin/env python3
"""Animate the flat chibi avatar (chibi.py) like a Live2D/VTuber model.

    .venv-liveportrait/Scripts/python.exe avatar2d/drive_signals.py        # once: motion curves
    .venv/Scripts/python.exe avatar2d/chibi_animate.py [--seconds 15]

LivePortrait warps pixels, and a flat chibi has nothing to warp: its mouth is
a single line. So the moving parts are redrawn every frame in the drawing's own
flat style, driven by the photoreal event performance (drive.npz: her real
mouth opening, blinks and head pose) and each line's expression
(timeline.npz from animate.py --timeline-out):

- Mouth: the drawn smile is painted out with skin colour, then a new mouth is
  drawn (4x supersampled): a smile arc when closed, an open shape with a dark
  interior and tongue as her real mouth opens; its curve follows the line's
  expression (neutral keeps the drawing's slight smile).
- Eyes: skin-coloured lids come down over the eye ovals as her real eyes
  close, with a lid line; "happy" lines add a squint.
- Head: a smooth deformation (like a Live2D deformer), not a cut: rotation from
  her roll, sideways shift from yaw, nod from pitch, fading to nothing down the
  neck so no seam can open. The body breathes and follows a little.
- Composited on the stage plate; soundtrack output/event/audio/full.wav.

Face features (eyes = two big dark ovals in the face, mouth = the dark curve
below them, skin colour) are found automatically; --debug-features writes an
overlay to check them.
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STYLE = ROOT / "output" / "avatar2d" / "style"
FPS = 25
SIZE = (1920, 1280)
SS = 4  # supersampling for drawn parts
# Mouth curve per expression: >0 smiles, <0 frowns (fraction of mouth width).
SMILE = {"neutral": 0.10, "smile": 0.16, "happy": 0.22, "surprised": 0.02, "concerned": -0.05}
SQUINT = {"happy": 0.15}   # more reads as sleepy/smug on these big eyes


def features(bgr, alpha):
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2]
    light = ((v > 200) & (alpha > 128)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(light)
    face_k = 1 + int(np.argmax(st[1:, 4]))
    face = (lab == face_k).astype(np.uint8)
    skin = np.median(bgr[face > 0], 0)
    fx, fy, fw, fh = st[face_k, :4]
    cnts, _ = cv2.findContours(face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.drawContours(np.zeros_like(face), [cv2.convexHull(max(cnts, key=cv2.contourArea))], -1, 1, -1)
    # Eyes: the two biggest compact (taller than wide) black blobs inside the face.
    dark = ((v < 70) & (hull > 0)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark)
    cand = [k for k in range(1, n) if st[k, 3] > st[k, 2] * 1.2 and st[k, 4] > 2000]
    eyes = sorted(sorted(cand, key=lambda k: -st[k, 4])[:2], key=lambda k: cen[k][0])
    if len(eyes) != 2:
        raise SystemExit("couldn't find two eyes on the chibi; check --debug-features")
    eye_boxes = [tuple(int(x) for x in st[k, :4]) for k in eyes]
    # Mouth: the biggest darkish stroke below the eyes and between their inner
    # edges (wider than that picks up hair strands at the cheeks).
    ix0 = eye_boxes[0][0] + eye_boxes[0][2]
    ix1 = eye_boxes[1][0]
    ey1 = max(b[1] + b[3] for b in eye_boxes)
    zone = np.zeros_like(face)
    zone[ey1 + 10:fy + fh, ix0:ix1] = 1
    mdark = ((v < 160) & (zone > 0) & (alpha > 128)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mdark)
    if n < 2:
        raise SystemExit("couldn't find the mouth on the chibi; check --debug-features")
    mk = 1 + int(np.argmax(st[1:, 4]))
    mouth = tuple(int(x) for x in st[mk, :4])
    line_col = np.median(bgr[lab == mk], 0)
    thick = max(3, int(round(st[mk, 4] / max(st[mk, 2], 1))))   # stroke area / length
    return dict(skin=skin, eyes=eye_boxes, mouth=mouth, line=line_col, thick=thick,
                chin=int(fy + fh), face=(int(fx), int(fy), int(fw), int(fh)))


def draw_mouth(img, f, open_, smile, interior, tongue):
    mx, my, mw, mh = f["mouth"]
    cx, cy = mx + mw / 2, my + mh / 2
    # Paint out the drawn mouth (its thin tips fade below the detection threshold,
    # so clear wider than the detected box).
    cv2.rectangle(img, (mx - 35, my - 10), (mx + mw + 35, my + mh + 10), f["skin"].tolist(), -1)
    w = mw * 1.15 * (1 - 0.15 * min(open_, 1))
    x0, y0 = int(cx - w / 2 - 20), int(cy - 20)
    cw, ch = int(w + 40), int(50 + mw * 1.1)
    canvas = np.zeros((ch * SS, cw * SS, 3), np.float32)
    mask = np.zeros((ch * SS, cw * SS), np.float32)
    t = np.linspace(-1, 1, 40)
    xs = (cx - x0 + t * w / 2) * SS
    curve = smile * w * (1 - t ** 2)                     # positive = corners up / centre down
    top = (cy - y0 + curve * (0.6 if open_ > 0.08 else 1.0)) * SS
    th = f["thick"] * SS
    if open_ <= 0.08:
        pts = np.stack([xs, top], 1).astype(np.int32)
        cv2.polylines(mask, [pts], False, 1.0, th, cv2.LINE_AA)
        canvas[:] = f["line"]
    else:
        depth = open_ * mw * 0.85
        bottom = top + (depth * (1 - t ** 2) ** 0.8) * SS
        poly = np.concatenate([np.stack([xs, top], 1), np.stack([xs[::-1], bottom[::-1]], 1)]).astype(np.int32)
        inner = np.zeros_like(mask)
        cv2.fillPoly(inner, [poly], 1.0, cv2.LINE_AA)
        canvas[:] = interior
        if open_ > 0.35:  # tongue: a lighter bump along the bottom
            tg = np.zeros_like(mask)
            cv2.ellipse(tg, (int(xs.mean()), int(bottom.max())), (int(w * 0.28 * SS), int(depth * 0.45 * SS)),
                        0, 180, 360, 1.0, -1, cv2.LINE_AA)
            tg *= inner
            canvas = canvas * (1 - tg[..., None]) + np.array(tongue, np.float32) * tg[..., None]
        ol = np.zeros_like(mask)
        cv2.polylines(ol, [poly], True, 1.0, th, cv2.LINE_AA)
        canvas = canvas * (1 - ol[..., None]) + np.array(f["line"], np.float32) * ol[..., None]
        mask = np.maximum(inner, ol)
    small = cv2.resize(canvas, (cw, ch), interpolation=cv2.INTER_AREA)
    a = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_AREA)[..., None]
    reg = img[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    img[y0:y0 + ch, x0:x0 + cw] = (a * small + (1 - a) * reg).round().clip(0, 255).astype(np.uint8)


def draw_lids(img, f, close):
    if close <= 0.02:
        return
    for (x, y, w, h) in f["eyes"]:
        pad = 6
        x0, y0, cw, ch = x - pad, y - pad, w + 2 * pad, h + 2 * pad
        lid_y = (pad + close * (h + pad)) * SS
        cover = np.zeros((ch * SS, cw * SS), np.float32)
        cv2.rectangle(cover, (0, 0), (cw * SS, int(lid_y)), 1.0, -1)
        oval = np.zeros_like(cover)
        cv2.ellipse(oval, (cw * SS // 2, ch * SS // 2), ((w // 2 + pad) * SS, (h // 2 + pad) * SS), 0, 0, 360, 1.0, -1)
        cover *= oval
        line = np.zeros_like(cover)
        half = (w // 2 + 2) * SS * np.sqrt(max(0.0, 1 - ((lid_y / SS - ch / 2) / (h / 2 + pad)) ** 2))
        tt = np.linspace(-1, 1, 24)
        sag = (3 + 6 * close) * SS * (1 - tt ** 2)       # lid line curves down a little
        pts = np.stack([cw * SS / 2 + tt * half, lid_y + sag], 1).astype(np.int32)
        cv2.polylines(line, [pts], False, 1.0, f["thick"] * SS, cv2.LINE_AA)
        cover_s = cv2.resize(cover, (cw, ch), interpolation=cv2.INTER_AREA)[..., None]
        line_s = cv2.resize(line, (cw, ch), interpolation=cv2.INTER_AREA)[..., None]
        reg = img[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
        reg = cover_s * f["skin"] + (1 - cover_s) * reg
        reg = line_s * np.array([20, 10, 5], np.float32) + (1 - line_s) * reg
        img[y0:y0 + ch, x0:x0 + cw] = reg.round().clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default=str(STYLE / "chibi_3_rgba.png"))
    ap.add_argument("--plate", default=str(STYLE / "plate.png"))
    ap.add_argument("--drive", default=str(ROOT / "output" / "avatar2d" / "drive.npz"))
    ap.add_argument("--timeline", default=str(ROOT / "output" / "avatar2d" / "timeline.npz"))
    ap.add_argument("--audio", default=str(ROOT / "output" / "event" / "audio" / "full.wav"))
    ap.add_argument("--out", default=str(ROOT / "output" / "avatar2d" / "event_avatar_chibi.mp4"))
    ap.add_argument("--head-motion", type=float, default=1.0, help="scale the head movement")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--debug-features", action="store_true")
    a = ap.parse_args()

    lay = cv2.resize(cv2.imread(a.layer, cv2.IMREAD_UNCHANGED), SIZE, interpolation=cv2.INTER_AREA)
    plate = cv2.resize(cv2.imread(a.plate), SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    char, alpha = lay[..., :3].copy(), lay[..., 3]
    f = features(char, alpha)
    print("features:", {k: v for k, v in f.items() if k in ("eyes", "mouth", "thick", "chin")})
    if a.debug_features:
        vis = char.copy()
        for (x, y, w, h) in f["eyes"] + [f["mouth"]]:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.line(vis, (0, f["chin"]), (SIZE[0], f["chin"]), (0, 255, 0), 2)
        cv2.imwrite(str(ROOT / "output" / "avatar2d" / "chibi_features.jpg"), vis)

    d = np.load(a.drive)
    tl = np.load(a.timeline)
    expr_w, names = tl["expr_w"], [str(x) for x in tl["expr_names"]]
    smile = expr_w @ np.array([SMILE[n] for n in names])
    squint = expr_w @ np.array([SQUINT.get(n, 0) for n in names])
    n = min(len(d["mouth"]), len(expr_w))
    if a.seconds:
        n = min(n, int(round(a.seconds * FPS)))
    interior = np.array([40, 20, 10], np.float32)               # dark navy, in the tint's family
    tongue = (0.5 * f["line"] + 0.5 * f["skin"]).astype(np.float32)

    # Head deformer weights: 1 above the chin, fading to 0 over the neck.
    H, W = SIZE[1], SIZE[0]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    fx, fy, fw, fh = f["face"]
    pivot = (fx + fw / 2, f["chin"] + 15)
    head_w = np.clip((pivot[1] + 25 - yy) / 50, 0, 1)
    ys_c = np.nonzero(alpha > 128)[0]
    feet = float(ys_c.max())

    enc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-i", a.audio, "-t", f"{n / FPS:.3f}",
         "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    for i in range(n):
        img = char.copy()
        mouth_open = float(np.clip((d["mouth"][i] - 0.12) / 0.88, 0, 1.1))
        draw_mouth(img, f, mouth_open, float(smile[i]), interior, tongue)
        close = float(np.clip((0.85 - d["eyes"][i]) / 0.6, 0, 1))
        close = max(close, float(squint[i]))
        draw_lids(img, f, close)

        # Head pose -> 2D: roll rotates, yaw shifts sideways, pitch nods (LivePortrait
        # pitch > 0 is chin up).
        k = a.head_motion
        ang = -float(d["roll"][i]) * 0.9 * k
        hdx = float(d["yaw"][i]) * 3.0 * k
        hdy = -float(d["pitch"][i]) * 2.0 * k
        Minv = cv2.getRotationMatrix2D(pivot, -ang, 1.0)   # inverse rotation for sampling
        sx = Minv[0, 0] * (xx - hdx) + Minv[0, 1] * (yy - hdy) + Minv[0, 2]
        sy = Minv[1, 0] * (xx - hdx) + Minv[1, 1] * (yy - hdy) + Minv[1, 2]
        breath = 1 + 0.006 * np.sin(2 * np.pi * i / FPS / 3.8)
        bx = 0.25 * hdx
        by_sx = xx - bx
        by_sy = feet - (feet - yy) / breath
        mapx = (head_w * sx + (1 - head_w) * by_sx).astype(np.float32)
        mapy = (head_w * sy + (1 - head_w) * by_sy).astype(np.float32)
        rgba = np.dstack([img, alpha])
        warped = cv2.remap(rgba, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        al = warped[..., 3:].astype(np.float32) / 255
        frame = al * warped[..., :3].astype(np.float32) + (1 - al) * plate
        enc.stdin.write(frame.round().clip(0, 255).astype(np.uint8).tobytes())
        if i % 250 == 0:
            print(f"frame {i}/{n}", flush=True)
    enc.stdin.close()
    if enc.wait():
        raise SystemExit("ffmpeg failed")
    print(f"wrote {a.out} ({n} frames, {n / FPS:.2f} s)")


if __name__ == "__main__":
    main()
