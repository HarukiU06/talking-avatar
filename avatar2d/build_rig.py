#!/usr/bin/env python3
"""Build the 2D puppet rig: one sprite per (expression, mouth, eyes) state.

    .venv-liveportrait/Scripts/python.exe avatar2d/build_rig.py [--src output/avatar2d/style/cel.png]

Every sprite is a LivePortrait single-image retargeting of the same illustration
(the edits the LivePortrait app exposes as sliders: smile, eyebrow, pout, grin,
eye- and lip-open ratio). They all start from one set of source keypoints and are
pasted back through the same face mask, so outside that mask every sprite is
pixel-identical to the illustration. Only the face box is stored per sprite;
animate.py composites it onto base.png and can crossfade any two cleanly.

Writes output/avatar2d/rig/: base.png, sprites/<expr>_<mouth>_<eyes>.png (face
box only), rig.json, and sheet.jpg (expression x mouth, eyes open) + eyes.jpg.

Tune the look in PRESETS / EYES below and re-run; a full rig is
90 sprites, well under a minute on the GPU.
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")  # LivePortrait's rich logging vs cp932 consoles
ROOT = Path(__file__).resolve().parent.parent
LP_DIR = Path(os.environ.get("LIVEPORTRAIT_DIR", ROOT / "LivePortrait")).resolve()

# Slider values per state, in the LivePortrait app's units, one table per look
# (--preset). Expression and mouth values add up; lip_ratio is a target opening;
# eyes are a fraction of the source's own eye opening (None = unchanged).
# Keys: smile, eyebrow, grin (lip_variation_two), purse (lip_variation_one:
# + rounds the lips, - widens them), pout (lip_variation_zero; it shifts a lip
# point sideways, so it reads lopsided on a big cartoon mouth).
PRESETS = {
    # Cel-shaded drawing (style/cel.png): small features, needs strong values to read.
    "flat": {
        "expressions": {
            "neutral":   {},
            "smile":     {"smile": 0.9},
            "happy":     {"smile": 1.3, "eyebrow": 10},
            "surprised": {"eyebrow": 25, "eye_ratio_add": 0.12},
            "concerned": {"eyebrow": -22, "smile": -0.25},
        },
        "mouths": {
            "closed": {},
            "a": {"lip_ratio": 0.45},
            "i": {"grin": 10, "lip_ratio": 0.15},
            "u": {"pout": 0.09, "lip_ratio": 0.12},
            "e": {"grin": 7, "lip_ratio": 0.3},
            "o": {"pout": 0.05, "lip_ratio": 0.38},
        },
    },
    # Disney/Pixar-style 3D cartoon (cartoonize.py): big features move a lot, so
    # gentler values. The model draws inner brows tilted down (reads as angry); an
    # eyebrow raise of +20 everywhere relaxes them into a soft arch (+30 already
    # reads as surprised), and "concerned" only dips a little below that.
    "cartoon3d": {
        "expressions": {
            "neutral":   {"eyebrow": 20},
            "smile":     {"smile": 0.5, "eyebrow": 20},
            "happy":     {"smile": 0.9, "eyebrow": 22},
            "surprised": {"eyebrow": 35, "eye_ratio_add": 0.08},
            "concerned": {"eyebrow": 8, "smile": -0.15},
        },
        "mouths": {
            "closed": {},
            "a": {"lip_ratio": 0.32},
            "i": {"purse": -8, "lip_ratio": 0.12},
            "u": {"purse": 10, "lip_ratio": 0.12},
            "e": {"purse": -4, "lip_ratio": 0.22},
            "o": {"purse": 14, "lip_ratio": 0.24},
        },
    },
    # Flat chibi mascot (chibi.py): tiny features; the brows hide under the bangs,
    # so expressions are mostly the mouth, kept gentle.
    "chibi": {
        "expressions": {
            "neutral":   {},
            "smile":     {"smile": 0.4},
            "happy":     {"smile": 0.8},
            "surprised": {"eyebrow": 10, "eye_ratio_add": 0.05},
            "concerned": {"eyebrow": -8, "smile": -0.1},
        },
        "mouths": {
            "closed": {},
            "a": {"lip_ratio": 0.32},
            "i": {"purse": -8, "lip_ratio": 0.12},
            "u": {"purse": 10, "lip_ratio": 0.12},
            "e": {"purse": -4, "lip_ratio": 0.22},
            "o": {"purse": 14, "lip_ratio": 0.24},
        },
    },
}
EYES = {"open": None, "half": 0.6, "closed": 0.0}


def load_pipeline():
    sys.path.insert(0, str(LP_DIR))
    os.chdir(LP_DIR)  # insightface / landmark model paths are cwd-relative in places
    from src.config.argument_config import ArgumentConfig
    from src.config.crop_config import CropConfig
    from src.config.inference_config import InferenceConfig
    from src.gradio_pipeline import GradioPipeline

    args = ArgumentConfig()
    pick = lambda cls: cls(**{k: v for k, v in args.__dict__.items() if hasattr(cls, k)})
    return GradioPipeline(inference_cfg=pick(InferenceConfig), crop_cfg=pick(CropConfig), args=args)


class Rigger:
    """execute_image_retargeting split in two: the source is prepared once, then
    each state is just the keypoint edit + warp/decode. Also keeps the full-res
    image (the app path downsizes to 1280 px)."""

    def __init__(self, pipe, img_rgb):
        from src.utils.camera import get_rotation_matrix
        from src.utils.crop import prepare_paste_back
        from src.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio

        self.p, self.w, self.img = pipe, pipe.live_portrait_wrapper, img_rgb
        crop = pipe.cropper.crop_source_image(img_rgb, pipe.cropper.crop_cfg)
        if crop is None:
            raise SystemExit("LivePortrait found no face in the illustration. Re-run with --src on "
                             "a less stylized image (face_paint.png), or rig the photo and stylize after.")
        I_s = self.w.prepare_source(crop["img_crop_256x256"])
        self.info = self.w.get_kp_info(I_s)
        self.f_s = self.w.extract_feature_3d(I_s)
        self.x_s = self.w.transform_keypoint(self.info)
        self.R_s = get_rotation_matrix(self.info["pitch"], self.info["yaw"], self.info["roll"])
        self.lmk, self.M_c2o = crop["lmk_crop"], crop["M_c2o"]
        h, w = img_rgb.shape[:2]
        self.mask = prepare_paste_back(self.w.inference_cfg.mask_crop, self.M_c2o, dsize=(w, h))
        self.src_eye = float(calc_eye_close_ratio(self.lmk[None]).mean())
        self.src_lip = float(calc_lip_close_ratio(self.lmk[None])[0][0])

    @torch.no_grad()
    def render(self, s: dict) -> np.ndarray:
        from src.utils.crop import paste_back
        dev = self.w.device
        delta = self.info["exp"].clone().to(dev)  # the app edits this in place; don't
        if s.get("smile"):
            delta = self.p.update_delta_new_smile(torch.tensor(s["smile"], device=dev), delta)
        if s.get("eyebrow"):
            delta = self.p.update_delta_new_eyebrow(torch.tensor(s["eyebrow"], device=dev), delta)
        if s.get("pout"):
            delta = self.p.update_delta_new_lip_variation_zero(torch.tensor(s["pout"], device=dev), delta)
        if s.get("purse"):
            delta = self.p.update_delta_new_lip_variation_one(torch.tensor(s["purse"], device=dev), delta)
        if s.get("grin"):
            delta = self.p.update_delta_new_lip_variation_two(torch.tensor(s["grin"], device=dev), delta)
        R = self.R_s.to(dev)
        x_d = self.info["scale"].to(dev) * (self.info["kp"].to(dev) @ R + delta) + self.info["t"].to(dev)
        x_s = self.x_s.to(dev)
        eye = s["eye_frac"] * self.src_eye if s.get("eye_frac") is not None else None
        if eye is None and s.get("eye_ratio_add"):
            eye = self.src_eye + s["eye_ratio_add"]
        if eye is not None:
            x_d = x_d + self.w.retarget_eye(x_s, self.w.calc_combined_eye_ratio([[eye]], self.lmk))
        if s.get("lip_ratio") is not None:
            x_d = x_d + self.w.retarget_lip(x_s, self.w.calc_combined_lip_ratio([[s["lip_ratio"]]], self.lmk))
        x_d = self.w.stitching(x_s, x_d)
        out = self.w.parse_output(self.w.warp_decode(self.f_s.to(dev), x_s, x_d)["out"])[0]
        return paste_back(out, self.M_c2o, self.img, self.mask)


def combine(expr: dict, mouth: dict, eye_target) -> dict:
    s = dict(expr)
    for k, v in mouth.items():
        s[k] = s.get(k, 0) + v
    if eye_target is not None:  # a blink overrides the expression's eye opening
        s["eye_frac"] = eye_target
        s.pop("eye_ratio_add", None)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "output" / "avatar2d" / "style" / "cel.png"))
    ap.add_argument("--preset", default="flat", choices=list(PRESETS))
    ap.add_argument("--out", default=str(ROOT / "output" / "avatar2d" / "rig"))
    a = ap.parse_args()
    out_dir = Path(a.out).resolve()
    EXPRESSIONS, MOUTHS = PRESETS[a.preset]["expressions"], PRESETS[a.preset]["mouths"]
    src = Path(a.src).resolve()
    (out_dir / "sprites").mkdir(parents=True, exist_ok=True)
    img = cv2.cvtColor(cv2.imread(str(src)), cv2.COLOR_BGR2RGB)

    rig = Rigger(load_pipeline(), img)
    ys, xs = np.nonzero(rig.mask[..., 0] if rig.mask.ndim == 3 else rig.mask)
    pad = 4
    box = [int(xs.min()) - pad, int(ys.min()) - pad, int(xs.max()) + 1 + pad, int(ys.max()) + 1 + pad]
    box = [max(box[0], 0), max(box[1], 0), min(box[2], img.shape[1]), min(box[3], img.shape[0])]
    x0, y0, x1, y1 = box
    print(f"source eye ratio {rig.src_eye:.2f}, lip ratio {rig.src_lip:.2f}; face box {box}")

    cv2.imwrite(str(out_dir / "base.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    worst_outside = 0
    tiles = {}
    for (en, e), (mn, m), (yn, y) in itertools.product(EXPRESSIONS.items(), MOUTHS.items(), EYES.items()):
        full = rig.render(combine(e, m, y))
        outside = full.copy()
        outside[y0:y1, x0:x1] = img[y0:y1, x0:x1]
        worst_outside = max(worst_outside, int(np.abs(outside.astype(int) - img).max()))
        face = full[y0:y1, x0:x1]
        cv2.imwrite(str(out_dir / "sprites" / f"{en}_{mn}_{yn}.png"), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        tiles[en, mn, yn] = face
    print(f"{len(tiles)} sprites; max pixel change outside the face box: {worst_outside} (should be 0)")

    (out_dir / "rig.json").write_text(json.dumps({
        "source": str(src), "preset": a.preset, "size": [img.shape[1], img.shape[0]], "box": box,
        "expressions": EXPRESSIONS, "mouths": MOUTHS, "eyes": EYES,
        "source_eye_ratio": rig.src_eye, "source_lip_ratio": rig.src_lip,
    }, indent=1), encoding="utf-8")

    def tile(img_rgb, label):
        t = cv2.cvtColor(cv2.resize(img_rgb, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        cv2.putText(t, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return t
    rows = [np.hstack([tile(tiles[en, mn, "open"], f"{en}/{mn}") for mn in MOUTHS]) for en in EXPRESSIONS]
    cv2.imwrite(str(out_dir / "sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(str(out_dir / "eyes.jpg"), np.hstack([tile(tiles["neutral", "closed", yn], yn) for yn in EYES]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
