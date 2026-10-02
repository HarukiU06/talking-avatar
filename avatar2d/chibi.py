#!/usr/bin/env python3
"""Put her, as a chibi mascot, into the stage visual of photos/image.jpg.

    .venv-infinitetalk/Scripts/python.exe avatar2d/chibi.py [--seeds 1,2,3,4,5,6]

The reference is a flat chibi mascot (big head, thick black outlines, flat
colour) tinted blue, standing on a dark, sparkly projected background. Built
in layers rather than by repainting the character in place (inpainting the
character directly painted white backdrop blobs into the sparkles and drew slim
anime girls, not the chunky mascot shape):

1. plate.png: the reference with the original character removed: its area is
   filled with real background copied sideways along the horizontal stage lines.
2. chibi_<seed>_natural.png: her as a chibi mascot on plain white, from Animagine
   XL 4.0 (SDXL anime/illustration model) with danbooru-style tags; IP-Adapter
   plus-face, fed her photo, nudges the likeness (short black bob etc.).
3. Cut out from the white (flood fill from the border, so whites inside her,
   like eyes, stay), tinted to the reference's monochrome blue by mapping each
   pixel's lightness to the original character's mean colour at that
   lightness (black outlines stay black), scaled to the original character's
   height and placed on its feet.

Writes output/avatar2d/style/chibi_<seed>.png (composite), chibi_<seed>_rgba.png
(her layer, transparent), plate.png, and chibi_sheet.jpg.
"""
import argparse
import gc
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "avatar2d" / "style"
CHECKPOINT = "cagliostrolab/animagine-xl-4.0"
# Animagine is trained on danbooru tags; quality tags last, as its model card recommends.
PROMPT = ("1girl, solo, chibi, chibi, super deformed, mascot, big head, full body, standing, front view, facing viewer, "
          "looking at viewer, own hands together, mature female, adult woman, short hair, black hair, bob cut, light smile, "
          "closed mouth, black sleeveless shirt, black skirt, thick outlines, flat color, simple coloring, "
          "white background, simple background, masterpiece, high score, great score, absurdres")
NEGATIVE = ("1boy, male, multiple views, open mouth, teeth, closed eyes, lowres, bad anatomy, bad hands, "
            "extra digits, cropped, worst quality, low quality, low score, bad score, average score, "
            "signature, watermark, text, blurry, realistic, 3d, gradient background, shadow")
PLATE_PROMPT = ("no humans, scenery, dark background, outer space, sparkle, glitter, floating particles, "
                "horizontal lines, blue theme, masterpiece, high score, absurdres")
PLATE_NEGATIVE = "1girl, 1boy, person, character, face, text, watermark, lowres, worst quality"


def character_mask(bgr: np.ndarray) -> np.ndarray:
    """uint8 mask (0/1) of the bright character on the dark background, full size."""
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    v = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 2], (0, 0), 3)
    fg = cv2.morphologyEx((v > 90).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = cv2.drawContours(np.zeros(fg.shape, np.uint8), cnts, -1, 1, -1)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def tint_lut(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean colour of the original character per lightness level (256x3, BGR)."""
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0][mask > 0]
    px = bgr[mask > 0].astype(np.float32)
    lut = np.zeros((256, 3), np.float32)
    have = np.zeros(256, bool)
    for l in range(256):
        sel = L == l
        if sel.sum() > 50:
            lut[l], have[l] = px[sel].mean(0), True
    idx = np.arange(256)
    for c in range(3):
        lut[:, c] = np.interp(idx, idx[have], lut[have, c])
    return lut


def cutout(bgr: np.ndarray) -> np.ndarray:
    """Alpha (0-1) of a character on a near-white background: flood fill the
    backdrop from the border so enclosed whites (eyes, highlights) stay. The
    model doesn't always draw pure white (light blue, beige), so the fill
    follows colour similarity rather than a white threshold."""
    h, w = bgr.shape[:2]
    ff = np.zeros((h, w), np.uint8)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for x, y in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0), (0, h // 2), (w - 1, h // 2)]:
        if mask[y + 1, x + 1] == 0:  # flood the backdrop by colour similarity, from the border
            cv2.floodFill(bgr.copy(), mask, (x, y), 0, (12, 12, 12), (12, 12, 12),
                          cv2.FLOODFILL_MASK_ONLY | (255 << 8) | 4)
    ff[mask[1:-1, 1:-1] == 255] = 2
    bg = (ff == 2).astype(np.uint8)
    fg = 1 - bg
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:
        fg = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    fg = cv2.erode(fg, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(fg.astype(np.float32), (0, 0), 1.2)


def face_image(path: str) -> Image.Image:
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    x, y, w, h = max(det.detectMultiScale(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 1.1, 5, minSize=(150, 150)),
                     key=lambda f: f[2] * f[3])
    m = int(0.35 * w)
    return Image.fromarray(rgb[max(y - m, 0):y + h + m, max(x - m, 0):x + w + m])


def load(cls, ip_scale=None):
    from transformers import CLIPVisionModelWithProjection
    kw = {}
    if ip_scale is not None:
        kw["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16)
    pipe = cls.from_pretrained(CHECKPOINT, torch_dtype=torch.float16, **kw)
    if ip_scale is not None:
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                             weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors")
        pipe.set_ip_adapter_scale(ip_scale)
    pipe.enable_model_cpu_offload()  # 12 GB laptop GPU
    return pipe


def free(pipe):
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(ROOT / "photos" / "image.jpg"))
    ap.add_argument("--face", default=str(ROOT / "photos" / "Finalimage.jpg"), help="her photo, for likeness")
    ap.add_argument("--seeds", default="1,2,3,4,5,6")
    ap.add_argument("--ip-scale", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]

    ref = cv2.imread(a.ref)
    H, W = ref.shape[:2]
    m = character_mask(ref)
    lut = tint_lut(ref, m)
    ys, xs = np.nonzero(m)
    feet_y, mid_x, height = int(ys.max()), int((xs.min() + xs.max()) // 2), int(ys.max() - ys.min())

    from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline

    # 1. Background plate, once: fill the character's area with real background from
    # the same rows further left/right. The stage lines run horizontally, so a
    # sideways copy keeps them continuous (an SDXL inpaint invented a glowing planet).
    plate_path = OUT / "plate.png"
    if not plate_path.exists():
        grow = cv2.dilate(m, np.ones((61, 61), np.uint8))
        soft = cv2.GaussianBlur(grow.astype(np.float32), (0, 0), 10)[..., None]
        gx0, gx1 = np.nonzero(grow.any(0))[0][[0, -1]]
        shift = int(gx1 - gx0) + 40
        left = np.roll(ref, shift, axis=1).astype(np.float32)     # pixels from the left of the hole
        right = np.roll(ref, -shift, axis=1).astype(np.float32)   # ... and from the right
        t = np.clip((np.arange(W) - gx0) / max(gx1 - gx0, 1), 0, 1)[None, :, None]
        fill = (1 - t) * left + t * right
        plate = soft * fill + (1 - soft) * ref.astype(np.float32)
        cv2.imwrite(str(plate_path), plate.round().clip(0, 255).astype(np.uint8))
        print("wrote", plate_path, flush=True)
    plate = cv2.imread(str(plate_path))

    # 2. Her, as a chibi on white.
    face = face_image(a.face)
    pipe = load(StableDiffusionXLPipeline, a.ip_scale)
    tiles = []
    for seed in seeds:
        img = pipe(prompt=PROMPT, negative_prompt=NEGATIVE, ip_adapter_image=face, width=1024, height=1024,
                   guidance_scale=6.0, num_inference_steps=a.steps,
                   generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        gen = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(OUT / f"chibi_{seed}_natural.png"), gen)

        # 3. Cut out, tint, scale to the original's height, stand on its feet.
        alpha = cutout(gen)
        gy, gx = np.nonzero(alpha > 0.5)
        if len(gy) == 0:
            print("no character found for seed", seed)
            continue
        bx0, by0, bx1, by1 = gx.min(), gy.min(), gx.max() + 1, gy.max() + 1
        k = height / (by1 - by0)
        char = lut[cv2.cvtColor(gen, cv2.COLOR_BGR2LAB)[..., 0]][by0:by1, bx0:bx1]
        al = alpha[by0:by1, bx0:bx1]
        cw, ch = int(round((bx1 - bx0) * k)), int(round((by1 - by0) * k))
        char = cv2.resize(char, (cw, ch), interpolation=cv2.INTER_AREA)
        al = cv2.resize(al, (cw, ch), interpolation=cv2.INTER_AREA)[..., None]
        px0, py0 = mid_x - cw // 2, feet_y - ch
        sx0, sy0 = max(0, -px0), max(0, -py0)
        px0, py0 = max(px0, 0), max(py0, 0)
        px1, py1 = min(px0 + cw - sx0, W), min(py0 + ch - sy0, H)
        out = plate.astype(np.float32)
        region = out[py0:py1, px0:px1]
        c = char[sy0:sy0 + (py1 - py0), sx0:sx0 + (px1 - px0)]
        aa = al[sy0:sy0 + (py1 - py0), sx0:sx0 + (px1 - px0)]
        out[py0:py1, px0:px1] = aa * c + (1 - aa) * region
        out = out.round().clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(OUT / f"chibi_{seed}.png"), out)
        layer = np.zeros((H, W, 4), np.uint8)
        layer[py0:py1, px0:px1, :3] = c.round().clip(0, 255).astype(np.uint8)
        layer[py0:py1, px0:px1, 3] = (aa[..., 0] * 255).round().astype(np.uint8)
        cv2.imwrite(str(OUT / f"chibi_{seed}_rgba.png"), layer)
        t = cv2.resize(out, (W // 5, H // 5), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"seed {seed}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        tiles.append(t)
        print("wrote chibi", seed, flush=True)
    free(pipe)
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(OUT / "chibi_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", OUT / "chibi_sheet.jpg")


if __name__ == "__main__":
    main()
