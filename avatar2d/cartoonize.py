#!/usr/bin/env python3
"""Turn the framed photo into a Disney/Pixar-style 3D cartoon character.

    .venv-infinitetalk/Scripts/python.exe avatar2d/cartoonize.py [--src output/event/frame/final_c74.png]

Runs in .venv-infinitetalk only because that venv already has a CUDA torch +
diffusers 0.40; nothing here touches InfiniteTalk. Models (setup_avatar2d.sh):
- Samaritan 3D Cartoon v4 (SDXL checkpoint tuned for the animated-film look);
- IP-Adapter plus-face (SDXL, ViT-H), fed a crop of her real face, which pulls
  the cartoon's face back toward hers.

It's img2img on the head-and-shoulders crop, so pose, framing, hair and top stay
where they are in the photo (the rig and the event framing rely on that). The
crop is generated at 1024x1024, scaled back and feathered into the original frame;
the wall is plain, so the seam doesn't show.

Writes output/avatar2d/style/pixar_s<strength>_<seed>.png for every
(strength, seed) and pixar_sheet.jpg. Higher strength = more cartoon, less
likeness. Pick one and pass it to build_rig.py --src.
"""
import argparse
import gc
import itertools
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "avatar2d" / "style"
CHECKPOINT = "GHArt/Samaritan_3d_Cartoon_V4.0_xl_fp16"
CROP = (440, 60, 1480, 1080)   # x0, y0, x1, y1 in the 1920x1080 frame: head and shoulders, square
FEATHER = 48
PRESMOOTH = 5
# SDXL's text encoders read only the first 77 tokens, so both prompts stay short
# and lead with what matters most.
PROMPT = ("3d animated film character, Disney Pixar style, friendly Japanese woman, short black bob, "
          "smooth flawless skin, soft gently arched eyebrows, kind relaxed expression, "
          "eyes looking straight at the viewer, natural skin tone, black sleeveless top, plain light grey background")
NEGATIVE = ("wrinkles, forehead lines, eye bags, nasolabial folds, frown, angry, furrowed brows, worried, "
            "cross-eyed, strabismus, rosy cheeks, blush, red nose, photo, realistic skin, anime, 2d, deformed, text")
EYES_PROMPT = ("Disney Pixar 3d character, symmetrical eyes looking straight at the viewer, "
               "soft gently arched relaxed eyebrows, kind friendly eyes, smooth skin")


def face_box(rgb: np.ndarray) -> tuple:
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = det.detectMultiScale(grey, 1.1, 5, minSize=(150, 150))
    if len(faces) == 0:
        raise SystemExit("no face found in the source frame")
    return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))


def presmooth(rgb: np.ndarray, passes: int = PRESMOOTH) -> np.ndarray:
    """Bilateral smoothing: img2img keeps the photo's fine lines (forehead creases,
    eye bags, smile lines) and the cartoon model renders them as hard wrinkles."""
    for _ in range(passes):
        rgb = cv2.bilateralFilter(rgb, 15, 30, 15)
    return rgb


def soft_mask(shape, boxes, blur: int) -> np.ndarray:
    m = np.zeros(shape, np.float32)
    for (bx0, by0, bx1, by1) in boxes:
        cv2.ellipse(m, ((bx0 + bx1) // 2, (by0 + by1) // 2), ((bx1 - bx0) // 2, (by1 - by0) // 2), 0, 0, 360, 1, -1)
    return cv2.GaussianBlur(m, (0, 0), blur)


def decheek(rgb: np.ndarray, face, amount: float) -> np.ndarray:
    """Pull red-excess skin (cheeks, nose tip) back to the face's median skin tint.
    The model paints heavy blush that no prompt removes. Works on Lab a* (red-green)
    inside the face, minus an ellipse around the mouth so the lips keep their colour."""
    if amount <= 0:
        return rgb
    x, y, w, h = face
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    region = soft_mask(rgb.shape[:2], [(x, y + int(0.15 * h), x + w, y + int(1.0 * h))], 12)
    region *= 1 - soft_mask(rgb.shape[:2], [(x + int(0.28 * w), y + int(0.68 * h),
                                            x + int(0.72 * w), y + int(0.92 * h))], 10)
    inside = region > 0.5
    base = float(np.median(lab[..., 1][inside]))
    excess = cv2.GaussianBlur(np.maximum(lab[..., 1] - (base + 2), 0), (0, 0), 6)
    lab[..., 1] -= amount * region * excess
    return cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def load(cls, ip_scale: float):
    from transformers import CLIPVisionModelWithProjection
    enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder",
                                                        torch_dtype=torch.float16)
    pipe = cls.from_pretrained(CHECKPOINT, image_encoder=enc, torch_dtype=torch.float16)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                         weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors")
    pipe.set_ip_adapter_scale(ip_scale)
    pipe.enable_model_cpu_offload()  # 12 GB laptop GPU; SDXL + encoder + adapter don't all fit at once
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "output" / "event" / "frame" / "final_c74.png"))
    ap.add_argument("--strengths", default="0.7")
    ap.add_argument("--seeds", default="1,2,3,4,5,6")
    ap.add_argument("--ip-scale", type=float, default=0.3, help="how hard IP-Adapter pulls toward her face")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--fix-eyes", type=float, default=0.0,
                    help="re-draw the eyes+brows band at this inpaint strength (0 = off). Tried at 0.6: it turned brown eyes blue-grey and left the brows as they were, so pick a seed with straight eyes instead and relax the brows in build_rig.py")
    ap.add_argument("--cheek-fix", type=float, default=0.8, help="0-1, how much blush to remove")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline
    pipe = load(StableDiffusionXLImg2ImgPipeline, a.ip_scale)

    frame = cv2.cvtColor(cv2.imread(a.src), cv2.COLOR_BGR2RGB)
    x0, y0, x1, y1 = CROP
    fx, fy, fw, fh = face_box(frame)
    soft = presmooth(frame)
    init = Image.fromarray(soft[y0:y1, x0:x1]).resize((1024, 1024), Image.LANCZOS)
    m = int(0.35 * fw)
    face = Image.fromarray(soft[max(fy - m, 0):fy + fh + m, max(fx - m, 0):fx + fw + m])
    k = 1024 / (x1 - x0)  # frame -> generation-crop scale
    band = [(int((fx - x0 + 0.02 * fw) * k), int((fy - y0 + 0.12 * fh) * k),
             int((fx - x0 + 0.98 * fw) * k), int((fy - y0 + 0.58 * fh) * k))]
    eye_mask = Image.fromarray((soft_mask((1024, 1024), band, 14) * 255).astype(np.uint8))

    # Feathered paste mask: 1 inside the crop, ramping to 0 at its edges (bottom edge
    # is the frame edge, so no ramp there).
    h, w = y1 - y0, x1 - x0
    mask = np.ones((h, w), np.float32)
    ramp = np.linspace(0, 1, FEATHER)
    mask[:, :FEATHER] *= ramp
    mask[:, -FEATHER:] *= ramp[::-1]
    mask[:FEATHER] *= ramp[:, None]
    mask = mask[..., None]

    strengths = [float(s) for s in a.strengths.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    jobs = list(itertools.product(strengths, seeds))
    imgs = []
    for strength, seed in jobs:
        imgs.append(pipe(prompt=PROMPT, negative_prompt=NEGATIVE, image=init, ip_adapter_image=face,
                         strength=strength, guidance_scale=6.0, num_inference_steps=a.steps,
                         generator=torch.Generator("cpu").manual_seed(seed)).images[0])
        print(f"generated s{strength:.2f}_{seed}", flush=True)
    if a.fix_eyes > 0:
        # A second pipeline, loaded only after the first is gone: two SDXL pipelines on
        # a 12 GB card spill into the driver's sysmem fallback and crawl (~30x slower).
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        inpaint = load(StableDiffusionXLInpaintPipeline, a.ip_scale)
        for i, (strength, seed) in enumerate(jobs):
            imgs[i] = inpaint(prompt=EYES_PROMPT, negative_prompt=NEGATIVE, image=imgs[i], mask_image=eye_mask,
                              ip_adapter_image=face, strength=a.fix_eyes, guidance_scale=6.0,
                              num_inference_steps=a.steps,
                              generator=torch.Generator("cpu").manual_seed(seed)).images[0]
            print(f"eyes fixed s{strength:.2f}_{seed}", flush=True)

    tiles = []
    for (strength, seed), img in zip(jobs, imgs):
        cart = np.asarray(img.resize((w, h), Image.LANCZOS)).astype(np.float32)
        out = frame.astype(np.float32).copy()
        out[y0:y1, x0:x1] = mask * cart + (1 - mask) * out[y0:y1, x0:x1]
        out = decheek(out.round().astype(np.uint8), (fx, fy, fw, fh), a.cheek_fix)
        name = f"pixar_s{strength:.2f}_{seed}"
        cv2.imwrite(str(OUT / f"{name}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        t = cv2.cvtColor(cv2.resize(out[y0:y1, x0:x1], (400, 400), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        cv2.putText(t, name, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(t)
        print("wrote", name, flush=True)
    rows = [np.hstack(tiles[i:i + len(seeds)]) for i in range(0, len(tiles), len(seeds))]
    cv2.imwrite(str(OUT / "pixar_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", OUT / "pixar_sheet.jpg")


if __name__ == "__main__":
    main()
