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
PROMPT = ("3d animated film character, Disney Pixar style, a friendly middle-aged Japanese woman with a "
          "short black bob haircut, calm pleasant neutral expression, relaxed eyebrows, natural skin tone, black sleeveless top, looking at the camera, "
          "soft studio lighting, plain light grey background, smooth stylized skin, expressive eyes, "
          "high quality 3d render")
NEGATIVE = ("photo, photorealistic, realistic skin texture, wrinkles, painting, sketch, anime, 2d, "
            "frown, worried, furrowed brows, sad, angry, red cheeks, blush, sunburn, deformed, distorted face, extra limbs, text, watermark, blurry, lowres")


def face_crop(rgb: np.ndarray) -> Image.Image:
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = det.detectMultiScale(grey, 1.1, 5, minSize=(150, 150))
    if len(faces) == 0:
        raise SystemExit("no face found in the source frame")
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    m = int(0.35 * w)
    return Image.fromarray(rgb[max(y - m, 0):y + h + m, max(x - m, 0):x + w + m])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "output" / "event" / "frame" / "final_c74.png"))
    ap.add_argument("--strengths", default="0.65,0.75")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--ip-scale", type=float, default=0.5, help="how hard IP-Adapter pulls toward her face")
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from diffusers import StableDiffusionXLImg2ImgPipeline
    from transformers import CLIPVisionModelWithProjection
    enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder",
                                                        torch_dtype=torch.float16)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(CHECKPOINT, image_encoder=enc, torch_dtype=torch.float16)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                         weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors")
    pipe.set_ip_adapter_scale(a.ip_scale)
    pipe.enable_model_cpu_offload()  # 12 GB laptop GPU; SDXL + encoder + adapter don't all fit at once

    frame = cv2.cvtColor(cv2.imread(a.src), cv2.COLOR_BGR2RGB)
    x0, y0, x1, y1 = CROP
    init = Image.fromarray(frame[y0:y1, x0:x1]).resize((1024, 1024), Image.LANCZOS)
    face = face_crop(frame)

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
    tiles = []
    for strength, seed in itertools.product(strengths, seeds):
        img = pipe(prompt=PROMPT, negative_prompt=NEGATIVE, image=init, ip_adapter_image=face,
                   strength=strength, guidance_scale=6.0, num_inference_steps=a.steps,
                   generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        cart = np.asarray(img.resize((w, h), Image.LANCZOS)).astype(np.float32)
        out = frame.astype(np.float32).copy()
        out[y0:y1, x0:x1] = mask * cart + (1 - mask) * out[y0:y1, x0:x1]
        out = out.round().astype(np.uint8)
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
