#!/usr/bin/env python3
"""Synthesize segments with Style-Bert-VITS2. Runs inside .venv-sbv2, never imported.

The subprocess entry point make_avatar.synthesize_speech_sbv2() shells out to,
with the same JSON-file contract as xtts_synth.py (arbitrary Japanese text
through a Windows command line invites quoting and cp932 encoding bugs):

    python sbv2_synth.py JOB_JSON          (cwd: the Style-Bert-VITS2 checkout)

JOB_JSON = {"model_dir": "...", "out_dir": "...", "length": 1.0, "pitch_scale": 1.0,
            "segments": [{"pieces": [["text", accent or null], ...], "style": "Neutral",
                          "style_weight": 1.0, "intonation_scale": 1.0}, ...]}
-> JOB_JSON + ".result" = {"sr": 44100, "files": ["...wav", ...]}  (one per segment)

A piece with an accent (0 = flat, n = pitch drops after mora n) has its
pyopenjtalk accent replaced; the rest keep pyopenjtalk's. The cwd matters: SBV2
finds its BERT model and user dictionary relative to the checkout.
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())                              # style_bert_vits2, from the checkout
sys.path.insert(0, str(Path(__file__).resolve().parent))     # script_markup (ours, pure Python)
from script_markup import accent_tones  # noqa: E402


def latest_checkpoint(model_dir: Path) -> Path:
    def step(path: Path) -> int:
        match = re.search(r"_s(\d+)$", path.stem)
        return int(match.group(1)) if match else -1

    found = sorted(model_dir.glob("*.safetensors"), key=step)
    if not found:
        raise FileNotFoundError(f"No .safetensors checkpoint in {model_dir}")
    return found[-1]


def phones_and_tones(pieces: list) -> tuple:
    """Pieces -> (given_phone, given_tone), with marked pieces' accents replaced."""
    from style_bert_vits2.nlp.japanese.g2p_utils import g2kata_tone, kata_tone2phone_tone
    from style_bert_vits2.nlp.japanese.normalizer import normalize_text

    kata_tone = []
    for text, accent in pieces:
        part = g2kata_tone(normalize_text(text))
        if accent is not None:
            morae = [kata for kata, _ in part]
            try:
                part = list(zip(morae, accent_tones(len(morae), accent)))
            except ValueError as exc:
                raise ValueError(f"{text}: {exc}") from None
        kata_tone.extend(part)
    phone_tone = kata_tone2phone_tone(kata_tone)
    return [p for p, _ in phone_tone], [t for _, t in phone_tone]


def main() -> int:
    job_path = sys.argv[1]
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)

    import soundfile as sf
    import torch
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
    from style_bert_vits2.nlp.japanese.user_dict import update_dict
    from style_bert_vits2.tts_model import TTSModel

    pyopenjtalk_worker.initialize_worker()
    try:
        update_dict()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Style-Bert-VITS2 on {device} ...", flush=True)
        bert_models.load_model(Languages.JP, device_map=device)
        bert_models.load_tokenizer(Languages.JP)
        model_dir = Path(job["model_dir"])
        model = TTSModel(model_path=latest_checkpoint(model_dir), config_path=model_dir / "config.json",
                         style_vec_path=model_dir / "style_vectors.npy", device=device)

        files, sr = [], None
        for i, segment in enumerate(job["segments"]):
            kwargs = {}
            if any(accent is not None for _, accent in segment["pieces"]):
                kwargs["given_phone"], kwargs["given_tone"] = phones_and_tones(segment["pieces"])
            sr, audio = model.infer(
                text="".join(text for text, _ in segment["pieces"]), language=Languages.JP,
                style=segment["style"], style_weight=segment["style_weight"],
                intonation_scale=segment["intonation_scale"], length=job["length"],
                pitch_scale=job["pitch_scale"], line_split=False, **kwargs)
            path = os.path.join(job["out_dir"], f"segment_{i:03d}.wav")
            sf.write(path, audio, sr)
            files.append(path)
    finally:
        pyopenjtalk_worker.terminate_worker()

    with open(job_path + ".result", "w", encoding="utf-8") as fh:
        json.dump({"sr": sr, "files": files}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
