#!/usr/bin/env python3
"""A/B the voice routes on two lines of the script, using the explicit kana
readings in readings_ja.txt, and score how close each one is to the reference.

The score is the cosine similarity of Chatterbox's speaker-encoder embeddings
(reference clip vs. generated audio). It measures timbre only - not
intonation or reading, which still need a listen - and the number that means
"same person" is shown by comparing the two halves of the reference itself.

    .venv/Scripts/python.exe event/voice_ab_kana.py --voice voice_samples/newtest.wav
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import make_avatar as ma  # noqa: E402


def readings() -> dict:
    out = {}
    for line in (ROOT / "event" / "readings_ja.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\d+)\s+(.+)", line)
        if m and not line.startswith("#"):
            out[int(m[1])] = m[2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="voice_samples/newtest.wav")
    ap.add_argument("--lines", default="2,3", help="comma-separated script line numbers to speak")
    a = ap.parse_args()

    text = "".join(readings()[int(n)] for n in a.lines.split(","))
    out = ROOT / "output" / "voice_ab" / "kana"
    out.mkdir(parents=True, exist_ok=True)
    print("text:", text)

    ref = ma.get_prepared_voice(a.voice)
    variants = {
        "chatterbox_kana.wav": lambda f: ma.synthesize_speech(text, "ja", a.voice, f, pause_ms=0, cfg_weight=0.5),
        "chatterbox_kana_cfg0.2.wav": lambda f: ma.synthesize_speech(text, "ja", a.voice, f, pause_ms=0, cfg_weight=0.2),
        "xtts_kana.wav": lambda f: ma.synthesize_speech_xtts(text, "ja", a.voice, f, pause_ms=0),
        "xtts_kana_seedvc.wav": lambda f: ma.convert_voice(
            ma.synthesize_speech_xtts(text, "ja", a.voice, str(out / "x.stage1.wav"), pause_ms=0), ref, f),
        "chatterbox_kana_seedvc.wav": lambda f: ma.convert_voice(
            ma.synthesize_speech(text, "ja", a.voice, str(out / "c.stage1.wav"), pause_ms=0, cfg_weight=0.5), ref, f),
    }
    done = []
    for name, run in variants.items():
        if (out / name).exists():  # generation is the slow part; keep what is already there
            done.append(name)
            print("have", name, flush=True)
            continue
        try:
            run(str(out / name))
            done.append(name)
            print("ok", name, flush=True)
        except Exception as exc:  # keep going: one broken route shouldn't hide the rest
            print("FAILED", name, exc, flush=True)
    for tmp in out.glob("*.stage1.wav"):
        tmp.unlink(missing_ok=True)

    # The trained speaker encoder that ships inside Chatterbox (a bare VoiceEncoder()
    # has random weights and scores everything ~1.0).
    enc = ma.get_tts_model().ve.eval()
    def emb(path_or_audio, sr=None):
        wav, rate = (sf.read(path_or_audio) if isinstance(path_or_audio, str) else (path_or_audio, sr))
        if wav.ndim > 1:
            wav = wav.mean(1)
        # Resample here: the encoder's own kaiser_fast path trips on this numba/resampy pair.
        wav = resample_poly(wav.astype(np.float64), enc.hp.sample_rate, int(rate)).astype(np.float32)
        return np.asarray(enc.embeds_from_wavs([wav], sample_rate=enc.hp.sample_rate, as_spk=True)).reshape(-1)

    def cos(x, y):
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

    ref_audio, ref_sr = sf.read(ref)
    half = len(ref_audio) // 2
    ref_emb = emb(ref)
    print("\nspeaker similarity to the reference (1.0 = identical):")
    print(f"  reference's own two halves   {cos(emb(ref_audio[:half], ref_sr), emb(ref_audio[half:], ref_sr)):.3f}   <- 'same person' level")
    for name in done:
        print(f"  {name:28s} {cos(ref_emb, emb(str(out / name))):.3f}")
    ma.release_tts_model()
    ma.cleanup_prepared_voices()
    print(f"\nlisten in {out}")


if __name__ == "__main__":
    main()
