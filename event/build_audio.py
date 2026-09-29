#!/usr/bin/env python3
"""Phase 1 of the event-host video: the audio.

One Chatterbox call per script line (language_id ja/en), pauses inserted as
digital silence, per-section WAVs plus a full WAV, all at one common gain
(-16 LUFS integrated over the full track). Reuses make_avatar's model loader
and voice-sample prep; run it with the main .venv:

    .venv/Scripts/python.exe event/build_audio.py

Raw per-line clips are cached, so re-running (or rebuilding with another
line-18 take via --line18 SEED:ENDING) only generates what is missing.
Every clip's seed is recorded in tts_log.json; a re-render is a deliberate
re-roll (delete its raw clip or change the seed).
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import make_avatar as ma  # noqa: E402

SCRIPT = ROOT / "event" / "script.txt"
OUT = ROOT / "output" / "event" / "audio"
RAW = OUT / "raw"

# Applied to the TTS text only (never to subtitles). Order matters: the
# longer 'AI晶子' has to go before the bare 'AI'.
PRONUNCIATION = [
    ("AI晶子", "エーアイあきこ"),
    ("AI", "エーアイ"),
    ("私", "わたくし"),  # every 私 in this script is the speaker
    ("第68期", "だいろくじゅうはっき"),
    ("第2部", "だいにぶ"),
    ("17時30分", "じゅうななじさんじゅっぷん"),
    ("株式会社", "かぶしきがいしゃ"),
    ("行います", "おこないます"),
    ("入退室", "にゅうたいしつ"),
    ("消費電力", "しょうひでんりょく"),
    ("懇親会", "こんしんかい"),
    ("同時通訳", "どうじつうやく"),
    ("来期", "らいき"),
    ("方針発表会", "ほうしんはっぴょうかい"),
    # Not in the requested list: pykakasi (which Chatterbox uses to read kanji)
    # gets these three wrong - はっぴょうなか / ほう / ふん.
    ("発表中", "はっぴょうちゅう"),
    ("思われた方", "おもわれたかた"),
    ("わたくしの分", "わたくしのぶん"),
]

FADE_S = 0.010          # click protection on every clip edge
TARGET_LUFS = -16.0
PEAK_CEILING = 0.891    # -1 dBFS
LINE18_ENDINGS = {"dash": "——", "bare": "", "comma": "、", "ellipsis": "…"}
LINE18_SEEDS = (181, 182, 183)
# ~2.5 GPU-minutes per second of audio, measured as ~7.5 min per streaming chunk
# (81 frames first, then 72 new frames each) plus ~1.5 min of model loading.
MIN_PER_CHUNK, LOAD_MIN, FPS = 7.5, 1.5, 25


def parse_script(path: Path):
    sections, current = {}, None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if re.fullmatch(r"S\d+", line):
            current = line
            sections[current] = []
            continue
        m = re.fullmatch(r"(\d+)\s+\[(ja|en)\]\s+(.*?)\s*\(([\d.]+)\)", line)
        if not m or current is None:
            raise ValueError(f"cannot parse script line: {raw!r}")
        sections[current].append({"n": int(m[1]), "lang": m[2], "text": m[3], "pause": float(m[4])})
    return sections


def text_hash(text: str) -> str:
    """Part of every raw-clip name, so a changed reading can never reuse a stale clip."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:6]


def strip_dash(text: str) -> str:
    return re.sub(r"[—―–─\-]+\s*$", "", text)


def to_tts_text(text: str, lang: str) -> str:
    if lang != "ja":
        return text
    for src, dst in PRONUNCIATION:
        text = text.replace(src, dst)
    return text


def synth(text, lang, voice, seed, cfg_weight, exaggeration, cache: Path, log: dict, key: str):
    """One Chatterbox call, seeded, cached as a raw WAV."""
    if cache.exists():
        audio, sr = sf.read(cache)
        return audio.astype(np.float32), sr
    import torch

    model = ma.get_tts_model()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    t = time.time()
    audio = model.generate(text, language_id=lang, audio_prompt_path=voice,
                           cfg_weight=cfg_weight, exaggeration=exaggeration).squeeze().cpu().numpy()
    log[key] = {"text_tts": text, "lang": lang, "seed": seed, "cfg_weight": cfg_weight,
                "exaggeration": exaggeration, "gen_seconds": round(time.time() - t, 1)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    sf.write(cache, audio, model.sr)
    return audio.astype(np.float32), model.sr


def tidy(clip: np.ndarray, sr: int) -> np.ndarray:
    """Trim the model's own leading/trailing silence (pauses are ours to set),
    then 10 ms fades so a splice can never click."""
    import librosa

    _, (a, b) = librosa.effects.trim(clip, top_db=45)
    guard = int(0.02 * sr)
    clip = clip[max(a - guard, 0): min(b + guard, len(clip))].copy()
    n = int(FADE_S * sr)
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    clip[:n] *= ramp
    clip[-n:] *= ramp[::-1]
    return clip


def build_section(lines, clips, sr):
    parts = []
    for ln in lines:
        parts.append(clips[ln["n"]])
        parts.append(np.zeros(int(round(ln["pause"] * sr)), dtype=np.float32))
    return np.concatenate(parts)


def chunks_for(duration_s: float) -> int:
    frames = int(np.ceil(duration_s * FPS))
    return 1 + max(0, int(np.ceil((frames - 81) / 72)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=str(ROOT / "voice_samples" / "testaudio.wav"),
                    help="JA reference sample (also the EN one unless --voice-en is given)")
    ap.add_argument("--voice-en", default=None, help="English reference sample (default: --voice, cfg_weight 0)")
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    ap.add_argument("--exaggeration", type=float, default=0.5)
    ap.add_argument("--line18", default="181:dash",
                    help="SEED:ENDING used in S6 (ending: dash/bare/comma/ellipsis)")
    args = ap.parse_args()

    sections = parse_script(SCRIPT)
    OUT.mkdir(parents=True, exist_ok=True)
    voice_ja = ma.get_prepared_voice(args.voice)
    voice_en = ma.get_prepared_voice(args.voice_en) if args.voice_en else voice_ja
    en_cfg = 0.5 if args.voice_en else 0.0  # spec: JA sample speaking English -> cfg_weight 0
    seed18, ending18 = args.line18.split(":")
    seed18 = int(seed18)
    log_path = OUT / "tts_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}

    tts_lines, readings, clips, sr = [], [], {}, None
    for name, lines in sections.items():
        for ln in lines:
            n, lang = ln["n"], ln["lang"]
            tts_text = to_tts_text(ln["text"], lang)
            if n == 18:  # the trailing dash is replaced by the ending picked with --line18
                tts_text = to_tts_text(strip_dash(ln["text"]), lang) + LINE18_ENDINGS[ending18]
            seed = seed18 if n == 18 else 100 + n
            tag = f"line_{n:02d}_seed{seed}_{text_hash(tts_text)}"
            cfg = en_cfg if lang == "en" else args.cfg_weight
            audio, sr = synth(tts_text, lang, voice_en if lang == "en" else voice_ja, seed, cfg,
                              args.exaggeration, RAW / f"{tag}.wav", log, tag)
            clips[n] = tidy(audio, sr)
            tts_lines.append(f"{n} [{lang}] {tts_text}")
            print(f"line {n:2d} [{lang}] seed {seed} {len(clips[n]) / sr:5.2f}s  {tts_text[:40]}", flush=True)

    # Line-18 alternatives, so the cut-off can be chosen by ear.
    takes_dir = OUT / "line18_takes"
    takes_dir.mkdir(exist_ok=True)
    text18 = next(l for l in sections["S6"] if l["n"] == 18)
    base18 = to_tts_text(strip_dash(text18["text"]), "ja")
    tail = np.zeros(int(round(text18["pause"] * sr)), dtype=np.float32)
    for seed in LINE18_SEEDS:
        for ending, suffix in LINE18_ENDINGS.items():
            tag = f"line_18_seed{seed}_{text_hash(base18 + suffix)}"
            audio, _ = synth(base18 + suffix, "ja", voice_ja, seed, args.cfg_weight, args.exaggeration,
                             RAW / f"{tag}.wav", log, tag)
            sf.write(takes_dir / f"take_seed{seed}_{ending}.wav", np.concatenate([tidy(audio, sr), tail]), sr)
    ma.release_tts_model()
    ma.cleanup_prepared_voices()
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")

    # Sections and full track, one common gain.
    built = {name: build_section(lines, clips, sr) for name, lines in sections.items()}
    full = np.concatenate(list(built.values()))
    import pyloudnorm as pyln

    measured = pyln.Meter(sr).integrated_loudness(full)
    gain = 10 ** ((TARGET_LUFS - measured) / 20)
    peak = float(np.abs(full).max()) * gain
    if peak > PEAK_CEILING:
        gain *= PEAK_CEILING / peak
    final_lufs = measured + 20 * np.log10(gain)
    final_peak_db = 20 * np.log10(float(np.abs(full).max()) * gain)
    for name, audio in built.items():
        sf.write(OUT / f"{name}.wav", audio * gain, sr, subtype="PCM_24")
    sf.write(OUT / "full.wav", full * gain, sr, subtype="PCM_24")
    (OUT / "script_tts.txt").write_text("\n".join(tts_lines) + "\n", encoding="utf-8")

    # Readings as pykakasi sees them: a quick check for mis-read kanji.
    import pykakasi

    kks = pykakasi.kakasi()
    for line in tts_lines:
        num, rest = line.split(" ", 1)
        if "[ja]" in rest:
            readings.append(f"{num}: " + "".join(t["hira"] for t in kks.convert(rest.split("] ", 1)[1])))
    (OUT / "readings.txt").write_text("\n".join(readings) + "\n", encoding="utf-8")

    report = [f"voice: {args.voice}  (EN cfg_weight {en_cfg})", f"sample rate {sr} Hz, "
              f"loudness {final_lufs:.1f} LUFS (target {TARGET_LUFS}), peak {final_peak_db:.1f} dBFS",
              "", "section  duration  chunks  est. render"]
    total_min = 0.0
    for name, audio in built.items():
        d = len(audio) / sr
        c = chunks_for(d)
        est = LOAD_MIN + c * MIN_PER_CHUNK
        total_min += est
        report.append(f"{name:7s} {d:7.1f}s  {c:5d}  {est:5.0f} min")
    report.append(f"total   {len(full) / sr:7.1f}s  (line 18 = seed {seed18}, ending {ending18})  ~{total_min / 60:.1f} h")
    text = "\n".join(report)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
