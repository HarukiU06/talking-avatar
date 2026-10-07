#!/usr/bin/env python3
"""
make_avatar.py — Turn a photo + your cloned voice into a talking-head video.

Pipeline:
  1. Text -> speech in your voice   (Chatterbox Multilingual TTS, or XTTS-v2;
     optionally converted toward your voice with Seed-VC)
  2. Photo/video + speech -> lip-synced video (LivePortrait + LatentSync by
     default; SadTalker or InfiniteTalk with --engine)

Usage:
  Single line:
    python make_avatar.py --photo photos/me.jpg --voice voice_samples/me.wav \
        --text "Hello, how are you?" --lang en --out output/hello.mp4

  Batch (pre-entered script of many lines/languages):
    python make_avatar.py --config config.yaml

See README.md for setup instructions and supported language codes.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
import soundfile as sf

# Where SadTalker was cloned by setup.sh. Override with the SADTALKER_DIR
# env var if you put it somewhere else.
SADTALKER_DIR = os.environ.get("SADTALKER_DIR", str(Path(__file__).parent / "SadTalker"))

# LatentSync lives in its own venv (see setup_latentsync.sh) — its pinned
# numpy/diffusers/transformers versions would conflict with chatterbox-tts's
# in the main .venv, so it's only ever invoked as a subprocess using its own
# venv's Python, never imported directly here.
LATENTSYNC_DIR = os.environ.get("LATENTSYNC_DIR", str(Path(__file__).parent / "LatentSync"))
LATENTSYNC_VENV = os.environ.get("LATENTSYNC_VENV", str(Path(__file__).parent / ".venv-latentsync"))

# LivePortrait, same isolation reasoning as LatentSync above (see
# setup_liveportrait.sh).
LIVEPORTRAIT_DIR = os.environ.get("LIVEPORTRAIT_DIR", str(Path(__file__).parent / "LivePortrait"))
LIVEPORTRAIT_VENV = os.environ.get("LIVEPORTRAIT_VENV", str(Path(__file__).parent / ".venv-liveportrait"))

# Wav2Lip, same isolation reasoning as LatentSync/LivePortrait above (see
# setup_wav2lip.sh). Optional refinement pass, --engine sadtalker only.
WAV2LIP_DIR = os.environ.get("WAV2LIP_DIR", str(Path(__file__).parent / "Wav2Lip"))
WAV2LIP_VENV = os.environ.get("WAV2LIP_VENV", str(Path(__file__).parent / ".venv-wav2lip"))

# InfiniteTalk, same isolation reasoning as the other three above (see
# setup_infinitetalk.sh). A self-contained --engine option: unlike sadtalker/
# latentsync, it generates lip sync, head motion AND facial expression in one
# audio+photo-driven pass, so it never calls render_video/run_latentsync/
# make_idle_motion_video.
INFINITETALK_LIGHTX2V_LORA = "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"
INFINITETALK_DIR = os.environ.get("INFINITETALK_DIR", str(Path(__file__).parent / "InfiniteTalk"))
INFINITETALK_VENV = os.environ.get("INFINITETALK_VENV", str(Path(__file__).parent / ".venv-infinitetalk"))
# One InfiniteTalk chunk is 81 frames at 25 fps (3.24s); it needs strictly more
# audio than that. See render_infinitetalk().
INFINITETALK_MIN_AUDIO_S = 3.4

# XTTS-v2, the alternative TTS behind --tts xtts. Created by setup_xtts.sh.
XTTS_VENV = os.environ.get("XTTS_VENV", str(Path(__file__).parent / ".venv-xtts"))

# Seed-VC, the voice-conversion stage behind --voice-convert. Created by
# setup_seedvc.sh.
SEEDVC_DIR = os.environ.get("SEEDVC_DIR", str(Path(__file__).parent / "seed-vc"))
SEEDVC_VENV = os.environ.get("SEEDVC_VENV", str(Path(__file__).parent / ".venv-seedvc"))

# Language codes supported by Chatterbox Multilingual V3 (see README section 5)
CHATTERBOX_LANGS = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
}

# Languages XTTS-v2 covers (--tts xtts). A subset of Chatterbox's: da/el/fi/he/
# ms/no/sv/sw have no XTTS equivalent. Kept in sync with xtts_synth.py's LANG_MAP.
XTTS_LANGS = {
    "ar", "cs", "de", "en", "es", "fr", "hi", "hu", "it", "ja", "ko", "nl",
    "pl", "pt", "ru", "tr", "zh",
}

_tts_model = None  # loaded once, reused across all lines in a batch
_prepared_voice_cache = {}  # source path -> cleaned copy, same reuse rationale


def get_tts_model():
    """Lazily load the Chatterbox Multilingual model (kept warm for batches)."""
    global _tts_model
    if _tts_model is None:
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Chatterbox Multilingual TTS on {device} ...")
        _tts_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    return _tts_model


def release_tts_model() -> None:
    """Drop the Chatterbox model and free its VRAM.

    Chatterbox is cached at module level so a batch doesn't reload it per
    line, but that means a single-line run keeps it resident (GPU included)
    through the whole video stage, which is far longer and far hungrier.
    LatentSync alone peaks near 11.8GB on a 12GB card, so a few spare GB of
    VRAM and the matching host allocations are not a rounding error here —
    two runs were killed for memory pressure before this was added.
    """
    global _tts_model
    if _tts_model is None:
        return
    _tts_model = None
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def split_sentences(text: str) -> list:
    """Split text into sentences on ./!/? followed by whitespace.

    Not abbreviation-aware ("Dr.", "U.S.") — adequate for typed scripts,
    not worth a full sentence-boundary NLP library for this project.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def synthesize_speech(text: str, lang: str, voice_sample: str, out_wav: str, pause_ms: int = 0,
                       cfg_weight: float = 0.5, exaggeration: float = 0.5,
                       prep_voice: bool = True) -> str:
    """Generate speech audio in the cloned voice, in the given language.

    pause_ms=0 (the default) sends the whole text in one generate() call and
    lets the model place its own pauses. Anything above 0 restores the older
    behaviour: split on sentence boundaries, synthesize each separately, and
    splice that much silence between them.

    Splicing was the original default, on the reasoning that a paragraph sent
    in one shot reads breathlessly. It buys pause *control* at a real cost:
    each sentence is generated with no knowledge of the one before it, so
    intonation resets to neutral at every boundary and the uniform gaps make
    the result sound like disconnected fragments rather than someone talking.
    Listener feedback on this project was that the spliced output sounded
    mechanical, so the default flipped. Raise --pause-ms if a given model
    genuinely runs sentences together.

    cfg_weight=0.5/exaggeration=0.5 are Chatterbox's own defaults and work
    well for most prompts. If the cloned voice keeps the accent of the
    reference clip's language instead of switching to the target language
    (or just sounds more American-accented than the reference), try
    lowering cfg_weight toward 0.0-0.3 — see README's "Voice accent &
    delivery" section. exaggeration controls delivery intensity/emotion
    (0.25-2.0); higher also speeds up pacing, so a low cfg_weight + high
    exaggeration combination needs both tuned together.
    """
    if lang not in CHATTERBOX_LANGS:
        raise ValueError(f"Unsupported language '{lang}'. Supported: {sorted(CHATTERBOX_LANGS)}")

    voice_sample = get_prepared_voice(voice_sample, enabled=prep_voice)
    model = get_tts_model()

    # pause_ms=0 -> one call for the whole text, so the model carries
    # intonation across sentence boundaries instead of restarting at each.
    segments = split_sentences(text) if pause_ms > 0 else [text.strip()]
    clips = [
        model.generate(segment, language_id=lang, audio_prompt_path=voice_sample,
                       cfg_weight=cfg_weight, exaggeration=exaggeration).squeeze().cpu().numpy()
        for segment in segments
    ]

    # torchaudio.save() in recent versions requires the separate torchcodec
    # package as its encoding backend; write with soundfile instead to avoid
    # that extra dependency.
    sf.write(out_wav, concat_with_pauses(clips, model.sr, pause_ms), model.sr)
    return out_wav


def get_prepared_voice(voice_sample: str, enabled: bool = True) -> str:
    """prepare_voice_sample(), memoized for the length of the process.

    A batch run reuses one reference clip for every line in the config, so
    without this the same trim/resample work would repeat per line — and each
    repeat would leave another temp file behind.
    """
    if not enabled:
        return voice_sample
    key = os.path.abspath(voice_sample)
    if key not in _prepared_voice_cache:
        _prepared_voice_cache[key] = prepare_voice_sample(voice_sample)
    return _prepared_voice_cache[key]


def cleanup_prepared_voices() -> None:
    """Delete the temp clips get_prepared_voice() created (never the originals)."""
    for original, prepared in _prepared_voice_cache.items():
        if prepared != original:
            try:
                os.remove(prepared)
            except OSError:
                pass
    _prepared_voice_cache.clear()


def concat_with_pauses(clips: list, sr: int, pause_ms: int) -> np.ndarray:
    """Join per-sentence audio clips with a fixed silence gap between them.

    Shared by every TTS backend: none of them has an explicit "pause here"
    control, so a whole paragraph sent in one call comes back as one
    breathless run-on. Splitting on sentence boundaries and splicing gives
    direct control over pacing instead of hoping the model infers it.
    """
    if not clips:
        return np.zeros(0, dtype=np.float32)
    silence = np.zeros(int(sr * pause_ms / 1000), dtype=np.float32)
    spliced = []
    for i, clip in enumerate(clips):
        spliced.append(clip)
        if i < len(clips) - 1:
            spliced.append(silence)
    return np.concatenate(spliced)


def synthesize_speech_xtts(text: str, lang: str, voice_sample: str, out_wav: str,
                           pause_ms: int = 0, prep_voice: bool = True) -> str:
    """Generate speech with XTTS-v2 instead of Chatterbox.

    Same contract as synthesize_speech(): same sentence splitting, same
    --pause-ms splicing. The knobs differ — XTTS has no cfg_weight or
    exaggeration equivalent, so those arguments are deliberately absent
    rather than silently ignored.

    Runs in .venv-xtts as a subprocess, like every other model here: coqui-tts
    and chatterbox-tts pin conflicting packages and cannot share a process.
    """
    if lang not in XTTS_LANGS:
        raise ValueError(
            f"XTTS doesn't support language '{lang}'. Supported: {sorted(XTTS_LANGS)}. "
            "Use --tts chatterbox for the others."
        )

    worker = Path(__file__).parent / "xtts_synth.py"
    python = get_venv_python(XTTS_VENV)
    voice_sample = get_prepared_voice(voice_sample, enabled=prep_voice)

    work_dir = tempfile.mkdtemp(prefix="xtts_")
    job_path = os.path.join(work_dir, "job.json")
    with open(job_path, "w", encoding="utf-8") as fh:
        json.dump({
            # See synthesize_speech(): 0 means hand over the whole text and let
            # the model place its own pauses. XTTS splits internally anyway.
            "sentences": split_sentences(text) if pause_ms > 0 else [text.strip()],
            "lang": lang,
            "speaker_wav": os.path.abspath(voice_sample),
            "out_dir": work_dir,
        }, fh)

    cmd = [python, str(worker), job_path]
    print("Running XTTS:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        with open(job_path + ".result", encoding="utf-8") as fh:
            result = json.load(fh)
        clips = [sf.read(f, dtype="float32")[0] for f in result["files"]]
        sf.write(out_wav, concat_with_pauses(clips, result["sr"], pause_ms), result["sr"])
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return out_wav


def denoise(audio: "np.ndarray", sr: int, strength: float = 0.6) -> "np.ndarray":
    """Strip stationary background noise (room tone, hiss, hum) from speech.

    This matters most on the *reference* clip, not the output. A voice
    conversion model has no way to separate "how this person sounds" from
    "what their room sounds like" — both are simply properties of the audio
    it is told to imitate — so it reproduces the reference's noise floor
    faithfully onto every line it generates. Measured on this project's own
    recording, the converted output's noise floor matched the source
    recording's within ~1dB per band across the spectrum, while the TTS audio
    going in was 3-6dB cleaner. Cleaning the reference therefore removes the
    noise at its origin; cleaning the output only attacks it after it has been
    baked in, and risks eroding the voice along with it.

    strength is noisereduce's prop_decrease. The default is 0.6, set by a
    listening test rather than by the measurements: 0.8 scored better on both
    noise floor and speaker similarity, but sounded over-processed to the
    person whose voice it is. Neither metric captures that, so the ear wins.
    """
    import noisereduce as nr

    return nr.reduce_noise(y=audio, sr=sr, stationary=True,
                           prop_decrease=strength).astype("float32")


def _best_window_start(audio: "np.ndarray", sr: int, seconds: float) -> int:
    """Index of the best `seconds`-long excerpt in `audio`.

    Scored on the share of frames carrying voice, minus a heavy penalty for
    clipping — clipping is unrecoverable and poisons a speaker embedding,
    whereas a merely quiet passage still carries timbre.
    """
    win = int(seconds * sr)
    if len(audio) <= win:
        return 0
    hop = max(1, int(2.0 * sr))   # slide 2s at a time; finer buys nothing
    frame = max(1, sr // 50)      # 20ms

    best_score, best_start = None, 0
    for start in range(0, len(audio) - win + 1, hop):
        chunk = audio[start:start + win]
        frames = chunk[: len(chunk) // frame * frame].reshape(-1, frame)
        peaks = np.abs(frames).max(axis=1)
        score = float((peaks > 0.02).mean()) - 10.0 * float((np.abs(chunk) >= 0.99).mean())
        if best_score is None or score > best_score:
            best_score, best_start = score, start
    return best_start


def pick_best_window(audio_path: str, seconds: float = 25.0,
                     target_sr: int = 22050, denoise_ref: float = 0.6) -> str:
    """Return the best `seconds`-long excerpt of a recording, as a temp wav.

    Seed-VC truncates its reference to the first 25 seconds
    (seed-vc/inference.py: `ref_audio[:sr * 25]`), so handing it a long
    recording does not give it more to work with — it silently uses the
    opening and discards the rest, after paying to decode all of it. What a
    long recording *is* good for is choice: 11 minutes contains many possible
    25-second windows, and the opening one is rarely the best (throat-clears,
    level-finding, room noise before the speaker settles).

    Windows are scored on the share of frames carrying voice, minus a penalty
    for clipping, which is unrecoverable and poisons the speaker embedding.
    Returns the original path unchanged when the file is already short enough,
    so callers can compare identity before deleting (as prepare_photo does).
    """
    import librosa

    if sf.info(audio_path).duration <= seconds and denoise_ref <= 0:
        return audio_path

    audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    best_start = _best_window_start(audio, target_sr, seconds)
    excerpt = audio[best_start:best_start + int(seconds * target_sr)]         if len(audio) > int(seconds * target_sr) else audio
    if denoise_ref > 0:
        excerpt = denoise(excerpt, target_sr, strength=denoise_ref)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, excerpt, target_sr)
    print(f"Seed-VC reference: using the best {seconds:.0f}s of "
          f"{sf.info(audio_path).duration / 60:.1f} min "
          f"(from {best_start / target_sr / 60:.1f} min in)")
    return tmp.name


def convert_voice(source_wav: str, target_voice: str, out_wav: str,
                  diffusion_steps: int = 25, cfg_rate: float = 0.7,
                  length_adjust: float = 1.0, denoise_ref: float = 0.6) -> str:
    """Replace the speaker identity in `source_wav` with the one in `target_voice`.

    This is the second half of the two-stage voice pipeline. Zero-shot TTS
    cloning has to produce natural prosody AND imitate a specific person from
    a few seconds of audio, and it audibly compromises on both. Splitting the
    job lets the TTS concentrate on sounding like a person talking, and lets
    a model built for speaker identity handle who that person is.

    Runs in .venv-seedvc as a subprocess, like every other model here.
    """
    inference_py = Path(SEEDVC_DIR) / "inference.py"
    if not inference_py.exists():
        raise FileNotFoundError(
            f"Couldn't find Seed-VC at {SEEDVC_DIR}. "
            "Run setup_seedvc.sh first, or set the SEEDVC_DIR environment variable."
        )

    # Seed-VC only ever reads the first 25s of the reference, so choose which
    # 25s that is rather than letting the file's opening decide.
    original_target = target_voice
    target_voice = pick_best_window(target_voice, denoise_ref=denoise_ref)

    python = get_venv_python(SEEDVC_VENV)
    # inference.py writes into a directory under a name it chooses, rather
    # than to a path you give it, so point it at a temp dir and move the
    # result where the caller actually wants it.
    result_dir = tempfile.mkdtemp(prefix="seedvc_")
    cmd = [
        python, "inference.py",
        "--source", os.path.abspath(source_wav),
        "--target", os.path.abspath(target_voice),
        "--output", result_dir,
        "--diffusion-steps", str(diffusion_steps),
        "--inference-cfg-rate", str(cfg_rate),
        "--length-adjust", str(length_adjust),
    ]
    print("Running Seed-VC:", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=SEEDVC_DIR, check=True)
        produced = sorted(Path(result_dir).glob("*.wav"))
        if not produced:
            raise RuntimeError("Seed-VC did not produce audio - check the log above.")
        shutil.move(str(produced[0]), out_wav)
    finally:
        shutil.rmtree(result_dir, ignore_errors=True)
        if target_voice != original_target:
            os.remove(target_voice)
    return out_wav


def synthesize(tts: str, text: str, lang: str, voice_sample: str, out_wav: str,
               pause_ms: int = 0, cfg_weight: float = 0.5,
               exaggeration: float = 0.5, prep_voice: bool = True) -> str:
    """Dispatch to the selected TTS backend (--tts)."""
    if tts == "xtts":
        return synthesize_speech_xtts(text, lang, voice_sample, out_wav,
                                      pause_ms=pause_ms, prep_voice=prep_voice)
    return synthesize_speech(text, lang, voice_sample, out_wav, pause_ms=pause_ms,
                             cfg_weight=cfg_weight, exaggeration=exaggeration,
                             prep_voice=prep_voice)


def _scratch_wav(final_path: str) -> str:
    """Temp path for a pipeline stage whose output feeds another stage."""
    return str(Path(final_path).with_suffix(".stage1.wav"))


def voice_compare(voice_sample: str, text: str, lang: str,
                  out_dir: str = None, pause_ms: int = 0,
                  vc_target: str = None) -> None:
    """Render the same line through every available voice setting, and stop.

    Choosing a voice is a listening decision that needs several candidates
    side by side, but the video step costs minutes per line while the audio
    costs seconds — so there is no reason to render video while deciding.
    Variants that can't run (XTTS not installed, or not supporting this
    language) are skipped with a note rather than aborting the rest.

    Results go under a subdirectory named after the reference clip, because
    the other comparison worth making is between two different reference
    recordings — and a single shared output directory silently overwrites
    the first one's results when you try.
    """
    out = Path(out_dir) if out_dir else Path("output/voice_ab") / Path(voice_sample).stem
    out.mkdir(parents=True, exist_ok=True)

    variants = [
        # (filename, description, callable)
        ("chatterbox_natural.wav",
         "Chatterbox, whole text in one pass (current default)",
         lambda f: synthesize_speech(text, lang, voice_sample, f, pause_ms=0,
                                     cfg_weight=0.5, prep_voice=True)),
        ("chatterbox_spliced450.wav",
         "Chatterbox, per-sentence + 450ms splices (the old default)",
         lambda f: synthesize_speech(text, lang, voice_sample, f, pause_ms=450,
                                     cfg_weight=0.5, prep_voice=True)),
        ("chatterbox_natural_cfg0.2.wav",
         "Chatterbox, whole text, low cfg-weight (less accent carry-over)",
         lambda f: synthesize_speech(text, lang, voice_sample, f, pause_ms=0,
                                     cfg_weight=0.2, prep_voice=True)),
        ("chatterbox_raw_reference.wav",
         "Chatterbox, whole text, reference clip left uncleaned",
         lambda f: synthesize_speech(text, lang, voice_sample, f, pause_ms=0,
                                     cfg_weight=0.5, prep_voice=False)),
        ("xtts_natural.wav",
         "XTTS-v2, whole text in one pass",
         lambda f: synthesize_speech_xtts(text, lang, voice_sample, f,
                                          pause_ms=0, prep_voice=True)),
        ("xtts_spliced450.wav",
         "XTTS-v2, per-sentence + 450ms splices",
         lambda f: synthesize_speech_xtts(text, lang, voice_sample, f,
                                          pause_ms=450, prep_voice=True)),
        # The two-stage contenders. These are the ones that should actually
        # sound like you; everything above is zero-shot cloning, which has to
        # invent prosody and identity at once.
        ("chatterbox_then_seedvc.wav",
         "Chatterbox, whole text, then Seed-VC conversion to your voice",
         lambda f: convert_voice(
             synthesize_speech(text, lang, voice_sample, _scratch_wav(f), pause_ms=0,
                               cfg_weight=0.5, prep_voice=True),
             vc_target or voice_sample, f)),
        ("xtts_then_seedvc.wav",
         "XTTS-v2, whole text, then Seed-VC conversion to your voice",
         lambda f: convert_voice(
             synthesize_speech_xtts(text, lang, voice_sample, _scratch_wav(f),
                                    pause_ms=0, prep_voice=True),
             vc_target or voice_sample, f)),
    ]

    print()
    print(f"Generating {len(variants)} voice variants into {out}/ ...")
    produced = []
    for filename, description, run in variants:
        target = str(out / filename)
        print()
        print(f"--- {filename}: {description}")
        try:
            run(target)
            produced.append((filename, description))
        except Exception as exc:
            print(f"    skipped: {exc}")

    for stage1 in out.glob("*.stage1.wav"):
        stage1.unlink(missing_ok=True)

    print()
    print("Listen to these and pick the one that sounds most like you:")
    for filename, description in produced:
        print(f"  {out / filename}")
        print(f"      {description}")
    print()
    print("Then generate video with the matching flags "
          "(--tts / --cfg-weight / --no-voice-prep).")


def prepare_voice_sample(voice_sample: str, target_sr: int = 24000,
                         max_seconds: float = 30.0) -> str:
    """Clean up a voice reference clip before it is used for cloning.

    Zero-shot TTS derives the whole speaker identity from this one clip, so
    its defects are inherited by every generated line — this is the single
    highest-leverage input in the pipeline, and a phone recording is usually
    wrong in four ways at once:

    - stereo: the two channels are summed into the speaker embedding as if
      they were one signal, which smears it when they differ at all
    - wrong sample rate: resampled internally anyway, but doing it here with
      a good resampler is better than whatever the model happens to use
    - leading/trailing silence and room tone: modelled as part of the voice
    - over-length: only the first N seconds are actually conditioned on, so
      trailing material is wasted rather than helpful

    Returns the original path unchanged if none of that applies, so callers
    can compare identity before deleting the result (same contract as
    prepare_photo).
    """
    import librosa

    info = sf.info(voice_sample)
    needs_work = (info.channels > 1 or info.samplerate != target_sr
                  or info.duration > max_seconds)

    audio, _ = librosa.load(voice_sample, sr=target_sr, mono=True)

    # top_db=30 trims room tone without eating quiet speech onsets; librosa's
    # default of 60 is lenient enough to leave audible silence in place.
    trimmed, _ = librosa.effects.trim(audio, top_db=30)
    if len(trimmed) == 0:  # a clip that is silent all the way through
        raise ValueError(
            f"{voice_sample} appears to be silent. Record 25-30s of clear speech "
            "in a quiet room, or pass --no-voice-prep to use the file as-is."
        )
    if len(trimmed) < len(audio):
        needs_work = True
    # The TTS reference is capped too, so when the recording is longer than the
    # cap, choose which excerpt to keep rather than defaulting to the opening.
    start = _best_window_start(trimmed, target_sr, max_seconds)
    audio = trimmed[start: start + int(target_sr * max_seconds)]

    peak = float(np.abs(audio).max())
    if peak > 0 and not 0.5 <= peak <= 0.99:
        # Normalise to a consistent headroom. Only when it's actually off:
        # re-scaling an already well-levelled clip just adds a rounding pass.
        audio = audio * (0.95 / peak)
        needs_work = True

    if not needs_work:
        return voice_sample

    duration = len(audio) / target_sr
    if duration < 6.0:
        print(f"Warning: after trimming, {voice_sample} is only {duration:.1f}s of speech. "
              "6-30s is the usable range for voice cloning; under ~10s the clone is "
              "noticeably weaker. Consider recording a longer sample.")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, audio, target_sr)
    print(f"Prepared voice reference: {info.channels}ch/{info.samplerate}Hz/"
          f"{info.duration:.1f}s -> mono/{target_sr}Hz/{duration:.1f}s")
    return tmp.name


def prepare_photo(photo: str, max_dim: int = 1024) -> str:
    """Downscale oversized source photos before handing them to SadTalker.

    SadTalker's crop/align step (src/utils/croper.py) runs face detection
    twice: once on the full image, then again on a tight face-only crop it
    computes itself. On very high-resolution photos (e.g. a modern phone
    selfie, often 3000px+) the crop math produces an off/too-tight second
    crop that the detector then fails on, even though the same photo detects
    fine on its own. Capping the long edge avoids that without any visible
    quality loss for a talking-head video.
    """
    from PIL import Image

    with Image.open(photo) as img:
        if max(img.size) <= max_dim:
            return photo
        img = img.convert("RGB")
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(tmp.name, quality=95)
        return tmp.name


def render_video(
    photo: str,
    audio: str,
    out_video: str,
    enhancer: bool = False,
    still: bool = True,
    size: int = 512,
    expression_scale: float = 1.0,
    preprocess: str = "crop",
    pose_style: int = 0,
) -> None:
    """Run SadTalker to animate the photo in sync with the audio.

    size=512 uses SadTalker's 512px checkpoint instead of its 256px default;
    rendering natively at 512 preserves the source face far better than
    rendering at 256 and letting the enhancer upscale.

    enhancer (GFPGAN) is off by default: it visibly rewrites the face —
    smoothing skin texture and subtly reshaping features — which reads as
    "that's not quite me" even though it looks sharper.
    """
    result_dir = tempfile.mkdtemp(prefix="sadtalker_")
    inference_py = str(Path(SADTALKER_DIR) / "inference.py")

    if not Path(inference_py).exists():
        raise FileNotFoundError(
            f"Couldn't find SadTalker at {SADTALKER_DIR}. "
            "Run setup.sh first, or set the SADTALKER_DIR environment variable."
        )

    prepared_photo = prepare_photo(photo)

    cmd = [
        sys.executable, inference_py,
        "--driven_audio", os.path.abspath(audio),
        "--source_image", os.path.abspath(prepared_photo),
        "--result_dir", result_dir,
        "--preprocess", preprocess,
        "--size", str(size),
        "--expression_scale", str(expression_scale),
        "--pose_style", str(pose_style),
    ]
    if still:
        cmd.append("--still")
    if enhancer:
        cmd += ["--enhancer", "gfpgan"]

    print("Running SadTalker:", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=SADTALKER_DIR, check=True)
    finally:
        if prepared_photo != photo:
            os.remove(prepared_photo)

    produced = sorted(Path(result_dir).glob("*.mp4"))
    if not produced:
        raise RuntimeError("SadTalker did not produce a video - check the log above.")

    out_path = Path(out_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced[-1]), out_path)
    shutil.rmtree(result_dir, ignore_errors=True)
    print(f"Saved video -> {out_path}")


def refine_lipsync(video_path: str, audio_path: str, out_path: str) -> None:
    """Run Wav2Lip on `video_path` to re-draw its mouth region tightly
    synced to `audio_path`, saving the result to `out_path`.

    Optional refinement pass for --engine sadtalker only: SadTalker predicts
    a low-dimensional 3D expression coefficient rather than generating mouth
    pixels from audio, so its lip sync visibly drifts from the audio.
    Wav2Lip re-renders just the mouth region directly from the audio
    waveform, tightening sync at the cost of a slightly lower-res, softer
    mouth patch than SadTalker's own output (it composites a small,
    upsampled crop back into each frame). Not meant to run on top of the
    default latentsync engine's output — LatentSync already regenerates the
    mouth from audio via diffusion at higher fidelity than Wav2Lip's GAN
    patch, so stacking this after it would be a quality regression, not an
    improvement.
    """
    inference_py = Path(WAV2LIP_DIR) / "inference.py"
    checkpoint = Path(WAV2LIP_DIR) / "checkpoints" / "wav2lip_gan.pth"
    if not inference_py.exists():
        raise FileNotFoundError(
            f"Couldn't find Wav2Lip at {WAV2LIP_DIR}. "
            "Run setup_wav2lip.sh first, or set the WAV2LIP_DIR environment variable."
        )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Couldn't find {checkpoint}. setup_wav2lip.sh prints instructions for "
            "downloading wav2lip_gan.pth manually (its host has no stable direct-"
            "download URL) — see README's Wav2Lip section."
        )

    python = get_venv_python(WAV2LIP_VENV)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        python, "inference.py",
        "--checkpoint_path", str(checkpoint),
        "--face", os.path.abspath(video_path),
        "--audio", os.path.abspath(audio_path),
        "--outfile", str(out.resolve()),
    ]
    print("Running Wav2Lip refinement:", " ".join(cmd))
    subprocess.run(cmd, cwd=WAV2LIP_DIR, check=True)
    print(f"Saved refined video -> {out}")


def make_looped_video(photo: str, duration_s: float, fps: int = 25) -> str:
    """Turn a still photo into a static "video" of the given duration.

    LatentSync only masks and regenerates the mouth region of each input
    frame — everything else (eyes, hair, background) is carried through from
    the input essentially unchanged. Feeding it the unwarped source photo
    (instead of SadTalker's re-animated output) keeps the rest of the face
    pixel-faithful; only the mouth gets synced to the audio.
    """
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", os.path.abspath(photo),
        "-c:v", "libx264", "-t", str(duration_s), "-pix_fmt", "yuv420p",
        "-vf", f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2",
        out,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def detect_border_crop(video: str, sample_frames: int = 300) -> str:
    """ffmpeg crop filter that removes uniform black borders, or "" if none.

    Phone recordings are routinely portrait content stored in a landscape
    frame (or the reverse), padded with black bars. Those bars are pure cost:
    they consume most of the width, and since the clip is then downscaled to
    fit LatentSync's memory budget, they shrink the face — the only part that
    matters — by the same factor. Stripping them first spends the whole
    resolution budget on the subject.

    Observed on this project's own test footage: a 1280x720 file whose actual
    content was 404x720, i.e. two thirds of every frame was black.
    """
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", os.path.abspath(video),
         "-vf", "cropdetect=24:2:0", "-frames:v", str(sample_frames), "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    crops = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", result.stderr or "")
    if not crops:
        return ""

    # cropdetect reports per frame; take the most common verdict rather than
    # the last, so one dark frame can't decide the crop for the whole clip.
    from collections import Counter
    w, h, x, y = Counter(crops).most_common(1)[0][0]
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
         os.path.abspath(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    full_w, full_h = (int(v) for v in probe.split("x"))

    # Ignore a crop that barely changes anything; only act on real bars.
    if int(w) >= full_w * 0.95 and int(h) >= full_h * 0.95:
        return ""
    print(f"Source video: cropping black borders {full_w}x{full_h} -> {w}x{h}")
    return f"crop={w}:{h}:{x}:{y}"


def fit_source_video(video: str, duration_s: float, max_dim: int = 768) -> str:
    """Prepare a real recording of the speaker to be lip-synced directly.

    This is the path that exists because animating a still photo does not
    work well. LatentSync regenerates only the mouth and passes everything
    else through, so when the input is genuine footage of the person, the
    output keeps their real head motion, real blinks and real expression —
    none of which a model has to invent, and none of which repeats. Driving a
    photo with a borrowed clip instead produces that clip's mannerisms on a
    loop, which is what "unnatural" usually turns out to mean here.

    Downscaled for the same reason as prepare_photo(max_dim=768): LatentSync
    loads every frame into RAM at once, so a 1080p phone recording exhausts
    system memory long before VRAM becomes the limit. Shorter-than-audio
    clips are ping-pong looped, which is far less noticeable on a 1-2 minute
    recording than on a 3-second one, but recording long enough to avoid
    looping altogether is better still.
    """
    if not Path(video).exists():
        raise FileNotFoundError(f"Source video not found: {video}")

    clip_duration = _video_duration(video)
    crop = detect_border_crop(video)
    scale = ((crop + "," if crop else "") +
             f"scale='min({max_dim},iw)':'min({max_dim},ih)'"
             ":force_original_aspect_ratio=decrease,"
             "scale=trunc(iw/2)*2:trunc(ih/2)*2")

    if clip_duration < duration_s:
        looped = _ping_pong_loop(video, duration_s)
        try:
            out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            subprocess.run(
                ["ffmpeg", "-y", "-i", looped, "-an", "-vf", scale,
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
                check=True, capture_output=True,
            )
        finally:
            os.remove(looped)
        print(f"Source video is {clip_duration:.1f}s for {duration_s:.1f}s of speech - "
              "ping-pong looped. Record a longer clip to avoid the repetition.")
        return out

    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    subprocess.run(
        ["ffmpeg", "-y", "-i", os.path.abspath(video), "-t", str(duration_s),
         "-an", "-vf", scale, "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
        check=True, capture_output=True,
    )
    return out


def make_idle_motion_video(photo: str, duration_s: float, motion_video: str = None,
                           motion_scale: float = 1.0) -> str:
    """Animate the photo with natural idle head motion + blinking, looped
    to match the audio's duration.

    Uses LivePortrait driven by its bundled "d0.mp4" example driving clip.
    LivePortrait's own bundled parametric motion templates (the .pkl files
    under assets/examples/driving/) are from an older schema version and
    fail against the current inference.py/live_portrait_pipeline.py with a
    KeyError (missing 'c_d_eyes_lst') — using an actual driving video makes
    the pipeline extract motion fresh instead of loading a stale format.
    By design (flag_relative_motion, on by default), only *relative* motion
    deltas are transferred from the driving clip onto the source photo — no
    appearance/identity information from the person in d0.mp4 ends up in the
    output, only their head-motion/expression trajectory.

    The clip is short (~3s), so it's ping-pong looped (forward, then
    backward, repeated) to cover arbitrary audio length without a jarring
    jump cut at the loop point, then trimmed to the exact duration.

    `motion_video` overrides the bundled clip with your own recording, and is
    the single biggest realism win available here. Every clip LivePortrait
    ships is an *expression demo* — pulled faces, tongue out, exaggerated
    eyes — because that is what they were recorded to show off. d0.mp4 is
    merely the mildest of them, not a natural idle, and at 3.1s it has to be
    looped ~10x for a 30s line, which reads as robotic repetition. A 15-20s
    clip of you simply sitting still, blinking and shifting slightly needs
    almost no looping and carries your own motion signature.

    `motion_scale` maps to LivePortrait's driving_multiplier, which is active
    by default here (its driving_option defaults to "expression-friendly",
    the mode in which that multiplier applies). Above 1.0 amplifies the
    transferred head/expression motion; use it when the driving clip is
    merely too subtle rather than static.
    """
    inference_py = Path(LIVEPORTRAIT_DIR) / "inference.py"
    if not inference_py.exists():
        raise FileNotFoundError(
            f"Couldn't find LivePortrait at {LIVEPORTRAIT_DIR}. "
            "Run setup_liveportrait.sh first, or set the LIVEPORTRAIT_DIR environment variable."
        )

    if motion_video:
        driving_clip = Path(motion_video)
        if not driving_clip.exists():
            raise FileNotFoundError(f"Motion video not found: {motion_video}")
    else:
        driving_clip = Path(LIVEPORTRAIT_DIR) / "assets" / "examples" / "driving" / "d0.mp4"

    # LivePortrait runs its full pipeline on every frame of the driving video,
    # at the source photo's resolution, so cost and peak RAM scale with the
    # clip's length rather than with the output's. A 20s 30fps recording is
    # ~600 frames where the bundled 3s clip is 78, which is enough to push a
    # 32GB machine into swapping mid-run. Anything past the audio's duration
    # is discarded by the loop/trim step below anyway, so trim first.
    trimmed_clip = None
    if _video_duration(driving_clip) > duration_s:
        trimmed_clip = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(driving_clip.resolve()), "-t", str(duration_s),
             "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", trimmed_clip],
            check=True, capture_output=True,
        )
        driving_clip = Path(trimmed_clip)

    python = get_venv_python(LIVEPORTRAIT_VENV)
    result_dir = tempfile.mkdtemp(prefix="liveportrait_")

    cmd = [
        python, "inference.py",
        "-s", os.path.abspath(photo),
        "-d", str(driving_clip.resolve()),
        "-o", result_dir,
    ]
    if motion_video:
        # LivePortrait's flag_crop_driving_video defaults to False, which is
        # right for its own pre-cropped example clips and wrong for a raw
        # phone recording — without it the face is never located in frame.
        cmd += ["--flag-crop-driving-video"]
    if motion_scale != 1.0:
        cmd += ["--driving-multiplier", str(motion_scale)]
    print("Running LivePortrait:", " ".join(cmd))
    try:
        # LivePortrait's rich-based progress bar writes an emoji directly via
        # Windows' legacy console API when it detects an interactive terminal,
        # which crashes with UnicodeEncodeError on a non-UTF-8 console locale
        # (e.g. Windows set to Japanese/cp932). Capturing output makes it detect
        # a non-interactive stream and fall back to plain-text progress instead.
        result = subprocess.run(cmd, cwd=LIVEPORTRAIT_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise subprocess.CalledProcessError(result.returncode, cmd)

        # Output naming is deterministic ({source_stem}--{driving_stem}.mp4,
        # see LivePortrait's src/live_portrait_pipeline.py), but glob as a
        # safety net against version drift in that convention.
        produced = [p for p in Path(result_dir).glob("*.mp4") if "_concat" not in p.name]
        if not produced:
            raise RuntimeError("LivePortrait did not produce a video - check the log above.")

        return _ping_pong_loop(str(produced[0]), duration_s)
    finally:
        shutil.rmtree(result_dir, ignore_errors=True)
        if trimmed_clip:
            os.remove(trimmed_clip)


def _video_duration(video) -> float:
    """Duration of `video` in seconds, via ffprobe."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(probe.stdout.strip())


def _concat_quote(path: str) -> str:
    """Quote a path for an ffmpeg concat list.

    The list wraps each path in single quotes, so a quote inside the path
    (an apostrophe in a user folder name, say) has to be closed, escaped and
    reopened, or ffmpeg misreads the rest of the line.
    """
    return "'" + os.path.abspath(path).replace("'", "'\\''") + "'"


def _ping_pong_loop(video: str, duration_s: float) -> str:
    """Extend `video` to `duration_s` by alternating forward/reversed
    playback (so the loop point doesn't jump), then trim to length."""
    clip_duration = _video_duration(video)
    reps = max(1, int(duration_s // clip_duration) + 2)  # +2: one for the reverse half, one for rounding

    reversed_clip = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-vf", "reverse", "-an", reversed_clip],
        check=True, capture_output=True,
    )

    concat_list = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8")
    for _ in range(reps):
        concat_list.write(f"file {_concat_quote(video)}\n")
        concat_list.write(f"file {_concat_quote(reversed_clip)}\n")
    concat_list.close()

    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list.name,
             "-t", str(duration_s), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", out],
            check=True, capture_output=True,
        )
    finally:
        os.remove(reversed_clip)
        os.remove(concat_list.name)
    return out


def get_venv_python(venv_dir: str) -> str:
    for candidate in ("Scripts/python.exe", "bin/python"):
        p = Path(venv_dir) / candidate
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"Couldn't find a Python executable in venv {venv_dir}. Run the matching setup_*.sh script first."
    )


def run_latentsync(
    video: str,
    audio: str,
    out_video: str,
    inference_steps: int = 20,
    guidance_scale: float = 1.5,
    resolution: int = 256,
) -> None:
    """Run LatentSync 1.5 to lip-sync `video` to `audio`.

    stage2.yaml and stage2_512.yaml are byte-identical apart from their
    `resolution:` field, so the same checkpoint loads under either and
    `resolution` just selects which config to pass. That is not a promise
    that 512 looks better: the checkpoint installed by setup_latentsync.sh is
    LatentSync 1.5, trained at 256 — running it at 512 quadruples the VRAM
    for the mouth region and may simply upsample its way to a softer result.
    1.6's checkpoint is the one actually trained at 512. Treat this as an
    A/B knob to try on your own footage, not a quality setting to raise.
    """
    inference_module_marker = Path(LATENTSYNC_DIR) / "scripts" / "inference.py"
    if not inference_module_marker.exists():
        raise FileNotFoundError(
            f"Couldn't find LatentSync at {LATENTSYNC_DIR}. "
            "Run setup_latentsync.sh first, or set the LATENTSYNC_DIR environment variable."
        )

    python = get_venv_python(LATENTSYNC_VENV)
    out_path = Path(out_video).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        python, "-m", "scripts.inference",
        "--unet_config_path", f"configs/unet/stage2{'_512' if resolution == 512 else ''}.yaml",
        "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
        "--video_path", os.path.abspath(video),
        "--audio_path", os.path.abspath(audio),
        "--video_out_path", str(out_path),
        "--inference_steps", str(inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--enable_deepcache",
    ]

    print("Running LatentSync:", " ".join(cmd))
    subprocess.run(cmd, cwd=LATENTSYNC_DIR, check=True)
    print(f"Saved video -> {out_path}")


def render_infinitetalk(
    photo: str,
    audio: str,
    out_video: str,
    scene_prompt: str,
    size: str = "480",
    sample_steps: int = 40,
    quant: str = "fp8",
    low_vram: bool = True,
    mode: str = "streaming",
    accel: str = "none",
    seed: int = 42,
) -> None:
    """Run InfiniteTalk to generate lip sync, head motion AND facial
    expression together from a single photo + audio + text scene prompt.

    Unlike sadtalker/latentsync, this is a self-contained engine: no
    separate driving-video or mouth-only refinement step. `scene_prompt`
    describes the desired delivery (e.g. a confident business speech) —
    InfiniteTalk (built on Wan2.1-I2V-14B) conditions on it the same way
    a text-to-video model would, so it actually shapes expression/motion,
    not just the mouth.

    `quant="fp8"` + `low_vram=True` (both on by default) trade speed for
    fitting this 14B-parameter model into a 12GB-class GPU — expect this to
    be noticeably slower than sadtalker/latentsync.

    The subprocess runs infinitetalk_run.py (ours) rather than InfiniteTalk's
    generate_infinitetalk.py directly: it applies a set of measured runtime
    patches (Windows commit-limit loader, transformers 5 and Python 3.11
    drift, RTX 50-series kernels, GPU allocator cap, lazy T5/CLIP) and then
    hands every argument through unchanged. See that file for the details.

    `accel="lightx2v"` loads the lightx2v step-distillation LoRA with the
    sampling settings InfiniteTalk's README gives for it (text CFG 1, audio
    CFG 2, shift 2; `sample_steps` should then be ~4). That is 2 DiT passes
    per step instead of 3, and 4 steps instead of 40.
    """
    inference_py = Path(INFINITETALK_DIR) / "generate_infinitetalk.py"
    worker = Path(__file__).parent / "infinitetalk_run.py"
    ckpt_dir = Path(INFINITETALK_DIR) / "weights" / "Wan2.1-I2V-14B-480P"
    wav2vec_dir = Path(INFINITETALK_DIR) / "weights" / "chinese-wav2vec2-base"
    infinitetalk_dir = Path(INFINITETALK_DIR) / "weights" / "InfiniteTalk" / "single" / "infinitetalk.safetensors"
    quant_dir = Path(INFINITETALK_DIR) / "weights" / "InfiniteTalk" / "quant_models" / f"infinitetalk_single_{quant}.safetensors"

    if not inference_py.exists():
        raise FileNotFoundError(
            f"Couldn't find InfiniteTalk at {INFINITETALK_DIR}. "
            "Run setup_infinitetalk.sh first, or set the INFINITETALK_DIR environment variable."
        )
    if not ckpt_dir.exists() or not wav2vec_dir.exists() or not infinitetalk_dir.exists():
        raise FileNotFoundError(
            f"Missing InfiniteTalk checkpoints under {Path(INFINITETALK_DIR) / 'weights'}. "
            "Run setup_infinitetalk.sh to download them (~35-40GB)."
        )

    python = get_venv_python(INFINITETALK_VENV)
    out = Path(out_video).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # InfiniteTalk refuses audio that isn't longer than one 81-frame chunk
    # (3.24s at 25 fps): wan/multitalk.py silently drops any embedding with
    # <= 81 frames and then asserts. Short lines are normal here, so pad with
    # silence to just over a chunk and trim the video back afterwards.
    speech_s = sf.info(audio).duration
    padded_audio = None
    if speech_s < INFINITETALK_MIN_AUDIO_S:
        data, sr = sf.read(audio, always_2d=True)
        pad = np.zeros((int((INFINITETALK_MIN_AUDIO_S - speech_s) * sr) + 1, data.shape[1]), dtype=data.dtype)
        padded_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(padded_audio, np.concatenate([data, pad]), sr)
        print(f"Padding {speech_s:.2f}s of speech with silence to {INFINITETALK_MIN_AUDIO_S}s - "
              "InfiniteTalk needs more than one 81-frame chunk of audio; the video is trimmed back.",
              flush=True)
        audio = padded_audio

    # InfiniteTalk takes its inputs as a JSON file, not individual CLI flags
    # for photo/audio — schema per its examples/single_example_image.json.
    # Paths go in with forward slashes: generate_infinitetalk.py derives a
    # scratch-directory name from cond_video with split('/'), which on a
    # backslashed Windows path yields the whole path and lands that scratch
    # directory next to the photo instead of under --audio_save_dir.
    input_json = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8")
    json.dump({
        "prompt": scene_prompt,
        "cond_video": Path(photo).resolve().as_posix(),
        "cond_audio": {"person1": Path(audio).resolve().as_posix()},
    }, input_json)
    input_json.close()

    # --save_file takes a base name (no extension); InfiniteTalk appends
    # ".mp4" itself. Use a temp stem so we control the final destination.
    save_stem = tempfile.NamedTemporaryFile(suffix="", delete=False).name
    os.remove(save_stem)  # only need a unique name, not the empty file itself

    # Its audio embeddings and resampled wav would otherwise accumulate under
    # InfiniteTalk/save_audio/ across runs.
    audio_scratch = tempfile.mkdtemp(prefix="infinitetalk_audio_")

    cmd = [
        python, str(worker),
        "--ckpt_dir", str(ckpt_dir),
        "--wav2vec_dir", str(wav2vec_dir),
        "--infinitetalk_dir", str(infinitetalk_dir),
        "--input_json", input_json.name,
        "--audio_save_dir", audio_scratch,
        "--size", f"infinitetalk-{size}",
        "--sample_steps", str(sample_steps),
        "--mode", mode,
        "--motion_frame", "9",
        "--base_seed", str(seed),
        "--save_file", save_stem,
    ]
    if quant == "fp8":
        cmd += ["--quant", quant, "--quant_dir", str(quant_dir)]
    if low_vram:
        cmd += ["--num_persistent_param_in_dit", "0"]
    if accel == "lightx2v":
        lora = Path(INFINITETALK_DIR) / "weights" / "lora" / INFINITETALK_LIGHTX2V_LORA
        if not lora.exists():
            raise FileNotFoundError(
                f"Missing the lightx2v LoRA at {lora}. Re-run setup_infinitetalk.sh to download it.")
        cmd += ["--lora_dir", str(lora), "--lora_scale", "1.0",
                "--sample_text_guide_scale", "1.0", "--sample_audio_guide_scale", "2.0",
                "--sample_shift", "2"]

    print("Running InfiniteTalk:", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=INFINITETALK_DIR, check=True)
    finally:
        os.remove(input_json.name)
        shutil.rmtree(audio_scratch, ignore_errors=True)
        if padded_audio:
            os.remove(padded_audio)

    produced = Path(f"{save_stem}.mp4")
    if not produced.exists():
        raise RuntimeError("InfiniteTalk did not produce a video - check the log above.")
    if padded_audio:
        # A prefix cut needs no re-encode: every packet before -t is decodable.
        trimmed = f"{save_stem}-trimmed.mp4"
        subprocess.run(["ffmpeg", "-y", "-i", str(produced), "-t", f"{speech_s:.3f}", "-c", "copy", trimmed],
                       check=True, capture_output=True)
        os.remove(produced)
        produced = Path(trimmed)
    shutil.move(str(produced), out)
    print(f"Saved video -> {out}")


def make_one(photo: str, voice_sample: str, text: str, lang: str, out_video: str,
             source_video: str = None,
             engine: str = "latentsync", pause_ms: int = 0, motion: str = "idle",
             refine_lipsync_pass: bool = False, cfg_weight: float = 0.5,
             exaggeration: float = 0.5, tts: str = "chatterbox",
             prep_voice: bool = True, motion_video: str = None,
             motion_scale: float = 1.0, keep_intermediates: bool = False,
             keep_tts_warm: bool = False, voice_convert: bool = False,
             vc_target: str = None, vc_steps: int = 25, vc_denoise: float = 0.6,
             **engine_opts) -> None:
    kept = []  # intermediates to report instead of delete, when asked

    def discard(path: str, label: str) -> None:
        """Delete a temp file, or keep and announce it under --keep-intermediates.

        The intermediates are where a quality problem is actually diagnosable:
        the driving video shows whether LivePortrait moved the head at all,
        the wav shows whether the voice clone is the problem. Both are
        normally deleted before anyone can look at them.
        """
        if keep_intermediates:
            kept.append((label, path))
        else:
            try:
                os.remove(path)
            except OSError:
                pass

    if engine != "latentsync" and (motion_video or motion_scale != 1.0):
        print(
            f"--motion-video/--motion-scale have no effect with --engine {engine}: "
            "they drive LivePortrait, which only runs in the latentsync pipeline; "
            "ignoring."
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        synthesize(tts, text, lang, voice_sample, tmp_wav, pause_ms=pause_ms,
                   cfg_weight=cfg_weight, exaggeration=exaggeration,
                   prep_voice=prep_voice)
        if not keep_tts_warm:
            # Speech is done; everything below is the video stage, which wants
            # every byte of VRAM it can get. run_batch keeps it warm instead,
            # since it synthesizes again for the next line.
            release_tts_model()

        if voice_convert:
            # Seed-VC's reference can be longer and messier than the TTS
            # prompt (it only has to establish identity, not be cloned from),
            # so vc_target defaults to the TTS voice sample but is worth
            # pointing at a longer recording when one exists.
            converted = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            convert_voice(tmp_wav, vc_target or voice_sample, converted,
                          diffusion_steps=vc_steps, denoise_ref=vc_denoise)
            discard(tmp_wav, "TTS speech before voice conversion")
            tmp_wav = converted

        if engine == "sadtalker":
            if refine_lipsync_pass:
                sadtalker_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                try:
                    render_video(photo, tmp_wav, sadtalker_out, **engine_opts)
                    refine_lipsync(sadtalker_out, tmp_wav, out_video)
                finally:
                    discard(sadtalker_out, "SadTalker render (before Wav2Lip refinement)")
            else:
                render_video(photo, tmp_wav, out_video, **engine_opts)
        elif engine == "infinitetalk":
            if refine_lipsync_pass:
                print(
                    "--refine-lipsync has no effect with --engine infinitetalk: it already "
                    "generates the mouth from audio at higher fidelity than Wav2Lip's patch "
                    "would; skipping."
                )
            # InfiniteTalk's cond_video field takes an image OR a video; with
            # real footage it re-drives the person's own recording instead of
            # animating a still, which is the mode worth using.
            render_infinitetalk(source_video or photo, tmp_wav, out_video, **engine_opts)
        else:
            if refine_lipsync_pass:
                print(
                    "--refine-lipsync has no effect with --engine latentsync (the default): "
                    "LatentSync already regenerates the mouth from audio at higher fidelity "
                    "than Wav2Lip's patch would; skipping."
                )
            # LatentSync loads every frame of the input video into memory at
            # once before processing. A long clip at a phone photo's full
            # resolution (e.g. 2316x3088) can exhaust system RAM well before
            # GPU VRAM becomes the limit — downscale first.
            duration_s = sf.info(tmp_wav).duration
            if source_video:
                # Real footage of the speaker: no motion to synthesize, so
                # LivePortrait is skipped entirely and LatentSync syncs the
                # mouth onto the person's own recording.
                prepared_photo = photo
                driving_video = fit_source_video(source_video, duration_s)
                try:
                    run_latentsync(driving_video, tmp_wav, out_video, **engine_opts)
                finally:
                    discard(driving_video, "prepared source video")
                return
            prepared_photo = prepare_photo(photo, max_dim=768)
            if motion == "idle":
                driving_video = make_idle_motion_video(
                    prepared_photo, duration_s,
                    motion_video=motion_video, motion_scale=motion_scale,
                )
            else:
                driving_video = make_looped_video(prepared_photo, duration_s)
            try:
                run_latentsync(driving_video, tmp_wav, out_video, **engine_opts)
            finally:
                discard(driving_video, "driving video (head motion before lip-sync)")
                if prepared_photo != photo:
                    os.remove(prepared_photo)
    finally:
        discard(tmp_wav, "synthesized speech")
        for label, path in kept:
            print(f"Kept {label}: {path}")


def run_batch(config_path: str, engine: str = "latentsync", pause_ms: int = 0,
              source_video: str = None,
              motion: str = "idle", refine_lipsync_pass: bool = False,
              cfg_weight: float = 0.5, exaggeration: float = 0.5,
              tts: str = "chatterbox", prep_voice: bool = True,
              motion_video: str = None, motion_scale: float = 1.0,
              keep_intermediates: bool = False, voice_convert: bool = False,
              vc_target: str = None, vc_steps: int = 25, vc_denoise: float = 0.6,
              **engine_opts) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    photo = cfg.get("photo")
    source_video = cfg.get("video", source_video)
    voice_sample = cfg.get("voice_sample")
    lines = cfg.get("lines")
    if not (photo or source_video):
        raise SystemExit(f"{config_path}: set 'photo' (or 'video').")
    if not voice_sample:
        raise SystemExit(f"{config_path}: set 'voice_sample'.")
    if not isinstance(lines, list) or not lines:
        raise SystemExit(f"{config_path}: 'lines' must be a non-empty list.")
    for i, line in enumerate(lines, 1):
        if not isinstance(line, dict) or not str(line.get("text", "")).strip():
            raise SystemExit(f"{config_path}: line {i} needs a 'text' field.")
    # CLI flags OR with matching top-level config.yaml keys (applies to the
    # whole batch, same as engine/motion/pause_ms above).
    refine_lipsync_pass = refine_lipsync_pass or bool(cfg.get("refine_lipsync", False))
    cfg_weight = cfg.get("cfg_weight", cfg_weight)
    exaggeration = cfg.get("exaggeration", exaggeration)
    tts = cfg.get("tts", tts)
    voice_convert = cfg.get("voice_convert", voice_convert)
    vc_target = cfg.get("vc_target", vc_target)
    motion_video = cfg.get("motion_video", motion_video)
    motion_scale = cfg.get("motion_scale", motion_scale)
    if engine == "infinitetalk" and "scene_prompt" in cfg:
        engine_opts["scene_prompt"] = cfg["scene_prompt"]

    for i, line in enumerate(lines, 1):
        text = line["text"]
        lang = line.get("lang", "en")
        out = line.get("output", f"output/line_{i}.mp4")
        print(f"\n[{i}/{len(lines)}] ({lang}) {text[:60]}{'...' if len(text) > 60 else ''}")
        make_one(photo, voice_sample, text, lang, out, source_video=source_video,
                 engine=engine, pause_ms=pause_ms,
                 motion=motion, refine_lipsync_pass=refine_lipsync_pass,
                 cfg_weight=cfg_weight, exaggeration=exaggeration, tts=tts,
                 prep_voice=prep_voice, motion_video=motion_video,
                 motion_scale=motion_scale, keep_intermediates=keep_intermediates,
                 keep_tts_warm=True, voice_convert=voice_convert,
                 vc_target=vc_target, vc_steps=vc_steps, vc_denoise=vc_denoise,
                 **engine_opts)

    release_tts_model()


def main():
    parser = argparse.ArgumentParser(description="Generate talking-avatar videos from text.")
    parser.add_argument("--config", help="YAML file with a batch of pre-entered lines")
    parser.add_argument("--photo", help="Path to your face photo")
    parser.add_argument("--video",
                        help="A real video of you instead of a photo - 1-2 min, front-facing, "
                             "NOT talking (just looking at the camera, blinking, small "
                             "movements). Your own head motion and expression are kept and "
                             "only the mouth is re-synced, which avoids the looped borrowed "
                             "mannerisms you get when animating a still photo. Works with "
                             "--engine latentsync (skips LivePortrait) and --engine "
                             "infinitetalk (video-to-video)")
    parser.add_argument("--voice", help="Path to your voice sample (6-30s, clean, wav/mp3)")
    parser.add_argument("--text", help="Text to speak")
    parser.add_argument("--lang", default="en", help="Language code (see README section 5)")
    parser.add_argument("--out", help="Output video path")

    parser.add_argument("--engine", default="latentsync",
                        choices=["latentsync", "sadtalker", "infinitetalk"],
                        help="latentsync (default): only regenerates the mouth, keeps the rest of "
                             "the face pixel-faithful to the source photo, no head/eye motion. "
                             "sadtalker: full-face reenactment with head motion/blinks, but visibly "
                             "reshapes the face (eyes, etc.) and has weaker lip sync. infinitetalk: "
                             "generates lip sync, head motion AND facial expression together from "
                             "audio + a text scene prompt (--scene-prompt) - most realistic, but a "
                             "14B-parameter model, much slower and requires setup_infinitetalk.sh")
    parser.add_argument("--tts", default="chatterbox", choices=["chatterbox", "xtts"],
                        help="Text-to-speech engine. chatterbox (default): 23 languages. "
                             "xtts: XTTS-v2, 17 languages, often clones timbre more "
                             "closely - needs ./setup_xtts.sh")
    parser.add_argument("--voice-convert", action="store_true",
                        help="Add a Seed-VC pass after the TTS that converts the generated "
                             "speech to your voice. Two stages beat zero-shot cloning alone: "
                             "the TTS handles natural delivery, Seed-VC handles identity. "
                             "Needs ./setup_seedvc.sh")
    parser.add_argument("--vc-target",
                        help="Reference recording for --voice-convert (default: whatever "
                             "--voice is). Point this at a longer recording of yourself if "
                             "you have one - it only has to establish identity, so it can be "
                             "longer and less pristine than the TTS reference")
    parser.add_argument("--vc-denoise", type=float, default=0.6,
                        help="How hard to denoise the Seed-VC reference before conversion "
                             "(0 = off, 0.6 = default, 1.0 = maximum). Conversion copies the "
                             "reference's room tone onto every line along with the voice, so "
                             "cleaning it here removes background noise at its source. "
                             "0.6 was chosen by ear; higher values measure better on both "
                             "noise floor and speaker similarity but start sounding "
                             "over-processed. Raise it if noise still gets through")
    parser.add_argument("--vc-steps", type=int, default=25,
                        help="Seed-VC diffusion steps (default 25). Higher is cleaner and "
                             "slower; 30-50 is upstream's suggestion for singing")
    parser.add_argument("--no-voice-prep", action="store_true",
                        help="Use the voice sample exactly as given. By default it is "
                             "downmixed to mono, resampled, silence-trimmed and level-matched "
                             "first, which usually improves the clone noticeably")
    parser.add_argument("--voice-compare", action="store_true",
                        help="Generate the same line with every voice setting into "
                             "output/voice_ab/<clip-name>/ and exit without rendering video - "
                             "use this to pick a voice before paying for the slow video step")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="Keep (and print the paths of) the synthesized speech and the "
                             "driving video instead of deleting them - the two files you "
                             "need to diagnose a voice or head-motion problem")
    parser.add_argument("--pause-ms", type=int, default=0,
                        help="0 (default): send the whole text to the TTS in one call and let it "
                             "place its own pauses, carrying intonation across sentence boundaries. "
                             "Above 0: split into sentences, synthesize each separately and splice "
                             "this much silence between them - exact pause control, but every "
                             "sentence restarts at neutral intonation and it sounds mechanical")
    parser.add_argument("--motion-video",
                        help="[latentsync] Your own 15-20s video of head idling/blinking, used "
                             "to drive the photo instead of LivePortrait's bundled 3s clip. "
                             "Biggest realism win available: real motion, and long enough to "
                             "need almost no looping")
    parser.add_argument("--motion-scale", type=float, default=1.0,
                        help="[latentsync] Amplify transferred motion (LivePortrait's "
                             "driving_multiplier). Use sparingly: it amplifies the deviation "
                             "from your source photo, so past ~1.2 the face visibly stops "
                             "looking like you. Prefer a better --motion-video")
    parser.add_argument("--motion", default="idle", choices=["idle", "none"],
                        help="[latentsync] idle (default): animate the photo with natural head "
                             "motion/blinking via LivePortrait before lip-syncing, instead of a "
                             "frozen frame. none: the old frozen-photo-loop behavior - use this if "
                             "idle motion introduces visible identity drift")
    parser.add_argument("--refine-lipsync", action="store_true",
                        help="[sadtalker] Run an extra Wav2Lip pass to re-draw the mouth region "
                             "tightly synced to the audio - fixes SadTalker's weaker lip sync at "
                             "the cost of a slightly softer, lower-res mouth patch. No effect with "
                             "the default latentsync engine or with infinitetalk (both already sync "
                             "the mouth at higher fidelity). Requires setup_wav2lip.sh")
    parser.add_argument("--cfg-weight", type=float, default=0.5,
                        help="[Chatterbox] CFG weight, 0.0-1.0 (default 0.5). Lower values (try "
                             "0.0-0.3) reduce the cloned voice's tendency to keep the reference "
                             "clip's accent - see README's Voice accent & delivery section")
    parser.add_argument("--exaggeration", type=float, default=0.5,
                        help="[Chatterbox] Delivery intensity/emotion, 0.25-2.0 (default 0.5, "
                             "neutral). Higher = more animated and also faster pacing - pair with a "
                             "lower --cfg-weight to compensate if it starts to sound rushed")

    # LatentSync-only options
    parser.add_argument("--inference-steps", type=int, default=20,
                        help="[latentsync] Diffusion steps, 20-50. Higher = better quality, slower")
    parser.add_argument("--latentsync-res", type=int, default=256, choices=[256, 512],
                        help="[latentsync] Resolution the mouth region is regenerated at. The "
                             "installed 1.5 checkpoint was trained at 256; 512 costs 4x the "
                             "VRAM and is not reliably sharper with these weights - A/B it")
    parser.add_argument("--guidance-scale", type=float, default=1.5,
                        help="[latentsync] Lip-sync accuracy vs. stability, 1.0-3.0. Higher can cause jitter")

    # SadTalker-only options
    parser.add_argument("--size", type=int, default=512, choices=[256, 512],
                        help="[sadtalker] Render resolution (512 keeps your face far more faithfully)")
    parser.add_argument("--enhancer", action="store_true",
                        help="[sadtalker] Run GFPGAN face enhancement. Sharper, but visibly reshapes/smooths the face")
    parser.add_argument("--expression-scale", type=float, default=1.0,
                        help="[sadtalker] Mouth/expression intensity. Try 1.2-1.5 if lip movement looks too subtle")
    parser.add_argument("--preprocess", default="crop", choices=["crop", "extcrop", "resize", "full", "extfull"],
                        help="[sadtalker] 'crop' = face only; 'full' = animate the face back into the whole original photo")
    parser.add_argument("--pose-style", type=int, default=0, help="[sadtalker] Head pose style, 0-45")
    parser.add_argument("--sadtalker-motion", action="store_true",
                        help="[sadtalker] Allow head movement (default keeps the head still)")

    # InfiniteTalk-only options
    DEFAULT_SCENE_PROMPT = (
        "A person confidently delivering a business speech to an audience, natural "
        "professional facial expressions, subtle head movements, direct eye contact "
        "with the camera, business attire, neutral office background."
    )
    parser.add_argument("--scene-prompt", default=DEFAULT_SCENE_PROMPT,
                        help="[infinitetalk] Text description of the desired delivery/scene - the "
                             "main lever for expression that 'fits the situation' (default: a "
                             "generic business-speech description)")
    parser.add_argument("--infinitetalk-size", default="480", choices=["480", "720"],
                        help="[infinitetalk] Render resolution, 480p (default) or 720p - 720p needs "
                             "significantly more VRAM/time")
    parser.add_argument("--infinitetalk-steps", type=int, default=None,
                        help="[infinitetalk] Diffusion sample steps (default 40, or 4 with "
                             "--infinitetalk-accel lightx2v). Higher = better quality, slower")
    parser.add_argument("--infinitetalk-seed", type=int, default=42,
                        help="[infinitetalk] Sampling seed (default 42, InfiniteTalk's own default). "
                             "The same photo/audio/seed reproduces the same video; change it to re-roll.")
    parser.add_argument("--infinitetalk-accel", default="none", choices=["none", "lightx2v"],
                        help="[infinitetalk] lightx2v: step-distillation LoRA, 4 steps with 2 "
                             "model passes each instead of 40 x 3 - roughly 15x faster")
    parser.add_argument("--infinitetalk-quant", default="fp8", choices=["fp8", "none"],
                        help="[infinitetalk] fp8 (default): quantized model, needed to fit a "
                             "12GB-class GPU. none: full precision, needs significantly more VRAM "
                             "and the ~32GB of Wan2.1 diffusion shards setup_infinitetalk.sh "
                             "downloads. (InfiniteTalk also publishes an int8 DiT, but no int8 "
                             "T5 to go with it, so it can't be loaded as-is.)")
    parser.add_argument("--infinitetalk-no-low-vram", action="store_true",
                        help="[infinitetalk] Disable CPU offloading (--num_persistent_param_in_dit "
                             "0 is on by default for 12GB-class GPUs) - only if you have VRAM to spare")
    parser.add_argument("--infinitetalk-mode", default="streaming", choices=["streaming", "clip"],
                        help="[infinitetalk] streaming (default): supports longer audio. clip: "
                             "single-chunk generation")
    args = parser.parse_args()

    if args.voice_compare:
        if not (args.voice and args.text):
            parser.error("--voice-compare needs --voice and --text.")
        try:
            voice_compare(args.voice, args.text, args.lang, pause_ms=args.pause_ms,
                          vc_target=args.vc_target)
        finally:
            cleanup_prepared_voices()
        return

    if args.engine == "sadtalker":
        engine_opts = dict(
            size=args.size,
            enhancer=args.enhancer,
            expression_scale=args.expression_scale,
            preprocess=args.preprocess,
            pose_style=args.pose_style,
            still=not args.sadtalker_motion,
        )
    elif args.engine == "infinitetalk":
        engine_opts = dict(
            scene_prompt=args.scene_prompt,
            size=args.infinitetalk_size,
            sample_steps=args.infinitetalk_steps or (4 if args.infinitetalk_accel == "lightx2v" else 40),
            accel=args.infinitetalk_accel,
            seed=args.infinitetalk_seed,
            quant=args.infinitetalk_quant,
            low_vram=not args.infinitetalk_no_low_vram,
            mode=args.infinitetalk_mode,
        )
    else:
        engine_opts = dict(
            inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            resolution=args.latentsync_res,
        )

    shared_opts = dict(
        source_video=args.video,
        engine=args.engine, pause_ms=args.pause_ms, motion=args.motion,
        refine_lipsync_pass=args.refine_lipsync, cfg_weight=args.cfg_weight,
        exaggeration=args.exaggeration, tts=args.tts,
        prep_voice=not args.no_voice_prep, motion_video=args.motion_video,
        motion_scale=args.motion_scale, keep_intermediates=args.keep_intermediates,
        voice_convert=args.voice_convert, vc_target=args.vc_target,
        vc_steps=args.vc_steps, vc_denoise=args.vc_denoise,
    )

    try:
        if args.config:
            run_batch(args.config, **shared_opts, **engine_opts)
        elif (args.photo or args.video) and args.voice and args.text and args.out:
            make_one(args.photo, args.voice, args.text, args.lang, args.out,
                     **shared_opts, **engine_opts)
        else:
            parser.error("Either --config, or --voice --text --out plus one of "
                         "--photo / --video, are required.")
    finally:
        cleanup_prepared_voices()


if __name__ == "__main__":
    main()
