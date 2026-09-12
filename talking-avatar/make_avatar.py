#!/usr/bin/env python3
"""
make_avatar.py — Turn a photo + your cloned voice into a talking-head video.

Pipeline:
  1. Text  -> speech in your voice   (Chatterbox Multilingual TTS)
  2. Photo + speech -> lip-synced video (SadTalker)

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
INFINITETALK_DIR = os.environ.get("INFINITETALK_DIR", str(Path(__file__).parent / "InfiniteTalk"))
INFINITETALK_VENV = os.environ.get("INFINITETALK_VENV", str(Path(__file__).parent / ".venv-infinitetalk"))

# XTTS-v2, the alternative TTS behind --tts xtts. Created by setup_xtts.sh.
XTTS_VENV = os.environ.get("XTTS_VENV", str(Path(__file__).parent / ".venv-xtts"))

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


def synthesize_speech(text: str, lang: str, voice_sample: str, out_wav: str, pause_ms: int = 450,
                       cfg_weight: float = 0.5, exaggeration: float = 0.5,
                       prep_voice: bool = True) -> str:
    """Generate speech audio in the cloned voice, in the given language.

    Chatterbox has no explicit "pause here" control — sent as one big
    generate() call, a multi-sentence paragraph comes out as one continuous,
    breathless read with no gap between sentences. Splitting on sentence
    boundaries and inserting explicit silence between the synthesized clips
    gives direct control over pacing instead of hoping the model infers it.

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

    clips = [
        model.generate(sentence, language_id=lang, audio_prompt_path=voice_sample,
                       cfg_weight=cfg_weight, exaggeration=exaggeration).squeeze().cpu().numpy()
        for sentence in split_sentences(text)
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
                           pause_ms: int = 450, prep_voice: bool = True) -> str:
    """Generate speech with XTTS-v2 instead of Chatterbox.

    Same contract as synthesize_speech(): same sentence splitting, same
    --pause-ms splicing. The knobs differ — XTTS has no cfg_weight or
    exaggeration equivalent, so those arguments are deliberately absent
    rather than silently ignored.

    Runs in .venv-xtts as a subprocess (see CLAUDE.md's cross-venv isolation
    rule); coqui-tts and chatterbox-tts cannot coexist in one process.
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
            "sentences": split_sentences(text),
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


def synthesize(tts: str, text: str, lang: str, voice_sample: str, out_wav: str,
               pause_ms: int = 450, cfg_weight: float = 0.5,
               exaggeration: float = 0.5, prep_voice: bool = True) -> str:
    """Dispatch to the selected TTS backend (--tts)."""
    if tts == "xtts":
        return synthesize_speech_xtts(text, lang, voice_sample, out_wav,
                                      pause_ms=pause_ms, prep_voice=prep_voice)
    return synthesize_speech(text, lang, voice_sample, out_wav, pause_ms=pause_ms,
                             cfg_weight=cfg_weight, exaggeration=exaggeration,
                             prep_voice=prep_voice)


def voice_compare(voice_sample: str, text: str, lang: str,
                  out_dir: str = "output/voice_ab", pause_ms: int = 450) -> None:
    """Render the same line through every available voice setting, and stop.

    Choosing a voice is a listening decision that needs several candidates
    side by side, but the video step costs minutes per line while the audio
    costs seconds — so there is no reason to render video while deciding.
    Variants that can't run (XTTS not installed, or not supporting this
    language) are skipped with a note rather than aborting the rest.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    variants = [
        # (filename, description, callable)
        ("chatterbox_raw_cfg0.5.wav",
         "Chatterbox, reference clip as-is, default settings",
         lambda f: synthesize_speech(text, lang, voice_sample, f, pause_ms=pause_ms,
                                     cfg_weight=0.5, prep_voice=False)),
        ("chatterbox_prepped_cfg0.5.wav",
         "Chatterbox, cleaned reference clip, default settings",
         lambda f: synthesize_speech(text, lang, voice_sample, f, pause_ms=pause_ms,
                                     cfg_weight=0.5, prep_voice=True)),
        ("chatterbox_prepped_cfg0.2.wav",
         "Chatterbox, cleaned clip, low cfg-weight (less accent carry-over)",
         lambda f: synthesize_speech(text, lang, voice_sample, f, pause_ms=pause_ms,
                                     cfg_weight=0.2, prep_voice=True)),
        ("xtts_prepped.wav",
         "XTTS-v2, cleaned reference clip",
         lambda f: synthesize_speech_xtts(text, lang, voice_sample, f,
                                          pause_ms=pause_ms, prep_voice=True)),
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
    audio = trimmed[: int(target_sr * max_seconds)]

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
        concat_list.write(f"file '{os.path.abspath(video)}'\n")
        concat_list.write(f"file '{os.path.abspath(reversed_clip)}'\n")
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
    """
    inference_py = Path(INFINITETALK_DIR) / "generate_infinitetalk.py"
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

    # InfiniteTalk takes its inputs as a JSON file, not individual CLI flags
    # for photo/audio — schema per its examples/single_example_image.json.
    input_json = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8")
    json.dump({
        "prompt": scene_prompt,
        "cond_video": os.path.abspath(photo),
        "cond_audio": {"person1": os.path.abspath(audio)},
    }, input_json)
    input_json.close()

    # --save_file takes a base name (no extension); InfiniteTalk appends
    # ".mp4" itself. Use a temp stem so we control the final destination.
    save_stem = tempfile.NamedTemporaryFile(suffix="", delete=False).name
    os.remove(save_stem)  # only need a unique name, not the empty file itself

    cmd = [
        python, "generate_infinitetalk.py",
        "--ckpt_dir", str(ckpt_dir),
        "--wav2vec_dir", str(wav2vec_dir),
        "--infinitetalk_dir", str(infinitetalk_dir),
        "--input_json", input_json.name,
        "--size", f"infinitetalk-{size}",
        "--sample_steps", str(sample_steps),
        "--mode", mode,
        "--motion_frame", "9",
        "--save_file", save_stem,
    ]
    if quant in ("fp8", "int8"):
        cmd += ["--quant", quant, "--quant_dir", str(quant_dir)]
    if low_vram:
        cmd += ["--num_persistent_param_in_dit", "0"]

    print("Running InfiniteTalk:", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=INFINITETALK_DIR, check=True)
    finally:
        os.remove(input_json.name)

    produced = Path(f"{save_stem}.mp4")
    if not produced.exists():
        raise RuntimeError("InfiniteTalk did not produce a video - check the log above.")
    shutil.move(str(produced), out)
    print(f"Saved video -> {out}")


def make_one(photo: str, voice_sample: str, text: str, lang: str, out_video: str,
             engine: str = "latentsync", pause_ms: int = 450, motion: str = "idle",
             refine_lipsync_pass: bool = False, cfg_weight: float = 0.5,
             exaggeration: float = 0.5, tts: str = "chatterbox",
             prep_voice: bool = True, motion_video: str = None,
             motion_scale: float = 1.0, keep_intermediates: bool = False,
             keep_tts_warm: bool = False, **engine_opts) -> None:
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
            render_infinitetalk(photo, tmp_wav, out_video, **engine_opts)
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
            prepared_photo = prepare_photo(photo, max_dim=768)
            duration_s = sf.info(tmp_wav).duration
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


def run_batch(config_path: str, engine: str = "latentsync", pause_ms: int = 450,
              motion: str = "idle", refine_lipsync_pass: bool = False,
              cfg_weight: float = 0.5, exaggeration: float = 0.5,
              tts: str = "chatterbox", prep_voice: bool = True,
              motion_video: str = None, motion_scale: float = 1.0,
              keep_intermediates: bool = False, **engine_opts) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    photo = cfg["photo"]
    voice_sample = cfg["voice_sample"]
    lines = cfg["lines"]
    # CLI flags OR with matching top-level config.yaml keys (applies to the
    # whole batch, same as engine/motion/pause_ms above).
    refine_lipsync_pass = refine_lipsync_pass or bool(cfg.get("refine_lipsync", False))
    cfg_weight = cfg.get("cfg_weight", cfg_weight)
    exaggeration = cfg.get("exaggeration", exaggeration)
    tts = cfg.get("tts", tts)
    motion_video = cfg.get("motion_video", motion_video)
    motion_scale = cfg.get("motion_scale", motion_scale)
    if engine == "infinitetalk" and "scene_prompt" in cfg:
        engine_opts["scene_prompt"] = cfg["scene_prompt"]

    for i, line in enumerate(lines, 1):
        text = line["text"]
        lang = line.get("lang", "en")
        out = line.get("output", f"output/line_{i}.mp4")
        print(f"\n[{i}/{len(lines)}] ({lang}) {text[:60]}{'...' if len(text) > 60 else ''}")
        make_one(photo, voice_sample, text, lang, out, engine=engine, pause_ms=pause_ms,
                 motion=motion, refine_lipsync_pass=refine_lipsync_pass,
                 cfg_weight=cfg_weight, exaggeration=exaggeration, tts=tts,
                 prep_voice=prep_voice, motion_video=motion_video,
                 motion_scale=motion_scale, keep_intermediates=keep_intermediates,
                 keep_tts_warm=True, **engine_opts)

    release_tts_model()


def main():
    parser = argparse.ArgumentParser(description="Generate talking-avatar videos from text.")
    parser.add_argument("--config", help="YAML file with a batch of pre-entered lines")
    parser.add_argument("--photo", help="Path to your face photo")
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
    parser.add_argument("--no-voice-prep", action="store_true",
                        help="Use the voice sample exactly as given. By default it is "
                             "downmixed to mono, resampled, silence-trimmed and level-matched "
                             "first, which usually improves the clone noticeably")
    parser.add_argument("--voice-compare", action="store_true",
                        help="Generate the same line with every voice setting into "
                             "output/voice_ab/ and exit without rendering video - use this "
                             "to pick a voice before paying for the slow video step")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="Keep (and print the paths of) the synthesized speech and the "
                             "driving video instead of deleting them - the two files you "
                             "need to diagnose a voice or head-motion problem")
    parser.add_argument("--pause-ms", type=int, default=450,
                        help="Silence inserted between sentences, in milliseconds. Chatterbox has no "
                             "built-in pause control, so long multi-sentence text is spliced from "
                             "separately-synthesized sentences with this much silence between them")
    parser.add_argument("--motion-video",
                        help="[latentsync] Your own 15-20s video of head idling/blinking, used "
                             "to drive the photo instead of LivePortrait's bundled 3s clip. "
                             "Biggest realism win available: real motion, and long enough to "
                             "need almost no looping")
    parser.add_argument("--motion-scale", type=float, default=1.0,
                        help="[latentsync] Amplify transferred head/expression motion "
                             "(LivePortrait's driving_multiplier). Try 1.2-1.5 if the head "
                             "moves too subtly")
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
    parser.add_argument("--infinitetalk-steps", type=int, default=40,
                        help="[infinitetalk] Diffusion sample steps (default 40). Higher = better "
                             "quality, slower")
    parser.add_argument("--infinitetalk-quant", default="fp8", choices=["fp8", "int8", "none"],
                        help="[infinitetalk] fp8 (default): quantized model, needed to fit a "
                             "12GB-class GPU. int8: alternate quantization, try this if fp8 crashes "
                             "on your GPU. none: full precision, needs significantly more VRAM")
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
            voice_compare(args.voice, args.text, args.lang, pause_ms=args.pause_ms)
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
            sample_steps=args.infinitetalk_steps,
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
        engine=args.engine, pause_ms=args.pause_ms, motion=args.motion,
        refine_lipsync_pass=args.refine_lipsync, cfg_weight=args.cfg_weight,
        exaggeration=args.exaggeration, tts=args.tts,
        prep_voice=not args.no_voice_prep, motion_video=args.motion_video,
        motion_scale=args.motion_scale, keep_intermediates=args.keep_intermediates,
    )

    try:
        if args.config:
            run_batch(args.config, **shared_opts, **engine_opts)
        elif args.photo and args.voice and args.text and args.out:
            make_one(args.photo, args.voice, args.text, args.lang, args.out,
                     **shared_opts, **engine_opts)
        else:
            parser.error("Either --config, or all of --photo --voice --text --out are required.")
    finally:
        cleanup_prepared_voices()


if __name__ == "__main__":
    main()
