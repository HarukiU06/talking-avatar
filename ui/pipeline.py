"""From what was picked in the app to a queued make_avatar.py job.

Option names, defaults and allowed values all come from make_avatar.py's own
argument parser, so the app and the command line can't drift apart: a
setting here is just the parser's `dest` name.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import make_avatar as ma
from ui import storage
from ui.jobs import Job, Stage

PARSER = ma.build_parser()
_ACTIONS = {a.dest: a for a in PARSER._actions}

# The one place the app deliberately differs from the command line: the
# lightx2v mode is ~15x faster for a small quality cost, which is the right
# first choice for someone clicking a button. The CLI keeps upstream's default.
APP_DEFAULTS = {"infinitetalk_accel": "lightx2v"}


def default(dest: str):
    return APP_DEFAULTS.get(dest, PARSER.get_default(dest))


def choices(dest: str) -> list:
    return list(_ACTIONS[dest].choices or [])


ENGINES = {
    "latentsync": ("Standard", "LatentSync"),
    "infinitetalk": ("Expressive", "InfiniteTalk"),
    "sadtalker": ("Lightweight", "SadTalker"),
}

ENGINE_INFO = {
    "latentsync": "Redraws only the mouth, so the rest of your face stays exactly as in your photo "
                  "or video. A few minutes per line on a recent NVIDIA GPU (8 GB+).",
    "infinitetalk": "A 14B video model that animates expression, head motion and lips together, "
                    "guided by a scene description. Slow: roughly 3 minutes of rendering per second "
                    "of speech in fast mode. Needs about 70 GB of disk and 32 GB of RAM.",
    "sadtalker": "Animates the whole face from a photo. Lighter and simpler, but the lip sync is "
                 "weaker and the face can drift from yours. Photo only.",
}

# Settings that only mean something to one engine. Left out of the command for
# the others, so the log shows only what affected the run.
ENGINE_OPTIONS = {
    "latentsync": {"motion", "motion_video", "motion_scale", "inference_steps", "guidance_scale",
                   "latentsync_res"},
    "sadtalker": {"size", "enhancer", "expression_scale", "preprocess", "pose_style",
                  "sadtalker_motion", "refine_lipsync"},
    "infinitetalk": {"scene_prompt", "infinitetalk_size", "infinitetalk_steps", "infinitetalk_seed",
                     "infinitetalk_accel", "infinitetalk_quant", "infinitetalk_no_low_vram",
                     "infinitetalk_mode"},
}
_ANY_ENGINE_OPTION = set().union(*ENGINE_OPTIONS.values())

LANGUAGE_NAMES = {
    "ar": "Arabic", "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi",
    "hu": "Hungarian", "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay",
    "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ru": "Russian",
    "sv": "Swedish", "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}


def languages(tts: str) -> list:
    """(name, code) pairs the given voice model can speak, sorted by name."""
    codes = ma.XTTS_LANGS if tts == "xtts" else ma.CHATTERBOX_LANGS
    return sorted(((LANGUAGE_NAMES.get(c, c), c) for c in codes), key=lambda pair: pair[0])


# Per video. Longer is minutes of speech in one render, which the video
# engines handle badly (memory, and InfiniteTalk drifts in colour past about
# a minute), and on Windows a whole script can outgrow the command line.
MAX_TEXT_CHARS = 3000


def clean_text(text: str) -> str:
    """One paragraph for the voice model: line breaks and runs of spaces collapsed."""
    return " ".join((text or "").split())


def paragraphs(text: str) -> list:
    return [p for p in (clean_text(chunk) for chunk in re.split(r"\n\s*\n", text or "")) if p]


def to_args(options: dict) -> list:
    """{dest: value} -> make_avatar.py arguments, leaving out anything at its default.

    Values go in as --flag=value, so text that starts with "-" can't be read
    as another option.
    """
    engine = options.get("engine") or PARSER.get_default("engine")
    args = []
    for dest, value in options.items():
        if dest in _ANY_ENGINE_OPTION and dest not in ENGINE_OPTIONS.get(engine, ()):
            continue
        action = _ACTIONS[dest]
        if value is None or value == "":
            continue
        flag = action.option_strings[-1]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                args.append(flag)
            continue
        if action.type is not None:
            value = action.type(value)
        if action.choices is not None and value not in action.choices:
            raise ValueError(f"{flag} must be one of {list(action.choices)}, not {value!r}")
        if value == action.default:
            continue
        args.append(f"{flag}={value}")
    return args


def plan_stages(options: dict) -> list:
    stages = [Stage("speech", "Speech")]
    if options.get("voice_convert"):
        stages.append(Stage("convert", "Voice match"))
    engine = options.get("engine")
    if engine == "sadtalker":
        stages.append(Stage("render", "Animate"))
        if options.get("refine_lipsync"):
            stages.append(Stage("refine", "Refine lips"))
    elif engine == "infinitetalk":
        stages.append(Stage("render", "Render"))
    else:
        if options.get("video"):
            stages.append(Stage("motion", "Prepare video"))
        elif options.get("motion", "idle") == "idle":
            stages.append(Stage("motion", "Head motion"))
        stages.append(Stage("lipsync", "Lip sync"))
    return stages


def needed_parts(options: dict) -> list:
    """Keys of ui.system.check_parts() this run depends on."""
    needed = ["ffmpeg", "xtts" if options.get("tts") == "xtts" else "chatterbox"]
    if options.get("voice_convert"):
        needed.append("seedvc")
    engine = options.get("engine")
    if engine == "sadtalker":
        needed.append("sadtalker")
        if options.get("refine_lipsync"):
            needed.append("wav2lip")
    elif engine == "infinitetalk":
        needed.append("infinitetalk")
    else:
        needed.append("latentsync")
        if not options.get("video") and options.get("motion", "idle") == "idle":
            needed.append("liveportrait")
    return needed


def problems(options: dict, parts: dict) -> list:
    """Reasons this can't run, worded for the person at the screen. Empty = good to go."""
    found = []
    if not clean_text(options.get("text")):
        found.append("Type what your avatar should say.")
    if not (options.get("photo") or options.get("video")):
        found.append("Add a photo or a video of your face.")
    if options.get("engine") == "sadtalker" and not options.get("photo"):
        found.append("The Lightweight engine needs a photo; it can't use a video.")
    if not options.get("voice"):
        found.append("Add a voice sample: 25-30 seconds of you speaking.")
    for key in ("photo", "video", "voice", "vc_target", "motion_video"):
        if options.get(key) and not Path(options[key]).is_file():
            found.append(f"The {key.replace('_', ' ')} file is no longer available. Add it again.")
    tts = options.get("tts") or "chatterbox"
    if options.get("lang") not in {code for _, code in languages(tts)}:
        name = LANGUAGE_NAMES.get(options.get("lang"), options.get("lang"))
        found.append(f"{'XTTS-v2' if tts == 'xtts' else 'Chatterbox'} can't speak {name}.")
    for key in needed_parts(options):
        part = parts.get(key)
        if part and not part.ready:
            found.append(f"{part.name} isn't installed yet. Run {part.setup} (see the Setup tab).")
    return found


def describe(options: dict) -> str:
    """Short human summary, for the queue and the library."""
    engine = ENGINES.get(options.get("engine"), ("", ""))[0]
    voice = "XTTS-v2" if options.get("tts") == "xtts" else "Chatterbox"
    if options.get("voice_convert"):
        voice += " + voice match"
    return f"{engine} · {voice} · {LANGUAGE_NAMES.get(options.get('lang'), options.get('lang'))}"


# Settings saved with a video and restored by "Use these settings": everything
# except the input files, which belong to the avatar.
_FILE_OPTIONS = {"photo", "video", "voice", "vc_target", "motion_video", "out", "text"}


def video_job(options: dict, avatar: str | None, taken: set) -> Job:
    text = clean_text(options["text"])
    out = storage.new_video_path(text, taken)
    options = {**options, "text": text, "out": str(out)}
    title = text if len(text) <= 60 else text[:57].rstrip() + "..."
    meta = {
        "text": text,
        "avatar": avatar,
        "summary": describe(options),
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "created_ts": time.time(),
        "settings": {k: v for k, v in options.items() if k not in _FILE_OPTIONS},
    }

    def save_details(job: Job) -> None:
        storage.write_video_meta(job.output, {**job.meta, "render_seconds": round(job.elapsed)})

    return Job(kind="video", title=title, args=to_args(options), stages=plan_stages(options),
               output=out, log_path=storage.LOGS_DIR / f"{out.stem}.log", meta=meta,
               on_done=save_details)


# --- Voice Lab ------------------------------------------------------------------

# The clips make_avatar.py --voice-compare writes, and the settings that
# reproduce each one in a real render. Clips it adds later still show up in
# the app; they just won't have a "Use this voice" button until listed here.
VOICE_VARIANTS = {
    "chatterbox_natural.wav": ("Chatterbox",
                               dict(tts="chatterbox", voice_convert=False, cfg_weight=0.5, pause_ms=0, no_voice_prep=False)),
    "chatterbox_natural_cfg0.2.wav": ("Chatterbox, less accent",
                                      dict(tts="chatterbox", voice_convert=False, cfg_weight=0.2, pause_ms=0, no_voice_prep=False)),
    "chatterbox_spliced450.wav": ("Chatterbox, sentence by sentence",
                                  dict(tts="chatterbox", voice_convert=False, cfg_weight=0.5, pause_ms=450, no_voice_prep=False)),
    "chatterbox_raw_reference.wav": ("Chatterbox, sample used as recorded",
                                     dict(tts="chatterbox", voice_convert=False, cfg_weight=0.5, pause_ms=0, no_voice_prep=True)),
    "chatterbox_then_seedvc.wav": ("Chatterbox + voice match",
                                   dict(tts="chatterbox", voice_convert=True, cfg_weight=0.5, pause_ms=0, no_voice_prep=False)),
    "xtts_natural.wav": ("XTTS-v2",
                         dict(tts="xtts", voice_convert=False, cfg_weight=0.5, pause_ms=0, no_voice_prep=False)),
    "xtts_spliced450.wav": ("XTTS-v2, sentence by sentence",
                            dict(tts="xtts", voice_convert=False, cfg_weight=0.5, pause_ms=450, no_voice_prep=False)),
    "xtts_then_seedvc.wav": ("XTTS-v2 + voice match",
                             dict(tts="xtts", voice_convert=True, cfg_weight=0.5, pause_ms=0, no_voice_prep=False)),
}


def voices_job(voice: str, text: str, lang: str, vc_target: str | None, label: str) -> Job:
    text = clean_text(text)
    out = storage.VOICE_TESTS_DIR / f"{time.strftime('%Y-%m-%d_%H%M%S')}_{storage.slug(label, 'voice', 24)}"
    options = {"voice_compare": True, "voice": voice, "text": text, "lang": lang,
               "vc_target": vc_target, "out": str(out)}
    return Job(kind="voices", title=f"Voice test: {text[:40]}", args=to_args(options),
               stages=[Stage("voices", "Voices")], output=out,
               log_path=storage.LOGS_DIR / f"{out.name}.log", meta={"variants": len(VOICE_VARIANTS)})
