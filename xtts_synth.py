#!/usr/bin/env python3
"""Synthesize sentences with XTTS-v2. Runs inside .venv-xtts, never imported.

This is the subprocess entry point make_avatar.py shells out to, following the
same isolation rule as every other model repo in this project:
coqui-tts pins transformers/numpy versions that conflict with chatterbox-tts in
.venv, so the two TTS engines can never share a process.

Input and output both go through a JSON file rather than CLI arguments. The
text being spoken is arbitrary user input in any of 17 languages — passing it
through a Windows command line invites both quoting bugs and the non-UTF-8
console encoding errors this project already works around elsewhere.

Usage:  python xtts_synth.py JOB_JSON
where JOB_JSON is {"sentences": [...], "lang": "en",
                   "speaker_wav": "...", "out_dir": "...",
                   "seeds": [...]}      # seeds optional, one per sentence
and the result is written back to JOB_JSON + ".result" as
                  {"sr": 24000, "files": ["...wav", ...]}
"""
import json
import os
import sys

# Accept the model license non-interactively. On a TTY the downloader prompts
# on stdin; from a subprocess that prompt would hang forever.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# XTTS-v2's own language codes. This is a subset of what Chatterbox covers —
# da/el/fi/he/ms/no/sv/sw have no XTTS equivalent — and it spells Chinese
# "zh-cn" where the rest of this project uses "zh".
LANG_MAP = {
    "ar": "ar", "cs": "cs", "de": "de", "en": "en", "es": "es", "fr": "fr",
    "hi": "hi", "hu": "hu", "it": "it", "ja": "ja", "ko": "ko", "nl": "nl",
    "pl": "pl", "pt": "pt", "ru": "ru", "tr": "tr", "zh": "zh-cn",
}


def main() -> int:
    job_path = sys.argv[1]
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)

    lang = LANG_MAP.get(job["lang"])
    if lang is None:
        print(f"XTTS does not support language '{job['lang']}'. "
              f"Supported: {sorted(LANG_MAP)} (use --tts chatterbox for the rest).",
              file=sys.stderr)
        return 2

    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading XTTS-v2 on {device} ...", flush=True)
    tts = TTS(MODEL).to(device)

    out_dir = job["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    files = []
    seeds = job.get("seeds")  # optional: one seed per sentence, for reproducible takes
    for i, sentence in enumerate(job["sentences"]):
        if seeds:
            torch.manual_seed(seeds[i])
            torch.cuda.manual_seed_all(seeds[i])
        path = os.path.join(out_dir, f"sentence_{i:03d}.wav")
        tts.tts_to_file(text=sentence, speaker_wav=job["speaker_wav"],
                        language=lang, file_path=path)
        files.append(path)

    # XTTS-v2 is fixed at 24kHz, but read it off the model rather than
    # hardcoding it — the caller splices these together and needs the real rate.
    sr = tts.synthesizer.output_sample_rate
    with open(job_path + ".result", "w", encoding="utf-8") as fh:
        json.dump({"sr": sr, "files": files}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
