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

A piece with an accent (0 = flat, n = pitch drops after mora n) gets that
accent in pyopenjtalk's own word analysis, so the particles and auxiliaries
attached to it follow as pyopenjtalk would place them; everything else keeps
pyopenjtalk's accents. The cwd matters: SBV2 finds its BERT model and user
dictionary relative to the checkout.
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())  # style_bert_vits2, from the checkout


def latest_checkpoint(model_dir: Path) -> Path:
    def step(path: Path) -> int:
        match = re.search(r"_s(\d+)$", path.stem)
        return int(match.group(1)) if match else -1

    found = sorted(model_dir.glob("*.safetensors"), key=step)
    if not found:
        raise FileNotFoundError(f"No .safetensors checkpoint in {model_dir}")
    return found[-1]


def segment_overrides(pieces: list, normalize) -> list:
    """(start, end, nucleus, word) character spans of the accented pieces in the normalized text."""
    spans, pos = [], 0
    for text, accent in pieces:
        norm = normalize(text)
        if accent is not None:
            spans.append((pos, pos + len(norm), accent, text))
        pos += len(norm)
    return spans


def apply_accent_overrides(nodes: list, overrides: list) -> None:
    """Set the accent of marked words in pyopenjtalk's word list, in place, before it builds labels.

    A word's accent decides its whole accent phrase, the particles and
    auxiliaries chained after it included (雨です ア'メデス, 飴です アメデ'ス),
    and pyopenjtalk keeps the phrase's nucleus on the phrase's first word, as
    "acc" counted in morae from the phrase start. Setting it there, rather than
    rewriting tones mora by mora afterwards, leaves the attached words to
    pyopenjtalk: rewriting the word alone left 雨です rising again on です.

    A flat override keeps a nucleus pyopenjtalk put on an attached auxiliary
    (only a flat word lets です carry it) and clears one inside the word.
    """
    starts, pos = [], 0
    for n in nodes:
        starts.append(pos)
        pos += len(n["string"])
    for start, end, nucleus, word in overrides:
        idx = [i for i, n in enumerate(nodes)
               if n["string"] and start <= starts[i] and starts[i] + len(n["string"]) <= end]
        if not idx or starts[idx[0]] != start or starts[idx[-1]] + len(nodes[idx[-1]]["string"]) != end:
            raise ValueError(f"{word}: can't set its accent, because pyopenjtalk reads the text around "
                             "it as different words. Put the whole word inside {...|...}.")
        word_morae = sum(nodes[i]["mora_size"] for i in idx)
        if nucleus > word_morae:
            raise ValueError(f"{word}: accent after mora {nucleus}, but it has {word_morae} morae")
        for i in idx[1:]:  # a word pyopenjtalk split stays one phrase
            nodes[i]["chain_flag"] = 1
        head = idx[0]
        while head > 0 and nodes[head]["chain_flag"] == 1:
            head -= 1
        offset = sum(nodes[i]["mora_size"] for i in range(head, idx[0]))
        if nucleus:
            nodes[head]["acc"] = offset + nucleus
        elif offset < nodes[head]["acc"] <= offset + word_morae:
            nodes[head]["acc"] = 0


class AccentedFrontend:
    """Stands in for pyopenjtalk_worker inside SBV2's g2p module for one infer() call.

    SBV2 reaches pyopenjtalk through that module's global, so swapping it at
    runtime changes the accents it sees without editing the checkout.
    """

    def __init__(self, real, text: str, overrides: list):
        self.real, self.text, self.overrides, self.applied = real, text, overrides, False

    def __getattr__(self, name):
        return getattr(self.real, name)

    def run_frontend(self, text: str) -> list:
        nodes = self.real.run_frontend(text)
        if text == self.text:
            apply_accent_overrides(nodes, self.overrides)
            self.applied = True
        return nodes


def main() -> int:
    job_path = sys.argv[1]
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)

    import soundfile as sf
    import torch
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.nlp.japanese import g2p as sbv2_g2p
    from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
    from style_bert_vits2.nlp.japanese.normalizer import normalize_text
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
            pieces = segment["pieces"]
            text = "".join(normalize_text(t) for t, _ in pieces)
            overrides = segment_overrides(pieces, normalize_text)
            frontend = AccentedFrontend(pyopenjtalk_worker, text, overrides) if overrides else None
            if frontend:
                sbv2_g2p.pyopenjtalk = frontend
            try:
                sr, audio = model.infer(
                    text=text, language=Languages.JP,
                    style=segment["style"], style_weight=segment["style_weight"],
                    intonation_scale=segment["intonation_scale"], length=job["length"],
                    pitch_scale=job["pitch_scale"], line_split=False)
            finally:
                sbv2_g2p.pyopenjtalk = pyopenjtalk_worker
            if frontend and not frontend.applied:
                raise RuntimeError(f"Accent marks in {text!r} never reached pyopenjtalk: SBV2 changed "
                                   "the text before analysing it.")
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
