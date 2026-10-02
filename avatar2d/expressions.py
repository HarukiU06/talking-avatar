#!/usr/bin/env python3
"""Pick a facial expression for every script line, from its text.

    .venv/Scripts/python.exe avatar2d/expressions.py [--script event/script.txt]

A local multilingual emotion classifier (MilaNLProc/xlm-emo-t: anger, fear,
joy, sadness) scores each line, and the scores map to the rig's expressions:

    joy >= HAPPY_AT           -> happy
    joy >= SMILE_AT           -> smile
    fear >= OTHER_AT          -> surprised
    anger/sadness >= OTHER_AT -> concerned
    otherwise                 -> neutral

The classifier has no "neutral" class (its four scores always sum to 1), so
calm, formal lines land on neutral only through these thresholds.

event/expressions.txt overrides any line ("<n> <expression>", # comments), and
wins over the classifier. Writes output/avatar2d/expressions.json: per line the
expression, who chose it, and the raw scores, so you can see what to override.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "avatar2d" / "expressions.json"
EXPRESSIONS = ("neutral", "smile", "happy", "surprised", "concerned")
HAPPY_AT, SMILE_AT, OTHER_AT = 0.95, 0.75, 0.6


def script_lines(path: Path) -> list:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\d+)\s+\[(\w+)\]\s+(.+?)\s+\(([\d.]+)\)\s*$", line)
        if m and not line.startswith("#"):
            out.append({"n": int(m[1]), "lang": m[2], "text": m[3], "pause": float(m[4])})
    return out


def overrides(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\d+)\s+(\w+)", line)
        if m and not line.startswith("#"):
            if m[2] not in EXPRESSIONS:
                raise SystemExit(f"{path}: line {m[1]}: unknown expression {m[2]!r} (use one of {EXPRESSIONS})")
            out[int(m[1])] = m[2]
    return out


def classify(scores: dict) -> str:
    if scores["joy"] >= HAPPY_AT:
        return "happy"
    if scores["joy"] >= SMILE_AT:
        return "smile"
    if scores["fear"] >= OTHER_AT:
        return "surprised"
    if scores["anger"] + scores["sadness"] >= OTHER_AT:
        return "concerned"
    return "neutral"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default=str(ROOT / "event" / "script.txt"))
    ap.add_argument("--overrides", default=str(ROOT / "event" / "expressions.txt"))
    a = ap.parse_args()

    from transformers import pipeline
    clf = pipeline("text-classification", model="MilaNLProc/xlm-emo-t", top_k=None)
    lines = script_lines(Path(a.script))
    over = overrides(Path(a.overrides))
    result = []
    for ln in lines:
        scores = {d["label"]: round(float(d["score"]), 3) for d in clf(ln["text"])[0]}
        auto = classify(scores)
        expr = over.get(ln["n"], auto)
        result.append({"n": ln["n"], "expression": expr, "by": "override" if ln["n"] in over else "classifier",
                       "classifier": auto, "scores": scores, "text": ln["text"]})
        print(f"{ln['n']:>2} {expr:<9} ({'override' if ln['n'] in over else 'auto'}; classifier {auto:<9} "
              + " ".join(f"{k} {v:.2f}" for k, v in sorted(scores.items())) + f")  {ln['text'][:30]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
