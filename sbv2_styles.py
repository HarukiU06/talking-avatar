#!/usr/bin/env python3
"""Nameable emotion styles for a trained Style-Bert-VITS2 voice. Runs in .venv-sbv2.

SBV2 learns no labelled emotions. Preprocessing stores one style vector per
training clip (<clip>.wav.npy) and a "style" is a point in that space. This
clusters them the way SBV2's own web UI does (KMeans; the mean vector first, as
"Neutral"), saves the result where inference reads it, and copies the clip
nearest each cluster centre to style_samples/ so the styles can be named by ear
in styles.yaml. A calm recording yields calm clusters: a style can't be more
emotional than anything in the training audio.

    python sbv2_styles.py --data models/sbv2-data/me --model models/sbv2/me --clusters 4
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--clusters", type=int, default=4)
    args = ap.parse_args()
    data, model = Path(args.data), Path(args.model)

    vec_files = sorted((data / "wavs").glob("*.wav.npy"))
    if len(vec_files) < args.clusters * 3:
        raise SystemExit(f"Only {len(vec_files)} clips: too few for {args.clusters} styles. "
                         "Lower --clusters or record more.")
    x = np.stack([np.load(f).reshape(-1) for f in vec_files])
    mean = x.mean(axis=0)
    km = KMeans(n_clusters=args.clusters, random_state=42, n_init="auto").fit(x)
    centroids = [x[km.labels_ == i].mean(axis=0) for i in range(args.clusters)]
    names = ["Neutral"] + [f"style_{i + 1}" for i in range(args.clusters)]

    np.save(model / "style_vectors.npy", np.stack([mean] + centroids))
    config_path = model / "config.json"
    if not config_path.exists():
        shutil.copy(data / "config.json", config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["data"]["num_styles"] = len(names)
    config["data"]["style2id"] = {n: i for i, n in enumerate(names)}
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    samples = model / "style_samples"
    samples.mkdir(exist_ok=True)
    for name, centre, members in [("Neutral", mean, np.arange(len(x)))] + [
            (names[i + 1], centroids[i], np.where(km.labels_ == i)[0]) for i in range(args.clusters)]:
        nearest = members[np.argmin(np.linalg.norm(x[members] - centre, axis=1))]
        shutil.copy(vec_files[nearest].with_suffix(""), samples / f"{name}.wav")

    lines = ["# Emotion tags for [tag]...[/tag] -> this voice's style names.",
             "# Listen to style_samples/*.wav, then rename the tags on the left",
             "# (happy, calm, ...). Keep the style names on the right as they are.",
             "neutral: Neutral"] + [f"{n}: {n}" for n in names[1:]]
    (model / "styles.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(names)} styles written to {model}; samples in {samples}")


if __name__ == "__main__":
    main()
