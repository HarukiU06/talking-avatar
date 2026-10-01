#!/usr/bin/env python3
"""Phase 4: join S1-S6 into the final 1920x1080 / 25 fps video.

    .venv/Scripts/python.exe event/assemble.py [--preview]

- Each 704x576 section render is scaled into the box it was cut from on the
  approved still (1100x900 at 410,180) and laid on the clean backdrop. The
  render's own wall is colour-matched to the backdrop frame by frame
  (per-channel gain from its top corners, smoothed) and the box edges are
  feathered, so there is no visible frame around the person.
- Sections sit on the timeline at their exact audio positions (full.wav is
  the section WAVs end to end), so the clean audio track is used directly and
  A/V stay in sync. A short section is padded by holding its last frame.
- Cuts are 6-frame dissolves centred on each section boundary, which falls in
  the silence after a line: the outgoing section's last frame / incoming
  section's first frame are held for the 3 frames each side needs.
- Hard cut at the end: no fade, no end card.
- Outputs: event_clean.mp4, and event_subtitled_en.mp4 with English subtitles
  (subtitles_en.txt, timed from the digital-silence pauses in full.wav) and the
  lower-third "司会 AI晶子" for the first 5 s of line 2. H.264 yuv420p + AAC.
--preview renders missing sections as a labelled still, to check the pipeline.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "output" / "event"
VIDEO, AUDIO, FRAME, OUT = EV / "video", EV / "audio", EV / "frame", EV / "final"
SECTIONS = ["S1", "S2", "S3", "S4", "S5", "S6"]
W, H, FPS = 1920, 1080, 25
BOX = (410, 180, 1510, 1080)          # where render_input.png was cut from final_c74.png
FEATHER = 48
DISSOLVE = 6


def section_frames(path: Path):
    """Yield RGB frames of a section render."""
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    size = 704 * 576 * 3
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(576, 704, 3)
    p.wait()


def first_frame(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(576, 704, 3)


def placeholder(label: str) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(str(FRAME / "render_input.png")), cv2.COLOR_BGR2RGB)
    img = (img * 0.5 + 64).astype(np.uint8)
    cv2.putText(img, f"{label} not rendered yet", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (200, 0, 0), 4)
    return img


class Compositor:
    def __init__(self):
        self.backdrop = cv2.cvtColor(cv2.imread(str(FRAME / "final_c74_backdrop.png")), cv2.COLOR_BGR2RGB).astype(np.float32)
        x0, y0, x1, y1 = BOX
        self.bw, self.bh = x1 - x0, y1 - y0
        ramp = lambda n: np.clip(np.arange(n, dtype=np.float32) / FEATHER, 0, 1)
        ax = np.minimum(ramp(self.bw), ramp(self.bw)[::-1])
        ay = ramp(self.bh)                       # top only; the bottom is the frame edge
        self.alpha = np.minimum(ax[None, :], ay[:, None])[..., None]
        # wall reference: backdrop under the render's top-corner patches
        self.patches = [(slice(0, 90), slice(0, 110)), (slice(0, 90), slice(self.bw - 110, self.bw))]
        under = self.backdrop[y0:y1, x0:x1]
        self.wall_ref = np.mean([under[p].reshape(-1, 3).mean(0) for p in self.patches], 0)
        self.gain = None

    def __call__(self, render: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = BOX
        big = cv2.resize(render, (self.bw, self.bh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        wall = np.mean([big[p].reshape(-1, 3).mean(0) for p in self.patches], 0)
        g = self.wall_ref / np.maximum(wall, 1.0)
        self.gain = g if self.gain is None else 0.9 * self.gain + 0.1 * g   # smoothed, no flicker
        big *= self.gain
        out = self.backdrop.copy()
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - self.alpha) + big * self.alpha
        return np.clip(out, 0, 255).astype(np.uint8)


def line_timings(full_wav: Path) -> list:
    """(start, end) seconds of each spoken line: the gaps are exact digital silence."""
    a, sr = sf.read(full_wav)
    z = np.concatenate([[0], (a == 0).astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(z))
    runs = [(s, e) for s, e in zip(edges[::2], edges[1::2]) if (e - s) / sr >= 0.2]
    segs, prev = [], 0
    for s, e in runs:
        if s > prev:
            segs.append((prev / sr, s / sr))
        prev = e
    if prev < len(a):
        segs.append((prev / sr, len(a) / sr))
    return segs


def ass_time(t: float) -> str:
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_ass(path: Path, timings: list) -> None:
    texts = {}
    for line in (ROOT / "event" / "subtitles_en.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\d+)\s+(.+)", line)
        if m and not line.startswith("#"):
            texts[int(m[1])] = m[2].strip()
    if len(timings) != len(texts):
        raise RuntimeError(f"{len(timings)} spoken lines in full.wav but {len(texts)} subtitles")
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,Arial,46,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,1,2,220,220,56,1
Style: Lower,Meiryo,54,&H00FFFFFF,&H00FFFFFF,&H00000000,&H9A2B2B2B,1,0,0,0,100,100,2,0,3,18,0,1,120,0,150,128

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for n, (s, e) in enumerate(timings, 1):
        events.append(f"Dialogue: 0,{ass_time(s)},{ass_time(e + 0.25)},Sub,,0,0,0,,{texts[n]}")
    s2 = timings[1][0]
    events.append(f"Dialogue: 1,{ass_time(s2)},{ass_time(s2 + 5)},Lower,,0,0,0,,{{\\fad(250,250)}}司会　AI晶子")
    path.write_text(head + "\n".join(events) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="stand in a labelled still for missing sections")
    ap.add_argument("--threads", type=int, default=0, help="limit x264 threads (e.g. while a render is running)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    durations = [sf.info(str(AUDIO / f"{s}.wav")).duration for s in SECTIONS]
    starts = np.round(np.concatenate([[0], np.cumsum(durations)]) * FPS).astype(int)
    total_frames = int(starts[-1])
    full = AUDIO / "full.wav"
    if abs(sf.info(str(full)).duration * FPS - total_frames) > 1:
        raise RuntimeError("full.wav does not match the section WAVs - rebuild the audio first")
    have = {s: (VIDEO / f"{s}.mp4").exists() for s in SECTIONS}
    if not all(have.values()) and not a.preview:
        raise SystemExit(f"missing renders: {[s for s, ok in have.items() if not ok]} (or use --preview)")

    firsts = {s: first_frame(VIDEO / f"{s}.mp4") if have[s] else placeholder(s) for s in SECTIONS}
    comp = Compositor()
    stem = "event_preview" if not all(have.values()) else "event_clean"
    clean = OUT / f"{stem}.mp4"
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-i", str(full), "-map", "0:v", "-map", "1:a",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-threads", str(a.threads),
                            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart",
                            "-t", f"{total_frames / FPS:.3f}", str(clean)], stdin=subprocess.PIPE)
    half = DISSOLVE // 2
    report, prev_last = [], None
    for k, s in enumerate(SECTIONS):
        n_frames = starts[k + 1] - starts[k]
        src = section_frames(VIDEO / f"{s}.mp4") if have[s] else iter(())
        last, got = firsts[s], 0
        for i in range(n_frames):
            frame = next(src, None)
            if frame is not None:
                last, got = frame, got + 1
            cur = comp(last)
            if k > 0 and i < half:                                  # fade in from the previous section
                t = (half + i + 0.5) / DISSOLVE
                cur = (comp(prev_last) * (1 - t) + cur * t).astype(np.uint8)
            if k + 1 < len(SECTIONS) and i >= n_frames - half:      # fade out into the next section
                t = (i - (n_frames - half) + 0.5) / DISSOLVE
                cur = (cur * (1 - t) + comp(firsts[SECTIONS[k + 1]]) * t).astype(np.uint8)
            enc.stdin.write(cur.tobytes())
        for _ in src:
            got += 1
        prev_last = last
        report.append({"section": s, "timeline_start_s": round(starts[k] / FPS, 2), "frames_on_timeline": int(n_frames),
                       "frames_rendered": got, "held_frames": int(max(n_frames - got, 0)),
                       "dropped_frames": int(max(got - n_frames, 0)), "rendered": have[s]})
        print(f"{s}: {n_frames} frames from {starts[k] / FPS:.2f}s ({got} rendered)", flush=True)
    enc.stdin.close()
    enc.wait()

    timings = line_timings(full)
    ass = OUT / "subtitles_en.ass"
    write_ass(ass, timings)
    srt = OUT / "subtitles_en.srt"
    texts = [l.split(" ", 1)[1] for l in (ROOT / "event" / "subtitles_en.txt").read_text(encoding="utf-8").splitlines()
             if re.match(r"\d+ ", l)]
    srt.write_text("\n".join(f"{i}\n{srt_time(s)} --> {srt_time(e + 0.25)}\n{t}\n"
                             for i, ((s, e), t) in enumerate(zip(timings, texts), 1)), encoding="utf-8")
    subbed = OUT / ("event_subtitled_en.mp4" if stem == "event_clean" else f"{stem}_subtitled_en.mp4")
    rel = ass.relative_to(ROOT).as_posix()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(clean), "-vf", f"ass={rel}",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-threads", str(a.threads), "-c:a", "copy",
                    "-movflags", "+faststart", str(subbed)], check=True, cwd=str(ROOT))

    def durations_of(p):
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration,nb_frames",
                              "-of", "json", str(p)], capture_output=True, text=True).stdout
        return {st["codec_type"]: (float(st["duration"]), st.get("nb_frames")) for st in json.loads(out)["streams"]}
    summary = {"timeline_frames": total_frames, "audio_s": round(sf.info(str(full)).duration, 3),
               "sections": report, "clean": str(clean), "subtitled": str(subbed),
               "clean_streams": durations_of(clean), "subtitled_streams": durations_of(subbed),
               "line_timings": [[round(s, 2), round(e, 2)] for s, e in timings]}
    (OUT / f"{stem}_report.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("timeline_frames", "audio_s", "clean_streams", "subtitled_streams")}))


if __name__ == "__main__":
    main()
