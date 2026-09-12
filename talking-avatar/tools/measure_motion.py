"""Measure how much a talking-head video actually moves, by region.

LatentSync only regenerates the mouth region, so a video can have a perfectly
animated mouth sitting in a completely frozen face. That reads as uncanny but
is easy to mistake for "bad lip sync" when judging by eye. This reports the
two separately, so the driving video and the final render can each be checked
for the motion they are supposed to contribute.

Frames are decoded through ffmpeg to a small grayscale buffer (no opencv
dependency - only numpy, which .venv already has).

IMPORTANT - what these numbers can and cannot be compared against. Regions
are a fixed proportional split of the frame, not a detected face box. So the
fraction of the frame the face occupies directly scales every number: a tight
512x512 face crop and a 576x768 portrait with background will report different
motion for the *same* head movement. That makes this tool valid for:

  - comparing pipeline outputs against each other (same photo, same framing)
  - before/after on one change (--motion-scale, --motion-video, ...)
  - the upper/mouth ratio within one video, which is what reveals the
    "frozen face, moving mouth" failure

and NOT valid for comparing a tightly-cropped reference clip against a
full-frame render and concluding one moves more. The frame size is printed
for each video so a mismatch is visible rather than silent.

Usage:  python tools/measure_motion.py VIDEO [VIDEO ...]
"""
import subprocess
import sys

import numpy as np

W = H = 128  # decode resolution; motion energy is scale-invariant enough at this size

# Proportional regions within a face-centered crop.
UPPER = (slice(0, int(H * 0.55)), slice(0, W))              # brow, eyes, nose, head outline
MOUTH = (slice(int(H * 0.55), int(H * 0.85)), slice(int(W * 0.25), int(W * 0.75)))


def source_size(path: str) -> str:
    """Native WxH, so a framing mismatch between compared videos is visible."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    return out[0] if out else "?"


def read_gray_frames(path: str) -> np.ndarray:
    """Decode `path` to an (n_frames, H, W) uint8 array via an ffmpeg pipe."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-vf", f"scale={W}:{H}", "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (W * H)
    if n < 2:
        raise RuntimeError(f"{path}: decoded {n} frames, need at least 2")
    return np.frombuffer(raw[: n * W * H], dtype=np.uint8).reshape(n, H, W)


def motion_energy(frames: np.ndarray) -> dict:
    """Mean absolute frame-to-frame difference, overall and per region.

    Reported on the 0-255 gray scale, so the numbers are directly comparable
    between videos: ~0.5 is a static image with encoder noise, single digits
    are visible motion.
    """
    diff = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16))
    return {
        "frames": len(frames),
        "overall": float(diff.mean()),
        "upper_face": float(diff[:, UPPER[0], UPPER[1]].mean()),
        "mouth": float(diff[:, MOUTH[0], MOUTH[1]].mean()),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    print(f"{'video':<44} {'size':>10} {'frames':>7} {'overall':>9} "
          f"{'upper':>9} {'mouth':>9} {'ratio':>7}")
    sizes = set()
    for path in sys.argv[1:]:
        try:
            size = source_size(path)
            m = motion_energy(read_gray_frames(path))
        except Exception as exc:  # a missing/corrupt file shouldn't abort the rest of the batch
            print(f"{path:<44} ERROR: {exc}")
            continue
        sizes.add(size)
        # upper/mouth ratio is the headline number: well below 1 means the
        # mouth is doing all the moving and the face is frozen.
        ratio = m["upper_face"] / m["mouth"] if m["mouth"] else float("nan")
        print(f"{path:<44} {size:>10} {m['frames']:>7} {m['overall']:>9.3f} "
              f"{m['upper_face']:>9.3f} {m['mouth']:>9.3f} {ratio:>7.2f}")

    if len(sizes) > 1:
        print()
        print("NOTE: these videos are not all the same size, so their absolute "
              "motion values are not directly comparable (see this script's "
              "docstring). The per-video upper/mouth ratio still is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
