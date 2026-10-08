"""What the app keeps on disk: saved avatars, finished videos, settings.

Everything lives next to the code, in places .gitignore already excludes
(avatars/, output/, app_settings.json), so nothing personal ends up in git.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AVATARS_DIR = ROOT / "avatars"
OUTPUT_DIR = ROOT / "output"
LOGS_DIR = OUTPUT_DIR / "logs"
VOICE_TESTS_DIR = OUTPUT_DIR / "voice_ab"
SETTINGS_PATH = ROOT / "app_settings.json"

# Names Windows refuses for a file or folder, whatever the extension.
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
             *(f"lpt{i}" for i in range(1, 10))}


def slug(text: str, fallback: str, max_len: int = 40) -> str:
    """A filesystem-safe name: letters and digits in any script, joined by '-'.

    Path separators and dots can't survive it, so the result never escapes
    the folder it's joined to.
    """
    name = re.sub(r"[^\w]+", "-", text).strip("-_").lower()[:max_len].strip("-_")
    if not name:
        return fallback
    return f"{name}_" if name in _RESERVED else name


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # never leaves a half-written file behind


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# --- avatars: a face and a voice saved under a name ------------------------------

# The file each avatar field is stored as, inside avatars/<name>/.
AVATAR_FILES = {"photo": "face", "video": "face_video", "voice": "voice", "vc_target": "voice_long"}


@dataclass
class Avatar:
    name: str
    folder: Path
    photo: str | None = None
    video: str | None = None
    voice: str | None = None
    vc_target: str | None = None
    face_kind: str = "photo"


def _load_avatar(meta_path: Path) -> Avatar | None:
    data = _read_json(meta_path)
    if not data:
        return None
    folder = meta_path.parent
    files = {}
    for key in AVATAR_FILES:
        name = data.get(key)
        # Only plain file names inside the avatar's own folder count, so an
        # edited avatar.json can't point the app at files elsewhere.
        if isinstance(name, str) and name and Path(name).name == name and (folder / name).is_file():
            files[key] = str(folder / name)
    face_kind = "video" if data.get("face_kind") == "video" else "photo"
    return Avatar(name=str(data.get("name") or folder.name), folder=folder, face_kind=face_kind, **files)


def list_avatars() -> list:
    if not AVATARS_DIR.is_dir():
        return []
    found = (_load_avatar(p) for p in AVATARS_DIR.glob("*/avatar.json"))
    return sorted((a for a in found if a), key=lambda a: a.name.lower())


def find_avatar(name: str | None) -> Avatar | None:
    if not name:
        return None
    return next((a for a in list_avatars() if a.name == name), None)


def save_avatar(name: str, face_kind: str = "photo", **files: str | None) -> Avatar:
    """Copy the given files into avatars/<name>/, replacing what was there.

    A field passed as None is removed from the avatar, so saving always
    leaves it matching what's on screen.
    """
    name = " ".join((name or "").split())
    if not name:
        raise ValueError("Give the avatar a name first.")
    existing = find_avatar(name)
    if existing:
        folder = existing.folder
    else:
        base = slug(name, "avatar")
        folder, n = AVATARS_DIR / base, 2
        while folder.exists():
            folder, n = AVATARS_DIR / f"{base}-{n}", n + 1
    folder.mkdir(parents=True, exist_ok=True)

    data = {"name": name, "face_kind": face_kind, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    for key, stem in AVATAR_FILES.items():
        source = files.get(key)
        if source and Path(source).is_file():
            target = folder / f"{stem}{Path(source).suffix.lower()}"
            if Path(source).resolve() != target.resolve():
                for old in folder.glob(f"{stem}.*"):
                    old.unlink()
                shutil.copy2(source, target)
            data[key] = target.name
        else:
            for old in folder.glob(f"{stem}.*"):
                old.unlink()
    _write_json(folder / "avatar.json", data)
    return _load_avatar(folder / "avatar.json")


def delete_avatar(name: str) -> None:
    avatar = find_avatar(name)
    if avatar and avatar.folder.resolve().parent == AVATARS_DIR.resolve():
        shutil.rmtree(avatar.folder, ignore_errors=True)


# --- finished videos ---------------------------------------------------------------

@dataclass
class Video:
    path: Path
    created: float
    meta: dict


def new_video_path(text: str, taken: set = frozenset()) -> Path:
    """output/<date>_<time>_<first words>.mp4, never an existing or queued one."""
    base = f"{time.strftime('%Y-%m-%d_%H%M%S')}_{slug(text, 'video', 32)}"
    path, n = OUTPUT_DIR / f"{base}.mp4", 2
    while path.exists() or str(path) in taken:
        path, n = OUTPUT_DIR / f"{base}-{n}.mp4", n + 1
    return path


def write_video_meta(video: Path, meta: dict) -> None:
    _write_json(Path(video).with_suffix(".json"), meta)


def list_videos() -> list:
    if not OUTPUT_DIR.is_dir():
        return []
    videos = []
    for path in OUTPUT_DIR.glob("*.mp4"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        meta = _read_json(path.with_suffix(".json"))
        videos.append(Video(path=path, created=meta.get("created_ts", mtime), meta=meta))
    return sorted(videos, key=lambda v: v.created, reverse=True)


def delete_video(path: str) -> None:
    video = Path(path).resolve()
    if video.parent != OUTPUT_DIR.resolve() or video.suffix.lower() != ".mp4":
        raise ValueError("Only videos in the output folder can be deleted here.")
    for related in (video, video.with_suffix(".json"), LOGS_DIR / f"{video.stem}.log"):
        related.unlink(missing_ok=True)


# --- settings --------------------------------------------------------------------

def load_settings() -> dict:
    return _read_json(SETTINGS_PATH)


def save_settings(values: dict) -> None:
    _write_json(SETTINGS_PATH, {**load_settings(), **values})


def open_folder(folder: Path) -> None:
    """Show a folder in Explorer / Finder / the desktop's file manager."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(folder)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
