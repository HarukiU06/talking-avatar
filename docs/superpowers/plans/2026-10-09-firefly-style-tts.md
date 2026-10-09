# Firefly-style Speech Controls + Style-Bert-VITS2 Japanese Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inline pauses, emotion tags, emphasis, pronunciation/accent fixes and speed/pitch for every TTS backend, plus a Style-Bert-VITS2 JP-Extra backend trained on the user's voice so Japanese pitch accent is right.

**Architecture:** A pure-Python parser (`script_markup.py`) turns marked-up text into segments; each backend synthesizes one clip per segment and `make_avatar.py` splices them with the requested pauses. SBV2 lives in its own venv and is driven only through `sbv2_synth.py` / `train_sbv2.py` subprocesses, like every other model repo here. `--tts auto` (new default) sends `ja` to SBV2 when a model trained on the same voice file (matched by SHA-256) exists. Any speech that isn't untouched Chatterbox output gets Chatterbox's own Perth watermark as the last audio step.

**Tech Stack:** Python 3.11, Chatterbox (`.venv`), Style-Bert-VITS2 JP-Extra (`.venv-sbv2`, torch 2.8.0+cu128), pyopenjtalk, ffmpeg `rubberband`, `perth`, pytest, Gradio (existing app).

**Spec:** `docs/superpowers/specs/2026-10-09-firefly-style-tts-design.md`

## Global Constraints

- Cross-venv rule: nothing from Style-Bert-VITS2 is ever imported into `.venv`; it runs only as `subprocess.run([get_venv_python(SBV2_VENV), ...], cwd=SBV2_DIR)`. `sbv2_synth.py` may import our own pure-Python `script_markup.py`.
- Never edit files inside the `Style-Bert-VITS2/` checkout (gitignored, vanishes on re-clone). Fixes go in `setup_sbv2.sh`. Adding a new file there (like `requirements.filtered.txt`, as `setup_seedvc.sh` does) is fine.
- `.venv-sbv2`: Python 3.11, `torch==2.8.0` + `torchaudio==2.8.0` from the cu128 index, installed **last** (SBV2 pins `torch<2.4`, which lacks sm_120; 2.9+ needs torchcodec + shared FFmpeg).
- `--tts` default `auto`; choices `auto|chatterbox|xtts|sbv2`. `auto` = `ja` → SBV2 if a trained model resolves, else Chatterbox (with a printed note); every other language → Chatterbox.
- Text with no markup and `--pause-ms 0` = exactly one TTS call (current behaviour).
- `--speed` 0.5–2.0 (default 1.0, multiplier); `--pitch` −12…+12 (default 0, semitones). `[pause N]` 0–10000 ms. Emotion intensity 0–3 (default 1).
- Watermark (`perth.PerthImplicitWatermarker`) is the last audio step — after Seed-VC, before the video engine — whenever the speech is not untouched Chatterbox output (`backend != "chatterbox" or speed != 1 or pitch != 0 or voice_convert`). Measured this session: rubberband erases a Perth mark (1.0 → 0.0); unmarked human speech reads 0.0, marked 1.0. Never add code that removes a watermark.
- Write audio with `soundfile`, not `torchaudio.save()`.
- SBV2 subprocesses get `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` (Japanese text in logs; Japanese-locale Windows consoles are cp932).
- Personal media and models never get committed: `.gitignore` gains `Style-Bert-VITS2/`, `models/`, `pronunciations.yaml`.
- Tests: `.venv/Scripts/python.exe -m pytest` from the repo root (Windows paths; on Linux/macOS `.venv/bin/python`).
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Pasted text that contains markup characters** ("5 * 3 = 15", "[Applause]") must fail with a clear message and character offset — and the app must say so *before* queuing a job, not minutes into it. Pinned in Task 4 (`test_errors`) and Task 8 (`test_problems_flags_markup_before_queueing`).
2. **The app's copy of a voice** (`avatars/<name>/voice.wav`, same bytes, different path and stem) must resolve to the model trained on the original file. Pinned in Task 2 (`test_model_found_by_content_for_a_renamed_copy`).
3. **A voice file rewritten in place** must not hit a stale hash cache and silently bind to the old model. Pinned in Task 2 (`test_sha256_changes_when_file_rewritten`).
4. **Retraining an existing `--name` from a different recording** must be refused, or the model folder would claim a voice it wasn't trained on. Pinned in Task 3 (`test_refuses_a_name_trained_from_another_recording`).
5. **A script that is only markup or starts/ends with a pause** ("[pause 500]", "[pause 500]Hi.") must keep the leading/trailing silence, and a script with nothing to say must raise a clear error instead of sending "" to a TTS. Pinned in Task 4 (`test_leading_and_trailing_pause`) and Task 5 (`test_script_segments_rejects_nothing_to_say`).

## File Map

| File | Status | Responsibility |
|---|---|---|
| `setup_sbv2.sh` | new | Clone SBV2, build `.venv-sbv2`, download BERT/pretrained/stock models |
| `script_markup.py` | new | Parse markup → `Segment`s; pronunciation dictionary; mora/accent helpers |
| `train_sbv2.py` | new | Resumable training driver (runs in `.venv`, steps in `.venv-sbv2`) |
| `sbv2_styles.py` | new | Cluster style vectors → `style_vectors.npy`, `styles.yaml`, samples (runs in `.venv-sbv2`) |
| `sbv2_synth.py` | new | Inference worker (runs in `.venv-sbv2`), job.json in / .result out |
| `make_avatar.py` | modify | Constants, model binding, audio helpers, per-backend segment synthesis, routing, flags, watermark |
| `ui/system.py`, `ui/pipeline.py`, `ui/layout.py`, `ui/jobs.py` | modify | SBV2 part, Auto choice, speed/pitch sliders, markup hint, early markup check |
| `tests/conftest.py` + `tests/test_*.py` | new | Unit tests |
| `pronunciations.example.yaml` | new | Example dictionary |
| `README.md`, `config.example.yaml`, `.gitignore` | modify | Docs, batch keys, ignores |

Task order: Task 1 → Task 2 → Task 3 (training then runs ~1–2 h **in the background**) → Tasks 4, 5, 6, 7, 8 while it trains → Task 9 once the trained model exists.

---

### Task 1: Style-Bert-VITS2 environment and stock-model smoke test

**Files:**
- Create: `setup_sbv2.sh`
- Modify: `.gitignore`
- Modify: `make_avatar.py` (constants block after the Seed-VC constants, ~line 76)

**Interfaces:**
- Produces: `make_avatar.SBV2_DIR`, `SBV2_VENV`, `SBV2_MODELS`, `SBV2_DATA` (str paths), `SBV2_LANGS = {"ja"}`, `sbv2_installed() -> bool`.

- [ ] **Step 1: Add the constants and `sbv2_installed()` to `make_avatar.py`**

Insert after the `SEEDVC_VENV = ...` line:

```python
# Style-Bert-VITS2 JP-Extra, the Japanese backend behind --tts sbv2 (and
# --tts auto for ja). Created by setup_sbv2.sh. Unlike the other TTS engines it
# can't clone zero-shot: train_sbv2.py trains it once on a recording of you,
# into SBV2_MODELS/<name>/. SBV2_DATA holds the sliced/transcribed training set.
SBV2_DIR = os.environ.get("SBV2_DIR", str(Path(__file__).parent / "Style-Bert-VITS2"))
SBV2_VENV = os.environ.get("SBV2_VENV", str(Path(__file__).parent / ".venv-sbv2"))
SBV2_MODELS = os.environ.get("SBV2_MODELS", str(Path(__file__).parent / "models" / "sbv2"))
SBV2_DATA = os.environ.get("SBV2_DATA", str(Path(__file__).parent / "models" / "sbv2-data"))
# JP-Extra is trained on Japanese only.
SBV2_LANGS = {"ja"}
```

And after `get_venv_python()` (search `def get_venv_python`), add:

```python
def sbv2_installed() -> bool:
    """setup_sbv2.sh has run: the checkout and its venv both exist."""
    return ((Path(SBV2_DIR) / "style_bert_vits2").is_dir()
            and any((Path(SBV2_VENV) / exe).exists() for exe in ("Scripts/python.exe", "bin/python")))
```

- [ ] **Step 2: Add the ignores**

In `.gitignore`, under `# Vendored third-party model repos (cloned by setup*.sh)` add `Style-Bert-VITS2/`; under `# User-specific inputs/outputs and local config` add:

```
models/
pronunciations.yaml
```

- [ ] **Step 3: Write `setup_sbv2.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "== Style-Bert-VITS2 setup (Japanese with correct pitch accent, --tts sbv2) =="

# Chatterbox and XTTS-v2 are multilingual models with no explicit model of
# Japanese pitch accent (高低アクセント); both were rejected by ear for Japanese
# intonation even with every reading spelled out in kana. Style-Bert-VITS2
# JP-Extra predicts accent from pyopenjtalk and accepts a per-mora override, so
# a wrong accent can also be fixed by hand ({雨|あ'め} in the script).
#
# It has no zero-shot cloning: train_sbv2.py trains it once on ~10 minutes of
# you speaking Japanese. After that, --lang ja uses it automatically.

if [ ! -d "Style-Bert-VITS2" ]; then
  git clone https://github.com/litagin02/Style-Bert-VITS2.git
fi

# 1. Python virtual environment. 3.11 is what the rest of this project uses,
# and pyopenjtalk-dict ships a prebuilt cp311 Windows wheel (no compiler needed).
PYTHON=""
for candidate in "py -3.11" python3.11 "py -3.10" python3.10 python3 python; do
  cmd=($candidate)
  if command -v "${cmd[0]}" &> /dev/null && "${cmd[@]}" --version &> /dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo "Python 3.10+ is required but wasn't found (or only the Microsoft"
  echo "Store alias stub is on PATH). Install Python from python.org, or"
  echo "disable the stub under Settings > Apps > Advanced app settings >"
  echo "App execution aliases, then re-run this script."
  exit 1
fi

echo "Using $($PYTHON --version) ($PYTHON)"
PYTHON_CMD=($PYTHON)
"${PYTHON_CMD[@]}" -m venv .venv-sbv2

if [ -f ".venv-sbv2/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv-sbv2/bin/activate
else
  # shellcheck disable=SC1091
  source .venv-sbv2/Scripts/activate
fi
python -m pip install --upgrade pip

# SBV2's configs/paths.yml needs a path Python understands; Git Bash's /c/...
# form would be read as a relative path on Windows.
ROOT_NATIVE="$(python -c 'import os; print(os.getcwd())')"

cd Style-Bert-VITS2
# requirements.txt pins torch<2.4 / torchaudio<2.4. Those builds have no sm_120
# kernels, so on an RTX 50-series card torch.cuda.is_available() is simply
# False. Filter the torch family out (same pattern as setup_seedvc.sh) and
# install a CUDA 12.8 build afterwards.
grep -viE '^[[:space:]]*torch(vision|audio)?([[:space:]]|[=<>!~]|$)' requirements.txt > requirements.filtered.txt
python -m pip install -r requirements.filtered.txt
cd ..

# Installed last so nothing in the requirements can pull torch back down.
# Pinned below 2.9 like setup_seedvc.sh/setup_xtts.sh: 2.9's torchaudio needs
# torchcodec and FFmpeg's shared libraries, which static Windows ffmpeg builds
# don't provide. torchvision only if something above pulled it in, pinned to
# match, or it drags torch back with it.
TORCH_PKGS=("torch==2.8.0" "torchaudio==2.8.0")
if python -m pip show torchvision > /dev/null 2>&1; then
  TORCH_PKGS+=("torchvision==0.23.0")
fi
python -m pip install "${TORCH_PKGS[@]}" --index-url https://download.pytorch.org/whl/cu128

# 2. Models: the Japanese BERT, the JP-Extra pretrained base that training
# starts from, and the stock JVNV voices (used to smoke-test the install before
# spending an hour on training). Trained voices go to models/sbv2/.
cd Style-Bert-VITS2
python initialize.py --dataset_root "$ROOT_NATIVE/models/sbv2-data" --assets_root "$ROOT_NATIVE/models/sbv2"
cd ..

echo ""
echo "Style-Bert-VITS2 setup complete. Next, train it on a recording of you"
echo "speaking Japanese (about 10 minutes is plenty):"
echo "  python train_sbv2.py --voice voice_samples/me.wav"
```

- [ ] **Step 4: Run it**

Run: `bash ./setup_sbv2.sh 2>&1 | tail -30` (timeout 30 min; several GB download).
Expected: ends with "Style-Bert-VITS2 setup complete."; `models/sbv2/` and `Style-Bert-VITS2/model_assets/jvnv-F1-jp/` exist.

If pip reports a dependency it cannot resolve against torch 2.8, fix it **in `setup_sbv2.sh`** (an extra pin or a filtered line), never inside the checkout, and re-run.

- [ ] **Step 5: Verify the GPU build**

Run: `.venv-sbv2/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), 'sm_120' in torch.cuda.get_arch_list())"`
Expected: `2.8.0+cu128 True True`

- [ ] **Step 6: Smoke-render with a stock voice**

Write this to the system temp dir as `sbv2_smoke.py` (throwaway, not committed):

```python
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
import soundfile as sf
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.tts_model import TTSModel

pyopenjtalk_worker.initialize_worker()
update_dict()
bert_models.load_model(Languages.JP, device_map="cuda")
bert_models.load_tokenizer(Languages.JP)
d = Path("model_assets/jvnv-F1-jp")
model = TTSModel(model_path=next(d.glob("*.safetensors")), config_path=d / "config.json",
                 style_vec_path=d / "style_vectors.npy", device="cuda")
sr, audio = model.infer(text="今日はいい天気ですね。", language=Languages.JP)
sf.write(sys.argv[1], audio, sr)
print("ok", sr, round(len(audio) / sr, 2))
pyopenjtalk_worker.terminate_worker()
```

Run (cwd must be the checkout): `cd Style-Bert-VITS2 && PYTHONUTF8=1 ../.venv-sbv2/Scripts/python.exe "$TEMP/sbv2_smoke.py" ../output/sbv2_smoke.wav; cd ..`
Expected: `ok 44100 <1.5–3.0>` and a playable `output/sbv2_smoke.wav`. Any import/CUDA error here is a setup bug: fix `setup_sbv2.sh`, re-run Step 4.

- [ ] **Step 7: Commit**

```bash
git add setup_sbv2.sh .gitignore make_avatar.py
git commit -m "Add setup_sbv2.sh: Style-Bert-VITS2 JP-Extra in its own venv" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Bind a trained model to a voice file (content hash)

**Files:**
- Create: `tests/conftest.py`, `tests/test_sbv2_binding.py`
- Modify: `make_avatar.py` (new functions after `sbv2_installed()`; `import hashlib` at the top)

**Interfaces:**
- Consumes: `SBV2_MODELS` (Task 1).
- Produces: `file_sha256(path: str) -> str`, `sbv2_model_ready(model_dir: Path) -> bool`, `find_sbv2_model(voice_sample: str | None, name: str | None = None) -> Path | None`.

- [ ] **Step 1: Install pytest and add the conftest**

Run: `.venv/Scripts/python.exe -m pip install pytest`

`tests/conftest.py`:

```python
import sys
from pathlib import Path

# make_avatar.py, script_markup.py and ui/ live at the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 2: Write the failing tests** — `tests/test_sbv2_binding.py`

```python
import json
from pathlib import Path

import make_avatar as ma


def _model(root: Path, name: str, sha: str | None = None, ready: bool = True) -> Path:
    d = root / name
    d.mkdir(parents=True)
    if ready:
        (d / "config.json").write_text("{}", encoding="utf-8")
        (d / f"{name}_e100_s1000.safetensors").write_bytes(b"x")
        (d / "styles.yaml").write_text("neutral: Neutral\n", encoding="utf-8")
    if sha:
        (d / "source.json").write_text(json.dumps({"sha256": sha}), encoding="utf-8")
    return d


def _voice(path: Path, data: bytes = b"RIFF-me-speaking") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_sha256_changes_when_file_rewritten(tmp_path):
    f = _voice(tmp_path / "v.wav", b"one")
    first = ma.file_sha256(str(f))
    f.write_bytes(b"a different take")
    assert ma.file_sha256(str(f)) != first


def test_model_found_by_content_for_a_renamed_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "SBV2_MODELS", str(tmp_path / "models"))
    voice = _voice(tmp_path / "voice_samples" / "newtest.wav")
    _model(tmp_path / "models", "newtest", sha=ma.file_sha256(str(voice)))
    copy = _voice(tmp_path / "avatars" / "me" / "voice.wav", voice.read_bytes())
    assert ma.find_sbv2_model(str(copy)) == tmp_path / "models" / "newtest"


def test_stem_fallback_without_source_json(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "SBV2_MODELS", str(tmp_path / "models"))
    _model(tmp_path / "models", "newtest")
    voice = _voice(tmp_path / "newtest.wav")
    assert ma.find_sbv2_model(str(voice)) == tmp_path / "models" / "newtest"


def test_other_voice_does_not_match(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "SBV2_MODELS", str(tmp_path / "models"))
    _model(tmp_path / "models", "newtest", sha="0" * 64)
    voice = _voice(tmp_path / "someone_else.wav")
    assert ma.find_sbv2_model(str(voice)) is None


def test_unfinished_model_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "SBV2_MODELS", str(tmp_path / "models"))
    voice = _voice(tmp_path / "newtest.wav")
    _model(tmp_path / "models", "newtest", sha=ma.file_sha256(str(voice)), ready=False)
    assert ma.find_sbv2_model(str(voice)) is None


def test_explicit_name(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "SBV2_MODELS", str(tmp_path / "models"))
    _model(tmp_path / "models", "other")
    assert ma.find_sbv2_model(None, "other") == tmp_path / "models" / "other"
    assert ma.find_sbv2_model(None, "missing") is None


def test_no_models_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "SBV2_MODELS", str(tmp_path / "nothing-here"))
    assert ma.find_sbv2_model(str(_voice(tmp_path / "v.wav"))) is None
```

- [ ] **Step 3: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sbv2_binding.py -v`
Expected: FAIL — `AttributeError: module 'make_avatar' has no attribute 'file_sha256'`.

- [ ] **Step 4: Implement**

Add `import hashlib` to the import block. After `sbv2_installed()`:

```python
_voice_hashes = {}  # (path, mtime_ns, size) -> sha256; the app re-checks on every change


def file_sha256(path: str) -> str:
    """SHA-256 of a file, cached per (path, mtime, size) so a rewrite is noticed."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if key not in _voice_hashes:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        _voice_hashes[key] = digest.hexdigest()
    return _voice_hashes[key]


def sbv2_model_ready(model_dir: Path) -> bool:
    """train_sbv2.py finished: styles.yaml is written by its last step."""
    return ((model_dir / "styles.yaml").is_file() and (model_dir / "config.json").is_file()
            and any(model_dir.glob("*.safetensors")))


def find_sbv2_model(voice_sample: str | None, name: str | None = None) -> Path | None:
    """The trained SBV2 voice for this voice sample, or None.

    By name when given; otherwise by content, then by file stem. Content comes
    first because the app stores every avatar's voice as avatars/<name>/voice.<ext>
    - a byte-for-byte copy under a stem that says nothing about whose it is.
    """
    root = Path(SBV2_MODELS)
    if name:
        return root / name if sbv2_model_ready(root / name) else None
    if not root.is_dir() or not voice_sample or not os.path.isfile(voice_sample):
        return None
    ready = [d for d in sorted(root.iterdir()) if d.is_dir() and sbv2_model_ready(d)]
    if not ready:
        return None
    digest = file_sha256(voice_sample)
    for model_dir in ready:
        try:
            source = json.loads((model_dir / "source.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if source.get("sha256") == digest:
            return model_dir
    by_stem = root / Path(voice_sample).stem
    return by_stem if by_stem in ready and not (by_stem / "source.json").exists() else None
```

(Stem fallback only applies to a model with no `source.json` — one that *has* a recorded hash belongs to that recording only. This is what `test_other_voice_does_not_match` and `test_stem_fallback_without_source_json` pin together.)

- [ ] **Step 5: Run to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sbv2_binding.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/test_sbv2_binding.py make_avatar.py
git commit -m "Bind trained SBV2 voices to voice files by content hash" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Training driver and style clustering; start training

**Files:**
- Create: `train_sbv2.py`, `sbv2_styles.py`, `tests/test_train_sbv2.py`

**Interfaces:**
- Consumes: `ma.SBV2_DIR/SBV2_VENV/SBV2_MODELS/SBV2_DATA`, `ma.sbv2_installed()`, `ma.get_venv_python()`, `ma.file_sha256()`, `ma.denoise()`.
- Produces: `models/sbv2/<name>/` containing `config.json` (with `data.style2id`), `<name>_e<E>_s<S>.safetensors`, `style_vectors.npy`, `source.json` (`{"sha256", "voice"}`), `styles.yaml` (`tag: StyleName` lines; `neutral: Neutral` always present), `style_samples/<StyleName>.wav`. In `train_sbv2.py`: `record_source(model_dir: Path, voice: str) -> None`, `drop_non_japanese(esd_list: Path) -> list[str]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_train_sbv2.py`

```python
import json

import pytest

import make_avatar as ma
import train_sbv2


def test_records_source_on_first_run(tmp_path):
    voice = tmp_path / "me.wav"
    voice.write_bytes(b"take one")
    train_sbv2.record_source(tmp_path / "model", str(voice))
    source = json.loads((tmp_path / "model" / "source.json").read_text(encoding="utf-8"))
    assert source["sha256"] == ma.file_sha256(str(voice))


def test_rerun_with_same_recording_is_fine(tmp_path):
    voice = tmp_path / "me.wav"
    voice.write_bytes(b"take one")
    train_sbv2.record_source(tmp_path / "model", str(voice))
    train_sbv2.record_source(tmp_path / "model", str(voice))


def test_refuses_a_name_trained_from_another_recording(tmp_path):
    first, second = tmp_path / "a.wav", tmp_path / "b.wav"
    first.write_bytes(b"take one")
    second.write_bytes(b"a completely different take")
    train_sbv2.record_source(tmp_path / "model", str(first))
    with pytest.raises(SystemExit, match="--name"):
        train_sbv2.record_source(tmp_path / "model", str(second))


def test_drop_non_japanese(tmp_path):
    esd = tmp_path / "esd.list"
    esd.write_text("a.wav|me|JP|こんにちは。\n"
                   "b.wav|me|JP|Hello there.\n"
                   "c.wav|me|JP|AIの話です。\n", encoding="utf-8")
    assert train_sbv2.drop_non_japanese(esd) == ["b.wav"]
    assert esd.read_text(encoding="utf-8").splitlines() == ["a.wav|me|JP|こんにちは。", "c.wav|me|JP|AIの話です。"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_train_sbv2.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train_sbv2'`.

- [ ] **Step 3: Write `train_sbv2.py`**

```python
#!/usr/bin/env python3
"""Train a Style-Bert-VITS2 JP-Extra voice on one recording of you.

    python train_sbv2.py --voice voice_samples/me.wav [--name me] [--epochs 100]

Afterwards `--lang ja` uses it automatically (make_avatar.py --tts auto, the
default) whenever --voice is this same recording - matched by content, so the
app's saved copy of it counts too.

Resumable: rerun the same command after an interruption and it continues at the
first step whose output is missing; training itself resumes from its last
checkpoint. Whisper's mistakes become training labels, so
`--stop-after transcribe` lets you correct models/sbv2-data/<name>/esd.list
before training on it.

Runs in .venv; every Style-Bert-VITS2 step runs in .venv-sbv2 as a subprocess
(see CLAUDE.md's cross-venv rule).
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import soundfile as sf

import make_avatar as ma

ROOT = Path(__file__).resolve().parent
STEPS = ("stage", "slice", "transcribe", "preprocess", "train", "styles")
_JAPANESE = re.compile(r"[぀-ヿ一-鿿]")
MIN_MINUTES = 5.0


def run_sbv2(args: list) -> None:
    cmd = [ma.get_venv_python(ma.SBV2_VENV), *args]
    print("Running Style-Bert-VITS2:", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    subprocess.run(cmd, cwd=ma.SBV2_DIR, check=True, env=env)


def record_source(model_dir: Path, voice: str) -> None:
    """Remember which recording this model is of; refuse to mix two recordings under one name."""
    model_dir.mkdir(parents=True, exist_ok=True)
    source_path = model_dir / "source.json"
    digest = ma.file_sha256(voice)
    if source_path.exists():
        recorded = json.loads(source_path.read_text(encoding="utf-8")).get("sha256")
        if recorded != digest:
            raise SystemExit(f"{model_dir} was trained from a different recording. "
                             "Pick another --name (or delete that folder to start over).")
        return
    source_path.write_text(json.dumps({"sha256": digest, "voice": os.path.abspath(voice)}, indent=2),
                           encoding="utf-8")


def drop_non_japanese(esd_list: Path) -> list:
    """Remove transcript lines with no Japanese in them; JP-Extra trains on Japanese only."""
    kept, dropped = [], []
    for line in esd_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        text = line.split("|", 3)[-1]
        (kept if _JAPANESE.search(text) else dropped).append(line)
    esd_list.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return [line.split("|", 1)[0] for line in dropped]


def stage(voice: str, staged: Path, denoise: float) -> None:
    """Mono 44.1 kHz wav, the rate JP-Extra trains at; optionally denoised first."""
    import librosa

    staged.parent.mkdir(parents=True, exist_ok=True)
    audio, _ = librosa.load(voice, sr=44100, mono=True)
    if denoise > 0:
        # The model learns the room along with the voice, exactly as a TTS
        # reference does (see make_avatar.denoise).
        audio = ma.denoise(audio, 44100, strength=denoise)
    sf.write(staged, audio, 44100)


def transcribe(name: str, data_dir: Path, device: str) -> None:
    # --use_hf_whisper: runs on the venv's torch 2.8. The default path uses the
    # pinned faster-whisper 0.10.1, whose CTranslate2 3.x build predates sm_120.
    run_sbv2(["transcribe.py", "--model_name", name, "--language", "jp", "--model", "large-v3",
              "--use_hf_whisper", "--device", device])
    esd = data_dir / "esd.list"
    dropped = drop_non_japanese(esd)
    if dropped:
        print(f"Dropped {len(dropped)} clip(s) with no Japanese in the transcript: {', '.join(dropped)}")
    minutes = sum(sf.info(data_dir / "raw" / line.split("|", 1)[0]).duration
                  for line in esd.read_text(encoding="utf-8").splitlines() if line.strip()) / 60
    print(f"Transcript: {esd} ({minutes:.1f} min of speech). Check it for mistakes before training.")
    if minutes < MIN_MINUTES:
        print(f"WARNING: under {MIN_MINUTES:.0f} minutes of usable speech; the voice will be less stable.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--voice", required=True, help="A recording of you speaking Japanese (~10 min is plenty)")
    ap.add_argument("--name", help="Model name (default: the voice file's name)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2, help="Raise if you have VRAM to spare (12 GB: 2-4)")
    ap.add_argument("--clusters", type=int, default=4, help="How many emotion styles to find in the recording")
    ap.add_argument("--denoise", type=float, default=0.0, help="noisereduce strength on the recording (0 = off)")
    ap.add_argument("--whisper-device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--stop-after", choices=STEPS)
    args = ap.parse_args()

    if not ma.sbv2_installed():
        raise SystemExit("Style-Bert-VITS2 isn't set up. Run ./setup_sbv2.sh first.")
    if not os.path.isfile(args.voice):
        raise SystemExit(f"No such file: {args.voice}")
    name = args.name or Path(args.voice).stem
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise SystemExit(f"--name {name!r}: use letters, digits, - and _ only (it becomes a folder name).")

    model_dir = Path(ma.SBV2_MODELS) / name
    data_dir = Path(ma.SBV2_DATA) / name
    staged = data_dir / "input" / f"{name}.wav"
    record_source(model_dir, args.voice)

    steps = {
        "stage": (lambda: staged.exists(),
                  lambda: stage(args.voice, staged, args.denoise)),
        "slice": (lambda: any((data_dir / "raw").glob("*.wav")),
                  lambda: run_sbv2(["slice.py", "--model_name", name, "-i", str(staged.parent),
                                    "-m", "2", "-M", "12"])),
        "transcribe": (lambda: (data_dir / "esd.list").exists(),
                       lambda: transcribe(name, data_dir, args.whisper_device)),
        "preprocess": (lambda: (data_dir / "train.list").exists() and (data_dir / "config.json").exists(),
                       lambda: run_sbv2(["preprocess_all.py", "-m", name, "--use_jp_extra",
                                         "-e", str(args.epochs), "-b", str(args.batch_size),
                                         "--yomi_error", "skip"])),
        "train": (lambda: any(model_dir.glob(f"{name}_e{args.epochs}_s*.safetensors")),
                  lambda: run_sbv2(["train_ms_jp_extra.py", "-m", str(data_dir),
                                    "-c", str(data_dir / "config.json"),
                                    "--assets_root", str(Path(ma.SBV2_MODELS))])),
        "styles": (lambda: (model_dir / "styles.yaml").exists(),
                   lambda: run_sbv2([str(ROOT / "sbv2_styles.py"), "--data", str(data_dir),
                                     "--model", str(model_dir), "--clusters", str(args.clusters)])),
    }
    for step in STEPS:
        done, run = steps[step]
        if done():
            print(f"[{step}] already done")
        else:
            print(f"[{step}] ...", flush=True)
            run()
        if step == args.stop_after:
            print(f"Stopped after {step}. Rerun the same command to continue.")
            return
    print(f"\nTrained: {model_dir}")
    print(f"Listen to {model_dir / 'style_samples'} and name the styles in {model_dir / 'styles.yaml'}.")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write `sbv2_styles.py`**

```python
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
```

- [ ] **Step 5: Run the unit tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_train_sbv2.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add train_sbv2.py sbv2_styles.py tests/test_train_sbv2.py
git commit -m "Add train_sbv2.py: resumable Style-Bert-VITS2 training on your recording" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Prepare the dataset and stop for transcript review**

Run: `.venv/Scripts/python.exe train_sbv2.py --voice voice_samples/newtest.wav --stop-after transcribe` (timeout 30 min)
Expected: `[stage] ... [slice] ... [transcribe] ...`, then `Transcript: ...esd.list (N min of speech)`, `Stopped after transcribe.`

Read `models/sbv2-data/newtest/esd.list`. Fix obvious mis-transcriptions in place (wrong kanji that would teach a wrong reading, hallucinated text on breaths/noise — delete such lines). Report to the user how many lines and minutes there are and what was fixed.

If transcription fails on CUDA, rerun with `--whisper-device cpu`.

- [ ] **Step 8: Start training in the background**

Run with `run_in_background: true`, timeout 7200000: `.venv/Scripts/python.exe train_sbv2.py --voice voice_samples/newtest.wav`
Note the start time. Continue with Task 4 while it runs. When it finishes, record wall-clock time and peak VRAM (from `nvidia-smi` sampled during the run) for the README. If it dies with CUDA OOM, rerun with `--batch-size 1` (resumes from the last checkpoint).

---

### Task 4: Script markup parser

**Files:**
- Create: `script_markup.py`, `tests/test_script_markup.py`

**Interfaces:**
- Produces:
  - `class MarkupError(ValueError)` with `.offset: int`
  - `@dataclass Piece(text: str, accent: int | None = None)`
  - `@dataclass Segment(pieces: list[Piece], emotion: str | None = None, intensity: float = 1.0, emphasis: bool = False, pause_after_ms: int = 0)` with property `text: str`
  - `parse(text: str, lang: str, pronunciations: dict | None = None) -> list[Segment]` (`pronunciations` = `{word: reading}` for this language)
  - `split_at_sentences(segments: list[Segment], pause_ms: int) -> list[Segment]`
  - `load_pronunciations(path: str | None) -> dict` (`{lang: {word: reading}}`)
  - `count_morae(kana: str) -> int`, `accent_tones(n_morae: int, nucleus: int) -> list[int]`

- [ ] **Step 1: Write the failing tests** — `tests/test_script_markup.py`

```python
import pytest

from script_markup import (MarkupError, Piece, accent_tones, count_morae, load_pronunciations,
                           parse, split_at_sentences)


def texts(segments):
    return [s.text for s in segments]


def test_plain_text_is_one_segment():
    segs = parse("Hello there. How are you? Fine!", "en")
    assert texts(segs) == ["Hello there. How are you? Fine!"]
    assert (segs[0].pause_after_ms, segs[0].emotion, segs[0].emphasis) == (0, None, False)


def test_japanese_plain_text_is_one_segment():
    assert texts(parse("こんにちは。今日はいい天気ですね。", "ja")) == ["こんにちは。今日はいい天気ですね。"]


@pytest.mark.parametrize("tag,ms", [("[pause 800]", 800), ("[pause 800ms]", 800), ("[pause 1.5s]", 1500),
                                    ("[ pause 250 ]", 250), ("[PAUSE 300]", 300)])
def test_pause_forms(tag, ms):
    segs = parse(f"One.{tag}Two.", "en")
    assert texts(segs) == ["One.", "Two."]
    assert [s.pause_after_ms for s in segs] == [ms, 0]


def test_leading_and_trailing_pause():
    segs = parse("[pause 500]Hi.[pause 200]", "en")
    assert [(s.text, s.pause_after_ms) for s in segs] == [("", 500), ("Hi.", 200)]


def test_back_to_back_pauses_add_up():
    assert parse("A.[pause 300][pause 200]B.", "en")[0].pause_after_ms == 500


def test_emotion_span():
    segs = parse("Hi. [happy]Great news![/happy] Bye.", "en")
    assert [(s.text, s.emotion) for s in segs] == [("Hi.", None), ("Great news!", "happy"), ("Bye.", None)]


def test_emotion_intensity():
    assert parse("[sad 0.5]Oh.[/sad]", "en")[0].intensity == 0.5


def test_pause_inside_emotion_keeps_emotion_on_both_sides():
    segs = parse("[happy]A.[pause 300]B.[/happy]", "en")
    assert [(s.text, s.emotion, s.pause_after_ms) for s in segs] == [("A.", "happy", 300), ("B.", "happy", 0)]


def test_emphasis_inside_emotion():
    segs = parse("[happy]I am *very* glad[/happy]", "en")
    assert [(s.text, s.emotion, s.emphasis) for s in segs] == [
        ("I am", "happy", False), ("very", "happy", True), ("glad", "happy", False)]


def test_adjacent_tags_make_no_empty_segments():
    assert texts(parse("[happy]A[/happy] [sad]B[/sad]", "en")) == ["A", "B"]


def test_pronunciation_inline_stays_in_one_segment():
    segs = parse("Hi {Ueno|oo-EH-no}!", "en")
    assert texts(segs) == ["Hi oo-EH-no!"]


def test_trailing_zero_is_text_outside_japanese():
    assert texts(parse("{v10|version 10}", "en")) == ["version 10"]


@pytest.mark.parametrize("reading,kana,accent", [("あ'め", "あめ", 1), ("あめ0", "あめ", 0), ("はし'", "はし", 2),
                                                 ("あめ", "あめ", None), ("きょ'う", "きょう", 1)])
def test_japanese_accent(reading, kana, accent):
    seg, = parse(f"今日は{{雨|{reading}}}です", "ja")
    assert seg.pieces[1] == Piece(kana, accent)
    assert seg.text == f"今日は{kana}です"


def test_dictionary_applies_longest_match_first():
    seg, = parse("雨宮さんと雨", "ja", {"雨": "あ'め", "雨宮": "あまみや"})
    assert [p.text for p in seg.pieces] == ["あまみや", "さんと", "あめ"]
    assert seg.pieces[2].accent == 1


def test_inline_beats_dictionary():
    seg, = parse("{雨|あめ0}", "ja", {"雨": "あ'め"})
    assert seg.pieces == [Piece("あめ", 0)]


def test_dictionary_respects_word_boundaries_outside_japanese():
    seg, = parse("Ueno and Uenomachi", "en", {"Ueno": "oo-EH-no"})
    assert seg.text == "oo-EH-no and Uenomachi"


def test_escapes():
    assert texts(parse(r"5 \* 3 \[sic\] \{x\}", "en")) == ["5 * 3 [sic] {x}"]


@pytest.mark.parametrize("text,lang,message", [
    ("Hello [happy]there", "en", "never closed"),
    ("[happy]a[/sad]", "en", "doesn't close"),
    ("[happy]a[sad]b[/sad][/happy]", "en", "can't nest"),
    ("a [Applause] b", "en", "unknown tag"),
    ("a [applause] b", "en", "never closed"),
    ("[pause soon]", "en", "unknown tag"),
    ("A[pause 11s]", "en", "10000"),
    ("[happy 4]a[/happy]", "en", "between 0 and 3"),
    ("5 * 3 = 15", "en", "never closed"),
    ("a [pause 3", "en", "no closing"),
    ("{雨}", "ja", "word|reading"),
    ("{雨|}", "ja", "word|reading"),
    ("{Ueno|oo'eh}", "en", "Japanese"),
    ("{雨|a'me}", "ja", "kana"),
    ("{雨|'あめ}", "ja", "after a mora"),
    ("{雨|あ'め'}", "ja", "one accent"),
])
def test_errors(text, lang, message):
    with pytest.raises(MarkupError, match=message):
        parse(text, lang)


def test_error_reports_offset():
    with pytest.raises(MarkupError) as err:
        parse("Hello [pause soon] world", "en")
    assert err.value.offset == 6
    assert "character 6" in str(err.value)


def test_split_at_sentences_matches_old_pause_ms():
    segs = split_at_sentences(parse("One. Two! Three?", "en"), 450)
    assert [(s.text, s.pause_after_ms) for s in segs] == [("One.", 450), ("Two!", 450), ("Three?", 0)]


def test_split_keeps_explicit_pause_and_emotion():
    segs = split_at_sentences(parse("[sad]One. Two.[/sad][pause 900]Three.", "en"), 450)
    assert [(s.text, s.emotion, s.pause_after_ms) for s in segs] == [
        ("One.", "sad", 450), ("Two.", "sad", 900), ("Three.", None, 0)]


@pytest.mark.parametrize("n,nucleus,tones", [(2, 0, [0, 1]), (2, 1, [1, 0]), (3, 2, [0, 1, 0]),
                                             (3, 3, [0, 1, 1]), (1, 0, [0]), (1, 1, [1])])
def test_accent_tones(n, nucleus, tones):
    assert accent_tones(n, nucleus) == tones


def test_accent_tones_rejects_nucleus_past_the_word():
    with pytest.raises(ValueError):
        accent_tones(2, 3)


@pytest.mark.parametrize("kana,n", [("きょう", 2), ("がっこう", 4), ("ラーメン", 4), ("あめ", 2)])
def test_count_morae(kana, n):
    assert count_morae(kana) == n


def test_load_pronunciations(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("ja:\n  雨: \"あ'め\"\nen:\n  Ueno: oo-EH-no\n", encoding="utf-8")
    assert load_pronunciations(str(f)) == {"ja": {"雨": "あ'め"}, "en": {"Ueno": "oo-EH-no"}}


def test_load_pronunciations_rejects_wrong_shape(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("- just a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="language"):
        load_pronunciations(str(f))
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_script_markup.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'script_markup'`.

- [ ] **Step 3: Implement `script_markup.py`**

```python
"""Firefly-style script markup, parsed into segments for the TTS backends.

    [pause 800]  [pause 800ms]  [pause 1.5s]    exact silence here (max 10 s)
    [happy]...[/happy]  [happy 1.5]...[/happy]  emotion span, intensity 0-3
    *word*                                      emphasis
    {surface|reading}                           say `reading` instead
    {雨|あ'め}  {飴|あめ0}                       Japanese accent: the pitch drops
                                                after the mora before ', or 0 = flat
    \\[  \\]  \\*  \\{  \\}  \\|  \\\\                  literal characters

The text is cut only at pauses and at emotion/emphasis boundaries. A run of
plain text, however many sentences long, stays one segment and so one TTS call:
splicing per sentence made speech sound mechanical (see
make_avatar.synthesize_speech), so splits happen only where the script asks.

Pure Python, no model imports: make_avatar.py uses it in .venv and
sbv2_synth.py imports accent_tones from it in .venv-sbv2.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_PAUSE_MS = 10_000
MAX_INTENSITY = 3.0

_PAUSE = re.compile(r"pause\s+(\d+(?:\.\d+)?)\s*(ms|s)?", re.IGNORECASE)
_OPEN = re.compile(r"([a-z0-9_]+)(?:\s+(\d+(?:\.\d+)?))?")
_CLOSE = re.compile(r"/([a-z0-9_]+)")
_ESCAPABLE = set("[]*{}|\\")
# The sentence rule --pause-ms has always used: ./!/? followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Small kana fuse with the kana before them into one mora; っ, ん and ー are
# morae of their own.
_SMALL_KANA = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")
_KANA = re.compile(r"[ぁ-ゖァ-ヺー]+")


class MarkupError(ValueError):
    """Malformed markup, reported with where it is in the text."""

    def __init__(self, message: str, text: str, offset: int):
        snippet = text[max(0, offset - 12): offset + 12].replace("\n", " ")
        super().__init__(f"{message} (character {offset}: ...{snippet}...)")
        self.offset = offset


@dataclass
class Piece:
    text: str
    # None: the TTS decides. 0: flat (heiban). n >= 1: pitch drops after mora n.
    accent: int | None = None


@dataclass
class Segment:
    pieces: list = field(default_factory=list)
    emotion: str | None = None
    intensity: float = 1.0
    emphasis: bool = False
    pause_after_ms: int = 0

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.pieces)


def count_morae(kana: str) -> int:
    return sum(1 for ch in kana if ch not in _SMALL_KANA)


def accent_tones(n_morae: int, nucleus: int) -> list:
    """Tokyo-dialect pitch per mora (0 low, 1 high) for an accent nucleus.

    0 (heiban): low, then high to the end. 1 (atamadaka): high, then low.
    n > 1: low, high through mora n, low after.
    """
    if n_morae < 1 or not 0 <= nucleus <= n_morae:
        raise ValueError(f"accent after mora {nucleus} doesn't fit a {n_morae}-mora word")
    if nucleus == 0:
        return [0] + [1] * (n_morae - 1)
    if nucleus == 1:
        return [1] + [0] * (n_morae - 1)
    return [0] + [1] * (nucleus - 1) + [0] * (n_morae - nucleus)


def parse_reading(reading: str, lang: str) -> Piece:
    """The reading half of {surface|reading}, with Japanese accent marks read off it."""
    if lang != "ja":
        if "'" in reading:
            raise ValueError("accent marks (') are for Japanese readings only")
        return Piece(reading)
    if reading.count("'") > 1:
        raise ValueError("a reading can mark only one accent nucleus (')")
    if "'" in reading:
        before, after = reading.split("'")
        if not _KANA.fullmatch(before + after):
            raise ValueError(f"a reading with an accent mark must be all kana: {reading!r}")
        if not before:
            raise ValueError("' goes after a mora: the pitch drops after it")
        return Piece(before + after, count_morae(before))
    if len(reading) > 1 and reading.endswith("0") and _KANA.fullmatch(reading[:-1]):
        return Piece(reading[:-1], 0)
    return Piece(reading)


def load_pronunciations(path: str | None) -> dict:
    """{lang: {word: reading}} from YAML; pronunciations.yaml next to this file by default, {} if absent."""
    if path is None:
        default = Path(__file__).resolve().parent / "pronunciations.yaml"
        if not default.exists():
            return {}
        path = str(default)
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or not all(isinstance(words, dict) for words in data.values()):
        raise ValueError(f"{path}: expected language codes mapping words to readings, "
                         "e.g.  ja: {雨: \"あ'め\"}")
    return {str(lang): {str(w): str(r) for w, r in words.items()} for lang, words in data.items()}


def _dictionary_pieces(text: str, lang: str, pron: dict) -> list:
    if not pron:
        return [Piece(text)]
    alternatives = "|".join(re.escape(k) for k in sorted(pron, key=len, reverse=True))
    # Japanese/Chinese have no spaces to mark word edges; elsewhere "Ueno" must
    # not rewrite the start of "Uenomachi".
    pattern = alternatives if lang in ("ja", "zh") else rf"(?<!\w)(?:{alternatives})(?!\w)"
    pieces, pos = [], 0
    for match in re.finditer(pattern, text):
        if match.start() > pos:
            pieces.append(Piece(text[pos:match.start()]))
        try:
            pieces.append(parse_reading(pron[match.group(0)], lang))
        except ValueError as exc:
            raise ValueError(f"pronunciations: {match.group(0)}: {exc}") from None
        pos = match.end()
    if pos < len(text):
        pieces.append(Piece(text[pos:]))
    return pieces


def _trim(pieces: list) -> list:
    out = [Piece(p.text, p.accent) for p in pieces if p.text]
    while out and out[0].accent is None and not out[0].text.strip():
        out.pop(0)
    while out and out[-1].accent is None and not out[-1].text.strip():
        out.pop()
    if out:
        out[0].text = out[0].text.lstrip()
        out[-1].text = out[-1].text.rstrip()
    return out


def parse(text: str, lang: str, pronunciations: dict | None = None) -> list:
    """Marked-up script -> segments. Raises MarkupError on malformed markup."""
    pron = pronunciations or {}
    segments, pieces, plain = [], [], []
    emotion = None       # (name, intensity, offset of its tag)
    emphasis_at = None   # offset of the open *, while inside one

    def flush_plain():
        if plain:
            pieces.extend(_dictionary_pieces("".join(plain), lang, pron))
            plain.clear()

    def flush_segment():
        flush_plain()
        trimmed = _trim(pieces)
        pieces.clear()
        if trimmed:
            segments.append(Segment(trimmed, emotion[0] if emotion else None,
                                    emotion[1] if emotion else 1.0, emphasis_at is not None))

    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in _ESCAPABLE:
            plain.append(text[i + 1])
            i += 2
            continue
        if ch == "[":
            end = text.find("]", i)
            if end < 0:
                raise MarkupError("'[' has no closing ']'", text, i)
            body = text[i + 1:end].strip()
            if m := _PAUSE.fullmatch(body):
                flush_segment()
                ms = float(m.group(1)) * (1000 if (m.group(2) or "ms").lower() == "s" else 1)
                if ms > MAX_PAUSE_MS:
                    raise MarkupError(f"pauses are at most {MAX_PAUSE_MS} ms", text, i)
                if segments:
                    segments[-1].pause_after_ms += int(round(ms))
                else:
                    segments.append(Segment([], pause_after_ms=int(round(ms))))
            elif m := _CLOSE.fullmatch(body):
                if not emotion or emotion[0] != m.group(1):
                    raise MarkupError(f"[/{m.group(1)}] doesn't close an open [{m.group(1)}]", text, i)
                flush_segment()
                emotion = None
            elif m := _OPEN.fullmatch(body):
                if emotion:
                    raise MarkupError(f"[{m.group(1)}] is inside [{emotion[0]}]: emotion spans can't nest",
                                      text, i)
                intensity = float(m.group(2)) if m.group(2) else 1.0
                if intensity > MAX_INTENSITY:
                    raise MarkupError(f"intensity must be between 0 and {MAX_INTENSITY:g}", text, i)
                flush_segment()
                emotion = (m.group(1), intensity, i)
            else:
                raise MarkupError(f"unknown tag [{body}] (write \\[ for a literal bracket)", text, i)
            i = end + 1
            continue
        if ch == "*":
            flush_segment()
            emphasis_at = None if emphasis_at is not None else i
            i += 1
            continue
        if ch == "{":
            end = text.find("}", i)
            if end < 0:
                raise MarkupError("'{' has no closing '}'", text, i)
            surface, sep, reading = text[i + 1:end].partition("|")
            if not sep or not surface.strip() or not reading.strip() or "|" in reading:
                raise MarkupError("expected {word|reading}", text, i)
            flush_plain()
            try:
                pieces.append(parse_reading(reading.strip(), lang))
            except ValueError as exc:
                raise MarkupError(str(exc), text, i) from None
            i = end + 1
            continue
        plain.append(ch)
        i += 1

    if emotion:
        raise MarkupError(f"[{emotion[0]}] is never closed", text, emotion[2])
    if emphasis_at is not None:
        raise MarkupError("'*' is never closed (write \\* for a literal asterisk)", text, emphasis_at)
    flush_segment()
    return segments


def split_at_sentences(segments: list, pause_ms: int) -> list:
    """--pause-ms: also cut each segment at sentence ends, with pause_ms of silence there.

    Explicit [pause]s keep their own length; only boundaries inside a segment
    get pause_ms, so a tag mid-sentence ("I am *very* glad") adds no gap.
    """
    out = []
    for seg in segments:
        current = []
        for piece in seg.pieces:
            if piece.accent is not None:
                current.append(piece)
                continue
            parts = _SENTENCE_END.split(piece.text)
            for k, part in enumerate(parts):
                if part:
                    current.append(Piece(part))
                if k < len(parts) - 1:
                    out.append(Segment(current, seg.emotion, seg.intensity, seg.emphasis, pause_ms))
                    current = []
        out.append(Segment(current, seg.emotion, seg.intensity, seg.emphasis, seg.pause_after_ms))
    return [s for s in out if s.pieces or s.pause_after_ms]
```

Note on `test_errors` case `"a [Applause] b"`: `_OPEN` only matches lowercase names, so a capitalised bracket is an *unknown tag*; a lowercase one parses as an emotion and then fails as *never closed*. Both are clear errors with an offset; the backend's tag check (Task 5) catches a closed-but-unknown name.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_script_markup.py -q`
Expected: all pass (≈55 tests).

- [ ] **Step 5: Commit**

```bash
git add script_markup.py tests/test_script_markup.py
git commit -m "Add script_markup.py: pauses, emotion tags, emphasis, pronunciation and accent" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Speech helpers — splicing, speed/pitch, watermark, Chatterbox emotion presets

**Files:**
- Modify: `make_avatar.py` (`concat_with_pauses` ~line 215; new helpers after it; `import script_markup` at the top)
- Create: `tests/test_speech_helpers.py`

**Interfaces:**
- Consumes: `script_markup.parse/split_at_sentences/load_pronunciations/Segment` (Task 4).
- Produces:
  - `concat_with_pauses(clips: list, sr: int, pause_ms: int | list) -> np.ndarray` (int = gap between every pair, as before; list = silence after each clip)
  - `script_segments(text: str, lang: str, pause_ms: int = 0, pronunciations: str | None = None) -> list[Segment]`
  - `check_emotions(segments: list, valid: set, backend: str) -> None` (raises `ValueError`)
  - `CHATTERBOX_EMOTIONS: dict[str, tuple[float, float]]`, `CHATTERBOX_EMPHASIS = 0.3`
  - `chatterbox_params(segment, exaggeration: float, cfg_weight: float) -> tuple[float, float]`
  - `apply_speed_pitch(wav: str, speed: float = 1.0, pitch: float = 0.0) -> None` (in place)
  - `apply_watermark(wav: str) -> None` (in place)
  - `needs_watermark(backend: str, speed: float, pitch: float, voice_convert: bool) -> bool`

- [ ] **Step 1: Write the failing tests** — `tests/test_speech_helpers.py`

```python
import os

import numpy as np
import pytest
import soundfile as sf

import make_avatar as ma
from script_markup import Piece, Segment

ONES_A = np.ones(10, np.float32)
ONES_B = np.ones(5, np.float32)


def test_concat_single_gap_matches_old_behaviour():
    out = ma.concat_with_pauses([ONES_A, ONES_B], 1000, 3)
    assert len(out) == 18 and out[10:13].sum() == 0


def test_concat_per_clip_gaps_with_trailing_silence():
    out = ma.concat_with_pauses([ONES_A, ONES_B], 1000, [2, 4])
    assert len(out) == 21 and out[-4:].sum() == 0


def test_concat_leading_pause_from_an_empty_clip():
    out = ma.concat_with_pauses([np.zeros(0, np.float32), ONES_A], 1000, [5, 0])
    assert len(out) == 15 and out[:5].sum() == 0


def test_concat_rejects_mismatched_gaps():
    with pytest.raises(ValueError):
        ma.concat_with_pauses([ONES_A, ONES_B], 1000, [1])


def seg(emotion=None, intensity=1.0, emphasis=False):
    return Segment([Piece("x")], emotion, intensity, emphasis)


def test_chatterbox_plain_segment_keeps_run_settings():
    assert ma.chatterbox_params(seg(), 0.7, 0.3) == (0.7, 0.3)


def test_chatterbox_happy_full_and_half():
    ex, cfg = ma.chatterbox_params(seg("happy"), 0.5, 0.5)
    assert (round(ex, 3), round(cfg, 3)) == (0.8, 0.4)
    ex, _ = ma.chatterbox_params(seg("happy", 0.5), 0.5, 0.5)
    assert round(ex, 3) == 0.65


def test_chatterbox_keeps_a_low_cfg_low():
    _, cfg = ma.chatterbox_params(seg("happy"), 0.5, 0.2)
    assert round(cfg, 3) == 0.1


def test_chatterbox_emphasis_and_clamping():
    ex, _ = ma.chatterbox_params(seg(emphasis=True), 0.5, 0.5)
    assert round(ex, 3) == 0.8
    # 0.5 + (1.1 - 0.5) * 3 + 0.3 = 2.6 -> 2.0;  0.1 + (0.3 - 0.5) * 3 = -0.5 -> 0.0
    ex, cfg = ma.chatterbox_params(seg("excited", 3.0, emphasis=True), 0.5, 0.1)
    assert ex == 2.0 and cfg == 0.0


def test_check_emotions_lists_valid_tags():
    with pytest.raises(ValueError, match="calm, happy"):
        ma.check_emotions([seg("joyful")], {"happy", "calm"}, "Chatterbox")
    ma.check_emotions([seg("happy"), seg()], {"happy"}, "Chatterbox")


def test_script_segments_rejects_nothing_to_say():
    with pytest.raises(ValueError, match="nothing to say"):
        ma.script_segments("[pause 500]", "en")


def test_script_segments_pause_ms_splits_sentences():
    segs = ma.script_segments("One. Two.", "en", pause_ms=300)
    assert [(s.text, s.pause_after_ms) for s in segs] == [("One.", 300), ("Two.", 0)]


def test_script_segments_reads_pronunciation_file(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("en:\n  Ueno: oo-EH-no\n", encoding="utf-8")
    assert ma.script_segments("Hi Ueno", "en", pronunciations=str(f))[0].text == "Hi oo-EH-no"


def _tone(path, seconds=2.0, sr=24000):
    t = np.arange(int(sr * seconds)) / sr
    sf.write(path, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)


def test_apply_speed_pitch_changes_duration(tmp_path):
    wav = str(tmp_path / "a.wav")
    _tone(wav)
    ma.apply_speed_pitch(wav, speed=1.25, pitch=2.0)
    assert abs(sf.info(wav).duration - 1.6) < 0.05


def test_apply_speed_pitch_noop_leaves_file_untouched(tmp_path):
    wav = str(tmp_path / "a.wav")
    _tone(wav)
    before = os.stat(wav).st_mtime_ns
    ma.apply_speed_pitch(wav)
    assert os.stat(wav).st_mtime_ns == before


@pytest.mark.parametrize("backend,speed,pitch,vc,expected", [
    ("chatterbox", 1.0, 0.0, False, False), ("chatterbox", 1.1, 0.0, False, True),
    ("chatterbox", 1.0, -1.0, False, True), ("chatterbox", 1.0, 0.0, True, True),
    ("sbv2", 1.0, 0.0, False, True), ("xtts", 1.0, 0.0, False, True)])
def test_needs_watermark(backend, speed, pitch, vc, expected):
    assert ma.needs_watermark(backend, speed, pitch, vc) is expected
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_speech_helpers.py -q`
Expected: FAIL — `AttributeError: module 'make_avatar' has no attribute 'chatterbox_params'` (and the concat list cases fail).

- [ ] **Step 3: Implement**

Add `import script_markup` after `import soundfile as sf`. Replace `concat_with_pauses` with:

```python
def concat_with_pauses(clips: list, sr: int, pause_ms) -> np.ndarray:
    """Join clips with silence between them.

    pause_ms is either one gap for between every pair (--pause-ms sentence
    splicing) or a list with the silence to put after each clip (markup
    [pause]s, which may also lead or trail: an empty first clip with a pause
    after it is leading silence).
    """
    if not clips:
        return np.zeros(0, dtype=np.float32)
    gaps = list(pause_ms) if isinstance(pause_ms, (list, tuple)) else [pause_ms] * (len(clips) - 1) + [0]
    if len(gaps) != len(clips):
        raise ValueError(f"{len(clips)} clips but {len(gaps)} pauses")
    spliced = []
    for clip, gap in zip(clips, gaps):
        spliced.append(np.asarray(clip, dtype=np.float32))
        if gap:
            spliced.append(np.zeros(int(sr * gap / 1000), dtype=np.float32))
    return np.concatenate(spliced)
```

After it, add:

```python
# Emotion tags on Chatterbox: tag -> (exaggeration, cfg_weight), starting
# points picked by ear. A tag sets exaggeration (scaled by intensity from the
# run's own value), but only *nudges* cfg_weight relative to Chatterbox's 0.5
# default, so a low --cfg-weight chosen to stop accent carry-over stays low.
CHATTERBOX_EMOTIONS = {
    "neutral": (0.5, 0.5),
    "calm": (0.35, 0.5),
    "happy": (0.8, 0.4),
    "excited": (1.1, 0.3),
    "sad": (0.4, 0.5),
    "angry": (1.2, 0.3),
    "serious": (0.45, 0.6),
}
# Neither Chatterbox nor SBV2 has an emphasis input; *word* is approximated
# with a livelier delivery of just that span.
CHATTERBOX_EMPHASIS = 0.3


def chatterbox_params(segment, exaggeration: float, cfg_weight: float) -> tuple:
    """(exaggeration, cfg_weight) for one segment of a Chatterbox run."""
    ex, cfg = exaggeration, cfg_weight
    if segment.emotion:
        tag_ex, tag_cfg = CHATTERBOX_EMOTIONS[segment.emotion]
        ex += (tag_ex - exaggeration) * segment.intensity
        cfg += (tag_cfg - 0.5) * segment.intensity
    if segment.emphasis:
        ex += CHATTERBOX_EMPHASIS
    return min(max(ex, 0.25), 2.0), min(max(cfg, 0.0), 1.0)


def check_emotions(segments: list, valid: set, backend: str) -> None:
    """Fail before any synthesis if the script uses a tag this backend can't do."""
    for segment in segments:
        if segment.emotion and segment.emotion not in valid:
            raise ValueError(f"[{segment.emotion}] isn't an emotion {backend} knows. "
                             f"Valid: {', '.join(sorted(valid))}")


def script_segments(text: str, lang: str, pause_ms: int = 0, pronunciations: str = None) -> list:
    """The text as script_markup segments, with the pronunciation dictionary and --pause-ms applied."""
    words = script_markup.load_pronunciations(pronunciations).get(lang, {})
    segments = script_markup.parse(text, lang, words)
    if pause_ms > 0:
        segments = script_markup.split_at_sentences(segments, pause_ms)
    if not any(s.text for s in segments):
        raise ValueError("There's nothing to say: the text is empty once the markup is taken out.")
    return segments


def apply_speed_pitch(wav: str, speed: float = 1.0, pitch: float = 0.0) -> None:
    """Change tempo and pitch (semitones) in place with ffmpeg's rubberband filter.

    For backends without native speed/pitch (Chatterbox, XTTS). rubberband with
    formants preserved keeps a pitch change from sounding like a different
    (or chipmunk) voice; atempo and librosa's phase vocoder sound metallic on
    speech. It erases the Chatterbox watermark (measured 1.0 -> 0.0), which is
    why apply_watermark() runs after every such change.
    """
    if speed == 1.0 and pitch == 0.0:
        return
    tmp = str(Path(wav).with_suffix(".rubberband.wav"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-af",
                    f"rubberband=tempo={speed}:pitch={2 ** (pitch / 12):.6f}"
                    ":formant=preserved:pitchq=quality", tmp], check=True)
    os.replace(tmp, wav)


_watermarker = None


def apply_watermark(wav: str) -> None:
    """Embed Chatterbox's own Perth watermark into a speech file, in place.

    Chatterbox marks what it generates, but SBV2 and XTTS output carry no mark,
    and rubberband and Seed-VC (which re-synthesizes the waveform) both erase
    it. So whatever reaches the video stage gets marked as the last audio step.
    """
    global _watermarker
    import perth

    if _watermarker is None:
        _watermarker = perth.PerthImplicitWatermarker()
    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    sf.write(wav, _watermarker.apply_watermark(audio, sample_rate=sr), sr)


def needs_watermark(backend: str, speed: float, pitch: float, voice_convert: bool) -> bool:
    """False only for Chatterbox output nothing has touched since Chatterbox marked it."""
    return backend != "chatterbox" or speed != 1.0 or pitch != 0.0 or voice_convert
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_speech_helpers.py -q`
Expected: all pass.

- [ ] **Step 5: Verify the watermark survives at every rate it will see**

SBV2 writes 44.1 kHz, Seed-VC 22.05/44.1 kHz. Run:

```bash
.venv/Scripts/python.exe - <<'EOF'
import librosa, perth, soundfile as sf, tempfile, os
import make_avatar as ma
w = perth.PerthImplicitWatermarker()
for sr in (22050, 24000, 44100):
    x, _ = librosa.load("voice_samples/newtest.wav", sr=sr, offset=60, duration=8)
    f = os.path.join(tempfile.gettempdir(), f"wm_{sr}.wav"); sf.write(f, x, sr)
    before = w.get_watermark(x, sample_rate=sr)
    ma.apply_watermark(f)
    y, _ = sf.read(f, dtype="float32")
    print(sr, "before", before, "after", w.get_watermark(y, sample_rate=sr))
EOF
```

Expected: every row `before 0.0 after 1.0`. If 44.1 kHz fails, resample to 24 kHz inside `apply_watermark` before marking (and re-run).

- [ ] **Step 6: Commit**

```bash
git add make_avatar.py tests/test_speech_helpers.py
git commit -m "Speech helpers: per-clip pauses, rubberband speed/pitch, watermark, emotion presets" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: SBV2 inference worker with accent overrides and styles

**Files:**
- Create: `sbv2_synth.py`, `tests/test_sbv2_job.py`
- Modify: `make_avatar.py` (new `load_sbv2_styles`, `sbv2_segment_job`, `synthesize_speech_sbv2` after `synthesize_speech_xtts`)

**Interfaces:**
- Consumes: `script_markup.accent_tones` (Task 4); `script_segments`, `check_emotions`, `concat_with_pauses` (Task 5); `SBV2_*`, `get_venv_python` (Task 1).
- Produces:
  - Worker contract: `python sbv2_synth.py JOB_JSON` with cwd `SBV2_DIR`; JOB_JSON = `{"model_dir": str, "out_dir": str, "length": float, "pitch_scale": float, "segments": [{"pieces": [[str, int|null], ...], "style": str, "style_weight": float, "intonation_scale": float}]}` → `JOB_JSON.result` = `{"sr": int, "files": [str, ...]}` (one per segment, in order).
  - `SBV2_STYLE_WEIGHT = 2.0`, `SBV2_EMPHASIS_INTONATION = 1.3`
  - `load_sbv2_styles(model_dir: Path) -> dict[str, str]` (tag → style; always has `"neutral"`)
  - `sbv2_segment_job(segment, styles: dict) -> dict`
  - `synthesize_speech_sbv2(text: str, lang: str, model_dir: Path, out_wav: str, pause_ms: int = 0, speed: float = 1.0, pitch: float = 0.0, pronunciations: str = None) -> str`

- [ ] **Step 1: Write the failing tests** — `tests/test_sbv2_job.py`

```python
import json

import pytest

import make_avatar as ma
from script_markup import Piece, Segment


def _model_dir(tmp_path, styles_yaml, style2id):
    d = tmp_path / "me"
    d.mkdir()
    (d / "styles.yaml").write_text(styles_yaml, encoding="utf-8")
    (d / "config.json").write_text(json.dumps({"data": {"style2id": style2id}}), encoding="utf-8")
    return d


def test_load_styles(tmp_path):
    d = _model_dir(tmp_path, "neutral: Neutral\nhappy: style_1\n", {"Neutral": 0, "style_1": 1})
    assert ma.load_sbv2_styles(d) == {"neutral": "Neutral", "happy": "style_1"}


def test_load_styles_adds_neutral(tmp_path):
    d = _model_dir(tmp_path, "happy: style_1\n", {"Neutral": 0, "style_1": 1})
    assert ma.load_sbv2_styles(d)["neutral"] == "Neutral"


def test_load_styles_rejects_unknown_style_name(tmp_path):
    d = _model_dir(tmp_path, "happy: Joy\n", {"Neutral": 0, "style_1": 1})
    with pytest.raises(ValueError, match="Joy"):
        ma.load_sbv2_styles(d)


def test_segment_job_plain():
    job = ma.sbv2_segment_job(Segment([Piece("こんにちは")]), {"neutral": "Neutral"})
    assert job == {"pieces": [["こんにちは", None]], "style": "Neutral", "style_weight": 1.0,
                   "intonation_scale": 1.0}


def test_segment_job_emotion_emphasis_and_accent():
    segment = Segment([Piece("今日は"), Piece("あめ", 1)], "happy", 1.5, True)
    job = ma.sbv2_segment_job(segment, {"neutral": "Neutral", "happy": "style_1"})
    assert job == {"pieces": [["今日は", None], ["あめ", 1]], "style": "style_1",
                   "style_weight": ma.SBV2_STYLE_WEIGHT * 1.5,
                   "intonation_scale": ma.SBV2_EMPHASIS_INTONATION}


def test_sbv2_is_japanese_only(tmp_path):
    with pytest.raises(ValueError, match="Japanese"):
        ma.synthesize_speech_sbv2("Hello", "en", tmp_path, str(tmp_path / "o.wav"))
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sbv2_job.py -q`
Expected: FAIL — `AttributeError: ... 'load_sbv2_styles'`.

- [ ] **Step 3: Implement the main-process side in `make_avatar.py`**

After `synthesize_speech_xtts`:

```python
# [tag] on SBV2: style_weight at intensity 1 (SBV2's own default is 1.0, which
# barely moves a voice trained on one calm recording). *word*: a wider pitch
# range for that span.
SBV2_STYLE_WEIGHT = 2.0
SBV2_EMPHASIS_INTONATION = 1.3


def load_sbv2_styles(model_dir: Path) -> dict:
    """styles.yaml: emotion tag -> this voice's style name (named by the user after listening)."""
    styles = yaml.safe_load((model_dir / "styles.yaml").read_text(encoding="utf-8")) or {}
    if not isinstance(styles, dict):
        raise ValueError(f"{model_dir / 'styles.yaml'}: expected lines like  happy: style_1")
    styles = {str(tag): str(style) for tag, style in styles.items()}
    styles.setdefault("neutral", "Neutral")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    known = set(config.get("data", {}).get("style2id", {}))
    unknown = sorted(set(styles.values()) - known)
    if unknown:
        raise ValueError(f"{model_dir / 'styles.yaml'} names styles this voice doesn't have: "
                         f"{', '.join(unknown)}. Available: {', '.join(sorted(known))}")
    return styles


def sbv2_segment_job(segment, styles: dict) -> dict:
    return {
        "pieces": [[p.text, p.accent] for p in segment.pieces],
        "style": styles[segment.emotion or "neutral"],
        "style_weight": SBV2_STYLE_WEIGHT * segment.intensity if segment.emotion else 1.0,
        "intonation_scale": SBV2_EMPHASIS_INTONATION if segment.emphasis else 1.0,
    }


def synthesize_speech_sbv2(text: str, lang: str, model_dir: Path, out_wav: str, pause_ms: int = 0,
                           speed: float = 1.0, pitch: float = 0.0, pronunciations: str = None) -> str:
    """Japanese with explicit pitch accent: Style-Bert-VITS2 JP-Extra trained on your voice.

    Chatterbox and XTTS have no model of Japanese pitch accent, and both were
    rejected by ear for Japanese intonation. SBV2 predicts accent with
    pyopenjtalk, and {雨|あ'め} markup overrides it per mora (given_phone /
    given_tone). Speed and pitch are native here (length, pitch_scale), so
    there's no rubberband pass.

    Runs in .venv-sbv2 as a subprocess (CLAUDE.md's cross-venv rule).
    """
    if lang not in SBV2_LANGS:
        raise ValueError(f"Style-Bert-VITS2 JP-Extra speaks Japanese only, not '{lang}'. "
                         "Use --tts chatterbox (or auto) for other languages.")
    segments = script_segments(text, lang, pause_ms, pronunciations)
    styles = load_sbv2_styles(model_dir)
    check_emotions(segments, set(styles), "this Style-Bert-VITS2 voice (see its styles.yaml)")

    work_dir = tempfile.mkdtemp(prefix="sbv2_")
    job_path = os.path.join(work_dir, "job.json")
    with open(job_path, "w", encoding="utf-8") as fh:
        json.dump({
            "model_dir": str(model_dir),
            "out_dir": work_dir,
            "length": 1.0 / speed,
            "pitch_scale": 2 ** (pitch / 12),
            "segments": [sbv2_segment_job(s, styles) for s in segments if s.text],
        }, fh, ensure_ascii=False)

    cmd = [get_venv_python(SBV2_VENV), str(Path(__file__).parent / "sbv2_synth.py"), job_path]
    print("Running Style-Bert-VITS2:", " ".join(cmd))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        subprocess.run(cmd, cwd=SBV2_DIR, check=True, env=env)
        with open(job_path + ".result", encoding="utf-8") as fh:
            result = json.load(fh)
        files = iter(result["files"])
        clips = [sf.read(next(files), dtype="float32")[0] if s.text else np.zeros(0, np.float32)
                 for s in segments]
        sf.write(out_wav, concat_with_pauses(clips, result["sr"], [s.pause_after_ms for s in segments]),
                 result["sr"])
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return out_wav
```

- [ ] **Step 4: Write `sbv2_synth.py`**

```python
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

A piece with an accent (0 = flat, n = pitch drops after mora n) has its
pyopenjtalk accent replaced; the rest keep pyopenjtalk's. The cwd matters: SBV2
finds its BERT model and user dictionary relative to the checkout.
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())                              # style_bert_vits2, from the checkout
sys.path.insert(0, str(Path(__file__).resolve().parent))     # script_markup (ours, pure Python)
from script_markup import accent_tones  # noqa: E402


def latest_checkpoint(model_dir: Path) -> Path:
    def step(path: Path) -> int:
        match = re.search(r"_s(\d+)$", path.stem)
        return int(match.group(1)) if match else -1

    found = sorted(model_dir.glob("*.safetensors"), key=step)
    if not found:
        raise FileNotFoundError(f"No .safetensors checkpoint in {model_dir}")
    return found[-1]


def phones_and_tones(pieces: list) -> tuple:
    """Pieces -> (given_phone, given_tone), with marked pieces' accents replaced."""
    from style_bert_vits2.nlp.japanese.g2p_utils import g2kata_tone, kata_tone2phone_tone
    from style_bert_vits2.nlp.japanese.normalizer import normalize_text

    kata_tone = []
    for text, accent in pieces:
        part = g2kata_tone(normalize_text(text))
        if accent is not None:
            morae = [kata for kata, _ in part]
            try:
                part = list(zip(morae, accent_tones(len(morae), accent)))
            except ValueError as exc:
                raise ValueError(f"{text}: {exc}") from None
        kata_tone.extend(part)
    phone_tone = kata_tone2phone_tone(kata_tone)
    return [p for p, _ in phone_tone], [t for _, t in phone_tone]


def main() -> int:
    job_path = sys.argv[1]
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)

    import soundfile as sf
    import torch
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
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
            kwargs = {}
            if any(accent is not None for _, accent in segment["pieces"]):
                kwargs["given_phone"], kwargs["given_tone"] = phones_and_tones(segment["pieces"])
            sr, audio = model.infer(
                text="".join(text for text, _ in segment["pieces"]), language=Languages.JP,
                style=segment["style"], style_weight=segment["style_weight"],
                intonation_scale=segment["intonation_scale"], length=job["length"],
                pitch_scale=job["pitch_scale"], line_split=False, **kwargs)
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
```

- [ ] **Step 5: Run the unit tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sbv2_job.py -q`
Expected: all pass.

- [ ] **Step 6: Verify the worker end to end on a stock voice, including accent override**

The trained voice may still be training, so use a copy of the stock JVNV voice (never write into the checkout):

```bash
.venv/Scripts/python.exe - <<'EOF'
import shutil, tempfile
from pathlib import Path
import librosa, numpy as np
import make_avatar as ma

stock = Path(ma.SBV2_DIR) / "model_assets" / "jvnv-F1-jp"
d = Path(tempfile.mkdtemp()) / "jvnv"
shutil.copytree(stock, d)
(d / "styles.yaml").write_text("neutral: Neutral\nhappy: Happy\nsad: Sad\n", encoding="utf-8")

def render(text, name, **kw):
    out = f"output/sbv2_check_{name}.wav"
    ma.synthesize_speech_sbv2(text, "ja", d, out, **kw)
    return out

def first_two_morae_f0(path):
    y, sr = librosa.load(path, sr=None)
    f0, voiced, _ = librosa.pyin(y, fmin=80, fmax=600, sr=sr)
    f0 = f0[voiced]
    half = len(f0) // 2
    return np.nanmedian(f0[:half]), np.nanmedian(f0[half:])

a = render("{雨|あ'め}", "ame_atamadaka")
b = render("{飴|あめ0}", "ame_heiban")
print("あ'め (expect first > second):", first_two_morae_f0(a))
print("あめ0 (expect first < second):", first_two_morae_f0(b))
render("こんにちは。[pause 700][happy]今日はとても楽しかったです！[/happy]", "markup")
render("今日はいい天気ですね。", "fast_high", speed=1.2, pitch=2)
print("ok")
EOF
```

Expected: the atamadaka line's first-half F0 is clearly higher than its second half; the heiban line's is lower; `ok`; four playable files in `output/`. If the two contours don't differ, `given_tone` isn't reaching the model — debug in `phones_and_tones` (print `kata_tone`) before going on.

- [ ] **Step 7: Commit**

```bash
git add sbv2_synth.py make_avatar.py tests/test_sbv2_job.py
git commit -m "Add sbv2_synth.py: Style-Bert-VITS2 worker with per-mora accent overrides" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Routing, flags, markup on every backend, watermark placement

**Files:**
- Modify: `make_avatar.py` — `synthesize_speech`, `synthesize_speech_xtts`, `synthesize`, `voice_compare`, `make_one`, `run_batch`, `build_parser`, `main`; delete `split_sentences`
- Create: `tests/test_routing.py`

**Interfaces:**
- Consumes: everything from Tasks 2, 5, 6.
- Produces:
  - `resolve_tts(tts: str, lang: str, voice_sample: str | None, sbv2_model: str | None = None) -> tuple[str, Path | None]`
  - `synthesize(tts, text, lang, voice_sample, out_wav, pause_ms=0, cfg_weight=0.5, exaggeration=0.5, prep_voice=True, speed=1.0, pitch=0.0, sbv2_model=None, pronunciations=None) -> str`
  - `synthesize_speech(..., prep_voice=True, speed=1.0, pitch=0.0, pronunciations=None)` and `synthesize_speech_xtts(..., prep_voice=True, speed=1.0, pitch=0.0, pronunciations=None)` (new trailing keywords)
  - `make_one(..., tts="auto", ..., speed=1.0, pitch=0.0, sbv2_model=None, pronunciations=None, **engine_opts)`; `run_batch` likewise; batch config keys `speed`, `pitch`, `sbv2_model`, `pronunciations`
  - Parser: `--tts {auto,chatterbox,xtts,sbv2}` (default `auto`), `--speed`, `--pitch`, `--sbv2-model`, `--pronunciations`

- [ ] **Step 1: Write the failing tests** — `tests/test_routing.py`

```python
from pathlib import Path

import pytest

import make_avatar as ma


@pytest.fixture
def sbv2_world(tmp_path, monkeypatch):
    """A fake installed SBV2 with one trained voice, bound to voice.wav by hash."""
    monkeypatch.setattr(ma, "sbv2_installed", lambda: True)
    monkeypatch.setattr(ma, "SBV2_MODELS", str(tmp_path / "models"))
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"me")
    model = tmp_path / "models" / "me"
    model.mkdir(parents=True)
    for name, body in (("config.json", "{}"), ("me_e100_s1.safetensors", "x"), ("styles.yaml", "neutral: Neutral")):
        (model / name).write_text(body, encoding="utf-8")
    (model / "source.json").write_text(f'{{"sha256": "{ma.file_sha256(str(voice))}"}}', encoding="utf-8")
    return voice, model


def test_auto_uses_sbv2_for_japanese_with_a_model(sbv2_world):
    voice, model = sbv2_world
    assert ma.resolve_tts("auto", "ja", str(voice)) == ("sbv2", model)


def test_auto_uses_chatterbox_for_other_languages(sbv2_world):
    voice, _ = sbv2_world
    assert ma.resolve_tts("auto", "en", str(voice)) == ("chatterbox", None)


def test_auto_falls_back_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "sbv2_installed", lambda: True)
    monkeypatch.setattr(ma, "SBV2_MODELS", str(tmp_path / "none"))
    assert ma.resolve_tts("auto", "ja", str(tmp_path / "v.wav")) == ("chatterbox", None)


def test_auto_with_a_missing_named_model_is_an_error(sbv2_world):
    voice, _ = sbv2_world
    with pytest.raises(FileNotFoundError, match="typo"):
        ma.resolve_tts("auto", "ja", str(voice), sbv2_model="typo")


def test_forced_sbv2_errors(sbv2_world, monkeypatch, tmp_path):
    voice, _ = sbv2_world
    with pytest.raises(ValueError, match="Japanese"):
        ma.resolve_tts("sbv2", "en", str(voice))
    other = tmp_path / "other.wav"
    other.write_bytes(b"someone else")
    with pytest.raises(FileNotFoundError, match="train_sbv2.py"):
        ma.resolve_tts("sbv2", "ja", str(other))
    monkeypatch.setattr(ma, "sbv2_installed", lambda: False)
    with pytest.raises(FileNotFoundError, match="setup_sbv2.sh"):
        ma.resolve_tts("sbv2", "ja", str(voice))


def test_forced_backends_pass_through():
    assert ma.resolve_tts("xtts", "ja", None) == ("xtts", None)
    assert ma.resolve_tts("chatterbox", "ja", None) == ("chatterbox", None)


def test_xtts_rejects_emotion_tags_before_running_anything():
    with pytest.raises(ValueError, match="XTTS has no emotion"):
        ma.synthesize_speech_xtts("[happy]Hi[/happy]", "en", "voice.wav", "out.wav")


def test_parser_defaults_and_ranges():
    parser = ma.build_parser()
    args = parser.parse_args([])
    assert (args.tts, args.speed, args.pitch, args.sbv2_model) == ("auto", 1.0, 0.0, None)
    assert parser.parse_args(["--pitch", "-2"]).pitch == -2.0
    for bad in (["--speed", "3"], ["--pitch", "12.5"], ["--tts", "firefly"]):
        with pytest.raises(SystemExit):
            parser.parse_args(bad)


def _record_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(ma, "synthesize", lambda *a, **k: calls.append("tts"))
    monkeypatch.setattr(ma, "release_tts_model", lambda: None)
    monkeypatch.setattr(ma, "convert_voice", lambda *a, **k: calls.append("seedvc"))
    monkeypatch.setattr(ma, "apply_watermark", lambda wav: calls.append("watermark"))
    monkeypatch.setattr(ma, "render_video", lambda *a, **k: calls.append("video"))
    return calls


@pytest.mark.parametrize("kwargs,expected", [
    (dict(voice_convert=True), ["tts", "seedvc", "watermark", "video"]),
    (dict(), ["tts", "video"]),
    (dict(speed=1.2), ["tts", "watermark", "video"]),
])
def test_watermark_is_the_last_audio_step(monkeypatch, tmp_path, kwargs, expected):
    calls = _record_pipeline(monkeypatch)
    ma.make_one("face.jpg", "voice.wav", "Hi", "en", str(tmp_path / "o.mp4"),
                engine="sadtalker", tts="chatterbox", **kwargs)
    assert calls == expected


def test_batch_reads_new_keys(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(ma, "make_one", lambda *a, **k: seen.update(k))
    monkeypatch.setattr(ma, "release_tts_model", lambda: None)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("photo: p.jpg\nvoice_sample: v.wav\nspeed: 1.1\npitch: -1\n"
                   "sbv2_model: me\npronunciations: p.yaml\nlines:\n  - text: hi\n", encoding="utf-8")
    ma.run_batch(str(cfg))
    assert (seen["speed"], seen["pitch"], seen["sbv2_model"], seen["pronunciations"]) == (1.1, -1, "me", "p.yaml")
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_routing.py -q`
Expected: FAIL — `AttributeError: ... 'resolve_tts'` and parser assertions.

- [ ] **Step 3: Rewrite the Chatterbox and XTTS synthesis bodies**

In `synthesize_speech`: change the signature to
`def synthesize_speech(text: str, lang: str, voice_sample: str, out_wav: str, pause_ms: int = 0, cfg_weight: float = 0.5, exaggeration: float = 0.5, prep_voice: bool = True, speed: float = 1.0, pitch: float = 0.0, pronunciations: str = None) -> str:`, append to the docstring:

```
    Script markup (script_markup.py) is honoured: one generate() call per
    segment, so text without markup is still a single call. Emotion tags map
    to CHATTERBOX_EMOTIONS; speed/pitch are a rubberband pass afterwards.
```

and replace everything from `voice_sample = get_prepared_voice(...)` to the end of the function with:

```python
    segments = script_segments(text, lang, pause_ms, pronunciations)
    check_emotions(segments, set(CHATTERBOX_EMOTIONS), "Chatterbox")

    voice_sample = get_prepared_voice(voice_sample, enabled=prep_voice)
    model = get_tts_model()
    clips = []
    for segment in segments:
        if not segment.text:  # a leading [pause]: silence only
            clips.append(np.zeros(0, dtype=np.float32))
            continue
        ex, cfg = chatterbox_params(segment, exaggeration, cfg_weight)
        clips.append(model.generate(segment.text, language_id=lang, audio_prompt_path=voice_sample,
                                    cfg_weight=cfg, exaggeration=ex).squeeze().cpu().numpy())

    # torchaudio.save() in recent versions requires the separate torchcodec
    # package as its encoding backend; write with soundfile instead to avoid
    # that extra dependency.
    sf.write(out_wav, concat_with_pauses(clips, model.sr, [s.pause_after_ms for s in segments]), model.sr)
    apply_speed_pitch(out_wav, speed, pitch)
    return out_wav
```

In `synthesize_speech_xtts`: signature gains `speed: float = 1.0, pitch: float = 0.0, pronunciations: str = None`. Right after the `XTTS_LANGS` check insert:

```python
    segments = script_segments(text, lang, pause_ms, pronunciations)
    if any(s.emotion or s.emphasis for s in segments):
        raise ValueError("XTTS has no emotion or emphasis control, so [tags] and *emphasis* can't be "
                         "honoured. Use --tts chatterbox, or take them out.")
```

replace the `"sentences": split_sentences(text) if pause_ms > 0 else [text.strip()],` line (and its comment) with `"sentences": [s.text for s in segments if s.text],`, and replace the two lines after `result = json.load(fh)` with:

```python
        files = iter(result["files"])
        clips = [sf.read(next(files), dtype="float32")[0] if s.text else np.zeros(0, np.float32)
                 for s in segments]
        sf.write(out_wav, concat_with_pauses(clips, result["sr"], [s.pause_after_ms for s in segments]),
                 result["sr"])
```

and add `apply_speed_pitch(out_wav, speed, pitch)` before `return out_wav`.

Delete `split_sentences()` (its only callers were these two functions; `script_markup.split_at_sentences` keeps the same boundary rule).

- [ ] **Step 4: Add `resolve_tts` and rewrite `synthesize`**

Replace `synthesize()` with:

```python
def resolve_tts(tts: str, lang: str, voice_sample: str | None, sbv2_model: str | None = None) -> tuple:
    """--tts -> (the backend that will actually run, its SBV2 model folder or None).

    auto: Japanese goes to Style-Bert-VITS2 when there's a voice trained on
    this recording (find_sbv2_model), everything else to Chatterbox. A forced
    backend that can't serve the request raises instead of falling back.
    """
    if tts == "auto":
        if lang in SBV2_LANGS and sbv2_installed():
            model = find_sbv2_model(voice_sample, sbv2_model)
            if model:
                return "sbv2", model
            if sbv2_model:
                raise FileNotFoundError(f"--sbv2-model {sbv2_model}: no finished model at "
                                        f"{Path(SBV2_MODELS) / sbv2_model}")
        return "chatterbox", None
    if tts == "sbv2":
        if lang not in SBV2_LANGS:
            raise ValueError(f"Style-Bert-VITS2 JP-Extra speaks Japanese only, not '{lang}'.")
        if not sbv2_installed():
            raise FileNotFoundError("Style-Bert-VITS2 isn't set up. Run ./setup_sbv2.sh first.")
        model = find_sbv2_model(voice_sample, sbv2_model)
        if model is None:
            name = sbv2_model or (Path(voice_sample).stem if voice_sample else "NAME")
            raise FileNotFoundError(
                f"No trained Style-Bert-VITS2 voice for {voice_sample} "
                f"(expected {Path(SBV2_MODELS) / name}/). Train one with: "
                f"python train_sbv2.py --voice {voice_sample}")
        return "sbv2", model
    return tts, None


def synthesize(tts: str, text: str, lang: str, voice_sample: str, out_wav: str,
               pause_ms: int = 0, cfg_weight: float = 0.5, exaggeration: float = 0.5,
               prep_voice: bool = True, speed: float = 1.0, pitch: float = 0.0,
               sbv2_model: str = None, pronunciations: str = None) -> str:
    """Dispatch to the backend --tts resolves to (see resolve_tts)."""
    backend, model_dir = resolve_tts(tts, lang, voice_sample, sbv2_model)
    if tts == "auto" and lang in SBV2_LANGS and backend == "chatterbox":
        print("Note: Japanese is using Chatterbox, whose pitch accent is often wrong. For correct "
              f"intonation: ./setup_sbv2.sh, then python train_sbv2.py --voice {voice_sample}")
    if backend == "sbv2":
        return synthesize_speech_sbv2(text, lang, model_dir, out_wav, pause_ms=pause_ms, speed=speed,
                                      pitch=pitch, pronunciations=pronunciations)
    if backend == "xtts":
        return synthesize_speech_xtts(text, lang, voice_sample, out_wav, pause_ms=pause_ms,
                                      prep_voice=prep_voice, speed=speed, pitch=pitch,
                                      pronunciations=pronunciations)
    return synthesize_speech(text, lang, voice_sample, out_wav, pause_ms=pause_ms,
                             cfg_weight=cfg_weight, exaggeration=exaggeration, prep_voice=prep_voice,
                             speed=speed, pitch=pitch, pronunciations=pronunciations)
```

- [ ] **Step 5: Watermark in `make_one` and `voice_compare`**

`make_one` signature: change `tts: str = "chatterbox"` to `tts: str = "auto"` and add `speed: float = 1.0, pitch: float = 0.0, sbv2_model: str = None, pronunciations: str = None,` before `**engine_opts`. Immediately after the `--motion-video` warning block (before `with tempfile.NamedTemporaryFile(...)`), add:

```python
    # Resolve the voice model up front: a forced backend that can't run should
    # fail now, not after minutes of work, and the watermark decision needs it.
    backend, _ = resolve_tts(tts, lang, voice_sample, sbv2_model)
```

Change the `synthesize(...)` call to pass `speed=speed, pitch=pitch, sbv2_model=sbv2_model, pronunciations=pronunciations`. After the `if voice_convert:` block (right before `if engine == "sadtalker":`), add:

```python
        if needs_watermark(backend, speed, pitch, voice_convert):
            # Last audio step on purpose: Seed-VC and rubberband both erase a
            # mark applied earlier (see apply_watermark).
            apply_watermark(tmp_wav)
```

In `voice_compare`, give each variant tuple a 4th element `mark` — `False` for the four `chatterbox_*` variants that are pure Chatterbox output, `True` for both `xtts_*` variants and both `*_then_seedvc` variants — change the loop to `for filename, description, run, mark in variants:` and after `run(target)` add:

```python
            if mark:
                apply_watermark(target)
```

- [ ] **Step 6: Parser, batch and main**

In `build_parser`, before `--tts`, add:

```python
    def float_in(lo: float, hi: float):
        def parse(value: str) -> float:
            number = float(value)
            if not lo <= number <= hi:
                raise argparse.ArgumentTypeError(f"must be between {lo:g} and {hi:g}")
            return number
        parse.__name__ = "number"  # argparse's error message names the type
        return parse
```

Replace the `--tts` argument with:

```python
    parser.add_argument("--tts", default="auto", choices=["auto", "chatterbox", "xtts", "sbv2"],
                        help="Text-to-speech engine. auto (default): Japanese uses your trained "
                             "Style-Bert-VITS2 voice when there is one (correct pitch accent), "
                             "everything else Chatterbox. chatterbox: 23 languages. xtts: XTTS-v2, "
                             "17 languages, needs ./setup_xtts.sh. sbv2: Style-Bert-VITS2, Japanese "
                             "only, needs ./setup_sbv2.sh and train_sbv2.py")
    parser.add_argument("--speed", type=float_in(0.5, 2.0), default=1.0,
                        help="Speaking rate, 0.5-2.0 (default 1.0)")
    parser.add_argument("--pitch", type=float_in(-12, 12), default=0.0,
                        help="Pitch shift in semitones, -12 to 12 (default 0)")
    parser.add_argument("--sbv2-model",
                        help="[sbv2] Which trained voice in models/sbv2/ to use (default: the one "
                             "trained on the --voice recording)")
    parser.add_argument("--pronunciations",
                        help="YAML of per-language word -> reading fixes (default: "
                             "pronunciations.yaml if it exists; see pronunciations.example.yaml)")
```

Extend `--text`'s help to: `"Text to speak. Supports markup: [pause 800], [happy]...[/happy], *emphasis*, {word|reading} (README: Script markup)"`.

`run_batch`: signature `tts: str = "auto"` and add `speed: float = 1.0, pitch: float = 0.0, sbv2_model: str = None, pronunciations: str = None,` before `**engine_opts`; after `motion_scale = cfg.get(...)` add:

```python
    speed = cfg.get("speed", speed)
    pitch = cfg.get("pitch", pitch)
    sbv2_model = cfg.get("sbv2_model", sbv2_model)
    pronunciations = cfg.get("pronunciations", pronunciations)
```

and pass `speed=speed, pitch=pitch, sbv2_model=sbv2_model, pronunciations=pronunciations,` in its `make_one(...)` call.

`main`: add to `shared_opts`: `speed=args.speed, pitch=args.pitch, sbv2_model=args.sbv2_model, pronunciations=args.pronunciations,`.

- [ ] **Step 7: Run all tests**

Run: `.venv/Scripts/python.exe -m pytest tests -q`
Expected: all pass. Also `grep -n "split_sentences" make_avatar.py` → no matches.

- [ ] **Step 8: Real-audio check of the no-markup path (regression)**

Run: `.venv/Scripts/python.exe make_avatar.py --voice-compare --voice voice_samples/newtest.wav --text "Hello, this is a quick test." --lang en --out output/regress_compare`
Expected: the Chatterbox variants generate as before (XTTS/Seed-VC variants generate or are skipped with a note, as before). Listen to `chatterbox_natural.wav`: identical in character to before this branch.

- [ ] **Step 9: Commit**

```bash
git add make_avatar.py tests/test_routing.py
git commit -m "--tts auto routes Japanese to Style-Bert-VITS2; markup, speed and pitch on every backend" -m "Watermark is now the last audio step, after Seed-VC, which also fixes Chatterbox + --voice-convert output reaching the video unmarked." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Desktop app

**Files:**
- Modify: `ui/system.py`, `ui/pipeline.py`, `ui/layout.py`, `ui/jobs.py`
- Create: `tests/test_ui_pipeline.py`

**Interfaces:**
- Consumes: `ma.resolve_tts`, `ma.find_sbv2_model`, `ma.sbv2_installed`, `ma.SBV2_LANGS`, `script_markup.parse`, parser dests `speed`, `pitch`.
- Produces: `pipeline.VOICE_NAMES: dict[str, str]`, `pipeline.speech_backend(options: dict) -> str`.

- [ ] **Step 1: Write the failing tests** — `tests/test_ui_pipeline.py`

```python
from pathlib import Path

import make_avatar as ma
from ui import pipeline


def test_languages_per_voice_model():
    assert pipeline.languages("sbv2") == [("Japanese", "ja")]
    assert pipeline.languages("auto") == pipeline.languages("chatterbox")


def test_problems_flags_markup_before_queueing():
    found = pipeline.problems({"text": "5 * 3 = 15", "lang": "en", "tts": "auto"}, {})
    assert any(p.startswith("Script markup:") for p in found)


def test_describe_and_parts_follow_the_resolved_backend(monkeypatch):
    monkeypatch.setattr(ma, "resolve_tts", lambda *a, **k: ("sbv2", Path("m")))
    options = {"tts": "auto", "lang": "ja", "engine": "latentsync", "voice": "v.wav"}
    assert "Style-Bert-VITS2" in pipeline.describe(options)
    assert "sbv2" in pipeline.needed_parts(options)


def test_to_args_passes_speed_and_pitch_and_omits_defaults():
    args = pipeline.to_args({"engine": "latentsync", "tts": "auto", "speed": 1.2, "pitch": -2})
    assert "--speed=1.2" in args and "--pitch=-2.0" in args
    assert not any(a.startswith("--tts") for a in args)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ui_pipeline.py -q`
Expected: FAIL (`languages("sbv2")` returns Chatterbox's list; no markup problem; describe says Chatterbox).

- [ ] **Step 3: `ui/system.py`** — add to the `parts` list after the `xtts` part:

```python
        Part("sbv2", "Style-Bert-VITS2",
             "Japanese with correct pitch accent, in a voice trained on yours (train_sbv2.py).",
             "./setup_sbv2.sh", ma.sbv2_installed()),
```

- [ ] **Step 4: `ui/pipeline.py`**

Add `import script_markup` after `import make_avatar as ma`. Replace `languages()`:

```python
VOICE_NAMES = {"auto": "Auto", "chatterbox": "Chatterbox", "xtts": "XTTS-v2", "sbv2": "Style-Bert-VITS2"}


def languages(tts: str) -> list:
    """(name, code) pairs the given voice model can speak, sorted by name."""
    codes = {"xtts": ma.XTTS_LANGS, "sbv2": ma.SBV2_LANGS}.get(tts, ma.CHATTERBOX_LANGS)
    return sorted(((LANGUAGE_NAMES.get(c, c), c) for c in codes), key=lambda pair: pair[0])


def speech_backend(options: dict) -> str:
    """The voice model a job will really use: auto resolves per language and voice file."""
    tts = options.get("tts") or default("tts")
    try:
        return ma.resolve_tts(tts, options.get("lang") or "en", options.get("voice"),
                              options.get("sbv2_model"))[0]
    except (ValueError, FileNotFoundError):
        return tts  # problems() explains why it can't run
```

In `needed_parts`, replace the first line with:

```python
    backend = speech_backend(options)
    needed = ["ffmpeg", "chatterbox"] + ([backend] if backend in ("xtts", "sbv2") else [])
```

(`chatterbox` stays needed: the main environment and the watermark come with it.)

In `problems`, replace the `tts = ...` block (the language check) with:

```python
    tts = options.get("tts") or default("tts")
    if options.get("lang") not in {code for _, code in languages(tts)}:
        name = LANGUAGE_NAMES.get(options.get("lang"), options.get("lang"))
        found.append(f"{VOICE_NAMES.get(tts, tts)} can't speak {name}.")
    elif tts == "sbv2" and ma.sbv2_installed() and options.get("voice") \
            and ma.find_sbv2_model(options["voice"], options.get("sbv2_model")) is None:
        found.append("This voice has no trained Style-Bert-VITS2 model yet. Train one with "
                     "train_sbv2.py (README: Style-Bert-VITS2), or pick Auto.")
    if clean_text(options.get("text")):
        try:
            script_markup.parse(options["text"], options.get("lang") or "en")
        except script_markup.MarkupError as exc:
            found.append(f"Script markup: {exc}")
```

In `describe`, replace the `voice = ...` line with `voice = VOICE_NAMES.get(speech_backend(options), "Chatterbox")`.

- [ ] **Step 5: `ui/layout.py`**

Replace the `settings["tts"] = tts = gr.Radio(...)` choices list with:

```python
                                [("Auto (Japanese uses your trained voice when there is one)", "auto"),
                                 ("Chatterbox (23 languages)", "chatterbox"),
                                 ("XTTS-v2 (17 languages, non-commercial)", "xtts"),
                                 ("Style-Bert-VITS2 (Japanese, trained voice)", "sbv2")],
```

Right after the `text = gr.Textbox(...)` line add:

```python
                            gr.Markdown("Pauses `[pause 800]` · emotion `[happy]…[/happy]` · emphasis "
                                        "`*word*` · pronunciation `{word|reading}`, Japanese accent "
                                        "`{雨|あ'め}`", elem_classes="note")
```

In the Advanced settings, after the row holding `pause_ms`/`vc_denoise`, add a row:

```python
                                with gr.Row():
                                    settings["speed"] = gr.Slider(
                                        0.5, 2, step=0.05, value=pipeline.default("speed"),
                                        label="Speed", info="1 is the voice's natural pace.")
                                    settings["pitch"] = gr.Slider(
                                        -12, 12, step=0.5, value=pipeline.default("pitch"),
                                        label="Pitch (semitones)", info="Small changes (±2) sound most natural.")
```

In `apply_settings`, change `tts_value = values.get("tts") or "chatterbox"` to `tts_value = values.get("tts") or pipeline.default("tts")`.

- [ ] **Step 6: `ui/jobs.py`** — add to `STAGE_MARKERS` after `("Running XTTS:", "speech"),`:

```python
    ("Running Style-Bert-VITS2:", "speech"),
```

- [ ] **Step 7: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests -q`
Expected: all pass.

- [ ] **Step 8: Launch the app and check the new controls**

Run in background: `.venv/Scripts/python.exe app.py --no-open --port 7861`
Then: `curl -s http://127.0.0.1:7861/config | grep -o "Style-Bert-VITS2[^\"]*\|Pitch (semitones)\|Auto (Japanese[^\"]*" | sort -u`
Expected: the Auto choice, the Style-Bert-VITS2 choice, and "Pitch (semitones)". Stop the app afterwards. (If `--port` isn't the flag name, check `python app.py --help`.)

- [ ] **Step 9: Commit**

```bash
git add ui tests/test_ui_pipeline.py
git commit -m "App: Auto voice model, Style-Bert-VITS2 choice, speed/pitch, markup hint and check" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Docs, trained-voice styles, and end-to-end verification

Starts once Task 3's background training has finished.

**Files:**
- Create: `pronunciations.example.yaml`
- Modify: `README.md`, `config.example.yaml`, local `CLAUDE.md` (gitignored; not committed)

- [ ] **Step 1: Confirm training finished and name the styles with the user**

Check: `ls models/sbv2/newtest/` shows `styles.yaml`, `style_samples/`, `newtest_e100_s*.safetensors`. Tell the user training time and peak VRAM. Ask them to listen to `models/sbv2/newtest/style_samples/*.wav` and say what each sounds like (happy, calm, …); write their names into the left side of `models/sbv2/newtest/styles.yaml`.

- [ ] **Step 2: Japanese A/B set**

```bash
.venv/Scripts/python.exe - <<'EOF'
import make_avatar as ma
from pathlib import Path
out = Path("output/ab_ja"); out.mkdir(parents=True, exist_ok=True)
lines = {
    "01_ame": "雨が降っています。",
    "02_ame_candy": "飴を買いました。",
    "03_hashi_bridge": "橋を渡ります。",
    "04_hashi_chopsticks": "箸で食べます。",
    "05_weather": "今日はいい天気ですね。散歩に行きましょう。",
    "06_meeting": "明日の会議は午後三時からです。",
}
voice = "voice_samples/newtest.wav"
for name, text in lines.items():
    ma.synthesize("chatterbox", text, "ja", voice, str(out / f"{name}_chatterbox_cfg0.2.wav"), cfg_weight=0.2)
    ma.synthesize("sbv2", text, "ja", voice, str(out / f"{name}_sbv2.wav"))
ma.release_tts_model()
print("done:", out)
EOF
```

Expected: 12 files in `output/ab_ja/`. Ask the user to listen and give the **intonation verdict by ear**; that verdict is this feature's success criterion. If accents are wrong on specific words, demonstrate the fix with `{word|reading}` and an accent mark.

- [ ] **Step 3: Markup measurements on Chatterbox (English)**

```bash
.venv/Scripts/python.exe - <<'EOF'
import numpy as np, soundfile as sf, librosa, perth
import make_avatar as ma
v = "voice_samples/newtest.wav"
ma.synthesize("chatterbox", "Hello.[pause 1200]This is [happy]great news[/happy], *really*.", "en", v, "output/mk_tags.wav", cfg_weight=0.0)
y, sr = sf.read("output/mk_tags.wav", dtype="float32")
quiet = np.abs(y) < 1e-4
runs, n = [], 0
for q in quiet:
    n = n + 1 if q else (runs.append(n) or 0)
print("longest digital silence ms:", round(max(runs + [n]) / sr * 1000))
ma.synthesize("chatterbox", "The quick brown fox jumps over the lazy dog.", "en", v, "output/mk_base.wav", cfg_weight=0.0)
import shutil
for name, kw in (("fast", dict(speed=1.2)), ("high", dict(pitch=2.0))):
    shutil.copy("output/mk_base.wav", f"output/mk_{name}.wav")
    ma.apply_speed_pitch(f"output/mk_{name}.wav", **kw)
print("duration ratio (expect ~0.83):", round(sf.info("output/mk_fast.wav").duration / sf.info("output/mk_base.wav").duration, 3))
f0 = lambda p: np.nanmedian(librosa.pyin(librosa.load(p, sr=24000)[0], fmin=60, fmax=500, sr=24000)[0])
print("F0 ratio (expect ~1.12):", round(f0("output/mk_high.wav") / f0("output/mk_base.wav"), 3))
w = perth.PerthImplicitWatermarker()
fast, fsr = sf.read("output/mk_fast.wav", dtype="float32")
print("rubberbanded, before watermark:", w.get_watermark(fast, sample_rate=fsr))
ma.apply_watermark("output/mk_fast.wav")
fast, fsr = sf.read("output/mk_fast.wav", dtype="float32")
print("after apply_watermark:", w.get_watermark(fast, sample_rate=fsr))
ma.convert_voice("output/mk_base.wav", v, "output/mk_vc.wav")
ma.apply_watermark("output/mk_vc.wav")
vc, vsr = sf.read("output/mk_vc.wav", dtype="float32")
print("Seed-VC then watermark:", w.get_watermark(vc, sample_rate=vsr))
ma.release_tts_model()
EOF
```

Expected: longest silence ≥ 1200 ms; duration ratio 0.81–0.85; F0 ratio 1.09–1.15; `before watermark: 0.0`, `after apply_watermark: 1.0`, `Seed-VC then watermark: 1.0`.

- [ ] **Step 4: One full Japanese video through the default pipeline**

Run (timeout 30 min): `.venv/Scripts/python.exe make_avatar.py --photo photos/image.jpg --voice voice_samples/newtest.wav --text "こんにちは。[pause 600]今日は{雨|あ'め}ですが、元気です。" --lang ja --out output/ja_sbv2_video.mp4`
Expected: log shows `Running Style-Bert-VITS2:` (not the Chatterbox note), then LivePortrait and LatentSync; `output/ja_sbv2_video.mp4` plays with lips in sync.

- [ ] **Step 5: `pronunciations.example.yaml`**

```yaml
# Copy to pronunciations.yaml (gitignored) to fix how words are read.
# Per language code: word -> reading. Applied wherever the word appears
# outside {word|reading} markup, which always wins.
#
# Japanese readings can also set the pitch accent:
#   ' after the mora where the pitch drops   雨 -> あ'め   (high-low)
#   a trailing 0 for a flat word             飴 -> あめ0   (low-high)
#   neither: just the reading; the accent is predicted as usual
ja:
  雨: "あ'め"
  飴: "あめ0"
en:
  Ueno: "oo-EH-no"
```

- [ ] **Step 6: `config.example.yaml`** — next to the existing `cfg_weight`/`exaggeration` keys add:

```yaml
# speed: 1.0          # 0.5-2.0
# pitch: 0            # semitones, -12 to 12
# sbv2_model: me      # which trained Style-Bert-VITS2 voice (default: the one trained on voice_sample)
# pronunciations: pronunciations.yaml
```

- [ ] **Step 7: README**

- In "3. Setup", after the Seed-VC subsection, add `### Style-Bert-VITS2 (optional, Japanese with correct pitch accent)`: why (Chatterbox/XTTS have no pitch-accent model), `./setup_sbv2.sh`, `python train_sbv2.py --voice voice_samples/me.wav` with the measured time/VRAM from Task 3, `--stop-after transcribe` to review `models/sbv2-data/<name>/esd.list`, naming styles in `styles.yaml` after listening to `style_samples/`, and that `--tts auto` then picks it up for `--lang ja` (also in the app, matched by file content).
- In "6. Tuning quality", add `### Script markup` with the syntax table from the spec §1, the per-backend table, the "splits only at tags" note, the emphasis and SBV2-emotion-range caveats, and `--speed`/`--pitch`.
- In "8. Troubleshooting", add: "Japanese sounds wrong / Chatterbox note in the log" → train SBV2; "`[x] isn't an emotion ... knows`" → valid tags; "Script markup error at character N" → escape with `\`.
- In the licence table: Style-Bert-VITS2 — AGPL-3.0, cloned at setup, run as a subprocess only; its Japanese BERT (deberta-v2-large-japanese-char-wwm) — CC-BY-SA-4.0; stock JVNV voices — see their model cards.
- In "Project layout": `setup_sbv2.sh`, `train_sbv2.py`, `sbv2_styles.py`, `sbv2_synth.py`, `script_markup.py`, `models/`, `tests/`.
- In "7. A note on responsible use": all generated speech, whichever model made it, carries Chatterbox's Perth watermark.

- [ ] **Step 8: Local `CLAUDE.md` (gitignored)** — add to "Known non-obvious workarounds": SBV2's `torch<2.4` pin is filtered out in `setup_sbv2.sh` (no sm_120); transcription uses `--use_hf_whisper` (pinned faster-whisper predates sm_120); rubberband and Seed-VC erase the Perth watermark, so `apply_watermark` is the last audio step; and add `setup_sbv2.sh` to the setup list and `--tts auto|sbv2` to the architecture section.

- [ ] **Step 9: Full test run and commit**

Run: `.venv/Scripts/python.exe -m pytest tests -q` → all pass.

```bash
git add README.md config.example.yaml pronunciations.example.yaml
git commit -m "Docs: script markup, Style-Bert-VITS2 setup and training" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 10: Finish the branch** — use superpowers:finishing-a-development-branch.
