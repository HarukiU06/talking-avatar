# Talking Avatar

Turn a photo of your face + a short voice sample into videos of "you" speaking
any text you type, in any of 23 languages. Everything runs locally — no cloud
service, no per-video cost, your face and voice never leave your computer.

**Pipeline:**
1. **Text → speech in your voice** — [Chatterbox Multilingual TTS](https://github.com/resemble-ai/chatterbox) (Resemble AI, MIT license) clones your voice from a short reference clip and speaks your text in it. Long, multi-sentence text is split into sentences and synthesized separately with a silence gap spliced between them (`--pause-ms`, default 450ms) — Chatterbox has no built-in pause control, so a whole paragraph sent in one shot comes out as one breathless run-on otherwise.
2. **Photo → idle-motion video** — by default, [LivePortrait](https://github.com/KwaiVGI/LivePortrait) (Kuaishou, MIT license) animates your still photo with natural head motion and blinking, driven by one of its bundled example clips (only the *motion* is transferred — the driving clip's own appearance/identity never appears in the output). The clip is short, so it's ping-pong looped to match your audio's length.
3. **Idle-motion video + speech → final video** — [LatentSync](https://github.com/bytedance/LatentSync) (ByteDance, Apache 2.0) lip-syncs that video to the audio, regenerating only the mouth region — everything else (eyes, face shape, the head motion from step 2) is carried through unchanged.

Pass `--motion none` to skip step 2 and lip-sync a frozen photo instead (useful if LivePortrait's motion ever looks off), or `--engine sadtalker` to use [SadTalker](https://github.com/Winfredy/SadTalker) (Apache 2.0) for the whole video step instead of LivePortrait+LatentSync — SadTalker is a single full-face reenactment model, simpler to run but with noticeably weaker lip sync and visible identity drift (it warps the whole face, not just the mouth, as a side effect of generating mouth motion).

For actual situational facial expression (not just lip sync — e.g. the face reading as "giving a confident business speech" rather than a static or generically animated one), pass `--engine infinitetalk` to use [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) (MeiGen-AI, Apache 2.0) instead — a single audio+photo+text-prompt model that generates lip sync, head motion, and expression together, conditioned on a `--scene-prompt` describing the delivery you want. It's a 14B-parameter model (built on Wan2.1-I2V), so it's noticeably slower and needs significantly more disk space than the other two engines — see its setup section below.

You "pre-enter" one or many lines of text (in a config file), walk away, and
come back to finished `.mp4` files.

---

## 1. Requirements

- **A computer with an NVIDIA GPU is strongly recommended.** LatentSync 1.5
  needs **8GB+ VRAM**; LivePortrait and SadTalker need less. All models can
  technically run on CPU, but generation will be slow (minutes per sentence
  instead of seconds).
  - No NVIDIA GPU? See "Alternatives" at the bottom — Apple Silicon (M-series)
    works too via `device="mps"`, just slower.
- Python 3.10 or 3.11
- ~16 GB free disk space (model weights across the latentsync/sadtalker
  engines) — add **~35-40 GB more** if you plan to use `--engine infinitetalk`
- [ffmpeg](https://ffmpeg.org/download.html) installed and on your PATH
- git
- **Windows only, if using the default engine**: a C++ compiler, needed to
  build one dependency (`insightface`, used by both LatentSync and
  LivePortrait) from source — no prebuilt Windows wheel exists for it.
  Install with:
  `winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"`

## 2. What to prepare

- **A clear, front-facing, well-lit photo of your face** — `photos/me.jpg`.
  Neutral expression, mouth closed, looking at the camera works best.
- **6–30 seconds of clean audio of your voice** — `voice_samples/me.wav`.
  Quiet room, no background music/noise, one continuous take, saved as
  `.wav` or `.mp3`. This is the sample the voice is cloned from — it does
  *not* need to be in the same language as what you'll generate later.

## 3. Setup

```bash
cd talking-avatar
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

`setup.sh` will:
- create a Python virtual environment (`.venv`)
- install PyTorch (edit the CUDA version in the script if yours differs —
  check with `nvidia-smi`)
- install `chatterbox-tts` (voice cloning TTS)
- clone SadTalker and download its model checkpoints (~2GB)
- check that ffmpeg is available

This step needs a real internet connection and will take a while the first
time (downloading model weights).

### LatentSync (default video engine)

LatentSync needs its own, separate virtual environment — its pinned
dependency versions conflict with chatterbox-tts's in `.venv`. Run this
after `setup.sh`:

```bash
./setup_latentsync.sh
```

`setup_latentsync.sh` will:
- clone LatentSync and create a second virtual environment (`.venv-latentsync`)
- install PyTorch and LatentSync's other dependencies
- build `insightface` from source (needs the C++ compiler mentioned above)
- download the LatentSync 1.5 checkpoints (~7GB) — not the repo's own
  default of 1.6, which needs more VRAM than most GPUs have

You don't need to activate `.venv-latentsync` yourself — `make_avatar.py`
calls into it directly as a subprocess.

### LivePortrait (default idle-motion engine)

Also needs its own virtual environment, for the same reason. Run:

```bash
./setup_liveportrait.sh
```

`setup_liveportrait.sh` will:
- clone LivePortrait and create a third virtual environment (`.venv-liveportrait`)
- install PyTorch and its other dependencies
- build `insightface` from source (same C++ compiler as above)
- download its pretrained weights (~1GB)

Same as LatentSync — `make_avatar.py` calls into `.venv-liveportrait`
directly, no need to activate it yourself.

### Wav2Lip (optional, `--refine-lipsync`)

Only needed if you plan to use `--engine sadtalker --refine-lipsync`. Skip
this if you're using the default LatentSync engine — it already syncs the
mouth from audio at higher fidelity than Wav2Lip's patch, so refinement
would make it *worse*, not better (`make_avatar.py` prints a warning and
skips it if you try).

```bash
./setup_wav2lip.sh
```

`setup_wav2lip.sh` will:
- clone Wav2Lip and create a fourth virtual environment (`.venv-wav2lip`)
- install PyTorch and Wav2Lip's other dependencies
- download the S3FD face-detection weights
- print a manual download step for `wav2lip_gan.pth` — its checkpoint is
  hosted on the authors' OneDrive with no stable direct-download URL, so
  this one step can't be scripted; follow the printed instructions once,
  save the file into `Wav2Lip/checkpoints/wav2lip_gan.pth`, and you're done

You don't need to activate `.venv-wav2lip` yourself — `make_avatar.py`
calls into it directly as a subprocess.

### InfiniteTalk (optional, `--engine infinitetalk`)

Only needed if you want situational facial expression (not just lip sync) —
see the pipeline description above. Skip this entirely if `latentsync` or
`sadtalker` already meet your needs; it's a much heavier install.

```bash
./setup_infinitetalk.sh
```

`setup_infinitetalk.sh` will:
- clone InfiniteTalk and create a fifth virtual environment (`.venv-infinitetalk`)
- install PyTorch and InfiniteTalk's other dependencies
- install `flash_attn` — **required, with no fallback, and no documented
  Windows support.** This is the step most likely to need manual
  troubleshooting: if it fails, the script prints options (install the full
  CUDA Toolkit + MSVC Build Tools and retry, look for a matching prebuilt
  Windows wheel, or run this engine from WSL2 instead, where wheel
  availability is much better). Everything else in the script still
  completes even if this step fails — fix it and re-run
  `pip install flash_attn==2.7.4.post1` inside `.venv-infinitetalk` afterward.
- download ~35-40GB of checkpoints: the Wan2.1-I2V-14B-480P base model, the
  chinese-wav2vec2-base audio encoder, and InfiniteTalk's own adapter
  weights (including the FP8-quantized model this project uses by default
  to fit a 12GB-class GPU)

You don't need to activate `.venv-infinitetalk` yourself — `make_avatar.py`
calls into it directly as a subprocess. Expect generation to be
significantly slower than `latentsync`/`sadtalker` — it's a 14B-parameter
model running quantized with CPU offload on a 12GB-class GPU; that's a
deliberate quality-over-speed tradeoff, not a bug.

## 4. Usage

### Quick single test

```bash
python make_avatar.py \
  --photo photos/me.jpg \
  --voice voice_samples/me.wav \
  --text "Hello, this is a test of my avatar." \
  --lang en \
  --out output/test.mp4
```

### Batch mode (pre-enter multiple lines/languages at once)

Edit `config.example.yaml`, save as `config.yaml`, then:

```bash
python make_avatar.py --config config.yaml
```

Each entry in the config produces its own `.mp4` in `output/`.

## 5. Supported languages

Chatterbox Multilingual V3 supports: `ar` Arabic, `da` Danish, `de` German,
`el` Greek, `en` English, `es` Spanish, `fi` Finnish, `fr` French, `he` Hebrew,
`hi` Hindi, `it` Italian, `ja` Japanese, `ko` Korean, `ms` Malay, `nl` Dutch,
`no` Norwegian, `pl` Polish, `pt` Portuguese, `ru` Russian, `sv` Swedish,
`sw` Swahili, `tr` Turkish, `zh` Chinese.

Use these codes in the `lang` field.

## 6. Tuning quality

**Speech pacing (all engines)**
- `--pause-ms` (default 450): silence inserted between sentences. Chatterbox
  has no built-in pause control, so long text is split into sentences,
  synthesized separately, and spliced back together with this much silence
  between them. Raise it for a more deliberate delivery, lower it for
  brisker pacing.

**Voice accent & delivery (all engines)**
- `--cfg-weight` (default 0.5, range 0.0-1.0): if the cloned voice sounds
  like it's speaking with the accent of your *reference clip's* language
  rather than the target language — or just sounds more strongly accented
  than you'd like (e.g. more American-sounding than your own voice) — lower
  this, try `0.0-0.3`.
- `--exaggeration` (default 0.5, range 0.25-2.0): controls delivery
  intensity/emotion. Higher values are more animated but also speak faster;
  if a high `--exaggeration` starts to sound rushed, pair it with a lower
  `--cfg-weight` to slow the pacing back down. Both are Chatterbox
  parameters, not something this project invents — for a genuinely
  different accent (not just neutralizing the current one), the accent has
  to come from the reference `voice_sample` itself; no parameter setting
  can add an accent that isn't in the sample.

**LivePortrait + LatentSync (default engine)**
- `--motion idle` (default): animates the photo with natural head motion
  and blinking before lip-syncing. `--motion none`: skips that and
  lip-syncs a frozen photo instead — use this if idle motion ever
  introduces visible identity drift on your particular photo.
- `--inference-steps` (default 20, try up to 50): more LatentSync diffusion
  steps ⇒ better quality, slower generation.
- `--guidance-scale` (default 1.5, range 1.0-3.0): higher ⇒ more accurate
  lip sync, but can introduce jitter/distortion past ~2.5.

**SadTalker** (`--engine sadtalker`)
- `--preprocess crop` (its default) keeps a tight face-only crop, which
  tends to give the cleanest results for an "avatar" look. `full` includes
  shoulders/more of the frame but the face is smaller.
- `--sadtalker-motion` allows head movement (default keeps the head still).
- `--enhancer` runs GFPGAN after generation — sharpens detail, but visibly
  reshapes/smooths the face (eyes, skin texture) — off by default for that
  reason.
- Expect noticeably weaker lip-sync accuracy than LatentSync — SadTalker
  drives a low-dimensional 3D expression coefficient rather than directly
  generating mouth pixels.
- `--refine-lipsync` adds a Wav2Lip pass after SadTalker that re-draws the
  mouth region directly from the audio waveform, tightening sync
  noticeably. Tradeoff: Wav2Lip composites a small, upsampled mouth crop
  back into each frame, so the mouth ends up visibly softer/lower-res than
  the rest of SadTalker's face — worth it when sync accuracy matters more
  than that. Requires `./setup_wav2lip.sh` (see section 3). No effect with
  the default `latentsync` engine — see that section.

**InfiniteTalk** (`--engine infinitetalk`)
- `--scene-prompt` is the main lever here: a text description of the
  delivery/situation you want (default is a generic business-speech
  description — see `config.example.yaml`). This is what actually shapes
  expression and head motion, not just the mouth — be specific about tone,
  setting, and body language for best results.
- `--infinitetalk-size {480,720}` (default `480`): 720p looks sharper but
  needs significantly more VRAM/time — start with 480p on a 12GB-class GPU.
- `--infinitetalk-steps` (default 40): more diffusion steps ⇒ better
  quality, slower.
- `--infinitetalk-quant {fp8,none}` (default `fp8`): keep this on unless you
  have considerably more than 12GB VRAM to spare.
- `--infinitetalk-no-low-vram`: disables CPU offloading — only useful if you
  have VRAM to spare; leave the default (offloading on) otherwise.
- No effect from `--refine-lipsync` here — InfiniteTalk's own audio-driven
  mouth generation is already higher-fidelity than Wav2Lip's patch.

## 7. A note on responsible use

This creates a synthetic video of a real face speaking words it never said.
Only use it with your own likeness/voice or with someone's explicit consent.
Chatterbox embeds an inaudible watermark in generated audio for exactly this
reason — don't try to strip it, and be transparent with anyone you share the
videos with about how they were made.

## 8. Troubleshooting

- **`CUDA out of memory`**: close other GPU programs, or lower audio length
  per clip. For SadTalker specifically, `--preprocess crop` (already
  default) helps too.
- **`insightface` fails to build** (LatentSync/LivePortrait setup): it has
  no prebuilt Windows wheel and needs a C++ compiler. Install Visual Studio
  Build Tools (see Requirements above), open a **new** terminal, and re-run
  the setup script that failed.
- **`UnicodeEncodeError` / `illegal multibyte sequence` during any setup or
  generation step**: some tool's console output (a deprecation warning, a
  progress bar) includes a character that can't be represented in a non-
  UTF-8 Windows console locale (common on Japanese/Chinese-locale Windows).
  This project's own scripts avoid this, but a new dependency version could
  reintroduce it in code we don't control — if you hit one, note which
  command triggered it; it's usually fixable by capturing that subprocess's
  output instead of letting it write to the console directly.
- **SadTalker produces a warped/blurry face**: use a higher-resolution,
  front-facing source photo (512x512 or larger, plain background) — or
  switch to the default engine, which doesn't warp the face at all.
- **No `ffmpeg` found**: `sudo apt install ffmpeg` (Linux), `brew install
  ffmpeg` (Mac), or download from ffmpeg.org and add to PATH (Windows).
- **Model download fails in `setup.sh`**: SadTalker's checkpoints are hosted
  on GitHub/HuggingFace — check your network/firewall isn't blocking those
  domains, then rerun `bash SadTalker/scripts/download_models.sh` directly.
- **Model download fails in `setup_latentsync.sh`**: re-run the two
  `huggingface-cli download` lines directly from inside `LatentSync/` with
  `.venv-latentsync` activated.
- **Model download fails in `setup_liveportrait.sh`**: re-run the `hf
  download` line directly from inside `LivePortrait/` with
  `.venv-liveportrait` activated.
- **`s3fd.pth` download fails in `setup_wav2lip.sh`**: search for
  "s3fd.pth download" — several mirrors of this face-detection weight
  circulate since the canonical one moves occasionally — and save it to
  `Wav2Lip/face_detection/detection/sfd/s3fd.pth`.
- **`wav2lip_gan.pth` not found / `--refine-lipsync` errors immediately**:
  this checkpoint isn't auto-downloaded (see section 3) — grab it from the
  "Model" table at
  [github.com/Rudrabha/Wav2Lip#getting-the-weights](https://github.com/Rudrabha/Wav2Lip)
  and save it to `Wav2Lip/checkpoints/wav2lip_gan.pth`.
- **`flash_attn` fails to install in `setup_infinitetalk.sh`**: often a
  Windows `MAX_PATH` (260-character) issue during extraction, not a missing
  compiler — flash-attn's source tree nests very deeply. The script already
  retries once with a short TEMP path; if it still fails, enable Windows
  long-path support (`New-ItemProperty -Path
  "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name
  "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force` as Administrator,
  then reboot) and retry. If it's a genuine compiler issue instead, install
  the CUDA Toolkit + MSVC Build Tools (same command as the `insightface`
  entry above) and retry `pip install flash_attn==2.7.4.post1` in
  `.venv-infinitetalk` in a new terminal, or run `--engine infinitetalk`
  from WSL2 instead, where wheel availability is much better.
- **`--engine infinitetalk` runs out of memory or is extremely slow**: this
  is expected on a 12GB-class GPU even with the defaults (`--infinitetalk-quant
  fp8`, offloading on) — it's a 14B-parameter model. Try
  `--infinitetalk-steps 20` for faster (lower-quality) generation, or a
  shorter line of text per run.
- **Model download fails in `setup_infinitetalk.sh`** with a `FileNotFoundError`
  pointing at a path under `.cache\huggingface\download\...`: this is the
  same Windows `MAX_PATH` issue as the `flash_attn` entry above, hit via
  `hf download`'s resumable-download cache instead — that cache lives
  *inside* `--local-dir`, so setting `HF_HOME` does not avoid it (confirmed
  by testing). The fix that actually works: move this whole project to a
  shorter path (e.g. `C:\avatar\` instead of nested under
  `Desktop\workspace\`) — this project was itself moved this way after
  hitting exactly this. Enabling Windows long-path support (see the
  `flash_attn` entry) should also fix it, if you'd rather do that than move
  the project.
- **`hf download` (or `hf --version`) exits immediately with no output at
  all**: the `hf` console-script `.exe` launcher itself can crash silently
  on some Windows setups, for reasons unrelated to this project.
  `setup_infinitetalk.sh` already works around this by invoking the same
  CLI through `python -c "from huggingface_hub.cli.hf import app; app()"`
  instead of the `hf` command directly — if you're re-running a download
  manually, do the same rather than typing `hf download ...` at the prompt.

## 9. Alternatives if this doesn't fit your machine

- **No GPU at all**: Chatterbox-Nano runs 3x realtime on CPU for the voice
  step, but the video step really wants a GPU. Consider a rented cloud GPU
  (RunPod, Lambda, Colab) for occasional batch runs instead of buying
  hardware.
- **Higher realism, more setup**: this project already integrates
  [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) as
  `--engine infinitetalk` (see section 3) for noticeably more lifelike,
  situation-appropriate expression than SadTalker/LatentSync, at the cost
  of much heavier compute (14B parameters) and a trickier install
  (`flash_attn`). If that still isn't enough, other EMO/OmniHuman-style
  diffusion avatar models circulate on GitHub and are worth watching, but
  aren't wired into this project.
