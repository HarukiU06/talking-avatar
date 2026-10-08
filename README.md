# Talking Avatar

Turn a photo of your face + a short voice sample into videos of "you" speaking
any text you type, in any of 23 languages. Everything runs locally — no cloud
service, no per-video cost, your face and voice never leave your computer.

**Pipeline:**
1. **Text → speech in your voice** — [Chatterbox Multilingual TTS](https://github.com/resemble-ai/chatterbox) (Resemble AI, MIT license) clones your voice from a short reference clip and speaks your text in it. Your reference clip is automatically cleaned up first (downmixed to mono, resampled, silence-trimmed, level-matched) — zero-shot cloning derives your entire voice identity from that one file, so its defects are inherited by every line you generate. `--tts xtts` swaps in [XTTS-v2](https://github.com/idiap/coqui-ai-TTS) as an alternative (17 languages instead of 23 — see section 3).

   **If the result doesn't sound enough like you, that is expected, and the fix is `--voice-convert`.** Zero-shot cloning has to invent natural delivery *and* imitate a specific person from a few seconds of audio, and it compromises on both — no parameter crosses that gap. `--voice-convert` adds a second stage, [Seed-VC](https://github.com/Plachtaa/seed-vc), which converts the generated speech's timbre to a reference recording of you. The TTS then only has to sound like a person talking, and Seed-VC decides who. `--voice-compare` renders the same line through every combination so you can pick by ear. By default the whole text goes to the TTS in one call so intonation carries across sentences; `--pause-ms` splits it into sentences with fixed silence between them instead (see section 6).
2. **Photo → idle-motion video** — by default, [LivePortrait](https://github.com/KwaiVGI/LivePortrait) (Kuaishou, MIT license) animates your still photo with natural head motion and blinking, driven by one of its bundled example clips (only the *motion* is transferred — the driving clip's own appearance/identity never appears in the output). The clip is short, so it's ping-pong looped to match your audio's length.
3. **Idle-motion video + speech → final video** — [LatentSync](https://github.com/bytedance/LatentSync) (ByteDance, Apache 2.0) lip-syncs that video to the audio, regenerating only the mouth region — everything else (eyes, face shape, the head motion from step 2) is carried through unchanged.

Pass `--motion none` to skip step 2 and lip-sync a frozen photo instead (useful if LivePortrait's motion ever looks off), or `--engine sadtalker` to use [SadTalker](https://github.com/Winfredy/SadTalker) (Apache 2.0) for the whole video step instead of LivePortrait+LatentSync — SadTalker is a single full-face reenactment model, simpler to run but with noticeably weaker lip sync and visible identity drift (it warps the whole face, not just the mouth, as a side effect of generating mouth motion).

For actual situational facial expression (not just lip sync — e.g. the face reading as "giving a confident business speech" rather than a static or generically animated one), pass `--engine infinitetalk` to use [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) (MeiGen-AI, Apache 2.0) instead — a single audio+photo+text-prompt model that generates lip sync, head motion, and expression together, conditioned on a `--scene-prompt` describing the delivery you want. It's a 14B-parameter model (built on Wan2.1-I2V), so it's noticeably slower and needs significantly more disk space than the other two engines — see its setup section below.

You can use it as a **desktop app** (`start.bat` on Windows, `./start.sh`
elsewhere): save your face and voice once as an avatar, type a script, press
**Generate**, and watch the progress while videos render in the background.
Or use the command line and "pre-enter" one or many lines of text in a config
file, walk away, and come back to finished `.mp4` files.

No suitable GPU? [`colab/talking_avatar_colab.ipynb`](colab/talking_avatar_colab.ipynb)
runs the InfiniteTalk engine on Google Colab instead (see section 3).

### Quick start

```bash
git clone https://github.com/HarukiU06/talking-avatar.git
cd talking-avatar
./setup.sh               # main environment: voice cloning (+ SadTalker)
./setup_latentsync.sh    # default lip-sync engine
./setup_liveportrait.sh  # default head-motion engine
./start.sh               # opens the app (on Windows you can double-click start.bat)
```

Prefer the command line? Put a photo in `photos/me.jpg` and a 25-30s voice
clip in `voice_samples/me.wav`, then:

```bash
source .venv/bin/activate   # Windows (Git Bash): source .venv/Scripts/activate
python make_avatar.py --photo photos/me.jpg --voice voice_samples/me.wav \
  --text "Hello, this is a test of my avatar." --lang en --out output/test.mp4
```

The rest of this README covers requirements, how to record good inputs, the
optional engines, and tuning. Contents: [1. Requirements](#1-requirements) ·
[2. What to prepare](#2-what-to-prepare) · [3. Setup](#3-setup) ·
[4. Usage](#4-usage) · [5. Languages](#5-supported-languages) ·
[6. Tuning](#6-tuning-quality) · [7. Responsible use](#7-a-note-on-responsible-use) ·
[8. Troubleshooting](#8-troubleshooting) · [9. Alternatives](#9-alternatives-if-this-doesnt-fit-your-machine) ·
[Project layout](#project-layout) · [License](#license)

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
  engines); **+2 GB** for `--tts xtts`, and **+70 GB** for `--engine
  infinitetalk`
- **32 GB of system RAM is the floor for `--engine infinitetalk`**, and it is
  tight: the quantized 14B model is streamed from system RAM on every
  diffusion step (~18 GB resident), so close memory-hungry apps first. The
  other engines are unaffected.
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
- **25–30 seconds of clean audio of your voice** — `voice_samples/me.wav`.
  Quiet room, no background music/noise, one continuous take, natural
  speaking tone, saved as `.wav` or `.mp3`. This is the sample the voice is
  cloned from — it does *not* need to be in the same language as what you'll
  generate later. 6s is the technical minimum, but the clone gets audibly
  better up to ~30s, and this one file determines the voice in every video
  you ever generate; it's worth a second take. Mono or stereo, any sample
  rate — the pipeline normalizes it.
- **Strongly recommended: 1–2 minutes of video of yourself not talking** —
  e.g. `photos/me.mp4`. Look at the camera, blink naturally, small nods and
  turns, slight expression shifts, mouth mostly closed. Pass it as
  `--video`, in place of `--photo`.

  **This is the single biggest quality decision in the whole pipeline.** With
  a photo, the head motion has to be invented — copied from a stock clip of a
  stranger and looped every few seconds, which is what produces eyes tracking
  side to side on a cycle and smiles appearing at random. With video, your
  own motion, blinks and expression are already there and only the mouth is
  re-synced. Nothing is invented, so nothing repeats.

  How to record it, in rough order of importance:
  - **Fill the frame with your head and shoulders.** Only the face region is
    actually used, so anything else is resolution thrown away. The clip is
    downscaled to 768px to fit in memory, and whatever fraction of the frame
    your face occupies is the fraction of that budget it gets.
  - **Don't let the camera letterbox it.** Phone video shot in one
    orientation and saved in the other arrives padded with black bars — one
    real example here was a 1280x720 file whose actual content was 404x720,
    two thirds of every frame black. The bars are detected and cropped
    automatically, but the pixels are already gone by then. Record in the
    orientation you intend to keep.
  - **Mouth closed, or nearly.** The mouth is the one region that gets
    replaced, and an original mouth that is already moving fights the new
    lip-sync.
  - **Longer than your typical line**, so no looping is needed at all.
  - **Wear the expression you want the avatar to have** — with
    `--engine latentsync` the expression passes through unchanged.
  - Even, front-on lighting; plain background; camera at eye level; hold
    reasonably still (small natural movement is the point, large movement is
    not).

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

### XTTS-v2 (optional, `--tts xtts`)

Only needed if Chatterbox's clone of your voice doesn't sound enough like
you. XTTS-v2 is a different zero-shot cloning model that often matches
timbre more closely; it's small (~2GB) and fast, so it's a cheap thing to
try.

```bash
./setup_xtts.sh
```

`setup_xtts.sh` will:
- create a sixth virtual environment (`.venv-xtts`) and install `coqui-tts`
  (the maintained fork of the archived `coqui-ai/TTS` package — installing
  plain `TTS` gets you the dead original)
- download the XTTS-v2 checkpoint (~2GB)

Then compare the two without paying for the slow video step:

```bash
python make_avatar.py --voice-compare   --voice voice_samples/me.wav --text "One sentence in your own words." --lang en
```

That writes one `.wav` per voice setting to
`output/voice_ab/<your-clip-name>/` (or the folder given with `--out`) —
Chatterbox with the cleaned and the raw clip, at a lower `--cfg-weight` and
sentence by sentence, XTTS whole and sentence by sentence, and Chatterbox and
XTTS each followed by Seed-VC: up to eight, skipping whatever isn't
installed — and exits. Listen, pick the one that
sounds most like you, and use the matching flags for real generation. The
app's Voice Lab tab does the same with a player for each.

Results are filed under the reference clip's name, so you can run this
against two different recordings of yourself and compare those too — often
the more useful comparison, since the reference clip matters more than any
setting.

**Languages:** XTTS covers 17 of Chatterbox's 23 — it has no Danish, Greek,
Finnish, Hebrew, Malay, Norwegian, Swedish or Swahili, and adds Czech and
Hungarian. `make_avatar.py` will tell you if you ask for one it can't do.

**Licensing:** XTTS-v2's *weights* are under the Coqui Public Model License,
which is non-commercial (the code is MPL-2.0). Chatterbox is MIT throughout.
If you plan to use the output commercially, stay on Chatterbox.

### Seed-VC (optional but recommended, `--voice-convert`)

The second stage of the voice pipeline, and the thing to install if the
cloned voice doesn't sound like you. It takes speech that has already been
generated and converts its speaker identity to a reference recording.

```bash
./setup_seedvc.sh
```

`setup_seedvc.sh` clones Seed-VC, creates `.venv-seedvc`, and installs
PyTorch (CUDA 12.8) plus its dependencies. Model checkpoints download
automatically on first run. It's light — upstream benchmarks it on an RTX
3060 Laptop — so it adds little to generation time.

```bash
python make_avatar.py --photo photos/me.jpg --voice voice_samples/me.wav   --voice-convert --vc-target voice_samples/me_long.wav   --text "..." --lang en --out output/test.mp4
```

`--vc-target` is the reference Seed-VC converts *towards*, and defaults to
whatever `--voice` is. It's worth pointing at a longer recording of
yourself if you have one: it only has to establish who you are, so it can be
longer and less pristine than the TTS's reference clip.

Why this rather than RVC: RVC needs a per-speaker training run, and on
RTX 50-series (Blackwell) cards it currently needs CUDA 12.8 plus
nightly-PyTorch workarounds that are still a moving target. Seed-VC works
zero-shot with no training, and ships a `train.py` if you later want to
fine-tune on a longer recording.

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
  weights (including the FP8-quantized DiT and T5 this project uses by
  default to fit a 12GB-class GPU)

You don't need to activate `.venv-infinitetalk` yourself — `make_avatar.py`
calls into it directly as a subprocess. That subprocess is
`infinitetalk_run.py` (part of this repo), not InfiniteTalk's own
`generate_infinitetalk.py`: it applies a set of runtime patches (Windows
memory behaviour, transformers 5, RTX 50-series kernels, GPU memory
management) and then hands every argument through. Nothing inside the
`InfiniteTalk/` checkout is modified, so it survives a re-clone; the file's
docstring explains each patch. Expect generation to be *much* slower than
`latentsync`/`sadtalker` — on a 12GB laptop GPU (RTX 5070 Ti) with 32GB
RAM, one 81-frame (3.2s) chunk at 480p costs about 2.5 minutes per
diffusion step (three DiT passes of ~48s each, for classifier-free
guidance), so the default 40 steps is roughly 1.5 hours per chunk, plus a
~40s VAE decode, and `--infinitetalk-mode streaming` chains one chunk per
~2.9s of audio. It's a 14B-parameter video model running quantized with
CPU offload; that's a deliberate quality-over-speed tradeoff, not a bug.
Start with `--infinitetalk-steps 8` (about 20 minutes per chunk) to check
that a photo and prompt work before committing to a long run.

`--infinitetalk-accel lightx2v` is the fast path: it loads the lightx2v
step-distillation LoRA (downloaded by `setup_infinitetalk.sh`) with the
settings InfiniteTalk's README gives for it — 4 steps, text CFG 1, audio
CFG 2 — so each step is two DiT passes instead of three and there are 4
steps instead of 40. Measured on the same machine: about 7.5 minutes per
chunk including the VAE decode, versus ~1.5 hours, so roughly 2.6 minutes
of rendering per second of speech. Distillation isn't bit-identical to 40
full-CFG steps; compare on your own photo if it matters.

No suitable GPU? `colab/talking_avatar_colab.ipynb` runs the same engine on
Google Colab (setup, model download, upload your photo and voice, render in
sections). It has not been tested on the free T4, which is much weaker than
the machine above; it starts with a one-chunk timing test for that reason.
It relies on `infinitetalk_run.py` falling back to PyTorch attention when
`flash_attn` isn't installed (`SKIP_FLASH_ATTN=1 ./setup_infinitetalk.sh`).

For anything longer than about a minute, render it in sections (one
`make_avatar.py` call per paragraph, or `lines:` in a config) and join them
with ffmpeg's concat demuxer: InfiniteTalk's README notes that single-image
generation drifts in colour beyond ~1 minute, and each section restarts
from the original photo.

## 4. Usage

### The app

Start it with **`start.bat`** (Windows: double-click it) or **`./start.sh`**
(macOS, Linux, Git Bash). It opens in its own window if Chrome or Edge is
installed, otherwise in your browser. The terminal window that starts with it
is the app's engine: leave it open while you work, and to stop the app close
it, press Ctrl+C in it, or use **Quit** on the Setup tab.

- **Create**: choose or make an avatar (a photo or a video of you, plus a
  25-30 s voice sample, either uploaded or recorded with your webcam and
  microphone), type the script, pick the language and engine, and press
  **Generate video**. Save the avatar once and it's there next time. Tick
  *One video per paragraph* to make a separate video from each paragraph.
- Videos render one at a time in the background, with live progress and the
  full log under *Details*. You can keep adding more, and closing the window
  doesn't stop anything: open the app again to see where it got to.
- **Library**: every video you've made, with the text and settings it was
  made with. *Use these settings* loads them back into Create.
- **Voice Lab**: the same sentence in every voice setting, side by side, so
  you can choose a voice before spending minutes on video (the app's version
  of `--voice-compare`).
- **Setup**: what's installed, which setup script adds what's missing, and
  your GPU.

The app runs `make_avatar.py` for every video, so the rest of this README
applies to it too: *Advanced settings* holds the same options as the command
line. `start.bat` and `start.sh` pass their arguments on to `app.py`:
`--browser` opens a normal browser tab, `--port N` picks the port, and
`--listen --auth USER:PASSWORD` makes the app reachable from other computers
on your network, behind a login (without one, anyone who can reach your
computer could use the app and see your videos).

Saved avatars go in `avatars/`, videos in `output/`, and the app's settings
in `app_settings.json`. Git ignores all three.

### Command line

#### Quick single test

```bash
python make_avatar.py \
  --video photos/me.mp4 \
  --voice voice_samples/me.wav \
  --voice-convert \
  --text "Hello, this is a test of my avatar." \
  --lang en \
  --out output/test.mp4
```

`--photo photos/me.jpg` works in place of `--video` if you have no footage,
but expect noticeably worse results — see section 2.

#### Batch mode (pre-enter multiple lines/languages at once)

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

With `--tts xtts` the set is different — 17 languages: `ar` Arabic, `cs`
Czech, `de` German, `en` English, `es` Spanish, `fr` French, `hi` Hindi,
`hu` Hungarian, `it` Italian, `ja` Japanese, `ko` Korean, `nl` Dutch, `pl`
Polish, `pt` Portuguese, `ru` Russian, `tr` Turkish, `zh` Chinese. Danish,
Greek, Finnish, Hebrew, Malay, Norwegian, Swedish and Swahili are
Chatterbox-only.

## 6. Tuning quality

**Speech pacing (all engines)**
- `--pause-ms` (default 0): at 0, the whole text goes to the TTS in one call
  and the model places its own pauses — it carries intonation across sentence
  boundaries, which is most of what makes speech sound like a person rather
  than a list. Above 0, text is split into sentences, each is synthesized
  separately, and this much silence is spliced between them. That buys exact
  pause control at a real cost: every sentence restarts at neutral intonation,
  and the uniform gaps sound mechanical. Splicing used to be the default here;
  it was changed after listening tests. Raise it only if a model genuinely
  runs sentences together.

**Which voice, and what it's cloned from (all engines)** — start here if the
generated voice doesn't sound like you. In rough order of impact:
- **`--voice-convert`.** The single biggest lever, and the only one that
  addresses the actual limitation rather than working around it. Zero-shot
  cloning must produce natural delivery and imitate you simultaneously, and
  compromises on both; this splits the job. Needs `./setup_seedvc.sh`
  (section 3). Pair it with `--vc-target` pointing at the longest clean
  recording of yourself you have.
- **The reference clip itself.** No parameter can add what isn't in the
  sample. 25–30s, quiet room, natural tone, one take. This matters more than
  every setting below it.
- `--pause-ms 0` (now the default) — see "Speech pacing" above. If your audio
  sounds like disconnected fragments, check you haven't raised this.
- The clip is cleaned automatically (mono, resampled, silence-trimmed,
  level-matched) before cloning — `--no-voice-prep` turns that off if you'd
  rather hand the model your file untouched.
- `--tts xtts` swaps Chatterbox for XTTS-v2, a different cloning model that
  often tracks timbre more closely. Needs `./setup_xtts.sh` (section 3).
- `--vc-denoise` (default 0.6, 0 = off) cleans the conversion reference
  before use. **If your generated audio has background noise, this is the
  control.** A conversion model cannot separate "how this person sounds" from
  "what their room sounds like" — both are just properties of the reference —
  so it copies the reference's room tone onto every line it generates. On this
  project's own recording the converted output's noise floor matched the
  source recording's within ~1dB per band, while the TTS audio going in was
  3-6dB cleaner. Cleaning the reference cut the output noise floor by ~13dB.
  It also *improved* speaker similarity (0.737 → 0.850 at strength 0.8),
  which is worth knowing because the opposite appears true if you measure
  against a noisy reference: a clip that reproduces the room tone scores
  higher for matching the noise, not the voice. Measure against a cleaned
  reference or the comparison is rigged.

  The default is nevertheless **0.6, not the metric-optimal 0.8** — 0.8 was
  judged over-processed in a listening test by the speaker themselves. Noise
  floor and speaker similarity both say "more is better" right up to 1.0, and
  neither measures whether the result still sounds like a human voice. Trust
  your ears over this section; raise it if noise still gets through, lower it
  if the voice starts sounding thin or watery.
- `--vc-steps` (default 25) is Seed-VC's quality/speed dial.
- `--voice-compare` renders the same sentence through all of the above into
  `output/voice_ab/<clip-name>/` and exits without making video. Use this to
  decide — the video step costs minutes, the audio costs seconds. Run it on
  two different recordings to compare those as well.
- `--keep-intermediates` keeps the synthesized `.wav` (and the driving
  video) from a real run instead of deleting them, so you can listen to
  exactly what the video was built from.

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
- `--motion-video path/to/your_idle_clip.mp4` — **the biggest realism win
  available.** LatentSync only regenerates the mouth; every other part of
  the face is carried through from the driving video unchanged. So if the
  driving video barely moves, you get a frozen face with a moving mouth,
  which is exactly what "unnatural" usually means here.

  The bundled default (`d0.mp4`) is wrong for this in two ways. It is 3.1
  seconds long, so it gets ping-pong looped ~10x for a normal line, which
  reads as robotic repetition. And every clip LivePortrait ships is an
  *expression demo* — the driver in `d0` grins broadly, so **your avatar
  grins broadly too, through whatever you typed.** LivePortrait transfers
  expression and head pose together; it cannot take the motion and leave the
  emotion. If your avatar looks like it's smiling through a serious line,
  this is why, and no parameter fixes it — only a different driving clip
  does. A 15–20s recording of your own head idling, with the expression you
  actually want, fixes all of it. See section 2 for how to record one.
- `--motion-scale` (default 1.0) amplifies the transferred motion.
  **Use sparingly.** It reliably increases movement (measured +43% and +48%
  upper-face motion at 1.5 on two different photos), but it amplifies the
  *deviation from your source photo*, not just the movement — at 1.5 the
  face shape, eye shape and smile visibly drift away from your actual face.
  It buys motion by spending likeness. Reach for a better driving clip
  first; use 1.1–1.2 only if the motion is genuinely too subtle after that,
  and check the result against your photo rather than trusting the motion
  numbers, which cannot see identity drift.
- `--motion idle` (default): animates the photo with natural head motion
  and blinking before lip-syncing. `--motion none`: skips that and
  lip-syncs a frozen photo instead — use this if idle motion ever
  introduces visible identity drift on your particular photo. Note that
  `--motion none` produces a completely static face by design.
- `--latentsync-res` (default 256) sets the resolution the mouth region is
  regenerated at. 512 is available, but the checkpoint this project
  installs is LatentSync 1.5, which was *trained* at 256 — 512 costs 4x the
  VRAM and is not reliably sharper with these weights. A/B it rather than
  assuming it's an upgrade.
- **Checking whether it actually moved:** `python tools/measure_motion.py
  output/your_video.mp4` reports frame-to-frame motion energy for the upper
  face and the mouth separately, plus their **ratio**. The ratio is the
  number to read: real human video sits around 0.6–0.75, and a ratio near 0
  means the face is frozen while only the mouth animates — the exact failure
  this section is about. Pass several files to compare them. The absolute
  values scale with how much of the frame your face fills, so they're only
  comparable between videos of the same size; the script warns you when
  they aren't.
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
  have considerably more than 12GB VRAM to spare. (`none` also needs the
  ~32GB of Wan2.1 diffusion shards, which `setup_infinitetalk.sh` downloads;
  they can be deleted from `InfiniteTalk/weights/Wan2.1-I2V-14B-480P/` if
  you only ever use `fp8`.)
- `--infinitetalk-no-low-vram`: disables CPU offloading — only useful if you
  have VRAM to spare; leave the default (offloading on) otherwise.
- No effect from `--refine-lipsync` here — InfiniteTalk's own audio-driven
  mouth generation is already higher-fidelity than Wav2Lip's patch.
- Lines shorter than about 3.3s of speech are padded with silence before
  hand-off (InfiniteTalk refuses audio shorter than one 81-frame chunk) and
  the video is trimmed back to the speech afterwards; you'll see a
  `Padding ...` line when this happens.

## 7. A note on responsible use

This creates a synthetic video of a real face speaking words it never said.
Only use it with your own likeness/voice or with someone's explicit consent.
Chatterbox embeds an inaudible watermark in generated audio for exactly this
reason — don't try to strip it, and be transparent with anyone you share the
videos with about how they were made. **XTTS-v2 (`--tts xtts`) does not
watermark its output**, so if you switch to it that safeguard is gone and
the transparency is entirely on you.

## 8. Troubleshooting

- **The app doesn't open a window**: open the address it prints in the
  terminal (`http://127.0.0.1:7860/` unless that port was taken) in any
  browser, or start it with `--browser`. If it says Gradio isn't installed,
  it's running with the wrong Python: start it with `start.bat` / `start.sh`,
  which use `.venv`, after running `setup.sh`.
- **The app refuses to generate and lists what's missing**: that's the
  pre-flight check. The Setup tab shows which engines are installed and the
  setup script for each one that isn't.

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
  `hf download` lines directly from inside `LatentSync/` with
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
- **`--engine infinitetalk` dies instantly with exit code `3221225477` and
  no Python traceback**, right after logging `Creating WanModel from ...`:
  that code is `0xC0000005`, a native access violation from running out of
  Windows *commit* (RAM + pagefile), not a GPU/CUDA/flash-attn problem.
  Upstream loads its quantized checkpoint through `optimum-quanto`'s
  `requantize()`, which first materializes every parameter of the 18.9B
  fp32 placeholder model (~75GB) before overwriting it with the 19.5GB of
  fp8 weights; Linux overcommits that, Windows charges it up front.
  `infinitetalk_run.py` replaces that loader, so this should no longer
  happen when running through `make_avatar.py`. If you see it anyway, you
  are probably invoking `generate_infinitetalk.py` directly — go through
  `make_avatar.py`, or run `infinitetalk_run.py` with the same arguments
  from inside `InfiniteTalk/`. To see the fault as a traceback rather than a
  bare exit code, add `python -X faulthandler`.
- **`--engine infinitetalk` takes many minutes per diffusion step (progress
  bar shows 300+ `s/it`) instead of under a minute**: on a 12GB-class card
  the whole run sits within a gigabyte of the VRAM limit, and recent NVIDIA
  Windows drivers default to *"CUDA - Sysmem Fallback Policy: Prefer Sysmem
  Fallback"*, which silently moves anything that doesn't fit into system
  RAM at PCIe speed instead of failing. `infinitetalk_run.py` caps PyTorch's
  allocations just below the free VRAM at startup so this doesn't happen,
  but that cap is computed from what is free *when the run starts*: close
  other GPU-using apps (browsers with hardware acceleration, games, other
  model runs) before launching. Setting the driver policy to *"Prefer No
  Sysmem Fallback"* in the NVIDIA Control Panel (Manage 3D settings) turns
  any remaining overflow into an explicit out-of-memory error, which is
  the better failure. System RAM matters too: with 32GB, the ~18GB of
  streamed weights plus everything else leaves little headroom, and paging
  shows up the same way.
- **`--engine infinitetalk` reports `CUDA out of memory`**: the 480p forward
  pass peaks at ~7.4GB and the final VAE decode needs ~8.2GB on its own, so
  a 12GB card has essentially no slack. Make sure nothing else is using the
  GPU, try a shorter line of text, or drop `--infinitetalk-steps`. If only
  the VAE decode fails, `infinitetalk_run.py` retries it on the CPU
  automatically (about 5 minutes per 81-frame chunk on a 16-thread CPU)
  rather than discarding the sampled clip.
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

## Project layout

| Path | What it is |
|---|---|
| `start.bat`, `start.sh` | Start the desktop app (`app.py`) with the `.venv` from `setup.sh` |
| `app.py`, `ui/` | The desktop app: screens, render queue, saved avatars, library |
| `make_avatar.py` | The pipeline: text → voice → video, single line or `--config` batch; the app runs it for every video |
| `config.example.yaml` | Template for batch mode; copy to `config.yaml` (gitignored) |
| `setup.sh`, `setup_*.sh` | One installer per engine, each with its own virtual environment |
| `xtts_synth.py` | XTTS-v2 worker, run inside `.venv-xtts` by `make_avatar.py` |
| `infinitetalk_run.py` | InfiniteTalk launcher with runtime patches, run inside `.venv-infinitetalk` |
| `tools/measure_motion.py` | Reports how much a rendered video's face actually moves |
| `colab/` | Google Colab notebook for the InfiniteTalk engine |

Your own inputs and results (`photos/`, `voice_samples/`, `output/`,
`config.yaml`, and the app's `avatars/` and `app_settings.json`) and every
downloaded model or virtual environment are gitignored, as are common audio/video/image files anywhere in the tree, so a
photo or voice recording can't be committed by accident.

## License

This project's own code is released under the [MIT License](LICENSE).

The models it downloads and runs are **not** part of this repository and keep
their own licenses: Chatterbox (MIT), LivePortrait (MIT), LatentSync
(Apache 2.0), SadTalker (Apache 2.0), InfiniteTalk (Apache 2.0), Wav2Lip
(research / non-commercial; see its repository), Seed-VC (GPL-3.0), and
XTTS-v2 (Coqui Public Model License, non-commercial). LatentSync and
LivePortrait also rely on InsightFace, whose pretrained face models are for
non-commercial research only. Check the license of each engine you use before
using its output commercially.
