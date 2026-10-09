# Firefly-style speech controls + a Japanese voice with correct pitch accent

Date: 2026-10-09 · Branch: `firefly-style-tts` (from `master` at a55fa42)

## Goal

Give the speech step the controls Adobe Firefly's *Generate Speech* offers —
inline pauses, emotion tags, per-word emphasis, pronunciation fixes, speed and
pitch — implemented locally on top of this project's own TTS backends, and fix
Japanese intonation, which every TTS route so far has failed by ear.

Firefly's feature set is taken from Adobe's public product page and help docs.
Firefly itself is not called, reverse-engineered, or depended on: everything
stays local, per the project's "no cloud services" rule.

### Why Japanese intonation needs a new backend

Chatterbox (cfg_weight 0.2, hiragana input) and XTTS-v2 (+ Seed-VC) were both
rejected by ear for Japanese intonation in earlier work, even with every kanji
reading spelled out, and the fallback was recording a human reading. The cause
is structural: both are multilingual models with no explicit model of Japanese
pitch accent (高低アクセント), so no parameter fixes it. Style-Bert-VITS2
JP-Extra (SBV2) predicts accent explicitly from pyopenjtalk and accepts a
per-mora accent override (`given_phone` / `given_tone`), so wrong accents can
also be corrected by hand.

### Success criteria

- Every control below works from `--text`, from batch configs, and from the app.
- Text with no markup produces exactly one TTS call per backend, as today.
- On a fixed Japanese A/B set, the user prefers SBV2's intonation over the best
  previous route (Chatterbox, cfg_weight 0.2) by ear.
- No change to video engines, the 23-language coverage, or any existing flag's meaning.

## Non-goals (this round)

- Training from the app (CLI only).
- A preset voice library (this project clones the user's voice; it has no stock voices).
- SBV2 for languages other than Japanese (JP-Extra is Japanese-only).
- Adding SBV2 to `--voice-compare`.

## 1. Script markup

```
こんにちは。[pause 800]今日は{雨|あ'め}ですね。[happy]でも元気です！[/happy] *本当に* ありがとう。
```

| Control | Syntax | Rules |
|---|---|---|
| Pause | `[pause 800]`, `[pause 800ms]`, `[pause 1.5s]` | 0–10000 ms. Splits the text there and splices exactly that much silence. |
| Emotion | `[happy]…[/happy]`, `[happy 1.5]…[/happy]` | Name: `[a-z0-9_]+`. Optional intensity 0–3 (default 1). No nesting of emotion spans; unclosed or mismatched → error. |
| Emphasis | `*word*` | No nesting. May sit inside an emotion span. |
| Pronunciation | `{surface\|reading}` | The reading replaces the surface text for synthesis. |
| Accent (ja only) | `{雨\|あ'め}`, `{飴\|あめ0}` | `'` after the mora where pitch drops (the accent nucleus); a trailing `0` means flat (heiban); neither means "reading only, OpenJTalk decides the accent". `'` or `0` on a non-`ja` line → error. |
| Literal characters | `\[`, `\*`, `\{` | Escapes. |

**Pronunciation dictionary**: `pronunciations.yaml` (path via `--pronunciations`,
default `pronunciations.yaml` in the repo root if present, gitignored as user
data; `pronunciations.example.yaml` committed):

```yaml
ja:
  雨: "あ'め"
en:
  Ueno: "oo-EH-no"
```

Applied to plain text only (never inside `{…}`), longest key first. Inline
`{…|…}` wins over the dictionary.

**Segmentation**: the parser cuts the text only at pauses and at emotion/emphasis
boundaries. A run of plain text — however many sentences — stays one segment,
so untagged text keeps the single-call behaviour the project chose because
spliced sentences sounded mechanical. `--pause-ms > 0` keeps its existing
meaning: each segment is additionally split into sentences with that gap; an
explicit `[pause]` at the same spot takes precedence.

**Speed / pitch** are global flags, not markup:
`--speed` (multiplier, 0.5–2.0, default 1.0) and `--pitch` (semitones, −12…+12,
default 0).

### How each backend realizes the controls

| Control | SBV2 (`ja`) | Chatterbox | XTTS |
|---|---|---|---|
| Pause | silence splice | silence splice | silence splice |
| Emotion | `style` = name looked up in the model's `styles.yaml`, `style_weight` = base weight × intensity | preset `(exaggeration, cfg_weight)` per tag, intensity scales the exaggeration delta from the run's base | error: XTTS has no emotion control (consistent with how XTTS already rejects cfg_weight/exaggeration rather than ignoring them) |
| Emphasis | segment with `intonation_scale` 1.3 | segment with exaggeration +0.3 | error, same reason |
| Pronunciation | reading substituted; accent → `given_phone`/`given_tone` (see §3) | reading substituted | reading substituted |
| Speed | native `length = 1/speed` | ffmpeg `rubberband` (formants preserved) | ffmpeg `rubberband` |
| Pitch | native `pitch_scale = 2^(st/12)` | ffmpeg `rubberband` | ffmpeg `rubberband` |

Chatterbox emotion presets — starting points, tuned by ear during implementation:
`calm (0.35, 0.5)`, `happy (0.8, 0.4)`, `excited (1.1, 0.3)`, `sad (0.4, 0.5)`,
`angry (1.2, 0.3)`, `serious (0.45, 0.6)`. Emphasis is the weakest control on every backend
(neither model has a real emphasis input) and costs a splice around the word.

SBV2 caveat: its styles are clusters of the user's own recording. A calm
recording yields calm clusters; `[angry]` cannot be more angry than anything
in the training audio.

## 2. Routing

`--tts` gains `auto` (new default) and `sbv2`:

- `auto`: `ja` → SBV2 if a trained model for this voice exists; otherwise, and
  for every other language, Chatterbox. Falling back for `ja` prints a warning
  naming the `train_sbv2.py` command.
- `sbv2`, `chatterbox`, `xtts`: forced; errors if the backend cannot serve the language.

`--sbv2-model NAME` selects `models/sbv2/NAME/`; default is the `--voice`
file's stem (`voice_samples/newtest.wav` → `newtest`).

## 3. Components

| Unit | Runs in | Responsibility |
|---|---|---|
| `script_markup.py` (new) | `.venv` | `parse(text, lang, pron) -> list[Segment]`, `load_pronunciations(path) -> dict`. `Segment` is a dataclass: `pieces` (list of `(text, accent)` where accent is `None`, an int nucleus position, or `0`), `emotion`, `intensity`, `emphasis`, `pause_after_ms`. Pure Python, no model imports. |
| `setup_sbv2.sh` (new) | creates `.venv-sbv2` | Clone Style-Bert-VITS2 into `Style-Bert-VITS2/` (gitignored, never patched, like `InfiniteTalk/`); install torch 2.8 cu128 (the version `.venv-seedvc` already runs on this RTX 50-series GPU), then SBV2's requirements; `initialize.py --skip_default_models --dataset_root models/sbv2-data --assets_root models/sbv2`. Overrides: `SBV2_DIR`, `SBV2_VENV`. |
| `train_sbv2.py` (new) | `.venv` driver, every step a subprocess into `.venv-sbv2` | `python train_sbv2.py --voice voice_samples/newtest.wav [--name newtest] [--epochs 100] [--clusters 4] [--stop-after STEP]`. Steps, each skipped when its output exists: **stage** (convert input to wav in a staging dir) → **slice** (`slice.py`, 2–12 s) → **transcribe** (`transcribe.py`, faster-whisper large-v3, `jp`; prints the transcript list path — Whisper mistakes become training labels, so `--stop-after transcribe` allows hand review) → **preprocess** (`preprocess_all.py --use_jp_extra -e EPOCHS`) → **train** (`train_ms_jp_extra.py`, resumes from its last checkpoint) → **styles** (`sbv2_styles.py`). |
| `sbv2_styles.py` (new) | `.venv-sbv2` | Clusters the per-utterance style vectors preprocessing wrote (KMeans, `--clusters`), writes `style_vectors.npy` and `style2id` into the model's config (mean = `Neutral`), copies the utterance nearest each centroid to `models/sbv2/NAME/style_samples/<style>.wav`, and writes `models/sbv2/NAME/styles.yaml` mapping tag names to styles (`neutral: Neutral`, `style_1: style_1`, …) for the user to rename after listening. |
| `sbv2_synth.py` (new) | `.venv-sbv2` | Worker with the `xtts_synth.py` contract: reads `job.json` (segments, model dir, speed, pitch), writes one wav per segment plus `job.json.result` (`files`, `sr`). Per segment: if any piece carries an accent, build `kata_tone` piecewise with SBV2's own `g2kata_tone`, overwrite the overridden pieces' tones with the Tokyo pattern for that nucleus (0 → low-high…; 1 → high-low…; n → low, high to mora n, low after), convert with `kata_tone2phone_tone`, and pass `given_phone`/`given_tone`; otherwise plain text. Always `line_split=False` (splitting is ours). |
| `make_avatar.py` | `.venv` | `synthesize()` parses markup, dispatches per segment to the backend, splices with a generalized `concat_with_pauses(clips, sr, gaps_ms)` (list of per-gap lengths; the fixed-gap call sites pass a constant list), applies rubberband when needed (non-SBV2), and watermarks (below). New flags: `--tts auto\|sbv2`, `--speed`, `--pitch`, `--sbv2-model`, `--pronunciations`. Batch configs accept `speed`, `pitch`, `sbv2_model`, `pronunciations`, like `cfg_weight` today. Adds `SBV2_DIR`/`SBV2_VENV`/`SBV2_MODELS` constants. |
| `ui/` | `.venv` | `system.py`: an "SBV2" part (venv + repo present). `layout.py`: a "Style-Bert-VITS2 (Japanese, trained voice)" choice in the voice-model radio, offered when a trained model exists; Speed and Pitch sliders in the advanced voice settings; a one-line markup hint under the Script box. `pipeline.py`: `languages("sbv2") == [("Japanese", "ja")]`; `clean_text` already preserves markup (it only collapses whitespace). |
| `tests/test_script_markup.py` (new) | `.venv`, pytest | See §5. |

**Watermark**: Chatterbox embeds a Perth watermark in its own output, but SBV2
and XTTS output carry none, and rubberband processing may degrade Chatterbox's.
The final speech wav of any route that is not unmodified Chatterbox output gets
`perth.PerthImplicitWatermarker` (already installed in `.venv` with Chatterbox)
applied as the last speech step, before Seed-VC/video. Never removed.

**Data flow**: text → `script_markup.parse` → backend (one call per segment) →
splice with per-segment pauses → [rubberband] → [watermark] → [Seed-VC if
`--voice-convert`] → video engine (unchanged).

**Licences** (README licence table): Style-Bert-VITS2 is AGPL-3.0, its Japanese
BERT is CC-BY-SA-4.0; like the other model repos it is cloned at setup and only
run as a subprocess, never imported or redistributed.

`.gitignore` additions: `Style-Bert-VITS2/`, `models/`, `pronunciations.yaml`.

## 4. Error handling

Fail loudly with an actionable message; only `auto` falls back, and it warns.

| Situation | Behaviour |
|---|---|
| `--tts sbv2`, venv/repo missing | Error: run `./setup_sbv2.sh` |
| `--tts sbv2`, no `models/sbv2/NAME/` | Error naming the path and the exact `train_sbv2.py` command |
| `--tts sbv2` with `--lang` ≠ `ja` | Error: JP-Extra is Japanese-only |
| `--tts auto`, `ja`, no model | Chatterbox + warning |
| Unknown emotion tag | Error listing valid tags for that backend (SBV2: names in `styles.yaml`) |
| Emotion/emphasis tags with XTTS | Error: XTTS has no emotion control |
| Malformed markup (unclosed/mismatched tag, nested emotion, bad pause value, accent mark on non-`ja`) | Error with the character offset and a snippet |
| Accent override whose reading isn't valid kana / nucleus beyond mora count | Error naming the word |
| `--speed` / `--pitch` out of range | argparse rejects |
| `train_sbv2.py` interrupted | Rerun resumes at the first incomplete step / last checkpoint |
| Transcription yields < 5 minutes of usable clips | Warning before training (quality will suffer), continue |

## 5. Testing

1. **Unit, test-first**: `tests/test_script_markup.py` covers plain text → one
   segment (the no-splice regression guard); pauses at start/middle/end and
   all unit forms; adjacent and back-to-back tags; emotion + emphasis +
   pronunciation combined; dictionary vs inline precedence and longest-match;
   Japanese text without spaces; accent parsing (`'` positions, `0`, none);
   every error row of §4 that the parser owns, with offsets. No GPU, < 1 s.
   pytest is installed in `.venv` only.
2. **Pipeline checks** (the project's "run a line, inspect the output" convention):
   - Chatterbox, English, one line using every tag: measure silence lengths in
     the wav against the `[pause]` values; `--speed 1.2` shortens duration by
     ~1/1.2; `--pitch 2` raises median F0 by ~2 semitones; watermark detectable
     with Perth's detector.
   - SBV2 after training on `voice_samples/newtest.wav`: a fixed set of ~5
     Japanese lines including accent minimal pairs (雨/飴, 橋/箸, 箸/端) rendered
     with Chatterbox cfg_weight 0.2 and with SBV2. Measured: pause lengths,
     durations, that a `given_tone` override changes the output. **Intonation
     verdict: the user, by ear.**
   - One full video: `--lang ja --tts auto` through LatentSync.
   - App: launch `app.py`, confirm the new choice and sliders, run one short job.

## Open risks

- Training time and VRAM on a 12 GB RTX 5070 Ti laptop GPU are estimated
  (~1–2 h for 100 epochs on ~10 minutes of audio), not measured; measure and
  record in the README.
- SBV2's pinned dependencies may need adjusting for torch 2.8 / sm_120; that
  is handled in `setup_sbv2.sh`, never by patching the clone.
- If `newtest.wav` contains English, those clips must be dropped after
  transcription (JP-Extra trains on Japanese only).
