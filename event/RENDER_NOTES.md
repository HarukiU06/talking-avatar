# AI晶子 event video: render notes

Final files (not in git, under `output/event/final/`):
- `event_subtitled_en.mp4`: 1920x1080, 25 fps, H.264 yuv420p + AAC 48 kHz, 110.80 s, English subtitles + lower-third 「司会 AI晶子」 (first 5 s of line 2)
- `event_clean.mp4`: same, no text
- `subtitles_en.srt` / `subtitles_en.ass`: the subtitles on their own

Video and audio are both 110.800 s (2770 frames). It ends with a hard cut after line 18's 「と——」.

## Pipeline
| Step | Script | Notes |
|---|---|---|
| Audio | `event/build_audio.py --route recording --recording voice_samples/Finalaudio.m4a --replace-line 1=output/event/line_takes/line01/chatterbox_cfg0.0_seed1.wav` | Human reading, split into 18 lines (2 blips dropped, line 8 re-joined at a 1.02 s mid-line pause), trimmed at the room's noise floor +10 dB, script pauses as digital silence, -16 LUFS with a soft limiter. Line 1 (English) is a Chatterbox EN take cloned from the reader, level-matched to the recorded lines. |
| Line 1 takes | `event/gen_line.py 1` | 9 takes; the one used is Chatterbox EN, cfg_weight 0, seed 1 |
| Framing | `event/frame_photo.py photos/Finalimage.jpg output/event/frame/final_c74.png --height 900 --crop-bottom 0.74` | Approved still. The render input `render_input.png` is its box (410,180)-(1510,1080) resized to 704x576 |
| Render | `event/render_sections.py S<n>` | InfiniteTalk, `--infinitetalk-accel lightx2v` (4 steps, text CFG 1, audio CFG 2, shift 2), fp8 + low VRAM, 480p bucket = 704x576, streaming, motion_frame 9 |
| Checks | `event/check_section.py S<n>` | contact sheet + metrics in `output/event/check/` |
| Assembly | `event/assemble.py` | 6-frame dissolves centred on section boundaries, clean `full.wav` as the soundtrack |

Scene prompt (all sections): "The person faces the camera as a poised event host: calm friendly expression, steady eye contact, small nods at phrase ends, natural blinking, a slight smile on jokes, mouth relaxed and closed in pauses. Static camera, chest-up, soft even lighting, no hand gestures."

## Sections
| Section | Lines | Audio | Chunks | Seed | Render time | SyncNet conf / offset | Background ΔE (worst) | Face colour ΔE |
|---|---|---|---|---|---|---|---|---|
| S1 | 1–3 | 14.60 s | 5 | 1001 | 53.4 min | 8.13 / -2 | 0.82 | 1.78 |
| S2 | 4–6 | 18.98 s | 7 | 1002 | 77.3 min | 7.70 / -1 | 1.01 | 0.30 |
| S3 | 7–8 | 20.63 s | 8 | 1003 | 90.4 min | 8.07 / -2 | 1.10 | 2.21 |
| S4 | 9–14 | 31.57 s | 11 | 1004 | 128.0 min | 8.10 / -2 | 0.87 | 2.66 |
| S5 | 15–16 | 12.60 s | 5 | 1005 | 55.4 min | 8.09 / -1 | 0.91 | 1.88 |
| S6 | 17–18 | 12.42 s | 5 | 1006 | 54.1 min | 7.75 / -1 | 1.71 | 2.61 |

Render time averaged ~11 min per chunk on an RTX 5070 Ti laptop (12 GB): ~100 s per diffusion step, plus the VAE decode, which ran out of GPU memory on most chunks and fell back to the CPU (~5 min). SyncNet offset is in frames; negative means the mouth leads the sound, 40–80 ms here.

Known looks, accepted as rendered: S2 has a broad toothy smile at about 0:15–0:17 (line 6), and S3 the same at the start (line 7).

## Redoing a section
1. Delete `output/event/video/S<n>.mp4`, or keep it and pick a new seed: `event/render_sections.py S<n> --seed S<n>=2000<n>`. A different seed re-rolls expression and motion; the same seed reproduces the section.
2. If the section's audio changed, the runner notices on its own: each video stores the hash of the WAV it was rendered from, and a stale one is set aside as `S<n>.stale-*.mp4` and re-rendered.
3. `event/check_section.py S<n>`, then `event/assemble.py`.
