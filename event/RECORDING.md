# Recording the AI晶子 script

About 2 minutes of speech. A phone is fine.

## Setup
- Quiet room with soft furnishings (a bedroom or a small meeting room with the door shut beats an empty office). Turn off fans and air conditioning.
- Phone 20–30 cm from the mouth, slightly to the side so breath doesn't hit it. Same position for the whole take.
- Do one 10-second test, listen back with headphones, and check that you don't hear hum or echo.
- Save as WAV if the app allows; m4a or mp3 also work.

## How to read
- Calm, deadpan MC delivery: steady pace, no acting on the jokes. The pauses do the comedy.
- **Leave a clear silence of about 2 seconds between lines.** The pauses in the final video are set later, so they don't need to be accurate, but the lines must not run into each other. Within a line, read normally (commas, breaths are fine).
- If you stumble, stop, wait 2 seconds and read **that whole line again**, then delete the bad attempt afterwards, or record one file per line instead (see below).
- Line 18 stops dead after 「と」: say 「…AI晶子と」 and stop, as if you were cut off. Don't let the pitch fall as if the sentence ended.

## Two ways to deliver
1. **One continuous take**, e.g. `recordings/take.wav`, all 18 lines with ~2 s gaps. The script checks that it finds exactly 18 lines and stops with a message otherwise.
2. **One file per line** in a folder, named `line_01.wav` … `line_18.wav`. Safest if re-takes are likely.

## Readings to watch
| Line | Text | Reading |
|---|---|---|
| 3 | 少々消費電力 | しょうしょう、しょうひでんりょく |
| 5, 14, 16 | 私 | わたくし |
| 8, 16, 18 | AI | エーアイ |
| 8 | 思われた方 | おもわれたかた |
| 10 | 第68期 / 株式会社 | だいろくじゅうはっき / かぶしきがいしゃ |
| 11 | 第2部 | だいにぶ |
| 13 | 17時30分 | じゅうななじさんじゅっぷん |
| 14 | 私の分 | わたくしのぶん |
| 17 | 発表中 / 入退室 | はっぴょうちゅう / にゅうたいしつ |
| 18 | AI晶子 | エーアイあきこ |

The full script with every reading is in `event/readings_ja.txt` (hiragana) and `event/script.txt` (original).

## After recording
```
.venv/Scripts/python.exe event/build_audio.py --route recording --recording recordings/take.wav
```
Add `--denoise-rec 0.5` if there's audible room noise, and `--seedvc-to voice_samples/newtest.wav` to move the timbre toward the newtest.wav voice (the reader's own intonation is kept).
