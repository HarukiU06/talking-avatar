import tempfile
import unittest
from pathlib import Path
from unittest import mock

import make_avatar as ma


class ResolveTts(unittest.TestCase):
    """A fake installed SBV2 with one trained voice, bound to voice.wav by hash."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        for name, value in (("sbv2_installed", lambda: True), ("SBV2_MODELS", str(self.tmp / "models"))):
            patcher = mock.patch.object(ma, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.voice = self.tmp / "voice.wav"
        self.voice.write_bytes(b"me")
        self.model = self.tmp / "models" / "me"
        self.model.mkdir(parents=True)
        for name, body in (("config.json", "{}"), ("me_e100_s1.safetensors", "x"),
                           ("styles.yaml", "neutral: Neutral")):
            (self.model / name).write_text(body, encoding="utf-8")
        (self.model / "source.json").write_text(f'{{"sha256": "{ma.file_sha256(str(self.voice))}"}}',
                                                encoding="utf-8")

    def test_auto_uses_sbv2_for_japanese_with_a_model(self):
        self.assertEqual(ma.resolve_tts("auto", "ja", str(self.voice)), ("sbv2", self.model))

    def test_auto_uses_chatterbox_for_other_languages(self):
        self.assertEqual(ma.resolve_tts("auto", "en", str(self.voice)), ("chatterbox", None))

    def test_auto_falls_back_without_a_model(self):
        other = self.tmp / "other.wav"
        other.write_bytes(b"someone else")
        self.assertEqual(ma.resolve_tts("auto", "ja", str(other)), ("chatterbox", None))

    def test_auto_falls_back_when_sbv2_is_not_installed(self):
        with mock.patch.object(ma, "sbv2_installed", lambda: False):
            self.assertEqual(ma.resolve_tts("auto", "ja", str(self.voice)), ("chatterbox", None))

    def test_auto_with_a_missing_named_model_is_an_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "typo"):
            ma.resolve_tts("auto", "ja", str(self.voice), sbv2_model="typo")

    def test_forced_sbv2_errors(self):
        with self.assertRaisesRegex(ValueError, "Japanese"):
            ma.resolve_tts("sbv2", "en", str(self.voice))
        other = self.tmp / "other.wav"
        other.write_bytes(b"someone else")
        with self.assertRaisesRegex(FileNotFoundError, "train_sbv2.py"):
            ma.resolve_tts("sbv2", "ja", str(other))
        with mock.patch.object(ma, "sbv2_installed", lambda: False):
            with self.assertRaisesRegex(FileNotFoundError, "setup_sbv2.sh"):
                ma.resolve_tts("sbv2", "ja", str(self.voice))

    def test_the_voice_match_recording_also_selects_the_trained_voice(self):
        short = self.tmp / "short.wav"
        short.write_bytes(b"a short clip of me")
        self.assertEqual(ma.resolve_tts("auto", "ja", str(short), vc_target=str(self.voice)),
                         ("sbv2", self.model))
        self.assertEqual(ma.resolve_tts("sbv2", "ja", str(short), vc_target=str(self.voice)),
                         ("sbv2", self.model))

    def test_make_one_watermarks_sbv2_speech_found_through_the_voice_match_recording(self):
        # make_one decides the watermark from its own resolve_tts; if it missed
        # vc_target while synthesize() honoured it, SBV2 speech would go out unmarked.
        short = self.tmp / "short.wav"
        short.write_bytes(b"a short clip of me")
        calls, seen = [], {}
        fakes = (("synthesize", lambda *a, **k: (seen.update(k), calls.append("tts"))),
                 ("release_tts_model", lambda: None),
                 ("apply_watermark", lambda wav: calls.append("watermark")),
                 ("render_video", lambda *a, **k: calls.append("video")))
        for name, fake in fakes:
            patcher = mock.patch.object(ma, name, fake)
            patcher.start()
            self.addCleanup(patcher.stop)
        ma.make_one("face.jpg", str(short), "雨です", "ja", str(self.tmp / "o.mp4"),
                    engine="sadtalker", tts="auto", vc_target=str(self.voice))
        self.assertEqual(calls, ["tts", "watermark", "video"])
        self.assertEqual(seen.get("vc_target"), str(self.voice))

    def test_forced_backends_pass_through(self):
        self.assertEqual(ma.resolve_tts("xtts", "ja", None), ("xtts", None))
        self.assertEqual(ma.resolve_tts("chatterbox", "ja", None), ("chatterbox", None))


class Backends(unittest.TestCase):
    def test_xtts_rejects_emotion_tags_before_running_anything(self):
        with self.assertRaisesRegex(ValueError, "XTTS has no emotion"):
            ma.synthesize_speech_xtts("[happy]Hi[/happy]", "en", "voice.wav", "out.wav")

    def test_chatterbox_rejects_unknown_tags_before_loading_the_model(self):
        with mock.patch.object(ma, "get_tts_model", side_effect=AssertionError("model loaded")):
            with self.assertRaisesRegex(ValueError, "isn't an emotion Chatterbox knows"):
                ma.synthesize_speech("[joyful]Hi[/joyful]", "en", "voice.wav", "out.wav")


class CheckScript(unittest.TestCase):
    """check_script is every refusal a backend makes before loading a model, for the app and batches."""

    def test_each_backend_refuses_what_it_cannot_say(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d)
            (model / "styles.yaml").write_text("neutral: Neutral\n", encoding="utf-8")
            (model / "config.json").write_text('{"data": {"style2id": {"Neutral": 0}}}', encoding="utf-8")
            cases = [("xtts", "en", "*Hi*", "XTTS has no emotion"),
                     ("chatterbox", "en", "[joyful]Hi[/joyful]", "isn't an emotion Chatterbox knows"),
                     ("sbv2", "ja", "[happy]雨[/happy]", "isn't an emotion this Style-Bert-VITS2 voice"),
                     ("sbv2", "en", "Hi", "Japanese only"),
                     ("xtts", "sw", "Hi", "XTTS doesn't support"),
                     ("chatterbox", "en", "[pause 500]", "nothing to say")]
            for backend, lang, text, error in cases:
                with self.subTest(backend=backend, text=text):
                    with self.assertRaisesRegex(ValueError, error):
                        ma.check_script(text, lang, backend, model)
            self.assertEqual(len(ma.check_script("[happy]Hi[/happy] there", "en", "chatterbox")), 2)


class Parser(unittest.TestCase):
    def test_defaults_and_ranges(self):
        parser = ma.build_parser()
        args = parser.parse_args([])
        self.assertEqual((args.tts, args.speed, args.pitch, args.sbv2_model), ("auto", 1.0, 0.0, None))
        self.assertEqual(parser.parse_args(["--pitch", "-2"]).pitch, -2.0)
        for bad in (["--speed", "3"], ["--pitch", "12.5"], ["--tts", "firefly"]):
            with self.subTest(bad=bad), mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    parser.parse_args(bad)


class Pipeline(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = str(Path(tmp.name) / "o.mp4")
        self.calls = []
        record = self.calls.append
        for name, fake in (("synthesize", lambda *a, **k: record("tts")),
                           ("release_tts_model", lambda: None),
                           ("convert_voice", lambda *a, **k: record("seedvc")),
                           ("apply_watermark", lambda wav: record("watermark")),
                           ("render_video", lambda *a, **k: record("video"))):
            patcher = mock.patch.object(ma, name, fake)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_watermark_is_the_last_audio_step(self):
        for kwargs, expected in [(dict(voice_convert=True), ["tts", "seedvc", "watermark", "video"]),
                                 (dict(), ["tts", "video"]),
                                 (dict(speed=1.2), ["tts", "watermark", "video"])]:
            with self.subTest(**kwargs):
                self.calls.clear()
                ma.make_one("face.jpg", "voice.wav", "Hi", "en", self.out,
                            engine="sadtalker", tts="chatterbox", **kwargs)
                self.assertEqual(self.calls, expected)

    def test_batch_reads_new_keys(self):
        seen = {}
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(ma, "make_one", lambda *a, **k: seen.update(k)):
            pron = Path(d) / "p.yaml"
            pron.write_text("en: {}\n", encoding="utf-8")
            cfg = Path(d) / "c.yaml"
            cfg.write_text("photo: p.jpg\nvoice_sample: v.wav\nspeed: 1.1\npitch: -1\n"
                           f"sbv2_model: me\npronunciations: {pron.as_posix()}\nlines:\n  - text: hi\n",
                           encoding="utf-8")
            ma.run_batch(str(cfg))
        self.assertEqual((seen["speed"], seen["pitch"], seen["sbv2_model"], seen["pronunciations"]),
                         (1.1, -1, "me", pron.as_posix()))

    def test_batch_checks_every_line_before_rendering_any(self):
        rendered = []
        bad_configs = {
            "line 2": 'lines:\n  - text: Hello.\n  - text: "[happy]Hi."\n',
            "line 3": "lines:\n  - text: Hello.\n  - text: Hi.\n  - text: Hola.\n    lang: xx\n",
            "speed": "speed: 0\nlines:\n  - text: Hello.\n",
            "pitch": "pitch: 13\nlines:\n  - text: Hello.\n",
        }
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(ma, "make_one", lambda *a, **k: rendered.append(a)):
            for expected, body in bad_configs.items():
                with self.subTest(expected):
                    cfg = Path(d) / "c.yaml"
                    cfg.write_text("photo: p.jpg\nvoice_sample: v.wav\ntts: chatterbox\n" + body,
                                   encoding="utf-8")
                    with self.assertRaisesRegex(SystemExit, expected):
                        ma.run_batch(str(cfg))
        self.assertEqual(rendered, [])


if __name__ == "__main__":
    unittest.main()
