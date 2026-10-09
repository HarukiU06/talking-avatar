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
            cfg = Path(d) / "c.yaml"
            cfg.write_text("photo: p.jpg\nvoice_sample: v.wav\nspeed: 1.1\npitch: -1\n"
                           "sbv2_model: me\npronunciations: p.yaml\nlines:\n  - text: hi\n", encoding="utf-8")
            ma.run_batch(str(cfg))
        self.assertEqual((seen["speed"], seen["pitch"], seen["sbv2_model"], seen["pronunciations"]),
                         (1.1, -1, "me", "p.yaml"))


if __name__ == "__main__":
    unittest.main()
