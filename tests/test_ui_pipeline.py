import unittest
from pathlib import Path
from unittest import mock

import make_avatar as ma
from ui import pipeline


class UiPipeline(unittest.TestCase):
    def test_languages_per_voice_model(self):
        self.assertEqual(pipeline.languages("sbv2"), [("Japanese", "ja")])
        self.assertEqual(pipeline.languages("auto"), pipeline.languages("chatterbox"))

    def test_problems_flags_markup_before_queueing(self):
        found = pipeline.problems({"text": "5 * 3 = 15", "lang": "en", "tts": "auto"}, {})
        self.assertTrue(any(p.startswith("Script markup:") for p in found), found)

    def test_describe_and_parts_follow_the_resolved_backend(self):
        options = {"tts": "auto", "lang": "ja", "engine": "latentsync", "voice": "v.wav"}
        with mock.patch.object(ma, "resolve_tts", lambda *a, **k: ("sbv2", Path("m"))):
            self.assertIn("Style-Bert-VITS2", pipeline.describe(options))
            self.assertIn("sbv2", pipeline.needed_parts(options))

    def test_to_args_passes_speed_and_pitch_and_omits_defaults(self):
        args = pipeline.to_args({"engine": "latentsync", "tts": "auto", "speed": 1.2, "pitch": -2})
        self.assertIn("--speed=1.2", args)
        self.assertIn("--pitch=-2.0", args)
        self.assertFalse(any(a.startswith("--tts") for a in args))


if __name__ == "__main__":
    unittest.main()
