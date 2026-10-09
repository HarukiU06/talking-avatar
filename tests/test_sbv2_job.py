import json
import tempfile
import unittest
from pathlib import Path

import make_avatar as ma
from script_markup import Piece, Segment


class Sbv2Styles(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def model_dir(self, styles_yaml: str, style2id: dict) -> Path:
        d = self.tmp / "me"
        d.mkdir()
        (d / "styles.yaml").write_text(styles_yaml, encoding="utf-8")
        (d / "config.json").write_text(json.dumps({"data": {"style2id": style2id}}), encoding="utf-8")
        return d

    def test_load_styles(self):
        d = self.model_dir("neutral: Neutral\nhappy: style_1\n", {"Neutral": 0, "style_1": 1})
        self.assertEqual(ma.load_sbv2_styles(d), {"neutral": "Neutral", "happy": "style_1"})

    def test_load_styles_adds_neutral(self):
        d = self.model_dir("happy: style_1\n", {"Neutral": 0, "style_1": 1})
        self.assertEqual(ma.load_sbv2_styles(d)["neutral"], "Neutral")

    def test_load_styles_rejects_unknown_style_name(self):
        d = self.model_dir("happy: Joy\n", {"Neutral": 0, "style_1": 1})
        with self.assertRaisesRegex(ValueError, "Joy"):
            ma.load_sbv2_styles(d)

    def test_sbv2_is_japanese_only(self):
        with self.assertRaisesRegex(ValueError, "Japanese"):
            ma.synthesize_speech_sbv2("Hello", "en", self.tmp, str(self.tmp / "o.wav"))


class Sbv2SegmentJob(unittest.TestCase):
    def test_plain(self):
        job = ma.sbv2_segment_job(Segment([Piece("こんにちは")]), {"neutral": "Neutral"})
        self.assertEqual(job, {"pieces": [["こんにちは", None]], "style": "Neutral", "style_weight": 1.0,
                               "intonation_scale": 1.0})

    def test_emotion_emphasis_and_accent(self):
        segment = Segment([Piece("今日は"), Piece("あめ", 1)], "happy", 1.5, True)
        job = ma.sbv2_segment_job(segment, {"neutral": "Neutral", "happy": "style_1"})
        self.assertEqual(job, {"pieces": [["今日は", None], ["あめ", 1]], "style": "style_1",
                               "style_weight": ma.SBV2_STYLE_WEIGHT * 1.5,
                               "intonation_scale": ma.SBV2_EMPHASIS_INTONATION})


if __name__ == "__main__":
    unittest.main()
