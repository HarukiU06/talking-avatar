import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import make_avatar as ma
from script_markup import Piece, Segment

ONES_A = np.ones(10, np.float32)
ONES_B = np.ones(5, np.float32)


def seg(emotion=None, intensity=1.0, emphasis=False):
    return Segment([Piece("x")], emotion, intensity, emphasis)


class ConcatWithPauses(unittest.TestCase):
    def test_single_gap_matches_old_behaviour(self):
        out = ma.concat_with_pauses([ONES_A, ONES_B], 1000, 3)
        self.assertEqual(len(out), 18)
        self.assertEqual(out[10:13].sum(), 0)

    def test_per_clip_gaps_with_trailing_silence(self):
        out = ma.concat_with_pauses([ONES_A, ONES_B], 1000, [2, 4])
        self.assertEqual(len(out), 21)
        self.assertEqual(out[-4:].sum(), 0)

    def test_leading_pause_from_an_empty_clip(self):
        out = ma.concat_with_pauses([np.zeros(0, np.float32), ONES_A], 1000, [5, 0])
        self.assertEqual(len(out), 15)
        self.assertEqual(out[:5].sum(), 0)

    def test_rejects_mismatched_gaps(self):
        with self.assertRaises(ValueError):
            ma.concat_with_pauses([ONES_A, ONES_B], 1000, [1])


class ChatterboxParams(unittest.TestCase):
    def test_plain_segment_keeps_run_settings(self):
        self.assertEqual(ma.chatterbox_params(seg(), 0.7, 0.3), (0.7, 0.3))

    def test_happy_full_and_half(self):
        ex, cfg = ma.chatterbox_params(seg("happy"), 0.5, 0.5)
        self.assertEqual((round(ex, 3), round(cfg, 3)), (0.8, 0.4))
        ex, _ = ma.chatterbox_params(seg("happy", 0.5), 0.5, 0.5)
        self.assertEqual(round(ex, 3), 0.65)

    def test_keeps_a_low_cfg_low(self):
        _, cfg = ma.chatterbox_params(seg("happy"), 0.5, 0.2)
        self.assertEqual(round(cfg, 3), 0.1)

    def test_emphasis_and_clamping(self):
        ex, _ = ma.chatterbox_params(seg(emphasis=True), 0.5, 0.5)
        self.assertEqual(round(ex, 3), 0.8)
        # 0.5 + (1.1 - 0.5) * 3 + 0.3 = 2.6 -> 2.0;  0.1 + (0.3 - 0.5) * 3 = -0.5 -> 0.0
        ex, cfg = ma.chatterbox_params(seg("excited", 3.0, emphasis=True), 0.5, 0.1)
        self.assertEqual((ex, cfg), (2.0, 0.0))

    def test_check_emotions_lists_valid_tags(self):
        with self.assertRaisesRegex(ValueError, "calm, happy"):
            ma.check_emotions([seg("joyful")], {"happy", "calm"}, "Chatterbox")
        ma.check_emotions([seg("happy"), seg()], {"happy"}, "Chatterbox")


class ScriptSegments(unittest.TestCase):
    def test_rejects_nothing_to_say(self):
        with self.assertRaisesRegex(ValueError, "nothing to say"):
            ma.script_segments("[pause 500]", "en")

    def test_pause_ms_splits_sentences(self):
        segs = ma.script_segments("One. Two.", "en", pause_ms=300)
        self.assertEqual([(s.text, s.pause_after_ms) for s in segs], [("One.", 300), ("Two.", 0)])

    def test_reads_pronunciation_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "p.yaml"
            f.write_text("en:\n  Ueno: oo-EH-no\n", encoding="utf-8")
            self.assertEqual(ma.script_segments("Hi Ueno", "en", pronunciations=str(f))[0].text,
                             "Hi oo-EH-no")


class SpeedPitchAndWatermark(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.wav = str(Path(tmp.name) / "a.wav")
        t = np.arange(48000) / 24000
        sf.write(self.wav, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 24000)

    def test_apply_speed_pitch_changes_duration(self):
        ma.apply_speed_pitch(self.wav, speed=1.25, pitch=2.0)
        self.assertLess(abs(sf.info(self.wav).duration - 1.6), 0.05)

    def test_apply_speed_pitch_noop_leaves_file_untouched(self):
        before = os.stat(self.wav).st_mtime_ns
        ma.apply_speed_pitch(self.wav)
        self.assertEqual(os.stat(self.wav).st_mtime_ns, before)

    def test_needs_watermark(self):
        for backend, speed, pitch, vc, expected in [
                ("chatterbox", 1.0, 0.0, False, False), ("chatterbox", 1.1, 0.0, False, True),
                ("chatterbox", 1.0, -1.0, False, True), ("chatterbox", 1.0, 0.0, True, True),
                ("sbv2", 1.0, 0.0, False, True), ("xtts", 1.0, 0.0, False, True)]:
            with self.subTest(backend=backend, speed=speed, pitch=pitch, vc=vc):
                self.assertIs(ma.needs_watermark(backend, speed, pitch, vc), expected)


if __name__ == "__main__":
    unittest.main()
