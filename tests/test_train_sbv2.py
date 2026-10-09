import json
import tempfile
import unittest
from pathlib import Path

import make_avatar as ma
import train_sbv2


class TrainSbv2(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def voice(self, name: str, data: bytes) -> str:
        path = self.tmp / name
        path.write_bytes(data)
        return str(path)

    def test_records_source_on_first_run(self):
        voice = self.voice("me.wav", b"take one")
        train_sbv2.record_source(self.tmp / "model", voice)
        source = json.loads((self.tmp / "model" / "source.json").read_text(encoding="utf-8"))
        self.assertEqual(source["sha256"], ma.file_sha256(voice))

    def test_rerun_with_same_recording_is_fine(self):
        voice = self.voice("me.wav", b"take one")
        train_sbv2.record_source(self.tmp / "model", voice)
        train_sbv2.record_source(self.tmp / "model", voice)

    def test_refuses_a_name_trained_from_another_recording(self):
        train_sbv2.record_source(self.tmp / "model", self.voice("a.wav", b"take one"))
        with self.assertRaisesRegex(SystemExit, "--name"):
            train_sbv2.record_source(self.tmp / "model", self.voice("b.wav", b"a completely different take"))

    def test_drop_non_japanese(self):
        esd = self.tmp / "esd.list"
        esd.write_text("a.wav|me|JP|こんにちは。\n"
                       "b.wav|me|JP|Hello there.\n"
                       "c.wav|me|JP|AIの話です。\n", encoding="utf-8")
        self.assertEqual(train_sbv2.drop_non_japanese(esd), ["b.wav"])
        self.assertEqual(esd.read_text(encoding="utf-8").splitlines(),
                         ["a.wav|me|JP|こんにちは。", "c.wav|me|JP|AIの話です。"])


if __name__ == "__main__":
    unittest.main()
