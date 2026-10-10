import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import make_avatar as ma


def make_model(root: Path, name: str, sha: str = None, ready: bool = True) -> Path:
    d = root / name
    d.mkdir(parents=True)
    if ready:
        (d / "config.json").write_text("{}", encoding="utf-8")
        (d / f"{name}_e100_s1000.safetensors").write_bytes(b"x")
        (d / "styles.yaml").write_text("neutral: Neutral\n", encoding="utf-8")
    if sha:
        (d / "source.json").write_text(json.dumps({"sha256": sha}), encoding="utf-8")
    return d


def make_voice(path: Path, data: bytes = b"RIFF-me-speaking") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class FindSbv2Model(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.models = self.tmp / "models"
        patcher = mock.patch.object(ma, "SBV2_MODELS", str(self.models))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_sha256_changes_when_file_rewritten(self):
        f = make_voice(self.tmp / "v.wav", b"one")
        first = ma.file_sha256(str(f))
        f.write_bytes(b"a different take")
        self.assertNotEqual(ma.file_sha256(str(f)), first)

    def test_model_found_by_content_for_a_renamed_copy(self):
        voice = make_voice(self.tmp / "voice_samples" / "newtest.wav")
        make_model(self.models, "newtest", sha=ma.file_sha256(str(voice)))
        copy = make_voice(self.tmp / "avatars" / "me" / "voice.wav", voice.read_bytes())
        self.assertEqual(ma.find_sbv2_model(str(copy)), self.models / "newtest")

    def test_model_found_by_the_voice_match_recording(self):
        # The app's natural setup: a short clip as the voice, the long training
        # recording as the voice-match target (avatars/<x>/voice_long.wav).
        long = make_voice(self.tmp / "avatars" / "me" / "voice_long.wav", b"ten minutes of me")
        make_model(self.models, "me", sha=ma.file_sha256(str(long)))
        short = make_voice(self.tmp / "avatars" / "me" / "voice.wav", b"thirty seconds of me")
        self.assertEqual(ma.find_sbv2_model(str(short), vc_target=str(long)), self.models / "me")
        self.assertIsNone(ma.find_sbv2_model(str(short)))

    def test_voice_hash_wins_over_the_voice_match_recording(self):
        short = make_voice(self.tmp / "short.wav", b"take one")
        long = make_voice(self.tmp / "long.wav", b"take two")
        make_model(self.models, "a_long", sha=ma.file_sha256(str(long)))
        make_model(self.models, "b_short", sha=ma.file_sha256(str(short)))
        self.assertEqual(ma.find_sbv2_model(str(short), vc_target=str(long)), self.models / "b_short")

    def test_stem_fallback_without_source_json(self):
        make_model(self.models, "newtest")
        voice = make_voice(self.tmp / "newtest.wav")
        self.assertEqual(ma.find_sbv2_model(str(voice)), self.models / "newtest")

    def test_other_voice_does_not_match(self):
        make_model(self.models, "newtest", sha="0" * 64)
        self.assertIsNone(ma.find_sbv2_model(str(make_voice(self.tmp / "someone_else.wav"))))

    def test_same_stem_but_other_recording_does_not_match(self):
        make_model(self.models, "newtest", sha="0" * 64)
        self.assertIsNone(ma.find_sbv2_model(str(make_voice(self.tmp / "newtest.wav"))))

    def test_unfinished_model_is_ignored(self):
        voice = make_voice(self.tmp / "newtest.wav")
        make_model(self.models, "newtest", sha=ma.file_sha256(str(voice)), ready=False)
        self.assertIsNone(ma.find_sbv2_model(str(voice)))

    def test_explicit_name(self):
        make_model(self.models, "other")
        self.assertEqual(ma.find_sbv2_model(None, "other"), self.models / "other")
        self.assertIsNone(ma.find_sbv2_model(None, "missing"))

    def test_no_models_folder(self):
        self.assertIsNone(ma.find_sbv2_model(str(make_voice(self.tmp / "v.wav"))))


if __name__ == "__main__":
    unittest.main()
