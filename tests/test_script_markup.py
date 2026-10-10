import tempfile
import unittest
from pathlib import Path

from script_markup import MarkupError, Piece, count_morae, load_pronunciations, parse, split_at_sentences


def texts(segments):
    return [s.text for s in segments]


class PlainText(unittest.TestCase):
    def test_plain_text_is_one_segment(self):
        segs = parse("Hello there. How are you? Fine!", "en")
        self.assertEqual(texts(segs), ["Hello there. How are you? Fine!"])
        self.assertEqual((segs[0].pause_after_ms, segs[0].emotion, segs[0].emphasis), (0, None, False))

    def test_japanese_plain_text_is_one_segment(self):
        self.assertEqual(texts(parse("こんにちは。今日はいい天気ですね。", "ja")),
                         ["こんにちは。今日はいい天気ですね。"])

    def test_escapes(self):
        self.assertEqual(texts(parse(r"5 \* 3 \[sic\] \{x\}", "en")), ["5 * 3 [sic] {x}"])


class Pauses(unittest.TestCase):
    def test_pause_forms(self):
        for tag, ms in [("[pause 800]", 800), ("[pause 800ms]", 800), ("[pause 1.5s]", 1500),
                        ("[ pause 250 ]", 250), ("[PAUSE 300]", 300)]:
            with self.subTest(tag=tag):
                segs = parse(f"One.{tag}Two.", "en")
                self.assertEqual(texts(segs), ["One.", "Two."])
                self.assertEqual([s.pause_after_ms for s in segs], [ms, 0])

    def test_leading_and_trailing_pause(self):
        segs = parse("[pause 500]Hi.[pause 200]", "en")
        self.assertEqual([(s.text, s.pause_after_ms) for s in segs], [("", 500), ("Hi.", 200)])

    def test_back_to_back_pauses_add_up(self):
        self.assertEqual(parse("A.[pause 300][pause 200]B.", "en")[0].pause_after_ms, 500)


class EmotionAndEmphasis(unittest.TestCase):
    def test_emotion_span(self):
        segs = parse("Hi. [happy]Great news![/happy] Bye.", "en")
        self.assertEqual([(s.text, s.emotion) for s in segs],
                         [("Hi.", None), ("Great news!", "happy"), ("Bye.", None)])

    def test_emotion_intensity(self):
        self.assertEqual(parse("[sad 0.5]Oh.[/sad]", "en")[0].intensity, 0.5)

    def test_pause_inside_emotion_keeps_emotion_on_both_sides(self):
        segs = parse("[happy]A.[pause 300]B.[/happy]", "en")
        self.assertEqual([(s.text, s.emotion, s.pause_after_ms) for s in segs],
                         [("A.", "happy", 300), ("B.", "happy", 0)])

    def test_emphasis_inside_emotion(self):
        segs = parse("[happy]I am *very* glad[/happy]", "en")
        self.assertEqual([(s.text, s.emotion, s.emphasis) for s in segs],
                         [("I am", "happy", False), ("very", "happy", True), ("glad", "happy", False)])

    def test_punctuation_left_after_a_tag_joins_the_previous_segment(self):
        # Chatterbox crashes on a segment that is only "." (IndexError in its alignment analyzer).
        segs = parse("Hello, *really*.", "en")
        self.assertEqual([(s.text, s.emphasis) for s in segs], [("Hello,", False), ("really.", True)])
        self.assertEqual(texts(parse("[happy]すごい[/happy]！", "ja")), ["すごい！"])

    def test_leading_punctuation_only_segment_is_dropped_but_its_pause_kept(self):
        segs = parse("。[pause 300]こんにちは", "ja")
        self.assertEqual([(s.text, s.pause_after_ms) for s in segs], [("", 300), ("こんにちは", 0)])

    def test_adjacent_tags_make_no_empty_segments(self):
        self.assertEqual(texts(parse("[happy]A[/happy] [sad]B[/sad]", "en")), ["A", "B"])


class Pronunciation(unittest.TestCase):
    def test_inline_stays_in_one_segment(self):
        self.assertEqual(texts(parse("Hi {Ueno|oo-EH-no}!", "en")), ["Hi oo-EH-no!"])

    def test_trailing_zero_is_text_outside_japanese(self):
        self.assertEqual(texts(parse("{v10|version 10}", "en")), ["version 10"])

    def test_japanese_accent(self):
        for reading, kana, accent in [("あ'め", "あめ", 1), ("あめ0", "あめ", 0), ("はし'", "はし", 2),
                                      ("あめ", "あめ", None), ("きょ'う", "きょう", 1)]:
            with self.subTest(reading=reading):
                seg, = parse(f"今日は{{雨|{reading}}}です", "ja")
                self.assertEqual(seg.pieces[1], Piece(kana, accent))
                self.assertEqual(seg.text, f"今日は{kana}です")

    def test_dictionary_applies_longest_match_first(self):
        seg, = parse("雨宮さんと雨", "ja", {"雨": "あ'め", "雨宮": "あまみや"})
        self.assertEqual([p.text for p in seg.pieces], ["あまみや", "さんと", "あめ"])
        self.assertEqual(seg.pieces[2].accent, 1)

    def test_dictionary_leaves_kanji_compounds_alone(self):
        # 梅雨 (つゆ), 雨天 (うてん), 雨戸 (あまど) are other words, not 雨 (あめ) + a kanji.
        pron = {"雨": "あ'め"}
        for text in ("梅雨です", "雨天です", "雨戸です"):
            with self.subTest(text=text):
                self.assertEqual(texts(parse(text, "ja", pron)), [text])
        seg, = parse("雨が降る", "ja", pron)
        self.assertEqual(seg.pieces[0], Piece("あめ", 1))

    def test_dictionary_key_ending_in_kana_may_touch_kanji(self):
        seg, = parse("あめ玉", "ja", {"あめ": "あめ0"})
        self.assertEqual(seg.pieces[0], Piece("あめ", 0))

    def test_inline_beats_dictionary(self):
        seg, = parse("{雨|あめ0}", "ja", {"雨": "あ'め"})
        self.assertEqual(seg.pieces, [Piece("あめ", 0)])

    def test_dictionary_respects_word_boundaries_outside_japanese(self):
        seg, = parse("Ueno and Uenomachi", "en", {"Ueno": "oo-EH-no"})
        self.assertEqual(seg.text, "oo-EH-no and Uenomachi")


class Errors(unittest.TestCase):
    def test_errors(self):
        for text, lang, message in [
            ("Hello [happy]there", "en", "never closed"),
            ("[happy]a[/sad]", "en", "doesn't close"),
            ("[happy]a[sad]b[/sad][/happy]", "en", "can't nest"),
            ("a [Applause] b", "en", "unknown tag"),
            ("a [applause] b", "en", "never closed"),
            ("[pause soon]", "en", "unknown tag"),
            ("A[pause 11s]", "en", "10000"),
            ("[happy 4]a[/happy]", "en", "between 0 and 3"),
            ("5 * 3 = 15", "en", "literal asterisk"),
            ("5 * 3 * 2 = 30", "en", "literal asterisk"),
            ("Rated 4* and 5* hotels", "en", "literal asterisk"),
            ("**bold** text", "en", "literal asterisk"),
            ("*word *", "en", "literal asterisk"),
            ("[1] and [2]", "en", "unknown tag"),
            ("a [pause 3", "en", "no closing"),
            ("{雨}", "ja", "word|reading"),
            ("{雨|}", "ja", "word|reading"),
            ("{Ueno|oo'eh}", "en", "Japanese"),
            ("{雨|a'me}", "ja", "kana"),
            ("{雨|'あめ}", "ja", "after a mora"),
            ("{雨|あ'め'}", "ja", "one accent"),
        ]:
            with self.subTest(text=text):
                with self.assertRaises(MarkupError) as err:
                    parse(text, lang)
                self.assertIn(message, str(err.exception))

    def test_bracket_errors_say_how_to_write_a_literal_bracket(self):
        for text in ("a [applause] b", "[1] and [2]"):
            with self.subTest(text=text):
                with self.assertRaises(MarkupError) as err:
                    parse(text, "en")
                self.assertIn("\[", str(err.exception))

    def test_error_reports_offset(self):
        with self.assertRaises(MarkupError) as err:
            parse("Hello [pause soon] world", "en")
        self.assertEqual(err.exception.offset, 6)
        self.assertIn("character 6", str(err.exception))


class SentenceSplitting(unittest.TestCase):
    def test_matches_old_pause_ms(self):
        segs = split_at_sentences(parse("One. Two! Three?", "en"), 450)
        self.assertEqual([(s.text, s.pause_after_ms) for s in segs],
                         [("One.", 450), ("Two!", 450), ("Three?", 0)])

    def test_keeps_explicit_pause_and_emotion(self):
        segs = split_at_sentences(parse("[sad]One. Two.[/sad][pause 900]Three.", "en"), 450)
        self.assertEqual([(s.text, s.emotion, s.pause_after_ms) for s in segs],
                         [("One.", "sad", 450), ("Two.", "sad", 900), ("Three.", None, 0)])


class Morae(unittest.TestCase):
    def test_count_morae(self):
        for kana, n in [("きょう", 2), ("がっこう", 4), ("ラーメン", 4), ("あめ", 2)]:
            with self.subTest(kana=kana):
                self.assertEqual(count_morae(kana), n)


class PronunciationFile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "p.yaml"

    def test_loads(self):
        self.path.write_text("ja:\n  雨: \"あ'め\"\nen:\n  Ueno: oo-EH-no\n", encoding="utf-8")
        self.assertEqual(load_pronunciations(str(self.path)),
                         {"ja": {"雨": "あ'め"}, "en": {"Ueno": "oo-EH-no"}})

    def test_rejects_wrong_shape(self):
        self.path.write_text("- just a list\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "language"):
            load_pronunciations(str(self.path))


if __name__ == "__main__":
    unittest.main()
