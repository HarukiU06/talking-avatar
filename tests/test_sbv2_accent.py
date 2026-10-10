import unittest

from sbv2_synth import apply_accent_overrides, segment_overrides


def node(string, acc, mora, chain):
    return {"string": string, "acc": acc, "mora_size": mora, "chain_flag": chain}


def ame_desu():
    # pyopenjtalk on "あめです。": one accent phrase, nucleus on デ (mora 3).
    return [node("あめ", 3, 2, -1), node("です", 1, 2, 1), node("。", 0, 0, 0)]


class SegmentOverrides(unittest.TestCase):
    def test_spans_in_normalized_text(self):
        pieces = [["今日は", None], ["あめ", 1], ["です", None]]
        self.assertEqual(segment_overrides(pieces, lambda t: t), [(3, 5, 1, "あめ")])

    def test_no_overrides(self):
        self.assertEqual(segment_overrides([["こんにちは", None]], lambda t: t), [])


class ApplyAccentOverrides(unittest.TestCase):
    def test_accented_word_keeps_attached_particles_low(self):
        nodes = ame_desu()
        apply_accent_overrides(nodes, [(0, 2, 1, "あめ")])
        self.assertEqual(nodes[0]["acc"], 1)  # ア'メデス: high, then low through です

    def test_flat_word_keeps_a_nucleus_that_falls_on_an_attached_auxiliary(self):
        nodes = ame_desu()
        apply_accent_overrides(nodes, [(0, 2, 0, "あめ")])
        self.assertEqual(nodes[0]["acc"], 3)  # アメデ'ス, as for 飴です

    def test_flat_word_clears_a_nucleus_inside_the_word(self):
        nodes = [node("あめ", 1, 2, -1), node("が", 0, 1, 1)]
        apply_accent_overrides(nodes, [(0, 2, 0, "あめ")])
        self.assertEqual(nodes[0]["acc"], 0)

    def test_word_that_heads_a_later_phrase(self):
        nodes = [node("今日", 1, 2, -1), node("は", 0, 1, 1), node("あめ", 0, 2, 0), node("が", 0, 1, 1)]
        apply_accent_overrides(nodes, [(3, 5, 1, "あめ")])
        self.assertEqual([n["acc"] for n in nodes], [1, 0, 1, 0])

    def test_word_chained_after_a_prefix(self):
        nodes = [node("お", 0, 1, -1), node("あめ", 0, 2, 1)]
        apply_accent_overrides(nodes, [(1, 3, 1, "あめ")])
        self.assertEqual(nodes[0]["acc"], 2)  # the phrase's nucleus is the word's 1st mora = phrase mora 2

    def test_word_split_across_nodes_is_joined_into_one_phrase(self):
        nodes = [node("あ", 1, 1, -1), node("め", 1, 1, 0), node("です", 1, 2, 1)]
        apply_accent_overrides(nodes, [(0, 2, 2, "あめ")])
        self.assertEqual((nodes[0]["acc"], nodes[1]["chain_flag"]), (2, 1))

    def test_nucleus_past_the_word_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "あめ"):
            apply_accent_overrides(ame_desu(), [(0, 2, 3, "あめ")])

    def test_word_that_does_not_line_up_with_pyopenjtalk_words_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "あ"):
            apply_accent_overrides(ame_desu(), [(0, 1, 1, "あ")])


if __name__ == "__main__":
    unittest.main()
