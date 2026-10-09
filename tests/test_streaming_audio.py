import unittest

from orion.audio.streaming import SpeechPhraseBuffer


class SpeechPhraseBufferTests(unittest.TestCase):
    def test_splits_sentences_and_preserves_final_tail(self):
        buffer = SpeechPhraseBuffer()

        self.assertEqual(buffer.feed("First answer. Second"), ["First answer."])
        self.assertEqual(buffer.feed(" part! Tail", final=True), ["Second part!", "Tail"])

    def test_does_not_split_decimal_or_common_abbreviation(self):
        buffer = SpeechPhraseBuffer()

        self.assertEqual(
            buffer.feed("Use 3.14, e.g. this value. Next."),
            ["Use 3.14, e.g. this value.", "Next."],
        )

    def test_bounds_punctuation_free_phrases_without_losing_unicode(self):
        buffer = SpeechPhraseBuffer(max_chars=8)

        self.assertEqual(buffer.feed("café déjà vu"), ["café"])
        self.assertEqual(buffer.feed("!", final=True), ["déjà vu!"])

    def test_flushes_tail_only_when_final(self):
        buffer = SpeechPhraseBuffer()

        self.assertEqual(buffer.feed("unfinished answer"), [])
        self.assertEqual(buffer.feed("", final=True), ["unfinished answer"])

    def test_punctuation_free_text_flushes_at_twenty_words(self):
        buffer = SpeechPhraseBuffer()
        words = [f"word{index}" for index in range(21)]

        self.assertEqual(buffer.feed(" ".join(words)), [" ".join(words[:20])])
        self.assertEqual(buffer.feed("", final=True), [words[20]])


if __name__ == "__main__":
    unittest.main()