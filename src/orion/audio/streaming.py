"""Bounded text framing for streamed speech output."""

from __future__ import annotations


class SpeechPhraseBuffer:
    _ABBREVIATIONS = (
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.",
        "e.g.", "i.e.", "vs.",
    )

    def __init__(self, max_chars: int = 240, max_words: int = 20):
        if max_chars < 1 or max_words < 8:
            raise ValueError("phrase limits must be positive and max_words at least 8")
        self.max_chars = max_chars
        self.max_words = max_words
        self._buffer = ""

    def feed(self, text: str, *, final: bool = False) -> list[str]:
        self._buffer += text
        phrases: list[str] = []
        while self._buffer:
            boundary = self._sentence_boundary()
            if boundary is not None and self._fits(self._buffer[:boundary]):
                phrase = self._buffer[:boundary].strip()
                self._buffer = self._buffer[boundary:].lstrip()
                if phrase:
                    phrases.append(phrase)
                continue
            if not self._fits(self._buffer):
                split = self._bounded_split()
                phrase = self._buffer[:split].strip()
                self._buffer = self._buffer[split:].lstrip()
                if phrase:
                    phrases.append(phrase)
                continue
            if final:
                phrase = self._buffer.strip()
                self._buffer = ""
                if phrase:
                    phrases.append(phrase)
            break
        return phrases

    def _fits(self, text: str) -> bool:
        return len(text) <= self.max_chars and len(text.split()) <= self.max_words

    def _bounded_split(self) -> int:
        character_limit = min(len(self._buffer), self.max_chars)
        word_end = 0
        words = 0
        for index, character in enumerate(self._buffer[:character_limit]):
            if character.isspace():
                continue
            if index == 0 or self._buffer[index - 1].isspace():
                words += 1
                if words > self.max_words:
                    return max(1, word_end)
            word_end = index + 1
        split = self._buffer.rfind(" ", 0, character_limit + 1)
        if split <= 0:
            return max(1, character_limit)
        return split

    def _sentence_boundary(self) -> int | None:
        for index, character in enumerate(self._buffer):
            if character not in ".!?":
                continue
            end = index + 1
            while end < len(self._buffer) and self._buffer[end] in "!?\"'”’)]}":
                end += 1
            if end < len(self._buffer) and not self._buffer[end].isspace():
                continue
            if character == "." and self._is_nonterminal_period(index):
                continue
            return end
        return None

    def _is_nonterminal_period(self, index: int) -> bool:
        before = self._buffer[:index + 1]
        after = self._buffer[index + 1:]
        if before[-2:-1].isdigit() and after[:1].isdigit():
            return True
        return any(before.endswith(abbreviation) for abbreviation in self._ABBREVIATIONS)