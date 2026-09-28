"""Streaming energy-based voice activity detection."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Iterator

from ..config import Settings


class VoiceActivityDetector:
    def __init__(self, settings: Settings):
        self.sample_rate = settings.audio_sample_rate
        self.block_frames = settings.audio_block_frames
        self.threshold = settings.vad_rms_threshold
        self.silence_chunks = max(
            1,
            math.ceil(
                settings.vad_silence_ms * self.sample_rate
                / (self.block_frames * 1000)
            ),
        )
        self.preroll_chunks = math.ceil(
            settings.vad_preroll_ms * self.sample_rate
            / (self.block_frames * 1000)
        )
        self.max_chunks = math.ceil(
            settings.max_utterance_seconds * self.sample_rate / self.block_frames
        )

    @staticmethod
    def _rms(chunk: bytes) -> float:
        samples = memoryview(chunk).cast("h")
        if not samples:
            return 0.0
        return math.sqrt(sum(sample * sample for sample in samples) / len(samples))

    def utterance_chunks(self, chunks: Iterable[bytes]) -> Iterator[bytes]:
        """Yield one speech utterance and stop at silence or configured maximum."""
        preroll: deque[bytes] = deque(maxlen=self.preroll_chunks)
        started = False
        silence_count = 0
        emitted = 0

        for chunk in chunks:
            if not started:
                preroll.append(chunk)
                if self._rms(chunk) < self.threshold:
                    continue
                started = True
                for buffered_chunk in preroll:
                    if emitted >= self.max_chunks:
                        return
                    yield buffered_chunk
                    emitted += 1
                preroll.clear()
                continue

            if emitted >= self.max_chunks:
                break
            yield chunk
            emitted += 1
            if self._rms(chunk) < self.threshold:
                silence_count += 1
                if silence_count >= self.silence_chunks:
                    break
            else:
                silence_count = 0
