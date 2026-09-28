"""Streaming VOSK transcription over bounded microphone chunks."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

from ..config import Settings
from .capture import AudioChunkSource, AudioGate, SoundDeviceChunkSource
from .vad import VoiceActivityDetector

logger = logging.getLogger(__name__)


class VoskTranscriber:
    def __init__(
        self,
        settings: Settings,
        *,
        source: AudioChunkSource | None = None,
        gate: AudioGate | None = None,
        model_factory: Callable | None = None,
        recognizer_factory: Callable | None = None,
    ):
        self.settings = settings
        self.gate = gate or AudioGate()
        self.source = source or SoundDeviceChunkSource(settings, gate=self.gate)
        self.vad = VoiceActivityDetector(settings)
        self._model_factory = model_factory
        self._recognizer_factory = recognizer_factory
        self._model = None

    def _load_model(self):
        if self._model is None:
            if not Path(self.settings.stt_model_path).is_dir():
                raise FileNotFoundError(
                    f"VOSK model directory not found: {self.settings.stt_model_path}"
                )
            model_factory = self._model_factory
            if model_factory is None:
                try:
                    from vosk import Model
                except ImportError as error:
                    raise RuntimeError("vosk is required for local speech recognition") from error
                model_factory = Model
            self._model = model_factory(str(self.settings.stt_model_path))
        return self._model

    def transcribe_chunks(self, chunks: Iterator[bytes]) -> str:
        recognizer_factory = self._recognizer_factory
        if recognizer_factory is None:
            try:
                from vosk import KaldiRecognizer
            except ImportError as error:
                raise RuntimeError("vosk is required for local speech recognition") from error
            recognizer_factory = KaldiRecognizer

        recognizer = recognizer_factory(self._load_model(), self.settings.audio_sample_rate)
        try:
            for chunk in self.vad.utterance_chunks(chunks):
                recognizer.AcceptWaveform(chunk)
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()
        result = json.loads(recognizer.FinalResult())
        return str(result.get("text", "")).strip()


class PushToTalkInput(VoskTranscriber):
    def __init__(self, settings: Settings, button, **kwargs):
        super().__init__(settings, **kwargs)
        self.button = button

    def next_utterance(self) -> str:
        self.gate.wait_until_open()
        self.button.wait_for_press()
        self.gate.wait_until_open()
        logger.info("Push-to-talk activated")
        return self.transcribe_chunks(self.source.chunks())


class WakeWordInput(VoskTranscriber):
    def __init__(self, settings: Settings, detector, **kwargs):
        super().__init__(settings, **kwargs)
        self.detector = detector

    def next_utterance(self) -> str:
        self.gate.wait_until_open()
        chunks = self.source.chunks()
        try:
            for chunk in chunks:
                if self.detector.detected(chunk):
                    logger.info("Wake word detected")
                    return self.transcribe_chunks(chunks)
            return ""
        finally:
            chunks.close()
