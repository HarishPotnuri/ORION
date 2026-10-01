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


class _TurnStopEvent:
    def __init__(self, service_stop: threading.Event, timeout_stop: threading.Event):
        self._events = (service_stop, timeout_stop)

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)


class VoskTranscriber:
    def __init__(
        self,
        settings: Settings,
        *,
        source: AudioChunkSource | None = None,
        gate: AudioGate | None = None,
        model_factory: Callable | None = None,
        recognizer_factory: Callable | None = None,
        stop_event: threading.Event | None = None,
    ):
        self.settings = settings
        self.gate = gate or AudioGate()
        self.source = source or SoundDeviceChunkSource(settings, gate=self.gate)
        self.vad = VoiceActivityDetector(settings)
        self._model_factory = model_factory
        self._recognizer_factory = recognizer_factory
        self._model = None
        self.stop_event = stop_event or threading.Event()

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

    def transcribe_chunks(
        self, chunks: Iterator[bytes], stop_event: _TurnStopEvent | None = None
    ) -> str:
        recognizer_factory = self._recognizer_factory
        if recognizer_factory is None:
            try:
                from vosk import KaldiRecognizer
            except ImportError as error:
                raise RuntimeError("vosk is required for local speech recognition") from error
            recognizer_factory = KaldiRecognizer

        try:
            recognizer = recognizer_factory(
                self._load_model(), self.settings.audio_sample_rate
            )
            for chunk in self.vad.utterance_chunks(
                self._until_stopped(chunks, stop_event)
            ):
                recognizer.AcceptWaveform(chunk)
        finally:
            self._close_chunks(chunks)
        result = json.loads(recognizer.FinalResult())
        return str(result.get("text", "")).strip()

    def _until_stopped(
        self, chunks: Iterator[bytes], stop_event: _TurnStopEvent | None = None
    ):
        for chunk in chunks:
            if self.stop_event.is_set() or (stop_event and stop_event.is_set()):
                break
            yield chunk

    def _turn_stop(self, timeout: float | None):
        timeout_stop = threading.Event()
        combined = _TurnStopEvent(self.stop_event, timeout_stop)
        timer = None
        if timeout is not None:
            if timeout <= 0:
                timeout_stop.set()
            else:
                timer = threading.Timer(timeout, timeout_stop.set)
                timer.daemon = True
                timer.start()
        return combined, timer

    @staticmethod
    def _close_chunks(chunks) -> None:
        close = getattr(chunks, "close", None)
        if close is not None:
            close()


class PushToTalkInput(VoskTranscriber):
    def __init__(self, settings: Settings, button, **kwargs):
        super().__init__(settings, **kwargs)
        self.button = button

    def next_utterance(self, timeout: float | None = None) -> str:
        turn_stop, timer = self._turn_stop(timeout)
        try:
            if turn_stop.is_set():
                return ""
            self.gate.wait_until_open()
            while not turn_stop.is_set():
                if getattr(self.button, "is_pressed", False):
                    break
                self.button.wait_for_press(timeout=0.2)
                if getattr(self.button, "is_pressed", False):
                    break
            if turn_stop.is_set():
                return ""
            self.gate.wait_until_open()
            logger.info("Push-to-talk activated")
            chunks = self.source.chunks(turn_stop)
            return self.transcribe_chunks(chunks, turn_stop)
        finally:
            if timer is not None:
                timer.cancel()


class WakeWordInput(VoskTranscriber):
    def __init__(self, settings: Settings, detector, **kwargs):
        super().__init__(settings, **kwargs)
        self.detector = detector

    def next_utterance(self, timeout: float | None = None) -> str:
        turn_stop, timer = self._turn_stop(timeout)
        if turn_stop.is_set():
            return ""
        self.gate.wait_until_open()
        chunks = self.source.chunks(turn_stop)
        try:
            for chunk in chunks:
                if turn_stop.is_set():
                    return ""
                if not self.gate.is_open:
                    continue
                if self.detector.detected(chunk):
                    logger.info("Wake word detected")
                    return self.transcribe_chunks(chunks, turn_stop)
            return ""
        finally:
            self._close_chunks(chunks)
            if timer is not None:
                timer.cancel()
