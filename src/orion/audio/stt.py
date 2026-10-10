"""Streaming VOSK transcription over bounded microphone chunks."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
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
        self.last_confidence: float | None = None

    def warmup(self) -> None:
        self._load_model()

    def validate_assets(self) -> None:
        if not Path(self.settings.stt_model_path).is_dir():
            raise FileNotFoundError("VOSK model directory not found")

    def transcribe(self, pcm: bytes, *, cancel_event: threading.Event, timeout=None) -> str:
        self.last_confidence = None
        if not pcm or cancel_event.is_set():
            return ""
        deadline = time.monotonic() + (self.settings.stt_timeout if timeout is None else timeout)
        recognizer_factory = self._recognizer_factory
        if recognizer_factory is None:
            from vosk import KaldiRecognizer

            recognizer_factory = KaldiRecognizer
        recognizer = recognizer_factory(self._load_model(), self.settings.audio_sample_rate)
        self._enable_word_confidence(recognizer)
        block_bytes = self.settings.audio_block_frames * 2
        for offset in range(0, len(pcm), block_bytes):
            if cancel_event.is_set():
                return ""
            if time.monotonic() >= deadline:
                raise TimeoutError("VOSK transcription timed out")
            recognizer.AcceptWaveform(pcm[offset:offset + block_bytes])
        if cancel_event.is_set():
            return ""
        result = json.loads(recognizer.FinalResult())
        self.last_confidence = self._word_confidence(result)
        return str(result.get("text", "")).strip()

    def close(self) -> None:
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

    def transcribe_chunks(
        self, chunks: Iterator[bytes], stop_event: _TurnStopEvent | None = None
    ) -> str:
        self.last_confidence = None
        recognizer_factory = self._recognizer_factory
        if recognizer_factory is None:
            try:
                from vosk import KaldiRecognizer
            except ImportError as error:
                raise RuntimeError("vosk is required for local speech recognition") from error
            recognizer_factory = KaldiRecognizer

        started_at = time.perf_counter()
        outcome = "failed"
        try:
            try:
                recognizer = recognizer_factory(
                    self._load_model(), self.settings.audio_sample_rate
                )
                self._enable_word_confidence(recognizer)
                for chunk in self.vad.utterance_chunks(
                    self._until_stopped(chunks, stop_event)
                ):
                    recognizer.AcceptWaveform(chunk)
            finally:
                self._close_chunks(chunks)
            result = json.loads(recognizer.FinalResult())
            transcript = str(result.get("text", "")).strip()
            self.last_confidence = self._word_confidence(result)
            outcome = "transcribed" if transcript else "empty"
            return transcript
        finally:
            logger.info(
                "Voice stage=stt elapsed_ms=%.1f outcome=%s",
                (time.perf_counter() - started_at) * 1000,
                outcome,
            )

    @staticmethod
    def _enable_word_confidence(recognizer) -> None:
        set_words = getattr(recognizer, "SetWords", None)
        if callable(set_words):
            set_words(True)

    @staticmethod
    def _word_confidence(result: object) -> float | None:
        words = result.get("result") if isinstance(result, dict) else None
        if not isinstance(words, list):
            return None
        confidences = [
            float(word["conf"])
            for word in words
            if isinstance(word, dict)
            and isinstance(word.get("conf"), (int, float))
            and not isinstance(word.get("conf"), bool)
            and math.isfinite(word["conf"])
            and 0 <= word["conf"] <= 1
        ]
        return min(confidences) if confidences else None

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
        wait_started = time.perf_counter()
        try:
            if turn_stop.is_set():
                logger.info("Voice stage=ptt_wait elapsed_ms=0.0 outcome=timeout")
                return ""
            self.gate.wait_until_open()
            while not turn_stop.is_set():
                if getattr(self.button, "is_pressed", False):
                    break
                self.button.wait_for_press(timeout=0.2)
                if getattr(self.button, "is_pressed", False):
                    break
            if turn_stop.is_set():
                logger.info(
                    "Voice stage=ptt_wait elapsed_ms=%.1f outcome=timeout",
                    (time.perf_counter() - wait_started) * 1000,
                )
                return ""
            self.gate.wait_until_open()
            logger.info("Push-to-talk activated")
            logger.info(
                "Voice stage=ptt_wait elapsed_ms=%.1f outcome=pressed",
                (time.perf_counter() - wait_started) * 1000,
            )
            chunks = self.source.chunks(turn_stop)
            return self.transcribe_chunks(chunks, turn_stop)
        finally:
            if timer is not None:
                timer.cancel()


class WhisperPushToTalkInput(PushToTalkInput):
    """Use existing PTT capture and VAD with the selected whisper.cpp model."""

    def __init__(self, settings: Settings, button, transcriber, **kwargs):
        super().__init__(settings, button, **kwargs)
        self.transcriber = transcriber

    def transcribe_chunks(self, chunks, stop_event: _TurnStopEvent | None = None) -> str:
        started_at = time.perf_counter()
        outcome = "failed"
        pcm = bytearray()
        try:
            for chunk in self.vad.utterance_chunks(self._until_stopped(chunks, stop_event)):
                pcm.extend(chunk)
            if self.stop_event.is_set() or (stop_event and stop_event.is_set()):
                outcome = "cancelled"
                return ""
            if not pcm:
                outcome = "empty"
                return ""
            transcript = self.transcriber.transcribe(
                bytes(pcm),
                cancel_event=stop_event or self.stop_event,
                timeout=self.settings.stt_timeout,
            )
            outcome = "transcribed" if transcript else "empty"
            return transcript
        finally:
            self._close_chunks(chunks)
            logger.info(
                "Voice stage=stt elapsed_ms=%.1f outcome=%s",
                (time.perf_counter() - started_at) * 1000,
                outcome,
            )

    def close(self) -> None:
        self.transcriber.close()
        super().close()


class WakeWordInput(VoskTranscriber):
    def __init__(self, settings: Settings, detector, *, on_wake=None, **kwargs):
        super().__init__(settings, **kwargs)
        self.detector = detector
        self.on_wake = on_wake

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
                    if self.on_wake is not None:
                        self.on_wake()
                    return self.transcribe_chunks(chunks, turn_stop)
            return ""
        finally:
            self._close_chunks(chunks)
            if timer is not None:
                timer.cancel()
