"""Bounded microphone chunk capture without audio-file persistence."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol

from ..config import Settings

logger = logging.getLogger(__name__)


class AudioBufferOverflow(RuntimeError):
    """Raised when capture cannot keep up without losing audio chunks."""


class AudioChunkSource(Protocol):
    def chunks(self, stop_event: threading.Event | None = None) -> Iterator[bytes]:
        """Yield bounded mono PCM chunks until stopped or the iterator closes."""


class AudioGate:
    """Prevent microphone activation while local speech output is active."""

    def __init__(self, *, allow_wakeword_during_speech: bool = False) -> None:
        self.allow_wakeword_during_speech = allow_wakeword_during_speech
        self._ready = threading.Event()
        self._ready.set()
        self._interrupt = threading.Event()
        self._turn_cancel = threading.Event()
        self._capture_cancel = threading.Event()
        self._lock = threading.Lock()
        self._generation = 0
        self._speaking_depth = 0
        self._active_turns = 0

    def wait_until_open(self) -> None:
        self._ready.wait()

    @property
    def is_open(self) -> bool:
        return self._ready.is_set()

    @property
    def wakeword_capture_allowed(self) -> bool:
        return self.is_open or self.allow_wakeword_during_speech

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def interrupt_event(self) -> threading.Event:
        return self._interrupt

    @property
    def turn_cancel_event(self) -> threading.Event:
        return self._turn_cancel

    @property
    def capture_cancel_event(self) -> threading.Event:
        return self._capture_cancel

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._speaking_depth > 0

    @property
    def turn_active(self) -> bool:
        with self._lock:
            return self._active_turns > 0

    def request_interrupt(self) -> None:
        with self._lock:
            if not self._ready.is_set():
                self._interrupt.set()
            if self._active_turns:
                self._turn_cancel.set()
            self._capture_cancel.set()

    def interrupt_playback(self) -> None:
        with self._lock:
            if self._speaking_depth:
                self._interrupt.set()
                if self._active_turns:
                    self._turn_cancel.set()

    def begin_capture(self) -> None:
        with self._lock:
            self._capture_cancel.clear()

    @contextmanager
    def interruptible_turn(self):
        with self._lock:
            if self._active_turns == 0:
                self._turn_cancel.clear()
            self._active_turns += 1
        try:
            yield self._turn_cancel
        finally:
            with self._lock:
                self._active_turns -= 1
                if self._active_turns == 0:
                    self._turn_cancel.clear()

    def wait_for_interrupt(self, timeout: float) -> bool:
        return self._interrupt.wait(timeout)

    @contextmanager
    def speaking(self):
        with self._lock:
            if self._speaking_depth == 0:
                self._interrupt.clear()
                self._generation += 1
                self._ready.clear()
            self._speaking_depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._speaking_depth -= 1
                if self._speaking_depth == 0:
                    self._generation += 1
                    self._interrupt.clear()
                    self._ready.set()


class SoundDeviceChunkSource:
    def __init__(
        self,
        settings: Settings,
        *,
        gate: AudioGate | None = None,
        stream_factory: Callable | None = None,
    ):
        self.settings = settings
        self.gate = gate
        self._stream_factory = stream_factory

    def _default_stream_factory(self) -> Callable:
        try:
            import sounddevice as sd
        except ImportError as error:
            raise RuntimeError("sounddevice is required for microphone capture") from error
        return sd.InputStream

    def chunks(self, stop_event: threading.Event | None = None) -> Iterator[bytes]:
        input_queue: queue.Queue[bytes] = queue.Queue(
            maxsize=self.settings.audio_queue_chunks
        )
        overflow = threading.Event()
        external_stop_event = stop_event
        stop_event = stop_event or threading.Event()

        def callback(indata, _frames, _time, status) -> None:
            if status:
                logger.warning("Microphone stream status: %s", status)
            if overflow.is_set() or (
                self.gate and not self.gate.wakeword_capture_allowed
            ):
                return
            try:
                input_queue.put_nowait(bytes(indata))
            except queue.Full:
                overflow.set()

        stream_factory = self._stream_factory or self._default_stream_factory()
        stream = stream_factory(
            samplerate=self.settings.audio_sample_rate,
            blocksize=self.settings.audio_block_frames,
            device=self.settings.audio_device,
            channels=1,
            dtype="int16",
            callback=callback,
        )
        try:
            with stream:
                while not stop_event.is_set() and not (
                    external_stop_event is not None and external_stop_event.is_set()
                ):
                    if overflow.is_set():
                        raise AudioBufferOverflow(
                            "Microphone chunk queue overflowed; utterance discarded"
                        )
                    if self.gate and not self.gate.wakeword_capture_allowed:
                        while True:
                            try:
                                input_queue.get_nowait()
                            except queue.Empty:
                                break
                        stop_event.wait(0.1)
                        continue
                    try:
                        yield input_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
        finally:
            if external_stop_event is None:
                stop_event.set()
