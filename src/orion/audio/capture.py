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

    def __init__(self) -> None:
        self._ready = threading.Event()
        self._ready.set()

    def wait_until_open(self) -> None:
        self._ready.wait()

    @property
    def is_open(self) -> bool:
        return self._ready.is_set()

    @contextmanager
    def speaking(self):
        self._ready.clear()
        try:
            yield
        finally:
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
        stop_event = stop_event or threading.Event()

        def callback(indata, _frames, _time, status) -> None:
            if status:
                logger.warning("Microphone stream status: %s", status)
            if overflow.is_set() or (self.gate and not self.gate.is_open):
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
                while not stop_event.is_set():
                    if overflow.is_set():
                        raise AudioBufferOverflow(
                            "Microphone chunk queue overflowed; utterance discarded"
                        )
                    if self.gate and not self.gate.is_open:
                        while True:
                            try:
                                input_queue.get_nowait()
                            except queue.Empty:
                                break
                        self.gate.wait_until_open()
                        continue
                    try:
                        yield input_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
        finally:
            stop_event.set()
