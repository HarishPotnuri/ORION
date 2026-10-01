"""Local KittenTTS output with in-memory playback only."""

from __future__ import annotations

import threading
import queue
from collections.abc import Callable
from typing import Any

from ..config import Settings
from .capture import AudioGate


class KittenSpeechOutput:
    def __init__(
        self,
        settings: Settings,
        gate: AudioGate,
        *,
        tts_factory: Callable | None = None,
        play: Callable | None = None,
        wait: Callable | None = None,
        stop: Callable | None = None,
    ):
        self.settings = settings
        self.gate = gate
        self._tts_factory = tts_factory
        self._play = play
        self._wait = wait
        self._stop = stop
        self._engine: Any = None
        self._lock = threading.Lock()

    def _load_engine(self):
        if self._engine is None:
            factory = self._tts_factory
            if factory is None:
                try:
                    from kittentts import KittenTTS
                except ImportError as error:
                    raise RuntimeError("kittentts is required for local speech output") from error
                factory = KittenTTS
            self._engine = factory("KittenML/kitten-tts-nano-0.2")
        return self._engine

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        with self._lock, self.gate.speaking():
            audio = self._load_engine().generate(text, voice=self.settings.tts_voice)
            play = self._play
            wait = self._wait
            stop = self._stop
            if play is not None and wait is not None and stop is None:
                stop = lambda: None
            if play is None or wait is None or stop is None:
                try:
                    import sounddevice as sd
                except ImportError as error:
                    raise RuntimeError("sounddevice is required for local speech playback") from error
                play = play or sd.play
                wait = wait or sd.wait
                stop = stop or sd.stop
            play(audio, samplerate=self.settings.tts_sample_rate)
            self._wait_for_playback(wait, stop)

    def _wait_for_playback(self, wait: Callable, stop: Callable) -> None:
        errors: queue.Queue[BaseException] = queue.Queue(maxsize=1)

        def wait_for_output() -> None:
            try:
                wait()
            except BaseException as error:
                errors.put(error)

        waiter = threading.Thread(
            target=wait_for_output, name="orion-tts-playback", daemon=True
        )
        waiter.start()
        while waiter.is_alive():
            if self.gate.wait_for_interrupt(0.05):
                stop()
                break
            waiter.join(0.05)
        waiter.join()
        if not errors.empty():
            raise errors.get_nowait()
