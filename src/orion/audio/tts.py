"""Local KittenTTS output with in-memory playback only."""

from __future__ import annotations

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
    ):
        self.settings = settings
        self.gate = gate
        self._tts_factory = tts_factory
        self._play = play
        self._wait = wait
        self._engine: Any = None

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
        with self.gate.speaking():
            audio = self._load_engine().generate(
                text, voice=self.settings.tts_voice
            )
            play = self._play
            wait = self._wait
            if play is None or wait is None:
                try:
                    import sounddevice as sd
                except ImportError as error:
                    raise RuntimeError("sounddevice is required for local speech playback") from error
                play = play or sd.play
                wait = wait or sd.wait
            play(audio, samplerate=self.settings.tts_sample_rate)
            wait()
