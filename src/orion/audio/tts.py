"""Local KittenTTS output with in-memory playback only."""

from __future__ import annotations

import array
import threading
import queue
import logging
import time
from collections.abc import Callable
from typing import Any

from ..config import Settings
from .capture import AudioGate

logger = logging.getLogger(__name__)


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
        output_stream_factory: Callable | None = None,
    ):
        self.settings = settings
        self.gate = gate
        self._tts_factory = tts_factory
        self._play = play
        self._wait = wait
        self._stop = stop
        self._output_stream_factory = output_stream_factory
        self._engine: Any = None
        self._lock = threading.Lock()

    @staticmethod
    def _resample(audio, source_rate: int, target_rate: int):
        if source_rate == target_rate or len(audio) < 2:
            return audio

        source = [float(sample) for sample in audio]
        target_length = max(1, round(len(source) * target_rate / source_rate))
        scale = source_rate / target_rate
        resampled = []
        for index in range(target_length):
            position = index * scale
            left = min(int(position), len(source) - 1)
            right = min(left + 1, len(source) - 1)
            fraction = position - left
            resampled.append(
                source[left] + (source[right] - source[left]) * fraction
            )
        return resampled

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

    def warmup(self) -> None:
        started = time.perf_counter()
        with self._lock:
            self._load_engine().generate("Ready.", voice=self.settings.tts_voice)
        logger.info("KittenTTS initialized in %.2f seconds", time.perf_counter() - started)

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        with self._lock, self.gate.speaking():
            audio = self._load_engine().generate(text, voice=self.settings.tts_voice)
            play = self._play
            wait = self._wait
            stop = self._stop
            sample_rate = self.settings.tts_sample_rate
            device = self.settings.audio_device
            if play is None and wait is None and stop is None:
                try:
                    import sounddevice as sd
                except ImportError as error:
                    raise RuntimeError("sounddevice is required for local speech playback") from error
                try:
                    sd.check_output_settings(
                        device=device,
                        channels=1,
                        dtype="float32",
                        samplerate=sample_rate,
                    )
                except sd.PortAudioError:
                    device_info = sd.query_devices(device, "output")
                    sample_rate = int(round(device_info["default_samplerate"]))
                    sd.check_output_settings(
                        device=device,
                        channels=1,
                        dtype="float32",
                        samplerate=sample_rate,
                    )
                audio = self._resample(
                    audio, self.settings.tts_sample_rate, sample_rate
                )
                self._play_interruptibly(audio, sample_rate, device, sd)
                return
            if play is not None and wait is not None and stop is None:
                stop = lambda: None
            if play is None or wait is None or stop is None:
                try:
                    import sounddevice as sd
                except ImportError as error:
                    raise RuntimeError("sounddevice is required for local speech playback") from error
                try:
                    sd.check_output_settings(
                        device=device,
                        channels=1,
                        dtype="float32",
                        samplerate=sample_rate,
                    )
                except sd.PortAudioError:
                    device_info = sd.query_devices(device, "output")
                    sample_rate = int(round(device_info["default_samplerate"]))
                    sd.check_output_settings(
                        device=device,
                        channels=1,
                        dtype="float32",
                        samplerate=sample_rate,
                    )
                audio = self._resample(
                    audio, self.settings.tts_sample_rate, sample_rate
                )
                play = play or sd.play
                wait = wait or sd.wait
                stop = stop or sd.stop
            play_options = {"samplerate": sample_rate}
            if device is not None:
                play_options["device"] = device
            play(audio, **play_options)
            self._wait_for_playback(wait, stop)

    def _play_interruptibly(self, audio, sample_rate: int, device, sounddevice) -> None:
        samples = array.array("f", audio)
        if not len(samples):
            return
        stream_factory = self._output_stream_factory or sounddevice.RawOutputStream
        block_frames = max(1, sample_rate // 50)
        with stream_factory(
            samplerate=sample_rate,
            device=device,
            channels=1,
            dtype="float32",
        ) as stream:
            interrupted = False
            for start in range(0, len(samples), block_frames):
                if self.gate.wait_for_interrupt(0):
                    interrupted = True
                    break
                stream.write(samples[start : start + block_frames].tobytes())
                if self.gate.wait_for_interrupt(0):
                    interrupted = True
                    break
            if interrupted:
                stream.abort()
        logger.info(
            "TTS playback outcome=%s",
            "interrupted" if interrupted else "completed",
        )

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
