"""Local KittenTTS output with in-memory playback only."""

from __future__ import annotations

import array
import json
import threading
import queue
import logging
import time
from collections.abc import Callable, Iterator
from itertools import chain
from pathlib import Path
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
        local_tts_factory: Callable | None = None,
        play: Callable | None = None,
        wait: Callable | None = None,
        stop: Callable | None = None,
        output_stream_factory: Callable | None = None,
    ):
        self.settings = settings
        self.gate = gate
        self._tts_factory = tts_factory
        self._local_tts_factory = local_tts_factory
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
            if self.settings.tts_model_path is not None:
                self._engine = self._load_local_engine(
                    self.settings.tts_model_path.expanduser()
                )
                return self._engine
            factory = self._tts_factory
            if factory is None:
                try:
                    from kittentts import KittenTTS
                except ImportError as error:
                    raise RuntimeError("kittentts is required for local speech output") from error
                factory = KittenTTS
            self._engine = factory("KittenML/kitten-tts-nano-0.2")
        return self._engine

    def _load_local_engine(self, model_directory: Path):
        root = model_directory.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Local KittenTTS model directory not found: {root}")
        try:
            with (root / "config.json").open(encoding="utf-8") as config_file:
                config = json.load(config_file)
        except (OSError, json.JSONDecodeError):
            raise ValueError("Local KittenTTS directory has an invalid config.json") from None
        if not isinstance(config, dict) or config.get("type") != "ONNX1":
            raise ValueError("Local KittenTTS config must describe an ONNX1 model")

        paths = []
        for field in ("model_file", "voices"):
            relative_path = config.get(field)
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(f"Local KittenTTS config is missing {field}")
            asset_path = (root / relative_path).resolve()
            try:
                asset_path.relative_to(root)
            except ValueError:
                raise ValueError("Local KittenTTS assets must stay inside the model directory") from None
            if not asset_path.is_file():
                raise FileNotFoundError(f"Local KittenTTS asset not found: {asset_path.name}")
            paths.append(str(asset_path))

        factory = self._local_tts_factory
        if factory is None:
            try:
                from kittentts.onnx_model import KittenTTS_1_Onnx
            except ImportError as error:
                raise RuntimeError("kittentts is required for local speech output") from error
            factory = KittenTTS_1_Onnx
        return factory(model_path=paths[0], voices_path=paths[1])

    def warmup(self) -> None:
        started = time.perf_counter()
        with self._lock:
            self._load_engine().generate("Ready.", voice=self.settings.tts_voice)
        logger.info("KittenTTS initialized in %.2f seconds", time.perf_counter() - started)

    def speak(self, text: str) -> bool:
        if not text.strip():
            return True
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
                return self._play_interruptibly(audio, sample_rate, device, sd)
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
            return self._wait_for_playback(wait, stop)

    def _play_interruptibly(self, audio, sample_rate: int, device, sounddevice) -> bool:
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
        return not interrupted

    def _wait_for_playback(self, wait: Callable, stop: Callable) -> bool:
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
        interrupted = False
        while waiter.is_alive():
            if self.gate.wait_for_interrupt(0.05):
                stop()
                interrupted = True
                break
            waiter.join(0.05)
        waiter.join()
        if not errors.empty():
            raise errors.get_nowait()
        return not interrupted


class PiperSpeechOutput:
    def __init__(
        self,
        settings: Settings,
        gate: AudioGate,
        *,
        voice_factory: Callable | None = None,
        output_stream_factory: Callable | None = None,
    ):
        self.settings = settings
        self.gate = gate
        self._voice_factory = voice_factory
        self._output_stream_factory = output_stream_factory
        self._voice: Any = None
        self._lock = threading.Lock()
        self._last_used = time.monotonic()

    def _load_voice(self):
        if self._voice is not None:
            return self._voice
        model_path = self.settings.tts_model_path
        if model_path is None:
            raise ValueError("Piper requires a local ORION_TTS_MODEL_PATH")
        model_path = model_path.expanduser().resolve()
        config_path = Path(f"{model_path}.json")
        if not model_path.is_file():
            raise FileNotFoundError(f"Piper voice model not found: {model_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"Piper voice config not found: {config_path}")

        factory = self._voice_factory or self._create_voice
        self._voice = factory(
            model_path=model_path,
            config_path=config_path,
            threads=self.settings.tts_threads,
        )
        return self._voice

    @staticmethod
    def _create_voice(*, model_path: Path, config_path: Path, threads: int):
        try:
            import onnxruntime
            from piper import PiperVoice
            from piper.config import PiperConfig
        except ImportError as error:
            raise RuntimeError("piper-tts and onnxruntime are required for Piper speech") from error

        with config_path.open(encoding="utf-8") as config_file:
            config = PiperConfig.from_dict(json.load(config_file))
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = threads
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        return PiperVoice(session, config)

    def speak(self, text: str) -> bool:
        if not text.strip():
            return True
        with self._lock, self.gate.speaking():
            voice = self._load_voice()
            source_rate = int(voice.config.sample_rate)
            if source_rate < 1:
                raise ValueError("Piper voice metadata has an invalid sample rate")
            completed = self._play_chunks(
                self._audio_chunks(voice, text, source_rate), source_rate
            )
            self._last_used = time.monotonic()
            return completed

    def _audio_chunks(self, voice, text: str, source_rate: int) -> Iterator[tuple[array.array, int]]:
        for chunk in voice.synthesize(text):
            if self.gate.wait_for_interrupt(0):
                return
            rate = int(getattr(chunk, "sample_rate", source_rate))
            samples = array.array("h")
            samples.frombytes(chunk.audio_int16_bytes)
            yield samples, rate

    def _play_chunks(
        self, chunks: Iterator[tuple[array.array, int]], source_rate: int
    ) -> bool:
        try:
            import sounddevice as sd
        except ImportError as error:
            raise RuntimeError("sounddevice is required for local speech playback") from error

        target_rate = self.settings.tts_sample_rate
        try:
            sd.check_output_settings(
                device=self.settings.audio_device,
                channels=1,
                dtype="float32",
                samplerate=target_rate,
            )
        except sd.PortAudioError:
            device_info = sd.query_devices(self.settings.audio_device, "output")
            target_rate = int(round(device_info["default_samplerate"]))
            sd.check_output_settings(
                device=self.settings.audio_device,
                channels=1,
                dtype="float32",
                samplerate=target_rate,
            )

        first_chunk = next(chunks, None)
        if first_chunk is None:
            return not self.gate.wait_for_interrupt(0)
        stream_factory = self._output_stream_factory or sd.RawOutputStream
        stream_options = {
            "samplerate": target_rate,
            "device": self.settings.audio_device,
            "channels": 1,
            "dtype": "float32",
        }
        block_samples = max(1, target_rate // 50)
        interrupted = False
        with stream_factory(**stream_options) as stream:
            for samples, chunk_rate in chain((first_chunk,), chunks):
                if self.gate.wait_for_interrupt(0):
                    interrupted = True
                    break
                normalized = [sample / 32768.0 for sample in samples]
                converted = KittenSpeechOutput._resample(
                    normalized, chunk_rate or source_rate, target_rate
                )
                for start in range(0, len(converted), block_samples):
                    if self.gate.wait_for_interrupt(0):
                        interrupted = True
                        break
                    block = array.array("f", converted[start:start + block_samples])
                    stream.write(block.tobytes())
                    if self.gate.wait_for_interrupt(0):
                        interrupted = True
                        break
                if interrupted:
                    break
            if interrupted:
                stream.abort()
        logger.info(
            "Piper playback outcome=%s",
            "interrupted" if interrupted else "completed",
        )
        return not interrupted

    def evict_if_idle(self, timeout: float) -> bool:
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if self._voice is None or time.monotonic() - self._last_used < timeout:
                return False
            self._voice = None
            logger.info("Evicted idle Piper voice")
            return True
        finally:
            self._lock.release()
