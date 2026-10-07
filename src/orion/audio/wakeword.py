"""Optional openWakeWord adapter loaded only when enabled."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time

from ..config import Settings
from .wakeword_assets import validate_wakeword_assets


def validate_settings(settings: Settings) -> None:
    """Validate configured wake assets without constructing the detector."""
    validate_wakeword_assets(
        wakeword_path=settings.wake_word_model_path,
        melspectrogram_path=settings.wake_word_melspectrogram_path,
        embedding_path=settings.wake_word_embedding_path,
        manifest_path=settings.wake_word_manifest_path,
        require_manifest=settings.wake_word_manifest_path is not None,
    )


class OpenWakeWordDetector:
    def __init__(
        self,
        model_name: str,
        threshold: float,
        *,
        model_factory: Callable | None = None,
        samples_factory: Callable | None = None,
        model_options: dict | None = None,
        cooldown_seconds: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not 0 <= threshold <= 1:
            raise ValueError("Wake-word threshold must be between 0 and 1")
        if cooldown_seconds < 0:
            raise ValueError("Wake-word cooldown cannot be negative")
        if model_factory is None:
            try:
                from openwakeword.model import Model
            except ImportError as error:
                raise RuntimeError(
                    "Install ORION wake-word dependencies to enable openWakeWord"
                ) from error
            model_factory = Model
        self._model = model_factory(wakeword_models=[model_name], **(model_options or {}))
        self._model_name = model_name
        self._threshold = threshold
        self._samples_factory = samples_factory
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._last_detection = float("-inf")
        self._latched = False

    @classmethod
    def from_settings(
        cls, settings: Settings, *, model_factory=None, samples_factory=None,
        session_factory=None, runtime_version=None, numpy_module=None,
    ):
        paths = (
            settings.wake_word_model_path,
            settings.wake_word_melspectrogram_path,
            settings.wake_word_embedding_path,
        )
        validate_wakeword_assets(
            wakeword_path=paths[0],
            melspectrogram_path=paths[1],
            embedding_path=paths[2],
            manifest_path=settings.wake_word_manifest_path,
            require_manifest=settings.wake_word_manifest_path is not None,
            session_factory=session_factory,
            runtime_version=runtime_version,
            numpy_module=numpy_module,
        )
        return cls(
            str(paths[0]), settings.wake_word_threshold,
            model_factory=model_factory, samples_factory=samples_factory,
            cooldown_seconds=settings.wake_word_cooldown_seconds,
            model_options={
                "inference_framework": "onnx",
                "melspec_model_path": str(paths[1]),
                "embedding_model_path": str(paths[2]),
                "ncpu": settings.wake_word_threads,
            },
        )

    def reset(self) -> None:
        self._model.reset()

    def detected(self, pcm_chunk: bytes) -> bool:
        if self._samples_factory is None:
            try:
                import numpy as np
            except ImportError as error:
                raise RuntimeError(
                    "Install ORION wake-word dependencies to enable openWakeWord"
                ) from error
            samples = np.frombuffer(pcm_chunk, dtype=np.int16)
        else:
            samples = self._samples_factory(pcm_chunk)
        scores = self._model.predict(samples)
        score_values = scores.values() if self._model_name not in scores else [
            scores[self._model_name]
        ]
        normalized_scores = [
            max(value.values(), default=0.0) if isinstance(value, dict) else value
            for value in score_values
        ]
        above_threshold = max(normalized_scores, default=0.0) >= self._threshold
        if not above_threshold:
            self._latched = False
            return False
        now = self._clock()
        if self._latched or now - self._last_detection < self._cooldown_seconds:
            return False
        self._latched = True
        self._last_detection = now
        return True
