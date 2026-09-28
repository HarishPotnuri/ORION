"""Optional openWakeWord adapter loaded only when enabled."""

from __future__ import annotations

import array
from collections.abc import Callable


class OpenWakeWordDetector:
    def __init__(
        self,
        model_name: str,
        threshold: float,
        *,
        model_factory: Callable | None = None,
        samples_factory: Callable | None = None,
    ):
        if model_factory is None:
            try:
                from openwakeword.model import Model
            except ImportError as error:
                raise RuntimeError(
                    "Install ORION wake-word dependencies to enable openWakeWord"
                ) from error
            model_factory = Model
        self._model = model_factory(wakeword_models=[model_name])
        self._model_name = model_name
        self._threshold = threshold
        self._samples_factory = samples_factory

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
        return max(normalized_scores, default=0.0) >= self._threshold
