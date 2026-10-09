"""Optional local ONNX text-embedding provider boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from ..config import Settings
from ..models import resolve_model_path


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OnnxTextEmbeddingProvider:
    """Run a configured ONNX model only when caller requests embeddings."""

    def __init__(self, model_path: Path, *, session_factory=None):
        self.model_path = Path(model_path)
        self._session_factory = session_factory
        self._session = None

    def _load_session(self):
        if self._session is not None:
            return self._session
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        factory = self._session_factory
        if factory is None:
            try:
                import onnxruntime
            except ImportError as error:
                raise RuntimeError("onnxruntime is required for local embeddings") from error
            factory = lambda path: onnxruntime.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
        session = factory(self.model_path)
        inputs = session.get_inputs()
        text_input = next((item for item in inputs if "string" in item.type), None)
        if text_input is None:
            raise RuntimeError(
                "BGE ONNX input contract needs tokenizer tensors; only direct text input is configured"
            )
        self._input_name = text_input.name
        self._session = session
        return session

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError("Embedding input must contain one or more strings")
        session = self._load_session()
        outputs = session.run(None, {self._input_name: list(texts)})
        if not outputs:
            raise RuntimeError("Embedding model returned no output")
        rows = outputs[0]
        return [[float(value) for value in row] for row in rows]


def create_embedding_provider(settings: Settings, *, session_factory=None):
    model_path = resolve_model_path(settings.model_root, settings.active_embedding_model)
    return OnnxTextEmbeddingProvider(model_path, session_factory=session_factory)
