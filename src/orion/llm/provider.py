"""Single-model local LLM lifecycle boundary."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..config import Settings
from ..core.contracts import ChatMessage, ModelStreamEvent
from ..models import get_model, resolve_model_path
from .runtime import LlamaCppServer, create_local_model


@runtime_checkable
class ILLMProvider(Protocol):
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def is_loaded(self) -> bool: ...
    def model_info(self) -> dict[str, object]: ...
    def generate(self, messages: Sequence[ChatMessage], *, max_tokens: int, timeout: float) -> str: ...
    def generate_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        timeout: float,
        cancel_event: threading.Event,
    ) -> Iterator[ModelStreamEvent]: ...


class LlamaCppProvider:
    """Own at most one llama.cpp server and its selected local GGUF."""

    def __init__(self, settings: Settings, *, server_factory=None, runtime_factory=None):
        self.settings = settings
        self._server_factory = server_factory or LlamaCppServer
        self._runtime_factory = runtime_factory or create_local_model
        self._server = None
        self._runtime = None
        self._lock = threading.RLock()
        self._active = 0
        self._last_used = time.monotonic()

    def load(self) -> None:
        with self._lock:
            if self._runtime is not None:
                return
            model_path = self._model_path()
            if not model_path.is_file():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if model_path.suffix.lower() != ".gguf":
                raise ValueError(f"Expected a .gguf model file: {model_path}")
            try:
                with model_path.open("rb") as model_file:
                    model_file.read(1)
            except OSError as error:
                raise OSError(f"Model file is not readable: {model_path}") from error
            server = self._server_factory(self.settings)
            self._server = server
            try:
                server.start()
                self._runtime = self._runtime_factory(self.settings)
            except BaseException:
                server.stop()
                self._server = None
                self._runtime = None
                raise

    def unload(self) -> None:
        with self._lock:
            if self._active:
                raise RuntimeError("Cannot unload an active LLM request")
            server = self._server
            if server is not None:
                server.stop()
            self._server = None
            self._runtime = None
            self._last_used = time.monotonic()

    def switch_model(self, model_id: str) -> None:
        spec = get_model(model_id)
        if spec.kind != "llm":
            raise ValueError(f"Selected model is not an LLM: {model_id}")
        target_path = resolve_model_path(self.settings.model_root, model_id).resolve()
        if not target_path.is_file() or target_path.suffix.lower() != ".gguf":
            raise FileNotFoundError(f"Model file not found or invalid: {target_path}")
        with self._lock:
            if self._active:
                raise RuntimeError("Cannot switch models during an active LLM request")
            if self._server is not None and not getattr(self._server, "owned_process", True):
                raise RuntimeError(
                    "Cannot switch an externally managed llama-server; stop it first"
                )
            self.unload()
            self.settings = replace(
                self.settings,
                active_llm_model=model_id,
                model_path=target_path,
                model_context_size=spec.context_size or self.settings.model_context_size,
            )
        self.load()

    def is_loaded(self) -> bool:
        with self._lock:
            return self._runtime is not None

    def model_info(self) -> dict[str, object]:
        spec = get_model(self.settings.active_llm_model)
        return {
            "id": spec.model_id,
            "display_name": spec.display_name,
            "type": spec.kind,
            "runtime": spec.runtime,
            "quantization": spec.quantization,
            "context_size": self.settings.model_context_size,
            "path": str(self._model_path()),
        }

    def generate(self, messages, *, max_tokens: int, timeout: float) -> str:
        with self._using_runtime() as runtime:
            return runtime.complete(messages, max_tokens=max_tokens, timeout=timeout)

    def generate_stream(
        self, messages, *, max_tokens, timeout, cancel_event
    ) -> Iterator[ModelStreamEvent]:
        with self._using_runtime() as runtime:
            yield from runtime.stream(
                messages,
                max_tokens=max_tokens,
                timeout=timeout,
                cancel_event=cancel_event,
            )

    def complete(self, messages, *, max_tokens: int, timeout: float) -> str:
        return self.generate(messages, max_tokens=max_tokens, timeout=timeout)

    def stream(self, messages, *, max_tokens, timeout, cancel_event):
        yield from self.generate_stream(
            messages,
            max_tokens=max_tokens,
            timeout=timeout,
            cancel_event=cancel_event,
        )

    def evict_if_idle(self, timeout: float) -> bool:
        with self._lock:
            if (
                self._runtime is None
                or self._active
                or time.monotonic() - self._last_used < timeout
            ):
                return False
            self.unload()
            return True

    def close(self) -> None:
        self.unload()

    def _model_path(self) -> Path:
        if self.settings.model_path is not None:
            return Path(self.settings.model_path).expanduser().resolve()
        return resolve_model_path(
            self.settings.model_root, self.settings.active_llm_model
        ).resolve()

    def _using_runtime(self):
        provider = self

        class RuntimeLease:
            def __enter__(self):
                with provider._lock:
                    provider.load()
                    provider._active += 1
                    return provider._runtime

            def __exit__(self, *_exc_info):
                with provider._lock:
                    provider._active -= 1
                    provider._last_used = time.monotonic()

        return RuntimeLease()
