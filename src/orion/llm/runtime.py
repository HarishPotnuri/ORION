"""Local-only llama.cpp and Ollama model runtimes."""

from __future__ import annotations

import json
import logging
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import urlencode, urlparse

from ..config import Settings
from ..core.contracts import ChatMessage
from ..core.policy import NetworkPolicy

_MAX_RESPONSE_BYTES = 1_048_576
logger = logging.getLogger(__name__)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


class LocalJsonTransport:
    """HTTP JSON client with proxies and redirects disabled."""

    def __init__(self, *, opener: Any | None = None):
        self._opener = opener or urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirectHandler()
        )

    def post(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                data = response.read(_MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Local model returned HTTP {error.code}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not reach local model: {error.reason}") from error
        if len(data) > _MAX_RESPONSE_BYTES:
            raise RuntimeError("Local model response exceeded the 1 MiB limit")
        try:
            result = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Local model returned invalid JSON") from error
        if not isinstance(result, dict):
            raise RuntimeError("Local model response must be a JSON object")
        return result

    def get(self, url: str, timeout: float) -> dict[str, Any]:
        request = urllib.request.Request(url, method="GET")
        try:
            with self._opener.open(request, timeout=timeout) as response:
                data = response.read(_MAX_RESPONSE_BYTES + 1)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
            return {}
        if len(data) > _MAX_RESPONSE_BYTES:
            return {}
        try:
            result = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return result if isinstance(result, dict) else {}

    def get_factual_json(
        self,
        url: str,
        *,
        params: dict[str, str | int] | None,
        allowed_hosts: set[str],
        timeout: float,
        policy: NetworkPolicy | None = None,
    ) -> dict[str, Any]:
        validator = policy or NetworkPolicy()
        validated_url = validator.validate_factual_url(url, allowed_hosts)
        if params:
            separator = "&" if "?" in validated_url else "?"
            validated_url += separator + urlencode(params)
        request = urllib.request.Request(
            validated_url,
            headers={"Accept": "application/json", "User-Agent": "ORION-local-assistant/0.1"},
            method="GET",
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                data = response.read(_MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Factual lookup returned HTTP {error.code}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Factual lookup failed: {error.reason}") from error
        if len(data) > _MAX_RESPONSE_BYTES:
            raise RuntimeError("Factual lookup response exceeded the 1 MiB limit")
        try:
            result = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Factual lookup returned invalid JSON") from error
        if not isinstance(result, dict):
            raise RuntimeError("Factual lookup response must be a JSON object")
        return result


class LocalModelRuntime:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: Any | None = None,
        policy: NetworkPolicy | None = None,
    ):
        self.settings = settings
        self.policy = policy or NetworkPolicy()
        self.endpoint = self.policy.validate_model_endpoint(settings.model_base_url)
        self.transport = transport or LocalJsonTransport()

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        timeout: float | None = None,
    ) -> str:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        return self._complete(
            messages,
            max_tokens=max_tokens,
            timeout=timeout or self.settings.max_total_task_time,
        )

    def is_ready(self) -> bool:
        probe_path = "/health" if self.settings.model_backend == "llama.cpp" else "/api/tags"
        return bool(
            self.transport.get(
                f"{self.endpoint}{probe_path}", timeout=min(2.0, self.settings.max_total_task_time)
            )
        )

    def _complete(
        self, messages: Sequence[ChatMessage], *, max_tokens: int, timeout: float
    ) -> str:
        raise NotImplementedError

    def _messages_payload(self, messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in messages]

    @staticmethod
    def _content(value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError("Local model returned an empty response")
        return value.strip()


class LlamaCppRuntime(LocalModelRuntime):
    def _complete(
        self, messages: Sequence[ChatMessage], *, max_tokens: int, timeout: float
    ) -> str:
        response = self.transport.post(
            f"{self.endpoint}/chat/completions",
            {
                "model": "local-model",
                "messages": self._messages_payload(messages),
                "max_tokens": max_tokens,
                "temperature": self.settings.model_temperature,
            },
            timeout=timeout,
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("llama.cpp returned an invalid chat response") from error
        return self._content(content)


class OllamaRuntime(LocalModelRuntime):
    def _complete(
        self, messages: Sequence[ChatMessage], *, max_tokens: int, timeout: float
    ) -> str:
        response = self.transport.post(
            f"{self.endpoint}/api/chat",
            {
                "model": self.settings.model_name,
                "messages": self._messages_payload(messages),
                "stream": False,
                "think": False,
                "format": "json",
                "options": {
                    "num_ctx": self.settings.model_context_size,
                    "num_thread": self.settings.model_threads,
                    "num_predict": max_tokens,
                    "temperature": self.settings.model_temperature,
                },
            },
            timeout=timeout,
        )
        try:
            content = response["message"]["content"]
        except (KeyError, TypeError) as error:
            raise RuntimeError("Ollama returned an invalid chat response") from error
        return self._content(content)


def create_local_model(
    settings: Settings,
    *,
    transport: Any | None = None,
    policy: NetworkPolicy | None = None,
) -> LocalModelRuntime:
    if settings.model_backend == "llama.cpp":
        runtime = LlamaCppRuntime
    elif settings.model_backend == "ollama":
        runtime = OllamaRuntime
    else:
        raise ValueError(f"Unsupported local model backend: {settings.model_backend}")
    return runtime(settings, transport=transport, policy=policy)


class LlamaCppServer:
    """Own and supervise one local llama.cpp server process."""

    def __init__(
        self,
        settings: Settings,
        *,
        health_probe: Callable[[str, float], bool] | None = None,
        popen: Callable[..., Any] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings
        self._endpoint = NetworkPolicy().validate_model_endpoint(settings.model_base_url)
        self._health_probe = health_probe or self._default_health_probe
        self._popen = popen or subprocess.Popen
        self._monotonic = monotonic
        self._sleep = sleep
        self._process: Any | None = None

    @property
    def owned_process(self) -> bool:
        return self._process is not None

    def start(self) -> None:
        if self.settings.model_backend != "llama.cpp":
            raise ValueError("LlamaCppServer requires ORION_MODEL_BACKEND=llama.cpp")
        health_url = self._health_url()
        if self._health_probe(health_url, 1.0):
            logger.info("Using existing local llama.cpp server")
            return
        if self._process is not None:
            if self._process.poll() is None:
                raise RuntimeError("Owned llama-server process is running but not ready")
            self._process = None
        if urlparse(self._endpoint).scheme != "http":
            raise ValueError("Managed llama-server requires an HTTP loopback endpoint")
        model_path = self.settings.model_path
        if model_path is None or not model_path.is_file():
            raise FileNotFoundError("ORION_MODEL_PATH must name a local GGUF model file")

        parsed = urlparse(self._endpoint)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8080
        command = [
            self.settings.llama_server_path,
            "-m",
            str(model_path),
            "--host",
            host,
            "--port",
            str(port),
            "-c",
            str(self.settings.model_context_size),
            "-t",
            str(self.settings.model_threads),
        ]
        self._process = self._popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = self._monotonic() + self.settings.llama_startup_timeout
        while self._monotonic() < deadline:
            if self._process.poll() is not None:
                return_code = self._process.returncode
                self._process = None
                raise RuntimeError(f"llama-server exited during startup ({return_code})")
            if self._health_probe(health_url, 1.0):
                logger.info("Local llama.cpp server is ready")
                return
            self._sleep(0.2)
        self.stop()
        raise TimeoutError("llama-server did not become ready before startup timeout")

    def stop(self, timeout: float = 5.0) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def __enter__(self) -> LlamaCppServer:
        self.start()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.stop()

    def _health_url(self) -> str:
        parsed = urlparse(self._endpoint)
        return f"{parsed.scheme}://{parsed.netloc}/health"

    @staticmethod
    def _default_health_probe(url: str, timeout: float) -> bool:
        response = LocalJsonTransport().get(url, timeout)
        return bool(response)
