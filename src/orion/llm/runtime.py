"""Local-only llama.cpp model runtime and server process."""

from __future__ import annotations

import codecs
from contextlib import contextmanager
import json
import logging
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Sequence
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

from ..config import Settings
from ..core.contracts import ChatMessage, ModelStreamEvent, ModelUsage
from ..core.policy import NetworkPolicy

_MAX_RESPONSE_BYTES = 1_048_576
_STREAM_CANCEL_POLL_INTERVAL = 0.05
_PINNED_LLAMA_COMMIT = "d81235049384534c167caea52b85a694f6103d14"
logger = logging.getLogger(__name__)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


@contextmanager
def _interruptible_socket(raw_socket, cancel_event):
    if raw_socket is None:
        yield
        return

    stopped = threading.Event()

    def interrupt_on_cancel():
        while not stopped.wait(_STREAM_CANCEL_POLL_INTERVAL):
            if cancel_event.is_set():
                try:
                    raw_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                return

    watcher = threading.Thread(target=interrupt_on_cancel, daemon=True)
    watcher.start()
    try:
        yield
    finally:
        stopped.set()
        watcher.join(_STREAM_CANCEL_POLL_INTERVAL * 2)


class SSEDecoder:
    """Decode server-sent events across arbitrary UTF-8 and line boundaries."""

    def __init__(self):
        self._utf8 = codecs.getincrementaldecoder("utf-8")("strict")
        self._buffer = ""
        self._data: list[str] = []

    def feed(self, chunk: bytes, *, final: bool = False) -> list[str]:
        self._buffer += self._utf8.decode(chunk, final=final)
        events: list[str] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._consume_line(line.rstrip("\r"), events)
        if final:
            if self._buffer:
                self._consume_line(self._buffer.rstrip("\r"), events)
                self._buffer = ""
            self._dispatch(events)
        return events

    def _consume_line(self, line: str, events: list[str]) -> None:
        if not line:
            self._dispatch(events)
        elif line.startswith(":"):
            return
        elif line.startswith("data:"):
            value = line[5:]
            self._data.append(value[1:] if value.startswith(" ") else value)

    def _dispatch(self, events: list[str]) -> None:
        if self._data:
            events.append("\n".join(self._data))
            self._data.clear()


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

    def stream(
        self,
        url: str,
        payload: dict[str, object],
        *,
        timeout: float,
        cancel_event,
    ) -> Iterator[ModelStreamEvent]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        deadline = time.monotonic() + timeout
        started = time.monotonic()
        first_token_at: float | None = None
        received_bytes = 0
        saw_done = False
        finish_reason: str | None = None
        completion_tokens: int | None = None
        decoder = SSEDecoder()

        try:
            with self._opener.open(request, timeout=timeout) as response:
                raw_socket = getattr(
                    getattr(getattr(response, "fp", None), "raw", None),
                    "_sock",
                    None,
                )
                with _interruptible_socket(raw_socket, cancel_event):
                    while not saw_done:
                        if cancel_event.is_set():
                            raise RuntimeError("Local model stream was cancelled")
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("Local model stream exceeded its deadline")
                        if raw_socket is not None:
                            raw_socket.settimeout(remaining)
                        try:
                            chunk = response.read1(min(4096, max(1, _MAX_RESPONSE_BYTES - received_bytes)))
                        except (socket.timeout, TimeoutError) as error:
                            if cancel_event.is_set():
                                raise RuntimeError("Local model stream was cancelled") from error
                            raise TimeoutError("Local model stream read exceeded its deadline") from error
                        except OSError as error:
                            if cancel_event.is_set():
                                raise RuntimeError("Local model stream was cancelled") from error
                            raise
                        if not chunk:
                            try:
                                pending_events = decoder.feed(b"", final=True)
                            except UnicodeDecodeError as error:
                                raise RuntimeError("Local model stream contains invalid UTF-8") from error
                            for data in pending_events:
                                if data == "[DONE]":
                                    saw_done = True
                                    break
                                parsed = self._parse_stream_data(data)
                                if parsed is None:
                                    continue
                                text, usage, finish_reason = parsed
                                now = time.monotonic()
                                if text:
                                    if first_token_at is None:
                                        first_token_at = now
                                    yield ModelStreamEvent(kind="text", text=text)
                                if usage is not None:
                                    completion_tokens = usage.completion_tokens
                                    yield ModelStreamEvent(kind="usage", usage=usage)
                            break
                        received_bytes += len(chunk)
                        if received_bytes > _MAX_RESPONSE_BYTES:
                            raise RuntimeError("Local model stream exceeded the 1 MiB limit")
                        try:
                            events = decoder.feed(chunk)
                        except UnicodeDecodeError as error:
                            raise RuntimeError("Local model stream contains invalid UTF-8") from error
                        for data in events:
                            if data == "[DONE]":
                                saw_done = True
                                break
                            parsed = self._parse_stream_data(data)
                            if parsed is None:
                                continue
                            text, usage, finish_reason = parsed
                            now = time.monotonic()
                            if text:
                                if first_token_at is None:
                                    first_token_at = now
                                yield ModelStreamEvent(kind="text", text=text)
                            if usage is not None:
                                completion_tokens = usage.completion_tokens
                                yield ModelStreamEvent(kind="usage", usage=usage)
                if not saw_done:
                    raise RuntimeError("Local model stream ended before [DONE]")
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Local model returned HTTP {error.code}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not reach local model: {error.reason}") from error
        finished_at = time.monotonic()
        generation_seconds = (
            finished_at - first_token_at if first_token_at is not None else 0.0
        )
        yield ModelStreamEvent(
            kind="completed",
            finish_reason=finish_reason,
            elapsed_ms=(finished_at - started) * 1000,
            first_token_ms=(first_token_at - started) * 1000 if first_token_at else None,
            tokens_per_second=(
                completion_tokens / generation_seconds
                if completion_tokens is not None and generation_seconds > 0
                else None
            ),
        )

    @staticmethod
    def _parse_stream_data(
        data: str,
    ) -> tuple[str, ModelUsage | None, str | None] | None:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as error:
            raise RuntimeError("Local model stream contains invalid event JSON") from error
        if not isinstance(payload, dict):
            raise RuntimeError("Local model stream event must be a JSON object")
        if "error" in payload:
            raise RuntimeError("Local model returned a stream error")
        usage_data = payload.get("usage")
        usage = None
        if usage_data is not None:
            if not isinstance(usage_data, dict):
                raise RuntimeError("Local model stream usage must be an object")
            values = []
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage_data.get(key)
                if value is not None and (not isinstance(value, int) or value < 0):
                    raise RuntimeError("Local model stream usage contains an invalid token count")
                values.append(value)
            usage = ModelUsage(*values)
        choices = payload.get("choices", [])
        if not isinstance(choices, list):
            raise RuntimeError("Local model stream choices must be an array")
        if not choices:
            return "", usage, None
        if not isinstance(choices[0], dict):
            raise RuntimeError("Local model stream choice must be an object")
        choice = choices[0]
        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            raise RuntimeError("Local model stream delta must be an object")
        text = delta.get("content", "")
        if text is None:
            text = ""
        elif not isinstance(text, str):
            raise RuntimeError("Local model stream content must be text")
        reason = delta.get("reasoning_content", delta.get("reasoning"))
        if reason is not None and not isinstance(reason, str):
            raise RuntimeError("Local model reasoning delta must be text")
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise RuntimeError("Local model finish reason must be text")
        return text, usage, finish_reason

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
        data = self._get_factual_bytes(
            url, params=params, allowed_hosts=allowed_hosts, timeout=timeout,
            policy=policy, accept="application/json",
        )
        try:
            result = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Factual lookup returned invalid JSON") from error
        if not isinstance(result, dict):
            raise RuntimeError("Factual lookup response must be a JSON object")
        return result

    def get_factual_rss(
        self,
        url: str,
        *,
        params: dict[str, str | int] | None,
        allowed_hosts: set[str],
        timeout: float,
        policy: NetworkPolicy | None = None,
    ) -> dict[str, Any]:
        data = self._get_factual_bytes(
            url, params=params, allowed_hosts=allowed_hosts, timeout=timeout,
            policy=policy, accept="application/rss+xml, application/xml, text/xml",
        )
        try:
            root = ElementTree.fromstring(data)
        except ElementTree.ParseError as error:
            raise RuntimeError("Factual lookup returned invalid RSS") from error
        if root.tag != "rss" or root.find("channel") is None:
            raise RuntimeError("Factual lookup response must be an RSS feed")
        return {"results": [
            {
                "title": item.findtext("title", ""),
                "content": item.findtext("description", ""),
                "url": item.findtext("link", ""),
                "published": item.findtext("pubDate", ""),
            }
            for item in root.findall("./channel/item")
        ]}

    def _get_factual_bytes(
        self,
        url: str,
        *,
        params: dict[str, str | int] | None,
        allowed_hosts: set[str],
        timeout: float,
        policy: NetworkPolicy | None,
        accept: str,
    ) -> bytes:
        validator = policy or NetworkPolicy()
        validated_url = validator.validate_factual_url(url, allowed_hosts)
        if params:
            separator = "&" if "?" in validated_url else "?"
            validated_url += separator + urlencode(params)
        request = urllib.request.Request(
            validated_url,
            headers={"Accept": accept, "User-Agent": "ORION-local-assistant/0.1"},
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
        return data


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
        if max_tokens > self.settings.model_answer_tokens:
            raise ValueError("max_tokens exceeds configured answer token budget")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        budget = timeout or self.settings.max_total_task_time
        started = time.monotonic()
        self._validate_prompt_budget(messages, max_tokens=max_tokens, timeout=budget)
        remaining = budget - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Prompt validation exceeded model deadline")
        return self._complete(
            messages,
            max_tokens=max_tokens,
            timeout=remaining,
        )

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        timeout: float,
        cancel_event,
    ) -> Iterator[ModelStreamEvent]:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if max_tokens > self.settings.model_answer_tokens:
            raise ValueError("max_tokens exceeds configured answer token budget")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        started = time.monotonic()
        self._validate_prompt_budget(messages, max_tokens=max_tokens, timeout=timeout)
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Prompt validation exceeded model deadline")
        return self._stream(
            messages,
            max_tokens=max_tokens,
            timeout=remaining,
            cancel_event=cancel_event,
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

    def _stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        timeout: float,
        cancel_event,
    ) -> Iterator[ModelStreamEvent]:
        raise NotImplementedError

    def _validate_prompt_budget(
        self, messages: Sequence[ChatMessage], *, max_tokens: int, timeout: float
    ) -> None:
        deadline = time.monotonic() + timeout
        server_endpoint = self.endpoint.removesuffix("/v1")
        template = self.transport.post(
            f"{server_endpoint}/apply-template",
            {
                "messages": self._messages_payload(messages),
            },
            timeout=timeout,
        )
        prompt = template.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeError("llama.cpp returned an invalid formatted prompt")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Prompt formatting exceeded model deadline")
        tokens = self.transport.post(
            f"{server_endpoint}/tokenize",
            {"content": prompt, "add_special": False, "parse_special": True},
            timeout=remaining,
        ).get("tokens")
        if not isinstance(tokens, list) or any(not isinstance(token, int) for token in tokens):
            raise RuntimeError("llama.cpp returned invalid prompt tokens")
        prompt_tokens = len(tokens)
        input_limit = (
            self.settings.max_context_tokens
            + self.settings.model_template_overhead_tokens
        )
        if prompt_tokens > input_limit:
            raise RuntimeError("Formatted prompt exceeded the configured input token budget")
        if prompt_tokens + max_tokens > self.settings.model_context_size:
            raise RuntimeError("Formatted prompt and output exceed model context size")

    def _messages_payload(self, messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
        system_contents = [
            message.content for message in messages if message.role == "system"
        ]
        payload = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role != "system"
        ]
        if system_contents:
            payload.insert(0, {"role": "system", "content": "\n\n".join(system_contents)})
        return payload

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
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "orion_decision",
                        "schema": {
                            "oneOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": {"const": "final"},
                                        "response": {"type": "string"},
                                        "memory_suggestions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "kind": {
                                                        "enum": ["fact", "preference", "decision"]
                                                    },
                                                    "content": {"type": "string"},
                                                    "topic": {"type": "string"},
                                                },
                                                "required": ["kind", "content"],
                                                "additionalProperties": False,
                                            },
                                            "maxItems": 5,
                                        },
                                    },
                                    "required": ["type", "response"],
                                    "additionalProperties": False,
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": {"const": "tool"},
                                        "name": {"type": "string"},
                                        "arguments": {"type": "object"},
                                    },
                                    "required": ["type", "name", "arguments"],
                                    "additionalProperties": False,
                                },
                            ],
                        },
                    },
                },
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_effort": "none",
            },
            timeout=timeout,
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("llama.cpp returned an invalid chat response") from error
        return self._content(content)

    def _stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        timeout: float,
        cancel_event,
    ) -> Iterator[ModelStreamEvent]:
        yield from self.transport.stream(
            f"{self.endpoint}/chat/completions",
            {
                "model": "local-model",
                "messages": self._messages_payload(messages),
                "max_tokens": max_tokens,
                "temperature": self.settings.model_temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_effort": "none",
            },
            timeout=timeout,
            cancel_event=cancel_event,
        )


def create_local_model(
    settings: Settings,
    *,
    transport: Any | None = None,
    policy: NetworkPolicy | None = None,
) -> LocalModelRuntime:
    if settings.model_backend != "llama.cpp":
        raise ValueError(f"Unsupported local model backend: {settings.model_backend}")
    return LlamaCppRuntime(settings, transport=transport, policy=policy)


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
        model_path = self.settings.model_path
        if model_path is None or not model_path.is_file():
            raise FileNotFoundError("ORION_MODEL_PATH must name a local GGUF model file")
        health_url = self._health_url()
        if self._health_probe(health_url, 1.0):
            self._verify_external_server(model_path)
            logger.info("Using existing local llama.cpp server")
            return
        if self._process is not None:
            if self._process.poll() is None:
                raise RuntimeError("Owned llama-server process is running but not ready")
            self._process = None
        if urlparse(self._endpoint).scheme != "http":
            raise ValueError("Managed llama-server requires an HTTP loopback endpoint")
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
            "--jinja",
            "--no-mmproj",
            "--reasoning-format",
            "deepseek",
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

    def _verify_external_server(self, model_path: Path) -> None:
        parsed = urlparse(self._endpoint)
        props = LocalJsonTransport().get(
            f"{parsed.scheme}://{parsed.netloc}/props", timeout=1.0
        )
        active_path = props.get("model_path")
        build_info = props.get("build_info")
        if not isinstance(active_path, str) or Path(active_path).resolve() != model_path.resolve():
            raise RuntimeError("Existing llama-server is not serving ORION_MODEL_PATH")
        if (
            not isinstance(build_info, str)
            or _PINNED_LLAMA_COMMIT[:7] not in build_info
        ):
            raise RuntimeError("Existing llama-server does not use the pinned llama.cpp build")

    @staticmethod
    def _default_health_probe(url: str, timeout: float) -> bool:
        response = LocalJsonTransport().get(url, timeout)
        return bool(response)
