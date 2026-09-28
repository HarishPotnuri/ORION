"""Interfaces between ORION core and local service adapters."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    content: str
    success: bool = True


@dataclass(frozen=True, slots=True)
class MemorySuggestion:
    kind: Literal["fact", "preference", "decision"]
    content: str
    topic: str = ""


class AudioInput(Protocol):
    def next_utterance(self) -> str:
        """Return one transcribed utterance without retaining its audio."""


class SpeechOutput(Protocol):
    def speak(self, text: str) -> None:
        """Speak text using a local speech engine."""


class LocalModel(Protocol):
    def complete(
        self, messages: Sequence[ChatMessage], *, max_tokens: int, timeout: float
    ) -> str:
        """Return a completion from the configured local model."""


class Tool(Protocol):
    name: str

    def execute(
        self,
        arguments: Mapping[str, object],
        *,
        timeout: float,
        cancel_event: threading.Event,
    ) -> ToolResult:
        """Execute one tool and honor its remaining execution time."""


class MemoryStore(Protocol):
    def recent_messages(self, limit: int) -> Sequence[ChatMessage]:
        """Return bounded short-term conversation context."""

    def search(self, query: str, *, limit: int) -> Sequence[str]:
        """Return relevant retained memories."""

    def save_recommendation(
        self, suggestion: MemorySuggestion, *, important: bool
    ) -> object:
        """Save a locally recommended durable memory after policy checks."""


class ReminderService(Protocol):
    def create(self, reminder: Mapping[str, object]) -> str:
        """Persist a reminder and return its identifier."""

    def list(self, *, status: str | None = None) -> Sequence[Mapping[str, object]]:
        """Return reminders filtered by optional status."""

    def cancel(self, reminder_id: str) -> bool:
        """Cancel a pending reminder and report whether it existed."""


class HardwareController(Protocol):
    def set_status(self, status: str) -> None:
        """Set the configured physical status indicator."""