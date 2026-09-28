"""Deterministic gates for explicit and model-recommended memory writes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class MemoryKind(StrEnum):
    EPISODE = "episode"
    FACT = "fact"
    PREFERENCE = "preference"
    DECISION = "decision"


class MemorySaveStatus(StrEnum):
    SAVED = "saved"
    BLOCKED_INCOGNITO = "blocked_incognito"
    REJECTED_SENSITIVE = "rejected_sensitive"
    REJECTED_KIND = "rejected_kind"
    REJECTED_NOT_IMPORTANT = "rejected_not_important"
    REJECTED_EMPTY = "rejected_empty"
    REJECTED_TOO_LONG = "rejected_too_long"


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    kind: MemoryKind
    content: str
    topic: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", MemoryKind(self.kind))


@dataclass(frozen=True, slots=True)
class MemorySaveResult:
    status: MemorySaveStatus
    memory_id: str | None = None
    explanation: str = ""


class MemoryWritePolicy:
    _sensitive_patterns = (
        re.compile(r"\b(password|passphrase|recovery code|one[- ]time code|pin code)\b", re.I),
        re.compile(r"\b(api[_ -]?key|access[_ -]?token|secret key|private key)\b", re.I),
        re.compile(r"\b(credit card|debit card|card number|cvv|cvc|bank account number)\b", re.I),
        re.compile(r"\b(social security number|national id|passport number|tax id)\b", re.I),
    )
    _durable_kinds = {MemoryKind.FACT, MemoryKind.PREFERENCE, MemoryKind.DECISION}
    _max_durable_chars = 2000
    _max_episode_chars = 4000

    def evaluate(
        self,
        candidate: MemoryCandidate,
        *,
        incognito: bool,
        explicit_request: bool = False,
        important: bool = False,
    ) -> MemorySaveResult:
        if incognito:
            return MemorySaveResult(
                MemorySaveStatus.BLOCKED_INCOGNITO,
                explanation="Incognito mode blocks every memory write.",
            )
        content = " ".join(candidate.content.split())
        if not content:
            return MemorySaveResult(MemorySaveStatus.REJECTED_EMPTY)
        if any(pattern.search(content) for pattern in self._sensitive_patterns):
            return MemorySaveResult(
                MemorySaveStatus.REJECTED_SENSITIVE,
                explanation="Credential or financial identifiers are not stored in memory.",
            )
        if candidate.kind == MemoryKind.EPISODE:
            maximum = self._max_episode_chars
            if not (explicit_request or important):
                return MemorySaveResult(MemorySaveStatus.REJECTED_NOT_IMPORTANT)
        elif candidate.kind in self._durable_kinds:
            maximum = self._max_durable_chars
            if not (explicit_request or important):
                return MemorySaveResult(MemorySaveStatus.REJECTED_NOT_IMPORTANT)
        else:
            return MemorySaveResult(MemorySaveStatus.REJECTED_KIND)
        if len(content) > maximum:
            return MemorySaveResult(MemorySaveStatus.REJECTED_TOO_LONG)
        return MemorySaveResult(MemorySaveStatus.SAVED)
