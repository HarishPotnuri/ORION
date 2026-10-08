"""Coordinate volatile turns, durable retrieval, Incognito, and memory writes."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Sequence

from ..config import Settings
from ..core.contracts import ChatMessage, MemorySuggestion
from .policy import (
    MemoryCandidate,
    MemoryKind,
    MemorySaveResult,
    MemorySaveStatus,
    MemoryWritePolicy,
)
from .store import MemoryRecord, SQLiteMemoryStore


@dataclass(frozen=True, slots=True)
class EpisodeDeleteResult:
    status: str
    matches: tuple[MemoryRecord, ...] = ()


class VolatileConversation:
    """Bounded recent turns held only in process memory."""

    def __init__(self, max_messages: int = 16):
        if max_messages < 1:
            raise ValueError("max_messages must be positive")
        self._messages: deque[ChatMessage] = deque(maxlen=max_messages)

    def add_message(self, message: ChatMessage) -> None:
        if message.role not in {"user", "assistant"}:
            raise ValueError("Conversation buffer accepts only user and assistant messages")
        self._messages.append(message)

    def add_turn(
        self, user_text: str, assistant_text: str, *, assistant_incomplete: bool = False
    ) -> None:
        if user_text.strip():
            self.add_message(ChatMessage(role="user", content=user_text.strip()))
        if assistant_text.strip():
            self.add_message(
                ChatMessage(
                    role="assistant",
                    content=assistant_text.strip(),
                    incomplete=assistant_incomplete,
                )
            )

    def recent_messages(self, limit: int) -> Sequence[ChatMessage]:
        if limit < 1:
            return ()
        return tuple(self._messages)[-limit:]

    def clear(self) -> None:
        self._messages.clear()


class MemoryManager:
    """Enforce write policy while exposing bounded recent context and FTS retrieval."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        *,
        max_recent_messages: int = 16,
        max_search_results: int = 5,
        policy: MemoryWritePolicy | None = None,
    ):
        if max_search_results < 1:
            raise ValueError("max_search_results must be positive")
        self.store = store
        self.conversation = VolatileConversation(max_recent_messages)
        self.max_search_results = max_search_results
        self.policy = policy or MemoryWritePolicy()
        self._incognito = False

    @classmethod
    def from_settings(cls, settings: Settings) -> MemoryManager:
        return cls(
            SQLiteMemoryStore(settings.database_path),
            max_recent_messages=settings.memory_recent_message_limit,
            max_search_results=settings.memory_search_limit,
        )

    @property
    def incognito(self) -> bool:
        return self._incognito

    def set_incognito(self, enabled: bool) -> None:
        self._incognito = bool(enabled)

    def add_turn(
        self, user_text: str, assistant_text: str, *, assistant_incomplete: bool = False
    ) -> None:
        self.conversation.add_turn(
            user_text, assistant_text, assistant_incomplete=assistant_incomplete
        )

    def recent_messages(self, limit: int) -> Sequence[ChatMessage]:
        return self.conversation.recent_messages(limit)

    def clear_conversation(self) -> None:
        self.conversation.clear()

    def search(self, query: str, *, limit: int) -> Sequence[str]:
        bounded_limit = min(limit, self.max_search_results)
        if bounded_limit < 1:
            return ()
        return tuple(
            f"[{record.kind}; {record.topic or 'general'}; id={record.id}] {record.content}"
            for record in self.store.search(query, limit=bounded_limit)
        )

    def save_explicit(self, candidate: MemoryCandidate) -> MemorySaveResult:
        return self._save(candidate, source="explicit", explicit_request=True)

    def save_explicit_request(self, request: str) -> MemorySaveResult:
        content = re.sub(
            r"^\s*(?:please\s+)?remember(?:\s+this|\s+that|\s+me)?\s*[:,]?\s*",
            "",
            request,
            flags=re.IGNORECASE,
        ).strip()
        if not content:
            return MemorySaveResult(
                MemorySaveStatus.REJECTED_EMPTY,
                explanation="No memory content followed the remember request.",
            )
        lowered = content.casefold()
        if re.search(r"\b(prefer|preference|like|love|dislike|hate)\b", lowered):
            kind = MemoryKind.PREFERENCE
        elif re.search(r"\b(decided|decision|chose|chosen|will use|agreed to)\b", lowered):
            kind = MemoryKind.DECISION
        else:
            kind = MemoryKind.FACT
        return self.save_explicit(MemoryCandidate(kind, content))

    def save_recommendation(
        self, candidate: MemorySuggestion | MemoryCandidate, *, important: bool
    ) -> MemorySaveResult:
        if isinstance(candidate, MemorySuggestion):
            candidate = MemoryCandidate(
                kind=candidate.kind,
                content=candidate.content,
                topic=candidate.topic,
            )
        return self._save(candidate, source="model_recommendation", important=important)

    def save_episode(
        self,
        summary: str,
        *,
        topic: str = "",
        explicitly_requested: bool = False,
        important: bool = False,
    ) -> MemorySaveResult:
        candidate = MemoryCandidate(kind=MemoryKind.EPISODE, content=summary, topic=topic)
        return self._save(
            candidate,
            source="episode_summary",
            explicit_request=explicitly_requested,
            important=important,
        )

    def _save(
        self,
        candidate: MemoryCandidate,
        *,
        source: str,
        explicit_request: bool = False,
        important: bool = False,
    ) -> MemorySaveResult:
        decision = self.policy.evaluate(
            candidate,
            incognito=self._incognito,
            explicit_request=explicit_request,
            important=important,
        )
        if decision.status != MemorySaveStatus.SAVED:
            return decision
        record = self.store.save(
            candidate.kind.value,
            candidate.content,
            topic=candidate.topic,
            source=source,
        )
        return MemorySaveResult(MemorySaveStatus.SAVED, memory_id=record.id)

    def list_memories(
        self, *, kind: str | None = None, limit: int = 50
    ) -> Sequence[MemoryRecord]:
        return self.store.list_memories(kind=kind, limit=limit)

    def delete_episode(self, memory_id: str) -> bool:
        return self.store.delete_episode(memory_id)

    def delete_episode_by_query(
        self, query: str, *, limit: int = 10
    ) -> EpisodeDeleteResult:
        result = self.find_episode_by_query(query, limit=limit)
        if result.status != "found":
            return result
        deleted = self.store.delete_episode(result.matches[0].id)
        return EpisodeDeleteResult("deleted" if deleted else "not_found", result.matches)

    def find_episode_by_query(
        self, query: str, *, limit: int = 10
    ) -> EpisodeDeleteResult:
        matches = tuple(self.store.search(query, kind="episode", limit=limit))
        if not matches:
            return EpisodeDeleteResult("not_found")
        if len(matches) > 1:
            return EpisodeDeleteResult("ambiguous", matches)
        return EpisodeDeleteResult("found", matches)

    def close(self) -> None:
        self.conversation.clear()
        self.store.close()
