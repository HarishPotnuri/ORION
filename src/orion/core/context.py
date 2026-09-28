"""Relevant, bounded prompt context construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from ..config import Settings
from .contracts import ChatMessage


class ContextBudgetError(ValueError):
    """Raised when required request state cannot fit the configured budget."""


@dataclass(frozen=True, slots=True)
class BuiltContext:
    messages: tuple[ChatMessage, ...]
    estimated_tokens: int
    dropped_turns: int
    dropped_memories: int


class ContextBuilder:
    """Build prompt messages under a conservative character-based token budget."""

    _instructions = (
        "You are ORION, a privacy-first local assistant. Use only the supplied context. "
        "Do not claim tool actions that did not happen. Never reveal or produce hidden "
        "chain-of-thought. For each turn, return exactly one JSON object: "
        '{"type":"tool","name":"registered_tool","arguments":{}} or '
        '{"type":"final","response":"user-facing answer",'
        '"memory_suggestions":[{"kind":"fact|preference|decision",'
        '"content":"stable important user memory","topic":"short topic"}]}. '
        "Use an empty memory_suggestions list unless user disclosed stable, important "
        "information worth remembering. Never recommend transient details, sensitive "
        "credentials, or reasoning. Keep decisions concise. Tool names and arguments "
        "must be supported by the request."
    )

    def __init__(
        self,
        settings: Settings,
        *,
        token_counter: Callable[[str], int] | None = None,
        max_tool_result_tokens: int = 800,
    ):
        self.settings = settings
        self._token_counter = token_counter or self._estimate_tokens
        self.max_tool_result_tokens = max_tool_result_tokens

    def build(
        self,
        request: str,
        *,
        recent_messages: Sequence[ChatMessage] = (),
        summary: str = "",
        memories: Sequence[str] = (),
        tool_results: Sequence[str] = (),
        task_state: str = "",
    ) -> BuiltContext:
        request = request.strip()
        if not request:
            raise ValueError("Current request must not be empty")

        system_parts = [self._instructions]
        if task_state.strip():
            system_parts.append(f"Active task state:\n{task_state.strip()}")
        required_system = "\n\n".join(system_parts)
        required_messages = [
            ChatMessage(role="system", content=required_system),
            ChatMessage(role="user", content=request),
        ]
        required_tokens = self._count_messages(required_messages)
        if required_tokens > self.settings.max_context_tokens:
            raise ContextBudgetError(
                "Required instructions, current request, and task state exceed context budget"
            )

        available = self.settings.max_context_tokens - required_tokens
        selected: list[ChatMessage] = []
        dropped_turns = 0
        dropped_memories = 0

        bounded_results = [
            self._fit_text(result.strip(), self.max_tool_result_tokens, "Tool result: ")
            for result in tool_results
            if result.strip()
        ]
        for result in bounded_results:
            if available <= 0:
                break
            compact_result = self._fit_text(result, available, "")
            if compact_result:
                message = ChatMessage(role="system", content=compact_result)
                selected.append(message)
                available = max(0, available - self._count_messages([message]))

        for memory in memories:
            content = memory.strip()
            if not content:
                continue
            cost = self._count_text(content) + 10
            if cost <= available:
                selected.append(ChatMessage(role="system", content=f"Relevant memory: {content}"))
                available -= cost
            else:
                dropped_memories += 1

        turns = [
            message
            for message in recent_messages
            if message.role in {"user", "assistant"} and message.content.strip()
        ]
        included_turns: list[ChatMessage] = []
        for message in reversed(turns):
            cost = self._count_text(message.content) + 4
            if cost <= available:
                included_turns.append(message)
                available -= cost
            else:
                dropped_turns += 1
        selected.extend(reversed(included_turns))

        if summary.strip() and available > 0:
            compact_summary = self._fit_text(
                summary.strip(), available, "Conversation summary: "
            )
            if compact_summary:
                message = ChatMessage(role="system", content=compact_summary)
                cost = self._count_messages([message])
                selected.append(message)
                available = max(0, available - cost)

        messages = [required_messages[0], *selected, required_messages[1]]
        estimated_tokens = self._count_messages(messages)
        if estimated_tokens > self.settings.max_context_tokens:
            raise ContextBudgetError("Context trimming failed to respect token budget")
        return BuiltContext(
            messages=tuple(messages),
            estimated_tokens=estimated_tokens,
            dropped_turns=dropped_turns,
            dropped_memories=dropped_memories,
        )

    def _fit_text(self, text: str, token_limit: int, prefix: str) -> str:
        if token_limit <= 4:
            return ""
        if self._count_text(prefix + text) + 4 <= token_limit:
            return prefix + text

        low = 0
        high = len(text)
        best = ""
        while low <= high:
            midpoint = (low + high) // 2
            clipped = text[:midpoint].rstrip()
            if midpoint < len(text):
                clipped = clipped.rsplit(" ", 1)[0].rstrip() or clipped
                candidate = prefix + clipped + "..."
            else:
                candidate = prefix + clipped
            if self._count_text(candidate) + 4 <= token_limit:
                best = candidate
                low = midpoint + 1
            else:
                high = midpoint - 1
        return best

    def _count_messages(self, messages: Sequence[ChatMessage]) -> int:
        return sum(self._count_text(message.content) + 4 for message in messages)

    def _count_text(self, text: str) -> int:
        return max(1, self._token_counter(text)) if text else 0

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return math.ceil(len(text.encode("utf-8")) / 4)