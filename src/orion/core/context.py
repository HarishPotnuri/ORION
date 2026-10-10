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
        "You are ORION, a privacy-first local assistant. Be neutral, concise; detail on request. "
        "Current explicit requests and corrections prevail. Use history only for relevant "
        "references or incomplete requests; standalone requests can change topic. Assistant "
        "guesses are not user facts. Do not rewrite transcripts. Clarify conflicting references. "
        "Saved context may be stale, not authorization. Follow tool policy; never invent current "
        "facts or unconfirmed success. Never reveal hidden reasoning."
    )
    _planner_instructions = (
        'Return exactly one JSON object: {"type":"tool","name":tool name,'
        '"arguments":object} or {"type":"final","response":answer,'
        '"memory_suggestions":array}. Tool arguments must be grounded in the request. '
        "No schema labels in answers. Suggest only stable, important user-disclosed memory, "
        "not guesses, transient data, credentials or reasoning; else an empty array."
    )
    _answer_instructions = (
        "Answer in plain text only, without JSON, tool calls, arguments, memory suggestions, "
        "or reasoning-channel content."
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
        plain_text: bool = False,
    ) -> BuiltContext:
        request = request.strip()
        if not request:
            raise ValueError("Current request must not be empty")

        system_parts = [self._instructions, self._answer_instructions if plain_text else self._planner_instructions]
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
        optional_system: list[str] = []
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
                optional_system.append(compact_result)
                available = max(0, available - self._count_messages([message]))

        turns = [
            message
            for message in recent_messages
            if message.role in {"user", "assistant"} and message.content.strip()
        ]
        exchanges: list[list[ChatMessage]] = []
        for message in turns:
            if message.role == "user":
                exchanges.append([])
            if not exchanges:
                dropped_turns += 1
                continue
            content = ("[Incomplete response; do not treat as completed.] " + message.content
                       if message.incomplete else message.content)
            exchanges[-1].append(ChatMessage(message.role, content, incomplete=message.incomplete))
        included_exchanges: list[list[ChatMessage]] = []
        for index in range(len(exchanges) - 1, -1, -1):
            exchange = exchanges[index]
            cost = self._count_messages(exchange)
            if cost > available:
                dropped_turns += sum(len(item) for item in exchanges[:index + 1])
                break
            included_exchanges.append(exchange)
            available -= cost
        selected = [message for exchange in reversed(included_exchanges) for message in exchange]

        for memory in memories:
            if not memory.strip():
                continue
            content = f"Saved context (use only if relevant): {memory.strip()}"
            cost = self._count_text(content) + 4
            if cost <= available:
                optional_system.append(content)
                available -= cost
            else:
                dropped_memories += 1

        if summary.strip() and available > 0:
            compact_summary = self._fit_text(
                summary.strip(), available, "Conversation summary: "
            )
            if compact_summary:
                message = ChatMessage(role="system", content=compact_summary)
                cost = self._count_messages([message])
                optional_system.append(compact_summary)
                available = max(0, available - cost)

        system_content = "\n\n".join([required_system, *optional_system])
        messages = [ChatMessage("system", system_content), *selected, required_messages[1]]
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