"""Bounded tool-using controller. Model output never exposes internal reasoning."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from .context import ContextBuilder
from .contracts import (
    ChatMessage,
    LocalModel,
    MemorySuggestion,
    MemoryStore,
    Tool,
    ToolResult,
)

logger = logging.getLogger(__name__)
_MAX_MODEL_RESPONSE_CHARS = 16_384
_MAX_ARGUMENT_DEPTH = 8


@dataclass(frozen=True, slots=True)
class AgentResult:
    response: str
    steps: int
    completed: bool
    reason: str


class AgentController:
    def __init__(
        self,
        settings: Settings,
        model: LocalModel,
        tools: Mapping[str, Tool],
        *,
        context_builder: ContextBuilder | None = None,
        memory: MemoryStore | None = None,
        clock: Callable[[], float] = time.monotonic,
        worker: Callable[[Callable[[], Any], float], Any] | None = None,
        max_tool_result_chars: int = 4000,
    ):
        self.settings = settings
        self.model = model
        self.tools = dict(tools)
        self.context_builder = context_builder or ContextBuilder(settings)
        self.memory = memory
        self._clock = clock
        self._worker = worker or self._run_with_timeout
        self.max_tool_result_chars = max_tool_result_chars

    def run(
        self,
        request: str,
        *,
        recent_messages: Sequence[ChatMessage] = (),
        memories: Sequence[str] = (),
        summary: str = "",
        initial_observations: Sequence[str] = (),
        initial_task_state: str = "No tool action has run.",
        allowed_tool_names: Sequence[str] | None = None,
    ) -> AgentResult:
        if not request.strip():
            return AgentResult("Please say what you need.", 0, False, "empty_request")

        if self.memory is not None:
            recent_messages = self.memory.recent_messages(8)
            memories = self.memory.search(request, limit=5)

        started = self._clock()
        deadline = started + self.settings.max_total_task_time
        tool_results = [self._compact_result(result) for result in initial_observations]
        task_state = initial_task_state
        allowed_tools = set(allowed_tool_names) if allowed_tool_names is not None else set(self.tools)
        seen_actions: set[str] = set()

        for step in range(1, self.settings.max_agent_steps + 1):
            remaining = deadline - self._clock()
            if remaining <= 0:
                return self._limit_result(step - 1, "task_timeout")

            try:
                context = self.context_builder.build(
                    request,
                    recent_messages=recent_messages,
                    summary=summary,
                    memories=memories,
                    tool_results=tool_results,
                    task_state=task_state,
                )
            except ValueError:
                logger.warning("Required task context exceeded configured budget")
                return self._limit_result(step - 1, "context_budget")

            remaining = deadline - self._clock()
            if remaining <= 0:
                return self._limit_result(step - 1, "task_timeout")
            decision = None
            invalid_response = False
            for attempt in range(2):
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return self._limit_result(step - 1, "task_timeout")
                try:
                    raw_response = self._worker(
                        lambda: self.model.complete(
                            context.messages,
                            max_tokens=min(512, self.settings.max_context_tokens),
                            timeout=remaining,
                        ),
                        remaining,
                    )
                except TimeoutError:
                    return self._limit_result(step - 1, "model_timeout")
                except Exception as error:
                    logger.warning(
                        "Local model call failed (%s), attempt %d of 2",
                        type(error).__name__,
                        attempt + 1,
                    )
                    if attempt == 0 and self._clock() < deadline:
                        continue
                    if invalid_response:
                        return self._limit_result(step, "invalid_model_response")
                    return self._limit_result(step - 1, "model_error")

                decision = self._parse_decision(raw_response)
                if decision is not None:
                    break
                invalid_response = True
                if attempt == 0:
                    logger.warning("Local model returned an invalid decision; retrying once")
            if decision is None:
                return self._limit_result(step, "invalid_model_response")
            if decision["type"] == "final":
                response = decision["response"].strip()
                if not response:
                    return self._limit_result(step, "empty_final_response")
                self._save_memory_suggestions(decision["memory_suggestions"])
                return AgentResult(response, step, True, "completed")

            name = decision["name"]
            arguments = decision["arguments"]
            action_key = self._action_key(name, arguments)
            if action_key in seen_actions:
                return self._limit_result(step, "repeated_action")
            seen_actions.add(action_key)

            tool = self.tools.get(name) if name in allowed_tools else None
            if tool is None:
                result = ToolResult(f"Tool '{name}' is not available.", success=False)
            else:
                remaining = deadline - self._clock()
                timeout = min(self.settings.max_tool_execution_time, remaining)
                if timeout <= 0:
                    return self._limit_result(step - 1, "task_timeout")
                cancel_event = threading.Event()
                try:
                    result = self._worker(
                        lambda: tool.execute(
                            arguments,
                            timeout=timeout,
                            cancel_event=cancel_event,
                        ),
                        timeout,
                    )
                    if not isinstance(result, ToolResult):
                        result = ToolResult(str(result), success=True)
                except TimeoutError:
                    cancel_event.set()
                    logger.warning("Tool %s exceeded its execution limit", name)
                    return self._limit_result(step, "tool_timeout")
                except Exception as error:
                    logger.warning("Tool %s failed (%s)", name, type(error).__name__)
                    result = ToolResult("Tool failed.", success=False)

            compact_result = self._compact_result(result.content)
            status = "succeeded" if result.success else "failed"
            tool_results.append(f"Tool {name} {status}: {compact_result}")
            task_state = f"Step {step}: called {name}; result {status}."

        return self._limit_result(self.settings.max_agent_steps, "max_steps")

    def _parse_decision(self, raw_response: str) -> dict[str, Any] | None:
        if not isinstance(raw_response, str) or len(raw_response) > _MAX_MODEL_RESPONSE_CHARS:
            return None
        try:
            decision = json.loads(raw_response)
        except json.JSONDecodeError:
            return None
        if not isinstance(decision, dict):
            return None
        if decision.get("type") == "final" and isinstance(decision.get("response"), str):
            return {
                "type": "final",
                "response": decision["response"],
                "memory_suggestions": self._parse_memory_suggestions(
                    decision.get("memory_suggestions", [])
                ),
            }
        if decision.get("type") != "tool":
            return None
        name = decision.get("name")
        arguments = decision.get("arguments")
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 100
            or not isinstance(arguments, dict)
            or self._depth(arguments) > _MAX_ARGUMENT_DEPTH
        ):
            return None
        return {"type": "tool", "name": name, "arguments": arguments}

    def _parse_memory_suggestions(self, value: object) -> tuple[MemorySuggestion, ...]:
        if not isinstance(value, list):
            return ()
        suggestions: list[MemorySuggestion] = []
        for item in value[:5]:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind")
            content = item.get("content")
            topic = item.get("topic", "")
            if (
                not isinstance(kind, str)
                or kind not in {"fact", "preference", "decision"}
                or not isinstance(content, str)
                or not content.strip()
                or not isinstance(topic, str)
            ):
                continue
            suggestions.append(
                MemorySuggestion(kind=kind, content=content, topic=topic)
            )
        return tuple(suggestions)

    def _save_memory_suggestions(
        self, suggestions: Sequence[MemorySuggestion]
    ) -> None:
        save_recommendation = getattr(self.memory, "save_recommendation", None)
        if save_recommendation is None:
            return
        for suggestion in suggestions:
            try:
                save_recommendation(suggestion, important=True)
            except Exception as error:
                logger.warning("Memory recommendation rejected (%s)", type(error).__name__)

    def _compact_result(self, content: str) -> str:
        compact = " ".join(str(content).split())
        if len(compact) <= self.max_tool_result_chars:
            return compact
        return compact[: self.max_tool_result_chars - 3].rsplit(" ", 1)[0] + "..."

    @staticmethod
    def _action_key(name: str, arguments: Mapping[str, object]) -> str:
        return json.dumps([name, arguments], sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _depth(value: object, current: int = 0) -> int:
        if isinstance(value, dict):
            return max(
                (AgentController._depth(item, current + 1) for item in value.values()),
                default=current,
            )
        if isinstance(value, list):
            return max(
                (AgentController._depth(item, current + 1) for item in value),
                default=current,
            )
        return current

    @staticmethod
    def _limit_result(steps: int, reason: str) -> AgentResult:
        return AgentResult(
            "I couldn't complete that request within my local limits. "
            "You can narrow the request or try again.",
            steps,
            False,
            reason,
        )

    @staticmethod
    def _run_with_timeout(operation: Callable[[], Any], timeout: float) -> Any:
        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                result_queue.put((True, operation()))
            except BaseException as error:
                result_queue.put((False, error))

        thread = threading.Thread(target=run, name="orion-bounded-call", daemon=True)
        thread.start()
        thread.join(max(0.0, timeout))
        if thread.is_alive():
            raise TimeoutError("Operation exceeded its time limit")
        succeeded, value = result_queue.get_nowait()
        if succeeded:
            return value
        raise value