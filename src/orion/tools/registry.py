"""Explicit registry for locally available and approved factual tools."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from types import MappingProxyType

from ..core.contracts import Tool, ToolResult


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()):
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        name = getattr(tool, "name", "")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Registered tools need a non-empty name")
        if name in self._tools:
            raise ValueError(f"Tool is already registered: {name}")
        self._tools[name] = tool

    @property
    def tools(self) -> Mapping[str, Tool]:
        return MappingProxyType(self._tools)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def execute(
        self,
        name: str,
        arguments: Mapping[str, object],
        *,
        timeout: float,
        cancel_event: threading.Event,
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"Tool '{name}' is not available.", success=False)
        return tool.execute(arguments, timeout=timeout, cancel_event=cancel_event)
