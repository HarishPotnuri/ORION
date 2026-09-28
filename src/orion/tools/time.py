"""Offline local date and time tool."""

from __future__ import annotations

import threading
from datetime import datetime
from zoneinfo import ZoneInfo

from ..config import Settings
from ..core.contracts import ToolResult


class LocalTimeTool:
    name = "time"

    def __init__(self, settings: Settings, *, clock=datetime.now):
        self.timezone = ZoneInfo(settings.reminder_timezone)
        self._clock = clock

    def execute(self, arguments, *, timeout: float, cancel_event: threading.Event) -> ToolResult:
        del timeout
        if cancel_event.is_set():
            return ToolResult("Time lookup cancelled.", success=False)
        now = self._clock(self.timezone)
        action = str(arguments.get("action", "time"))
        if action == "date":
            return ToolResult(now.strftime("%A, %d %B %Y") + f" {now.tzname()}")
        return ToolResult(
            f"{now.strftime('%A, %d %B %Y at %I:%M %p')} {now.tzname()}"
        )
