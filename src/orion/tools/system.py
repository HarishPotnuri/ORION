"""Offline, bounded operating-system status tool."""

from __future__ import annotations

import os
import platform
import shutil
import threading
from pathlib import Path

from ..core.contracts import ToolResult


class SystemStatusTool:
    name = "system_status"

    def __init__(self, *, data_path: str | Path = "."):
        self.data_path = Path(data_path).expanduser()

    def execute(self, arguments, *, timeout: float, cancel_event: threading.Event) -> ToolResult:
        del arguments, timeout
        if cancel_event.is_set():
            return ToolResult("System status lookup cancelled.", success=False)
        try:
            disk = shutil.disk_usage(self.data_path)
        except OSError:
            disk = None
        details = [
            f"OS: {platform.system()} {platform.release()}",
            f"CPU cores: {os.cpu_count() or 'unknown'}",
        ]
        if disk:
            details.append(f"Disk free: {disk.free // (1024**3)} GiB")
        memory = self._available_memory()
        if memory is not None:
            details.append(f"Available memory: {memory // (1024**2)} MiB")
        load_average = getattr(os, "getloadavg", None)
        if load_average:
            try:
                one_minute, _, _ = load_average()
                details.append(f"Load average: {one_minute:.2f}")
            except OSError:
                pass
        return ToolResult("; ".join(details))

    @staticmethod
    def _available_memory() -> int | None:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            return int(page_size * available_pages)
        except (AttributeError, OSError, ValueError):
            return None
