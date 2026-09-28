"""Construct ORION's small built-in tool set from central settings."""

from __future__ import annotations

from ..config import Settings
from ..core.policy import NetworkPolicy
from ..llm.runtime import LocalJsonTransport
from .registry import ToolRegistry
from .system import SystemStatusTool
from .time import LocalTimeTool
from .weather import OpenMeteoWeatherTool
from .web_search import SearxngSearchTool


def create_default_tools(
    settings: Settings,
    *,
    transport: LocalJsonTransport | None = None,
    policy: NetworkPolicy | None = None,
) -> ToolRegistry:
    shared_transport = transport or LocalJsonTransport()
    shared_policy = policy or NetworkPolicy()
    return ToolRegistry(
        (
            LocalTimeTool(settings),
            SystemStatusTool(data_path=settings.database_path.parent),
            OpenMeteoWeatherTool(
                settings, transport=shared_transport, policy=shared_policy
            ),
            SearxngSearchTool(
                settings, transport=shared_transport, policy=shared_policy
            ),
        )
    )
