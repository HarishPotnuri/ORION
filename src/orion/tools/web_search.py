"""Allowlisted SearXNG factual search adapter."""

from __future__ import annotations

import json
import re
import threading
from typing import Any
from urllib.parse import urlparse

from ..config import Settings
from ..core.contracts import ToolResult
from ..core.policy import NetworkPolicy
from ..llm.runtime import LocalJsonTransport


class SearxngSearchTool:
    name = "web_search"

    def __init__(
        self,
        settings: Settings,
        *,
        transport: Any | None = None,
        policy: NetworkPolicy | None = None,
        result_limit: int = 5,
        max_field_chars: int = 600,
    ):
        self.settings = settings
        self.transport = transport or LocalJsonTransport()
        self.policy = policy or NetworkPolicy()
        self.result_limit = result_limit
        self.max_field_chars = max_field_chars

    def execute(self, arguments, *, timeout: float, cancel_event: threading.Event) -> ToolResult:
        if not self.settings.searxng_base_url:
            return ToolResult(
                "Web search is not configured. Configure an HTTPS SearXNG instance and exact allowed host.",
                success=False,
            )
        query = self._query(str(arguments.get("query", "")))
        if not query:
            return ToolResult("Please specify what to search for.", success=False)
        if cancel_event.is_set():
            return ToolResult("Web search cancelled.", success=False)
        if timeout <= 0:
            return ToolResult("Web search timed out.", success=False)

        parsed = urlparse(self.settings.searxng_base_url)
        allowed_hosts = set(self.settings.searxng_allowed_hosts)
        if parsed.hostname is None or parsed.hostname.lower() not in allowed_hosts:
            return ToolResult("Web search is blocked by the configured host policy.", success=False)
        try:
            response = self.transport.get_factual_json(
                f"{self.settings.searxng_base_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": "news" if arguments.get("category") == "news" else "general",
                },
                allowed_hosts=allowed_hosts,
                timeout=timeout,
                policy=self.policy,
            )
            raw_results = response.get("results")
            if not isinstance(raw_results, list):
                return ToolResult("Search provider returned no usable results.", success=False)
            results = []
            for item in raw_results[: self.result_limit]:
                if not isinstance(item, dict):
                    continue
                title = self._compact(item.get("title", ""))
                snippet = self._compact(item.get("content", ""))
                url = self._compact(item.get("url", ""))
                if title or snippet:
                    results.append({"title": title, "snippet": snippet, "source": url})
            if not results:
                return ToolResult("Search returned no results.", success=False)
            return ToolResult(
                json.dumps({"query": query, "results": results}, ensure_ascii=False, separators=(",", ":"))
            )
        except Exception as error:
            return ToolResult(f"Web search failed ({type(error).__name__}).", success=False)

    def _compact(self, value: object) -> str:
        text = " ".join(str(value).split())
        if len(text) <= self.max_field_chars:
            return text
        return text[: self.max_field_chars - 3].rsplit(" ", 1)[0] + "..."

    @staticmethod
    def _query(request: str) -> str:
        query = " ".join(request.split()).strip()
        query = re.sub(
            r"^(?:please\s+)?(?:search(?: the web)?(?: for)?|look\s+up|find(?:\s+out)?|browse(?:\s+for)?)\s+",
            "",
            query,
            flags=re.IGNORECASE,
        )
        return query.strip(" ?.!\t")[:300]
