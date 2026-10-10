"""Public RSS search and news with an optional allowlisted SearXNG adapter."""

from __future__ import annotations

import json
import re
import threading
import time
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

from ..config import Settings
from ..core.contracts import ToolResult
from ..core.policy import NetworkPolicy
from ..llm.runtime import LocalJsonTransport
from ._factual import factual_failure


class _FeedText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


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
        if not self.settings.online_tools_enabled:
            return ToolResult(
                "Web search is disabled in offline mode. Enable online tools to use it.",
                success=False,
            )
        query = self._query(str(arguments.get("query", "")))
        if not query:
            return ToolResult("Please specify what to search for.", success=False)
        if cancel_event.is_set():
            return ToolResult("Web search cancelled.", success=False)
        if timeout <= 0:
            return ToolResult("Web search timed out.", success=False)
        category = arguments.get("category", "general")
        if category not in {"general", "news"}:
            return ToolResult("Search category must be general or news.", success=False)
        deadline = time.monotonic() + timeout

        provider = "SearXNG" if self.settings.searxng_base_url else "Public RSS"
        endpoint = self.settings.searxng_base_url
        allowed_hosts = set(self.settings.searxng_allowed_hosts)
        if endpoint:
            endpoint = f"{endpoint.rstrip('/')}/search"
            params = {"q": query, "format": "json", "categories": category}
            fetch = self.transport.get_factual_json
        else:
            allowed_hosts = {"www.bing.com", "news.google.com"}
            endpoint = "https://www.bing.com/search"
            params = {"q": query, "format": "rss"}
            if category == "news":
                endpoint = "https://news.google.com/rss/search"
                params = {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
            fetch = self.transport.get_factual_rss
        try:
            self.policy.validate_factual_url(endpoint, allowed_hosts)
            response = fetch(
                endpoint,
                params=params,
                allowed_hosts=allowed_hosts,
                timeout=timeout,
                policy=self.policy,
            )
            if cancel_event.is_set():
                return ToolResult("Web search cancelled.", success=False)
            if time.monotonic() >= deadline:
                raise TimeoutError
            if not isinstance(response, dict):
                raise ValueError("Invalid search response")
            raw_results = response.get("results")
            if not isinstance(raw_results, list):
                return ToolResult("Search provider returned no usable results.", success=False)
            results = []
            for item in raw_results[: self.result_limit]:
                if not isinstance(item, dict):
                    continue
                title = self._compact(item.get("title", ""))
                snippet = self._compact(item.get("content", ""))
                url = item.get("url", "")
                if not isinstance(url, str) or len(url) > 2048:
                    continue
                source = urlparse(url)
                if source.scheme not in {"http", "https"} or not source.hostname or source.username or source.password:
                    continue
                if title or snippet:
                    result = {"title": title, "snippet": snippet, "source": url}
                    if item.get("published"):
                        result["published"] = self._compact(item["published"])
                    results.append(result)
            if not results:
                return ToolResult("Search returned no results.", success=False)
            return ToolResult(
                json.dumps({"query": query, "results": results}, ensure_ascii=False, separators=(",", ":"))
            )
        except Exception as error:
            return factual_failure("Web search", provider, error)

    def _compact(self, value: object) -> str:
        parser = _FeedText()
        parser.feed(value if isinstance(value, str) else "")
        text = " ".join(" ".join(parser.parts).split())
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
