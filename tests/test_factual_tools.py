import json
import threading
from dataclasses import replace
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest
from orion.config import Settings
from orion.core.policy import NetworkPolicyError
from orion.llm.runtime import LocalJsonTransport
from orion.tools.weather import OpenMeteoWeatherTool
from orion.tools.web_search import SearxngSearchTool


def search_settings():
    return replace(
        Settings(online_tools_enabled=True),
        searxng_base_url="https://search.example.test",
        searxng_allowed_hosts=("search.example.test",),
    )


def execute(tool, arguments=None, timeout=2):
    return tool.execute(
        arguments or {"query": "weather in Pune", "location": "Pune"},
        timeout=timeout,
        cancel_event=threading.Event(),
    )


def test_weather_rejects_empty_forecast():
    transport = Mock()
    transport.get_factual_json.side_effect = [
        {"results": [{"name": "Pune", "latitude": 18.52, "longitude": 73.85}]},
        {"current": {}, "daily": {}},
    ]
    result = execute(OpenMeteoWeatherTool(Settings(online_tools_enabled=True), transport=transport))
    assert not result.success
    assert "invalid" in result.content.lower() or "incomplete" in result.content.lower()


def test_search_rejects_ungrounded_results():
    transport = Mock()
    transport.get_factual_json.return_value = {"results": [{"title": "Unverified headline"}]}
    result = execute(SearxngSearchTool(search_settings(), transport=transport))
    assert not result.success


@pytest.mark.parametrize("tool_class", [OpenMeteoWeatherTool, SearxngSearchTool])
def test_provider_timeout_is_actionable(tool_class):
    transport = Mock()
    transport.get_factual_json.side_effect = TimeoutError("secret provider details")
    result = execute(tool_class(search_settings(), transport=transport))
    assert not result.success
    assert "timed out" in result.content.lower()
    assert "secret" not in result.content


def test_search_news_category_uses_same_provider():
    transport = Mock()
    transport.get_factual_json.return_value = {
        "results": [{
            "title": "Verified headline",
            "content": "Reported fact",
            "url": "https://publisher.test/story",
        }]
    }
    result = execute(
        SearxngSearchTool(search_settings(), transport=transport),
        {"query": "science", "category": "news"},
    )

    assert result.success
    assert json.loads(result.content)["results"][0]["title"] == "Verified headline"
    transport.get_factual_json.assert_called_once()
    assert transport.get_factual_json.call_args.kwargs["params"]["categories"] == "news"