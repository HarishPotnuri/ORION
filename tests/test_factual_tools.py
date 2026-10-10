import json
import io
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


@pytest.mark.parametrize("category, endpoint", [
    ("general", "https://www.bing.com/search"),
    ("news", "https://news.google.com/rss/search"),
])
def test_default_search_and_news_use_public_rss(category, endpoint):
    transport = Mock()
    transport.get_factual_rss.return_value = {"results": [{
        "title": "Science &amp; technology",
        "content": '<a href="https://publisher.test/story">Verified report</a>',
        "url": "https://publisher.test/story",
        "published": "Sat, 10 Oct 2026 10:00:00 GMT",
    }]}
    result = execute(SearxngSearchTool(Settings(), transport=transport),
                     {"query": "science", "category": category})
    assert result.success
    item = json.loads(result.content)["results"][0]
    assert item["title"] == "Science & technology"
    assert item["snippet"] == "Verified report"
    assert item["published"] == "Sat, 10 Oct 2026 10:00:00 GMT"
    assert transport.get_factual_rss.call_args.args[0] == endpoint
    transport.get_factual_json.assert_not_called()


def test_rss_transport_parses_source_and_publication_date():
    opener = Mock()
    opener.open.return_value = io.BytesIO(b'''<rss><channel><item>
        <title>Science</title><description>A report</description>
        <link>https://publisher.test/story</link><pubDate>Today</pubDate>
        </item></channel></rss>''')
    response = LocalJsonTransport(opener=opener).get_factual_rss(
        "https://news.google.com/rss/search", params={"q": "science"},
        allowed_hosts={"news.google.com"}, timeout=2,
    )
    assert response["results"] == [{"title": "Science", "content": "A report",
                                     "url": "https://publisher.test/story", "published": "Today"}]
    assert "q=science" in opener.open.call_args.args[0].full_url


@pytest.mark.parametrize(
    "payload", [b"not XML", b"<html/>", b"<rss/>", b"x" * 1_048_577],
    ids=["invalid-xml", "not-rss", "missing-channel", "oversized"],
)
def test_rss_transport_rejects_invalid_or_oversized_feeds(payload):
    opener = Mock()
    opener.open.return_value = io.BytesIO(payload)
    with pytest.raises(RuntimeError):
        LocalJsonTransport(opener=opener).get_factual_rss(
            "https://news.google.com/rss/search", params=None,
            allowed_hosts={"news.google.com"}, timeout=2,
        )


def test_rss_transport_blocks_unapproved_hosts_before_network_access():
    opener = Mock()
    with pytest.raises(NetworkPolicyError):
        LocalJsonTransport(opener=opener).get_factual_rss(
            "https://unapproved.test/feed", params=None,
            allowed_hosts={"news.google.com"}, timeout=2,
        )
    opener.open.assert_not_called()