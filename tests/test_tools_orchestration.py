import json
import io
import tempfile
import threading
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from orion.config import Settings
from orion.core.capabilities import CapabilityPolicy
from orion.core.contracts import ChatMessage, ModelStreamEvent, ToolResult
from orion.core.orchestrator import RequestOrchestrator
from orion.core.policy import NetworkPolicyError
from orion.tools.factory import create_default_tools
from orion.tools.registry import ToolRegistry
from orion.tools.system import SystemStatusTool
from orion.tools.time import LocalTimeTool
from orion.tools.weather import OpenMeteoWeatherTool
from orion.tools.web_search import SearxngSearchTool
from orion.memory.manager import MemoryManager
from orion.memory.store import SQLiteMemoryStore
from orion.reminders.manager import ReminderManager
from orion.app import _LazyLocalModel, _run_request


class ScriptedModel:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, messages, *, max_tokens, timeout):
        self.calls.append((tuple(messages), max_tokens, timeout))
        return self.response


class ScriptedStreamingModel:
    def __init__(self, *events):
        self.events = events
        self.stream_calls = []

    def complete(self, *_args, **_kwargs):
        raise AssertionError("Tool-free stream must not call complete()")

    def stream(self, messages, *, max_tokens, timeout, cancel_event):
        self.stream_calls.append(tuple(messages))
        yield from self.events


class PlanningModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, *, max_tokens, timeout):
        self.calls.append((tuple(messages), max_tokens, timeout))
        return self.responses.pop(0)


class FakeTool:
    def __init__(self, name, result):
        self.name = name
        self.result = result
        self.calls = []

    def execute(self, arguments, *, timeout, cancel_event):
        self.calls.append((dict(arguments), timeout, cancel_event.is_set()))
        return self.result


class FakeFactualTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get_factual_json(self, url, *, params, allowed_hosts, timeout, policy):
        self.calls.append((url, params, allowed_hosts, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class CapabilityPolicyTests(unittest.TestCase):
    def test_rejects_broad_long_horizon_analysis(self):
        assessment = CapabilityPolicy().assess(
            "Analyze 10 years of global semiconductor trends and predict which company dominates the next five years."
        )

        self.assertFalse(assessment.supported)
        self.assertEqual(assessment.reason, "broad_long_horizon_analysis")
        self.assertIn("can't reliably", assessment.response)

    def test_allows_narrow_comparison(self):
        self.assertTrue(
            CapabilityPolicy().assess("Compare these two benchmark results").supported
        )


class LocalToolTests(unittest.TestCase):
    def test_lazy_local_model_forwards_stream_to_initialized_runtime(self):
        runtime = Mock()
        runtime.stream.return_value = iter(
            [ModelStreamEvent(kind="completed", finish_reason="stop")]
        )
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.touch()
            with patch("orion.llm.provider.LlamaCppServer") as server_type, patch(
                "orion.llm.provider.create_local_model", return_value=runtime
            ) as create_runtime:
                model = _LazyLocalModel(replace(Settings(), model_path=model_path))

                events = list(
                    model.stream(
                        [ChatMessage("user", "Hello")],
                        max_tokens=16,
                        timeout=1,
                        cancel_event=threading.Event(),
                    )
                )
                model.close()

        self.assertEqual(events[0].kind, "completed")
        server_type.return_value.start.assert_called_once()
        server_type.return_value.stop.assert_called_once()
        create_runtime.assert_called_once()
        runtime.stream.assert_called_once()

    def test_stream_failure_records_only_delivered_text_as_incomplete(self):
        request = "Explain Kafka briefly"
        memory = MemoryManager(SQLiteMemoryStore(":memory:"))
        model = ScriptedStreamingModel(
            ModelStreamEvent(kind="text", text="Kafka moves events."),
            ModelStreamEvent(kind="failed", error="stream_failed"),
        )
        orchestrator = RequestOrchestrator(
            Settings(), model, ToolRegistry([]), memory=memory
        )

        events = list(orchestrator.stream(request))
        messages = memory.recent_messages(8)
        assistant = next(message for message in messages if message.role == "assistant")

        self.assertEqual(events[-1].kind, "failed")
        self.assertEqual(assistant.content, "Kafka moves events.")
        self.assertTrue(assistant.incomplete)
        self.assertEqual(sum(message.role == "assistant" for message in messages), 1)
        self.assertEqual(len(model.stream_calls), 1)
        memory.close()

    def test_completed_stream_records_full_answer_once(self):
        request = "Explain Kafka briefly"
        memory = MemoryManager(SQLiteMemoryStore(":memory:"))
        model = ScriptedStreamingModel(
            ModelStreamEvent(kind="text", text="Kafka moves events."),
            ModelStreamEvent(kind="completed", finish_reason="stop"),
        )
        orchestrator = RequestOrchestrator(
            Settings(), model, ToolRegistry([]), memory=memory
        )

        events = list(orchestrator.stream(request))
        messages = memory.recent_messages(8)
        assistant = next(message for message in messages if message.role == "assistant")

        self.assertEqual(events[-1].kind, "completed")
        self.assertEqual(assistant.content, "Kafka moves events.")
        self.assertFalse(assistant.incomplete)
        self.assertEqual(sum(message.role == "assistant" for message in messages), 1)
        memory.close()

    def test_interrupted_voice_stream_records_only_reported_delivery(self):
        request = "Explain Kafka briefly"
        memory = MemoryManager(SQLiteMemoryStore(":memory:"))
        model = ScriptedStreamingModel(
            ModelStreamEvent(kind="text", text="Kafka moves events. Unsaid tail."),
            ModelStreamEvent(kind="completed", finish_reason="stop"),
        )
        orchestrator = RequestOrchestrator(
            Settings(), model, ToolRegistry([]), memory=memory
        )
        delivered_text = []
        cancel_event = threading.Event()
        stream = orchestrator.stream(
            request,
            cancel_event=cancel_event,
            delivered_text=delivered_text,
        )

        self.assertEqual(next(stream).text, "Kafka moves events. Unsaid tail.")
        delivered_text.append("Kafka moves events.")
        delivered_text.append("More facts.")
        cancel_event.set()
        stream.close()
        assistant = next(
            message for message in memory.recent_messages(8)
            if message.role == "assistant"
        )

        self.assertEqual(assistant.content, "Kafka moves events. More facts.")
        self.assertTrue(assistant.incomplete)
        self.assertEqual(sum(message.role == "assistant" for message in memory.recent_messages(8)), 1)
        memory.close()

    def test_interrupted_buffered_voice_stream_records_only_reported_delivery(self):
        memory = MemoryManager(SQLiteMemoryStore(":memory:"))
        fixed = datetime(2026, 9, 28, 13, 5, tzinfo=ZoneInfo("Asia/Kolkata"))
        model = ScriptedModel('unexpected model response')
        orchestrator = RequestOrchestrator(
            Settings(),
            model,
            ToolRegistry([LocalTimeTool(Settings(), clock=lambda zone: fixed.astimezone(zone))]),
            memory=memory,
        )
        delivered_text = []
        cancel_event = threading.Event()
        stream = orchestrator.stream(
            "What time is it?",
            cancel_event=cancel_event,
            delivered_text=delivered_text,
        )

        self.assertEqual(next(stream).kind, "text")
        delivered_text.append("The local time is 1:05 PM.")
        cancel_event.set()
        stream.close()
        assistant = next(
            message for message in memory.recent_messages(8)
            if message.role == "assistant"
        )

        self.assertEqual(assistant.content, "The local time is 1:05 PM.")
        self.assertTrue(assistant.incomplete)
        self.assertEqual(model.calls, [])
        memory.close()

    def test_planner_logs_completion_reason_without_request_text(self):
        request = "Explain Kafka briefly"
        model = ScriptedModel(
            '{"type":"final","response":"Kafka is a distributed event platform."}'
        )
        orchestrator = RequestOrchestrator(Settings(), model, ToolRegistry([]))

        with self.assertLogs("orion.core.orchestrator", level="INFO") as logs:
            result = orchestrator.handle(request)

        self.assertTrue(result.completed)
        self.assertIn("reason=completed", "\n".join(logs.output))
        self.assertNotIn(request, "\n".join(logs.output))

    def test_time_tool_is_offline_and_timezone_aware(self):
        settings = Settings()
        fixed = datetime(2026, 9, 28, 13, 5, tzinfo=ZoneInfo("Asia/Kolkata"))
        tool = LocalTimeTool(settings, clock=lambda zone: fixed.astimezone(zone))

        result = tool.execute(
            {"action": "time"}, timeout=1, cancel_event=threading.Event()
        )

        self.assertIn("01:05 PM IST", result.content)

    def test_system_status_is_offline(self):
        tool = SystemStatusTool()
        result = tool.execute({}, timeout=1, cancel_event=threading.Event())

        self.assertTrue(result.success)
        self.assertIn("OS:", result.content)
        self.assertIn("CPU cores:", result.content)

    def test_local_tool_route_does_not_call_model(self):
        model = ScriptedModel('{"type":"final","response":"unexpected"}')
        time_tool = LocalTimeTool(
            Settings(),
            clock=lambda zone: datetime(2026, 9, 28, 13, 5, tzinfo=zone),
        )
        orchestrator = RequestOrchestrator(
            Settings(), model, ToolRegistry([time_tool])
        )

        result = orchestrator.handle("What time is it?")

        self.assertEqual(result.route.action, "time")
        self.assertIn("01:05 PM IST", result.response)
        self.assertEqual(model.calls, [])


class FactualToolTests(unittest.TestCase):
    def test_offline_tools_block_transport_even_when_search_is_configured(self):
        settings = Settings(
            online_tools_enabled=False,
            searxng_base_url="https://search.example.test",
            searxng_allowed_hosts=("search.example.test",),
        )
        for tool_class in (OpenMeteoWeatherTool, SearxngSearchTool):
            with self.subTest(tool=tool_class.name):
                transport = FakeFactualTransport([])
                tool = tool_class(settings, transport=transport)

                result = tool.execute(
                    {"query": "weather in Pune", "location": "Pune"},
                    timeout=3,
                    cancel_event=threading.Event(),
                )

                self.assertFalse(result.success)
                self.assertIn("disabled in offline mode", result.content)
                self.assertEqual(transport.calls, [])

    def test_weather_geocodes_then_fetches_compact_data(self):
        transport = FakeFactualTransport(
            [
                {
                    "results": [
                        {
                            "name": "Pune",
                            "admin1": "Maharashtra",
                            "country": "India",
                            "latitude": 18.52,
                            "longitude": 73.85,
                        }
                    ]
                },
                {
                    "timezone": "Asia/Kolkata",
                    "current": {
                        "time": "2026-09-28T12:00",
                        "temperature_2m": 28.0,
                        "weather_code": 1,
                        "wind_speed_10m": 5.0,
                    },
                    "daily": {
                        "time": ["2026-09-28"],
                        "temperature_2m_max": [30.0],
                        "temperature_2m_min": [22.0],
                    },
                },
            ]
        )
        tool = OpenMeteoWeatherTool(Settings(online_tools_enabled=True), transport=transport)

        result = tool.execute(
            {"query": "What's the weather tomorrow in Pune?"},
            timeout=8,
            cancel_event=threading.Event(),
        )

        data = json.loads(result.content)
        self.assertEqual(data["location"], "Pune, Maharashtra, India")
        self.assertEqual(data["current"]["temperature_2m"], 28.0)
        self.assertEqual(transport.calls[0][0], "https://geocoding-api.open-meteo.com/v1/search")
        self.assertEqual(transport.calls[1][0], "https://api.open-meteo.com/v1/forecast")
        self.assertEqual(transport.calls[1][1]["forecast_days"], 2)

    def test_weather_requires_location_without_network_access(self):
        transport = FakeFactualTransport([])
        tool = OpenMeteoWeatherTool(Settings(online_tools_enabled=True), transport=transport)

        result = tool.execute(
            {"query": "What's the weather?"}, timeout=3, cancel_event=threading.Event()
        )

        self.assertFalse(result.success)
        self.assertIn("specify a city", result.content)
        self.assertEqual(transport.calls, [])

    def test_search_defaults_to_public_rss_when_not_configured(self):
        transport = unittest.mock.Mock()
        transport.get_factual_rss.return_value = {
            "results": [{"title": "Weather", "url": "https://source.test/weather"}]
        }
        tool = SearxngSearchTool(Settings(online_tools_enabled=True), transport=transport)

        result = tool.execute(
            {"query": "search for local weather"},
            timeout=3,
            cancel_event=threading.Event(),
        )

        self.assertTrue(result.success)
        self.assertEqual(transport.get_factual_rss.call_args.args[0], "https://www.bing.com/search")

    def test_search_compacts_results_and_limits_fields(self):
        settings = replace(
            Settings(online_tools_enabled=True),
            searxng_base_url="https://search.example.test",
            searxng_allowed_hosts=("search.example.test",),
        )
        transport = FakeFactualTransport(
            [{"results": [{"title": "A result", "content": "Useful fact " * 100, "url": "https://source.test/item"}]}]
        )
        tool = SearxngSearchTool(settings, transport=transport, max_field_chars=50)

        result = tool.execute(
            {"query": "Search the web for current facts"},
            timeout=3,
            cancel_event=threading.Event(),
        )

        data = json.loads(result.content)
        self.assertEqual(data["query"], "current facts")
        self.assertLessEqual(len(data["results"][0]["snippet"]), 50)
        self.assertEqual(transport.calls[0][2], {"search.example.test"})

    def test_remote_and_unallowlisted_search_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "allowlisted HTTPS host"):
            replace(
                Settings(),
                searxng_base_url="http://search.example.test",
                searxng_allowed_hosts=("search.example.test",),
            ).validate()
        with self.assertRaisesRegex(ValueError, "allowlisted HTTPS host"):
            replace(
                Settings(),
                searxng_base_url="https://search.example.test",
                searxng_allowed_hosts=("other.example.test",),
            ).validate()

    def test_transport_rejects_unallowlisted_url_before_network_call(self):
        from orion.llm.runtime import LocalJsonTransport

        class NeverOpen:
            def open(self, *_args, **_kwargs):
                raise AssertionError("network must not be called")

        transport = LocalJsonTransport(opener=NeverOpen())
        with self.assertRaises(NetworkPolicyError):
            transport.get_factual_json(
                "https://attacker.example/data",
                params=None,
                allowed_hosts={"api.weather.gov"},
                timeout=1,
            )


class OrchestrationTests(unittest.TestCase):
    def test_transcript_correction_is_one_step_and_never_executes_tools_or_saves_memory(self):
        memory = MemoryManager(SQLiteMemoryStore(":memory:"))
        tool = FakeTool("reminder", ToolResult("must not run"))
        model = ScriptedModel(json.dumps({
            "type": "final",
            "response": "CORRECTION: Remind me to call at five PM",
            "memory_suggestions": [{
                "kind": "fact", "content": "unconfirmed correction", "topic": ""
            }],
        }))
        orchestrator = RequestOrchestrator(
            Settings(), model, ToolRegistry([tool]), memory=memory
        )

        correction = orchestrator.suggest_transcript_correction(
            "Remind me to call at file PM"
        )

        self.assertEqual(correction, "Remind me to call at five PM")
        self.assertEqual(tool.calls, [])
        self.assertEqual(memory.list_memories(), [])
        self.assertEqual(len(model.calls), 1)
        memory.close()

    def test_transcript_correction_rejects_uncertain_or_unchanged_output(self):
        for response in ("UNCERTAIN", "CORRECTION: file"):
            with self.subTest(response=response):
                model = ScriptedModel(json.dumps({
                    "type": "final", "response": response, "memory_suggestions": []
                }))
                orchestrator = RequestOrchestrator(
                    Settings(), model, ToolRegistry([])
                )

                self.assertIsNone(
                    orchestrator.suggest_transcript_correction("file")
                )

    def test_explicit_news_route_uses_news_category_buffered_and_streamed(self):
        for streamed in (False, True):
            with self.subTest(streamed=streamed):
                settings = replace(
                    Settings(online_tools_enabled=True),
                    searxng_base_url="https://search.example.test",
                    searxng_allowed_hosts=("search.example.test",),
                )
                transport = FakeFactualTransport([{"results": [{
                    "title": "Space mission launches", "content": "A reported launch.",
                    "url": "https://publisher.test/story",
                }]}])
                model = ScriptedModel('{"type":"final","response":"A space mission launched."}')
                orchestrator = RequestOrchestrator(
                    settings, model, create_default_tools(settings, transport=transport)
                )
                if streamed:
                    events = list(orchestrator.stream("Latest space news"))
                    self.assertEqual(events[-1].kind, "completed")
                else:
                    result = orchestrator.handle("Latest space news")
                    self.assertTrue(result.completed)
                    self.assertEqual(result.route.action, "news")
                self.assertEqual(len(transport.calls), 1)
                self.assertEqual(transport.calls[0][1]["categories"], "news")
                prompt = " ".join(message.content for message in model.calls[0][0])
                self.assertIn("https://publisher.test/story", prompt)

    def test_news_disabled_and_provider_failures_do_not_call_model(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                settings = replace(
                    Settings(online_tools_enabled=enabled),
                    searxng_base_url="https://search.example.test",
                    searxng_allowed_hosts=("search.example.test",),
                )
                transport = FakeFactualTransport([TimeoutError("private details")])
                model = ScriptedModel("must not run")
                result = RequestOrchestrator(
                    settings, model, create_default_tools(settings, transport=transport)
                ).handle("Latest space news")
                self.assertFalse(result.completed)
                self.assertIn("timed out" if enabled else "offline", result.response)
                self.assertEqual(model.calls, [])

    def test_offline_direct_routes_block_registered_fake_tools_and_model(self):
        for name, request in (
            ("weather", "What's the weather in Pune?"),
            ("web_search", "Search for current space news"),
        ):
            with self.subTest(tool=name):
                tool = FakeTool(name, ToolResult("must not run"))
                model = ScriptedModel('{"type":"final","response":"must not run"}')
                orchestrator = RequestOrchestrator(Settings(online_tools_enabled=False), model, ToolRegistry([tool]))

                result = orchestrator.handle(request)

                self.assertEqual(result.route.kind.value, "ONLINE_FACTUAL_LOOKUP")
                self.assertFalse(result.completed)
                self.assertIn("disabled in offline mode", result.response)
                self.assertEqual(tool.calls, [])
                self.assertEqual(model.calls, [])

    def test_offline_planner_blocks_fake_factual_tools_but_runs_local_step(self):
        for name, phrase in (("weather", "weather in Pune"), ("web_search", "latest space news")):
            with self.subTest(tool=name):
                factual = FakeTool(name, ToolResult("must not run"))
                time_tool = FakeTool("time", ToolResult("It is 1 PM."))
                model = PlanningModel(
                    json.dumps({"type": "tool", "name": name, "arguments": {"query": phrase}}),
                    json.dumps({"type": "tool", "name": "time", "arguments": {"action": "time"}}),
                    json.dumps({"type": "final", "response": "Lookup is disabled in offline mode. It is 1 PM."}),
                )
                orchestrator = RequestOrchestrator(
                    Settings(online_tools_enabled=False), model, ToolRegistry([factual, time_tool])
                )

                result = orchestrator.handle(f"Check {phrase} and tell me the time")

                self.assertEqual(result.route.action, "multi_step")
                self.assertTrue(result.completed)
                self.assertEqual(factual.calls, [])
                self.assertEqual(len(time_tool.calls), 1)
                self.assertIn("offline mode", result.response)
                for messages, _, _ in model.calls:
                    prompt = " ".join(message.content for message in messages)
                    self.assertIn("Offline mode: weather and web search are disabled", prompt)
                    self.assertIn("No online lookup has run", prompt)
                    self.assertNotIn(f"{name}: arguments", prompt)
                observation = " ".join(message.content for message in model.calls[1][0])
                self.assertIn(f"Tool {name} failed", observation)

    def test_clear_conversation_preserves_saved_memory_and_reminders(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(), database_path=Path(directory) / "orion.sqlite3")
            memory = MemoryManager.from_settings(settings)
            reminders = ReminderManager.from_settings(settings)
            model = ScriptedModel('{"type":"final","response":"Fresh answer."}')
            orchestrator = RequestOrchestrator(
                settings, model, ToolRegistry([]), memory=memory, reminders=reminders
            )
            try:
                memory.save_explicit_request("Remember that my favorite color is green")
                reminder = reminders.create_from_request(
                    "Remind me tomorrow at 1 PM about project meeting"
                )
                saved = memory.list_memories()
                for command in (
                    "clear conversation history",
                    "Please wipe conversational memory!",
                    "reset our conversation",
                    "clear chat history",
                ):
                    with self.subTest(command=command):
                        memory.add_turn("misunderstood old request", "stale old reply")
                        result = orchestrator.handle(command)
                        self.assertTrue(result.completed)
                        self.assertEqual(result.route.action, "clear_conversation")
                        self.assertIn("Conversation history cleared", result.response)
                        self.assertEqual(memory.recent_messages(16), ())
                        self.assertEqual(memory.list_memories(), saved)
                        self.assertIsNotNone(reminders.get(reminder.id))
                        self.assertEqual(model.calls, [])

                orchestrator.handle("Explain gravity briefly")
                contents = "\n".join(message.content for message in model.calls[0][0])
                self.assertNotIn("misunderstood old request", contents)
                self.assertNotIn("stale old reply", contents)
                self.assertNotIn("Conversation history cleared", contents)
            finally:
                reminders.close()
                memory.close()

    def test_clear_conversation_without_memory_fails_without_model_call(self):
        model = ScriptedModel('{"type":"final","response":"unexpected"}')
        orchestrator = RequestOrchestrator(Settings(), model, ToolRegistry([]))

        result = orchestrator.handle("clear conversation history")

        self.assertFalse(result.completed)
        self.assertEqual(result.response, "Memory is not available in this session.")
        self.assertEqual(model.calls, [])

    def test_planner_can_combine_weather_and_reminder_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(online_tools_enabled=True), database_path=Path(directory) / "orion.sqlite3")
            model = PlanningModel(
                json.dumps({
                    "type": "tool",
                    "name": "weather",
                    "arguments": {"query": "weather in Pune"},
                }),
                json.dumps({
                    "type": "tool",
                    "name": "reminder",
                    "arguments": {
                        "action": "create",
                        "title": "carry an umbrella",
                        "when": "tomorrow at 8 AM",
                    },
                }),
                json.dumps({"type": "final", "response": "Rain is expected. I set your umbrella reminder."}),
            )
            weather = FakeTool("weather", ToolResult('{"forecast":"rain"}'))
            reminders = ReminderManager.from_settings(settings)
            orchestrator = RequestOrchestrator(
                settings, model, ToolRegistry([weather]), reminders=reminders
            )
            try:
                result = orchestrator.handle(
                    "Check the weather in Pune and set a reminder tomorrow at 8 AM to carry an umbrella"
                )

                self.assertEqual(result.route.kind.value, "LLM_REASONING")
                self.assertIn("set your umbrella reminder", result.response)
                self.assertEqual(len(weather.calls), 1)
                saved = reminders.list(status="pending")
                self.assertEqual(len(saved), 1)
                self.assertEqual(saved[0].title, "carry an umbrella")
                self.assertEqual(len(model.calls), 3)
            finally:
                reminders.close()

    def test_planner_authorizes_plural_reminders_and_system_status(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(), database_path=Path(directory) / "orion.sqlite3")
            model = PlanningModel(
                json.dumps({"type": "tool", "name": "reminder", "arguments": {"action": "list"}}),
                json.dumps({"type": "tool", "name": "system_status", "arguments": {}}),
                json.dumps({"type": "final", "response": "No reminders. System is healthy."}),
            )
            system_status = FakeTool("system_status", ToolResult("System is healthy."))
            reminders = ReminderManager.from_settings(settings)
            orchestrator = RequestOrchestrator(
                settings, model, ToolRegistry([system_status]), reminders=reminders
            )
            try:
                result = orchestrator.handle("List my reminders and show system status")

                self.assertTrue(result.completed)
                self.assertEqual(result.route.action, "multi_step")
                self.assertEqual(len(system_status.calls), 1)
                prompt = " ".join(message.content for message in model.calls[1][0])
                self.assertIn("Tool reminder succeeded: []", prompt)
                self.assertNotIn("not available", prompt)
            finally:
                reminders.close()

    def test_reminder_title_with_connectors_does_not_initialize_model(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(), database_path=Path(directory) / "orion.sqlite3")
            model = ScriptedModel('{"type":"final","response":"unexpected"}')
            reminders = ReminderManager.from_settings(settings)
            orchestrator = RequestOrchestrator(
                settings, model, ToolRegistry([]), reminders=reminders
            )
            try:
                result = orchestrator.handle(
                    "Remind me tomorrow at 8 AM to check weather and news"
                )

                self.assertTrue(result.completed)
                self.assertEqual(result.route.kind.value, "REMINDER")
                self.assertEqual(model.calls, [])
                self.assertEqual(reminders.list(status="pending")[0].title, "check weather and news")
            finally:
                reminders.close()

    def test_planner_can_combine_existing_local_query_forms(self):
        model = PlanningModel(
            json.dumps({"type": "tool", "name": "system_status", "arguments": {}}),
            json.dumps({"type": "tool", "name": "time", "arguments": {"action": "time"}}),
            json.dumps({"type": "final", "response": "Memory is available. It is 1 PM."}),
        )
        system_status = FakeTool("system_status", ToolResult("Memory is available."))
        time_tool = FakeTool("time", ToolResult("It is 1 PM."))
        orchestrator = RequestOrchestrator(
            Settings(), model, ToolRegistry([system_status, time_tool])
        )

        result = orchestrator.handle("How much free memory do I have and tell me the time")

        self.assertTrue(result.completed)
        self.assertEqual(result.route.action, "multi_step")
        self.assertEqual(len(system_status.calls), 1)
        self.assertEqual(len(time_tool.calls), 1)

    def test_planner_can_reschedule_then_add_grounded_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(), database_path=Path(directory) / "orion.sqlite3")
            reminders = ReminderManager.from_settings(settings)
            reminder = reminders.create_from_request(
                "Remind me tomorrow at 1 PM about project meeting"
            )
            model = PlanningModel(
                json.dumps({"type": "tool", "name": "reminder", "arguments": {
                    "action": "reschedule", "query": "project meeting", "when": "tomorrow at 3 PM",
                }}),
                json.dumps({"type": "tool", "name": "reminder", "arguments": {
                    "action": "add_notes", "query": "project meeting", "notes": "bring test results",
                }}),
                json.dumps({"type": "final", "response": "I updated the time and notes."}),
            )
            orchestrator = RequestOrchestrator(
                settings, model, ToolRegistry([]), reminders=reminders
            )
            try:
                result = orchestrator.handle(
                    "Reschedule my project meeting reminder to tomorrow at 3 PM and add notes: bring test results"
                )

                self.assertTrue(result.completed)
                stored = reminders.get(reminder.id)
                self.assertEqual(stored.due_at.astimezone(ZoneInfo(settings.reminder_timezone)).hour, 15)
                self.assertEqual(stored.notes, "bring test results")
                self.assertEqual(len(model.calls), 3)
                self.assertIn("Authorized tools:", model.calls[2][0][0].content)
                self.assertIn("Never guess missing reminder", model.calls[2][0][0].content)
            finally:
                reminders.close()

    def test_incomplete_reschedule_asks_without_changing_reminder(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(), database_path=Path(directory) / "orion.sqlite3")
            reminders = ReminderManager.from_settings(settings)
            reminder = reminders.create_from_request(
                "Remind me tomorrow at 1 PM about project meeting"
            )
            model = ScriptedModel(json.dumps({
                "type": "final", "response": "When should I move the project meeting reminder?",
            }))
            orchestrator = RequestOrchestrator(
                settings, model, ToolRegistry([]), reminders=reminders
            )
            try:
                result = orchestrator.handle("Reschedule my project meeting reminder")

                self.assertTrue(result.completed)
                self.assertIn("When should", result.response)
                self.assertEqual(len(model.calls), 1)
                self.assertEqual(reminders.get(reminder.id), reminder)
            finally:
                reminders.close()

    def test_incomplete_reminder_uses_model_to_ask_for_required_details(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(), database_path=Path(directory) / "orion.sqlite3")
            model = ScriptedModel(
                json.dumps({"type": "final", "response": "What time should I set the reminder for?"})
            )
            reminders = ReminderManager.from_settings(settings)
            orchestrator = RequestOrchestrator(
                settings, model, create_default_tools(settings), reminders=reminders
            )
            try:
                result = orchestrator.handle("Remind me to call the dentist")

                self.assertEqual(len(model.calls), 1)
                self.assertIn("What time", result.response)
                self.assertEqual(reminders.list(status="pending"), [])
            finally:
                reminders.close()

    def test_agent_factual_query_cannot_expand_beyond_user_request(self):
        model = PlanningModel(
            json.dumps({
                "type": "tool",
                "name": "weather",
                "arguments": {"query": "weather in London"},
            }),
            json.dumps({"type": "final", "response": "I could not look up that location."}),
        )
        weather = FakeTool("weather", ToolResult("should not be called"))
        orchestrator = RequestOrchestrator(Settings(online_tools_enabled=True), model, ToolRegistry([weather]))

        result = orchestrator.handle(
            "Check the weather in Pune and tell me today's date"
        )

        self.assertTrue(result.completed)
        self.assertEqual(weather.calls, [])
        observation = " ".join(message.content for message in model.calls[1][0])
        self.assertIn("must come from the user's request", observation)

    def test_online_result_is_passed_to_local_model_not_cloud(self):
        model = ScriptedModel(
            json.dumps({"type": "final", "response": "Verified search result."})
        )
        search = FakeTool("web_search", ToolResult('{"results":[{"title":"Verified result"}]}'))
        orchestrator = RequestOrchestrator(
            Settings(online_tools_enabled=True), model, ToolRegistry([search])
        )

        result = orchestrator.handle("Search for Python documentation")

        self.assertEqual(result.response, "Verified search result.")
        self.assertEqual(len(search.calls), 1)
        model_prompt = " ".join(m.content for m in model.calls[0][0])
        self.assertIn("Verified result", model_prompt)
        self.assertIn("Interpret only the supplied factual result", model_prompt)

    def test_incomplete_weather_data_does_not_invoke_model(self):
        model = ScriptedModel(' {"type":"final","response":"guess"}')
        weather = FakeTool("weather", ToolResult('{"location":"Pune"}'))
        orchestrator = RequestOrchestrator(Settings(), model, ToolRegistry([weather]))
        result = orchestrator.handle("What's the weather in Pune?")
        self.assertFalse(result.completed)
        self.assertIn("incomplete forecast", result.response)
        self.assertEqual(model.calls, [])

    def test_failed_online_lookup_does_not_invoke_model(self):
        model = ScriptedModel('{"type":"final","response":"guess"}')
        weather = FakeTool("weather", ToolResult("Weather lookup failed.", success=False))
        orchestrator = RequestOrchestrator(
            Settings(online_tools_enabled=True), model, ToolRegistry([weather])
        )

        result = orchestrator.handle("What's the weather tomorrow in Pune?")

        self.assertFalse(result.completed)
        self.assertEqual(result.response, "Weather lookup failed.")
        self.assertEqual(len(weather.calls), 1)
        self.assertEqual(model.calls, [])

    def test_weather_response_does_not_depend_on_planner(self):
        model = ScriptedModel("invalid planner decision")
        weather = FakeTool("weather", ToolResult(json.dumps({
            "location": "Hyderabad, Telangana, India",
            "timezone": "Asia/Kolkata",
            "current": {"temperature_2m": 29, "apparent_temperature": 31, "time": "2026-10-10T09:00"},
            "daily_units": {"temperature_2m_max": "\u00b0C"},
            "daily": [
                {"temperature_2m_min": 23, "temperature_2m_max": 32, "precipitation_probability_max": 20},
                {"temperature_2m_min": 24, "temperature_2m_max": 33},
            ],
        })))
        orchestrator = RequestOrchestrator(Settings(), model, ToolRegistry([weather]))
        result = orchestrator.handle("what's the weather in hyderabad")
        self.assertTrue(result.completed)
        self.assertIn("29 degrees Celsius", result.response)
        self.assertIn("Hyderabad", result.response)
        self.assertIn("20 percent", result.response)
        events = list(orchestrator.stream("what's the weather tomorrow in hyderabad"))
        self.assertIn("high is 33 degrees Celsius", events[0].text)
        self.assertEqual(model.calls, [])
        self.assertEqual(len(weather.calls), 2)

    def test_capability_fallback_skips_factual_and_model_calls(self):
        model = ScriptedModel('{"type":"final","response":"guess"}')
        search = FakeTool("web_search", ToolResult("facts"))
        orchestrator = RequestOrchestrator(Settings(), model, ToolRegistry([search]))

        result = orchestrator.handle(
            "Analyze 10 years of global semiconductor trends and predict which company dominates the next five years."
        )

        self.assertFalse(result.completed)
        self.assertIn("can't reliably", result.response)
        self.assertEqual(search.calls, [])
        self.assertEqual(model.calls, [])

    def test_registry_contains_only_explicit_default_tool_names(self):
        tools = create_default_tools(Settings())

        self.assertEqual(
            tools.names(),
            ("system_status", "time", "weather", "web_search"),
        )

    def test_time_route_and_reminder_crud_do_not_call_model(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(), database_path=Path(directory) / "orion.sqlite3")
            model = ScriptedModel('{"type":"final","response":"unexpected"}')
            memory = MemoryManager.from_settings(settings)
            reminders = ReminderManager.from_settings(settings)
            tools = create_default_tools(settings)
            orchestrator = RequestOrchestrator(
                settings, model, tools, memory=memory, reminders=reminders
            )
            try:
                time_result = orchestrator.handle("What time is it?")
                created = orchestrator.handle(
                    "Remind me tomorrow at 1 PM about project meeting. Discuss API migration."
                )
                listed = orchestrator.handle("What reminders do I have today?")
                tomorrow_list = orchestrator.handle("What reminders do I have for tmrw?")
                missed = orchestrator.handle("Did I miss any reminders yet?")
                time_follow_up = orchestrator.handle("at 2PM")
                updated = orchestrator.handle(
                    "Add notes to project meeting reminder: bring performance test results."
                )
                rescheduled = orchestrator.handle(
                    "Reschedule my project meeting reminder to tomorrow at 3 PM"
                )
                cancelled = orchestrator.handle("Cancel my project meeting reminder")

                self.assertIn("IST", time_result.response)
                self.assertIn("Reminder set: project meeting", created.response)
                self.assertIn("no pending reminders for today", listed.response)
                self.assertIn("project meeting", tomorrow_list.response)
                self.assertEqual(missed.response, "You have no overdue reminders.")
                self.assertIn("one-shot request", time_follow_up.response)
                self.assertEqual(updated.response, "Updated reminder: project meeting.")
                self.assertEqual(rescheduled.response, "Updated reminder: project meeting.")
                self.assertEqual(cancelled.response, "Reminder cancelled.")
                self.assertEqual(model.calls, [])
            finally:
                memory.close()
                reminders.close()

    def test_voice_confirmation_guards_reminder_cancel_and_memory_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(), database_path=Path(directory) / "orion.sqlite3")
            memory = MemoryManager.from_settings(settings)
            reminders = ReminderManager.from_settings(settings)
            memory.save_episode(
                "We chose SQLite for the launch decision.",
                topic="launch decision",
                explicitly_requested=True,
            )
            reminder = reminders.create_from_request(
                "Remind me tomorrow at 1 PM about project meeting",
                now=datetime(2026, 9, 28, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
            )
            answers = iter([False, False, True, True])
            prompts = []
            orchestrator = RequestOrchestrator(
                settings,
                ScriptedModel('{"type":"final","response":"unexpected"}'),
                create_default_tools(settings),
                memory=memory,
                reminders=reminders,
                confirm_action=lambda prompt: prompts.append(prompt) or next(answers),
            )
            try:
                denied_cancel = orchestrator.handle("Cancel my project meeting reminder")
                denied_delete = orchestrator.handle("Forget what we discussed about launch decision")
                confirmed_cancel = orchestrator.handle("Cancel my project meeting reminder")
                confirmed_delete = orchestrator.handle("Forget what we discussed about launch decision")

                self.assertFalse(denied_cancel.completed)
                self.assertFalse(denied_delete.completed)
                self.assertEqual(reminders.get(reminder.id).status.value, "cancelled")
                self.assertIn("I deleted", confirmed_delete.response)
                self.assertTrue(any("project meeting" in prompt for prompt in prompts))
                self.assertTrue(any("SQLite" in prompt for prompt in prompts))
                self.assertEqual(memory.list_memories(kind="episode"), [])
            finally:
                memory.close()
                reminders.close()

    def test_typed_reminder_reschedule_and_notes_update(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(), database_path=Path(directory) / "orion.sqlite3")
            model = ScriptedModel('{"type":"final","response":"unexpected"}')
            manager = ReminderManager.from_settings(settings)
            tool_registry = create_default_tools(settings)
            reminder = manager.create_from_request(
                "Remind me tomorrow at 1 PM about project meeting",
                now=datetime(2026, 9, 28, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
            )
            target_day = datetime.now(ZoneInfo("Asia/Kolkata")) + timedelta(days=1)
            orchestrator = RequestOrchestrator(
                settings, model, tool_registry, reminders=manager
            )
            try:
                updated = orchestrator.handle(
                    f"Reschedule my project meeting reminder to {target_day:%A} at 3 PM"
                )
                noted = orchestrator.handle(
                    "Add notes to the project meeting reminder: bring performance test results"
                )

                stored = manager.get(reminder.id)
                self.assertEqual(updated.response, "Updated reminder: project meeting.")
                self.assertEqual(
                    stored.due_at.astimezone(ZoneInfo("Asia/Kolkata")).weekday(),
                    target_day.weekday(),
                )
                self.assertEqual(stored.due_at.astimezone(ZoneInfo("Asia/Kolkata")).hour, 15)
                self.assertIn("Updated reminder", noted.response)
                self.assertEqual(stored.notes, "bring performance test results")
                self.assertEqual(model.calls, [])
            finally:
                manager.close()

    def test_ambiguous_reminder_cancel_requests_clarification(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(), database_path=Path(directory) / "orion.sqlite3")
            manager = ReminderManager.from_settings(settings)
            now = datetime(2026, 9, 28, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
            first = manager.create_from_request(
                "Remind me tomorrow at 1 PM about project meeting", now=now
            )
            second = manager.create_from_request(
                "Remind me tomorrow at 2 PM about project meeting review", now=now
            )
            orchestrator = RequestOrchestrator(
                settings,
                ScriptedModel('{"type":"final","response":"unexpected"}'),
                create_default_tools(settings),
                reminders=manager,
            )
            try:
                result = orchestrator.handle("Cancel my project meeting reminder")
                self.assertFalse(result.completed)
                self.assertIn("several matching reminders", result.response)
                self.assertEqual(manager.get(first.id).status.value, "pending")
                self.assertEqual(manager.get(second.id).status.value, "pending")
            finally:
                manager.close()

    def test_typed_cli_local_clock_request_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(), database_path=Path(directory) / "orion.sqlite3")
            output = io.StringIO()

            with redirect_stdout(output), patch.object(
                _LazyLocalModel,
                "complete",
                side_effect=AssertionError("model should stay unloaded"),
            ):
                _run_request(settings, "What time is it?")

            self.assertIn("IST", output.getvalue())


if __name__ == "__main__":
    unittest.main()
