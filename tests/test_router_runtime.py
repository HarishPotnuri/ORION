import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from orion.config import Settings
from orion.core.contracts import ChatMessage, ModelStreamEvent
from orion.core.policy import NetworkPolicy, NetworkPolicyError
from orion.core.router import OrionRouter, RouteKind
from orion.llm.runtime import LlamaCppServer, LocalJsonTransport, SSEDecoder, create_local_model


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.stream_events = [
            ModelStreamEvent(kind="text", text="Local streamed answer."),
            ModelStreamEvent(kind="completed", finish_reason="stop"),
        ]

    def post(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        if url.endswith("/apply-template"):
            return {"prompt": "\n".join(message["content"] for message in payload["messages"])}
        if url.endswith("/tokenize"):
            size = max(1, (len(payload["content"].encode("utf-8")) + 3) // 4)
            return {"tokens": list(range(size))}
        return self.response

    def stream(self, url, payload, *, timeout, cancel_event):
        self.calls.append((url, payload, timeout))
        yield from self.stream_events


class FakeStreamResponse:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def read1(self, _size):
        return next(self.chunks, b"")


class FakeStreamOpener:
    def __init__(self, chunks):
        self.response = FakeStreamResponse(chunks)

    def open(self, _request, timeout):
        self.timeout = timeout
        return self.response


class SSEDecoderTests(unittest.TestCase):
    def test_decodes_utf8_and_events_across_arbitrary_byte_splits(self):
        decoder = SSEDecoder()
        payload = 'data: {"choices":[{"delta":{"content":"caf\u00e9"}}]}\r\n\r\ndata: [DONE]\n\n'.encode()
        events = []
        for byte in payload:
            events.extend(decoder.feed(bytes([byte])))

        self.assertEqual(
            events,
            ['{"choices":[{"delta":{"content":"caf\u00e9"}}]}', "[DONE]"],
        )

    def test_stream_emits_text_usage_and_finish_metadata(self):
        chunks = [bytes([byte]) for byte in (
            b'data: {"choices":[{"delta":{"role":"assistant","content":null}}]}\n\n'
            b'data: {"choices":[{"delta":{"reasoning_content":"hidden"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"Hello \xe2\x98\x83"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}}\n\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            b'data: [DONE]\n\n'
        )]
        opener = FakeStreamOpener(chunks)
        transport = LocalJsonTransport(opener=opener)

        events = list(
            transport.stream(
                "http://127.0.0.1:8080/v1/chat/completions",
                {"messages": []},
                timeout=3,
                cancel_event=threading.Event(),
            )
        )

        self.assertEqual([event.text for event in events if event.kind == "text"], ["Hello ☃"])
        usage = next(event.usage for event in events if event.kind == "usage")
        self.assertEqual(usage.total_tokens, 15)
        completed = events[-1]
        self.assertEqual(completed.kind, "completed")
        self.assertEqual(completed.finish_reason, "stop")
        self.assertTrue(opener.response.closed)

    def test_stream_survives_delay_before_first_sse_chunk(self):
        class DelayedHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                time.sleep(0.3)
                events = (
                    b'data: {"choices":[{"delta":{"role":"assistant","content":null}}]}\n\n'
                    b'data: {"choices":[{"delta":{"content":"Ready."}}]}\n\n'
                    b'data: [DONE]\n\n'
                )
                self.wfile.write(f"{len(events):X}\r\n".encode() + events + b"\r\n0\r\n\r\n")
                self.wfile.flush()

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), DelayedHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            transport = LocalJsonTransport()
            events = list(
                transport.stream(
                    f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                    {"messages": []},
                    timeout=3,
                    cancel_event=threading.Event(),
                )
            )
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=1)

        self.assertEqual([event.text for event in events if event.kind == "text"], ["Ready."])
        self.assertEqual(events[-1].kind, "completed")

    def test_rejects_stream_without_done_marker(self):
        opener = FakeStreamOpener([b'data: {"choices":[]}\n\n'])
        transport = LocalJsonTransport(opener=opener)

        with self.assertRaisesRegex(RuntimeError, r"ended before \[DONE\]"):
            list(
                transport.stream(
                    "http://127.0.0.1:8080/v1/chat/completions",
                    {"messages": []},
                    timeout=3,
                    cancel_event=threading.Event(),
                )
            )
        self.assertTrue(opener.response.closed)

    def test_rejects_malformed_event_json(self):
        opener = FakeStreamOpener([b"data: not-json\n\n"])
        transport = LocalJsonTransport(opener=opener)

        with self.assertRaisesRegex(RuntimeError, "invalid event JSON"):
            list(
                transport.stream(
                    "http://127.0.0.1:8080/v1/chat/completions",
                    {"messages": []},
                    timeout=3,
                    cancel_event=threading.Event(),
                )
            )
        self.assertTrue(opener.response.closed)

    def test_cancellation_closes_active_sse_response(self):
        cancellation = threading.Event()

        class CancelOnReadResponse(FakeStreamResponse):
            def read1(self, _size):
                cancellation.set()
                raise TimeoutError()

        class CancelOnReadOpener:
            def __init__(self):
                self.response = CancelOnReadResponse([])

            def open(self, _request, timeout):
                return self.response

        opener = CancelOnReadOpener()
        transport = LocalJsonTransport(opener=opener)

        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            list(
                transport.stream(
                    "http://127.0.0.1:8080/v1/chat/completions",
                    {"messages": []},
                    timeout=3,
                    cancel_event=cancellation,
                )
            )
        self.assertTrue(opener.response.closed)


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.router = OrionRouter()

    def test_routes_simple_command_without_model(self):
        decision = self.router.route("Mute yourself")
        self.assertEqual(decision.kind, RouteKind.SIMPLE_COMMAND)
        self.assertEqual(decision.action, "mute_speaker")

    def test_routes_local_time_and_date(self):
        self.assertEqual(
            self.router.route("What time is it?").action,
            "time",
        )
        self.assertEqual(
            self.router.route("What is today's date?").action,
            "date",
        )
        self.assertEqual(
            self.router.route("What should I do today?").kind,
            RouteKind.LLM_REASONING,
        )
        system_status = self.router.route("Show system status")
        self.assertEqual(system_status.kind, RouteKind.LOCAL_TOOL)
        self.assertEqual(system_status.action, "system_status")

    def test_routes_reminder_operations(self):
        cases = {
            "Remind me tomorrow at 1 PM about the meeting": "create",
            "What reminders do I have today?": "list",
            "What reminders do I have for tmrw?": "list",
            "did I miss any reminders yet?": "missed",
            "Cancel my project meeting reminder": "cancel",
            "Reschedule my meeting reminder": "update",
            "What did I tell you when I created the reminder?": "query",
        }
        for request, action in cases.items():
            with self.subTest(request=request):
                decision = self.router.route(request)
                self.assertEqual(decision.kind, RouteKind.REMINDER)
                self.assertEqual(decision.action, action)

        self.assertEqual(
            self.router.route("at 2PM").action,
            "follow_up",
        )

    def test_routes_combined_operations_to_bounded_planner(self):
        requests = (
            "Check the weather in Pune and set a reminder tomorrow at 8 AM to carry an umbrella",
            "Find my project meeting reminder and move it to Friday",
            "Reschedule my project meeting reminder to Friday and add notes: bring test results",
            "Search recent news and tell me today's date",
        )
        for request in requests:
            with self.subTest(request=request):
                self.assertEqual(
                    self.router.route(request).kind,
                    RouteKind.LLM_REASONING,
                )

    def test_keeps_single_operations_with_connectors_on_fast_paths(self):
        cases = (
            ("Remind me tomorrow at 8 AM to check weather and news", RouteKind.REMINDER),
            ("Show system status and available memory", RouteKind.LOCAL_TOOL),
            ("Remember that I check the weather and news", RouteKind.MEMORY_UPDATE),
            ("Search for weather and news", RouteKind.ONLINE_FACTUAL_LOOKUP),
            ("Check the weather in Trinidad and Tobago", RouteKind.ONLINE_FACTUAL_LOOKUP),
        )
        for request, expected_kind in cases:
            with self.subTest(request=request):
                self.assertEqual(self.router.route(request).kind, expected_kind)

    def test_routes_compound_local_questions_to_planner(self):
        decision = self.router.route("What time is it and what is the system status?")

        self.assertEqual(decision.kind, RouteKind.LLM_REASONING)
        self.assertEqual(decision.action, "multi_step")

    def test_routes_memory_and_online_requests(self):
        self.assertEqual(
            self.router.route("What did we decide about Orion's memory?").kind,
            RouteKind.MEMORY_QUERY,
        )
        self.assertEqual(
            self.router.route("Remember that I prefer short answers").kind,
            RouteKind.MEMORY_UPDATE,
        )
        weather = self.router.route("What's the weather tomorrow?")
        self.assertEqual(weather.kind, RouteKind.ONLINE_FACTUAL_LOOKUP)
        self.assertEqual(weather.action, "weather")
        self.assertEqual(
            self.router.route("Search the web for recent Python changes").kind,
            RouteKind.ONLINE_FACTUAL_LOOKUP,
        )

    def test_routes_incognito_controls_and_memory_deletion(self):
        self.assertEqual(
            self.router.route("Turn incognito mode on").action,
            "on",
        )
        self.assertEqual(
            self.router.route("Incognito off").action,
            "off",
        )
        self.assertEqual(
            self.router.route("Forget what we discussed about Orion memory").kind,
            RouteKind.MEMORY_DELETE,
        )
        self.assertEqual(
            self.router.route("Remember this: I prefer quiet alerts").kind,
            RouteKind.MEMORY_UPDATE,
        )

    def test_unmatched_request_uses_local_reasoning_route(self):
        self.assertEqual(
            self.router.route("Why is my Kafka consumer slowing down?").kind,
            RouteKind.LLM_REASONING,
        )


class NetworkPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = NetworkPolicy()

    def test_accepts_only_local_model_endpoints(self):
        for endpoint in (
            "http://localhost:8080/v1",
            "http://127.0.0.1:8080/v1",
            "http://[::1]:8080/v1",
        ):
            with self.subTest(endpoint=endpoint):
                self.assertEqual(
                    self.policy.validate_model_endpoint(endpoint), endpoint
                )
        for endpoint in (
            "https://api.example.com/v1",
            "http://0.0.0.0:8080/v1",
            "http://user:secret@localhost:8080/v1",
        ):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(NetworkPolicyError):
                    self.policy.validate_model_endpoint(endpoint)

    def test_factual_lookup_requires_exact_https_allowlist(self):
        self.assertEqual(
            self.policy.validate_factual_url(
                "https://api.weather.gov/", {"api.weather.gov"}
            ),
            "https://api.weather.gov/",
        )
        with self.assertRaises(NetworkPolicyError):
            self.policy.validate_factual_url(
                "http://api.weather.gov/", {"api.weather.gov"}
            )
        with self.assertRaises(NetworkPolicyError):
            self.policy.validate_factual_url(
                "https://api.weather.gov.attacker.test/", {"api.weather.gov"}
            )


class LocalRuntimeTests(unittest.TestCase):
    def test_llama_cpp_sends_chat_only_to_local_endpoint(self):
        transport = FakeTransport(
            {"choices": [{"message": {"content": "Local answer."}}]}
        )
        settings = Settings()
        runtime = create_local_model(settings, transport=transport)

        with patch(
            "orion.llm.runtime.time.monotonic",
            side_effect=[100.0, 100.0, 100.25, 100.5],
        ):
            answer = runtime.complete(
                [ChatMessage(role="user", content="Explain the issue")],
                max_tokens=64,
                timeout=3.0,
            )

        self.assertEqual(answer, "Local answer.")
        url, payload, timeout = transport.calls[-1]
        self.assertEqual(url, "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(payload["messages"][0]["content"], "Explain the issue")
        self.assertEqual(payload["max_tokens"], 64)
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        decision_schema = payload["response_format"]["json_schema"]["schema"]
        final_schema, tool_schema = decision_schema["oneOf"]
        self.assertEqual(final_schema["properties"]["type"], {"const": "final"})
        self.assertEqual(final_schema["required"], ["type", "response"])
        self.assertEqual(tool_schema["properties"]["type"], {"const": "tool"})
        self.assertEqual(tool_schema["required"], ["type", "name", "arguments"])
        self.assertEqual(timeout, 2.5)
        self.assertIn("/apply-template", transport.calls[0][0])
        self.assertEqual(transport.calls[0][2], 3.0)
        self.assertIn("/tokenize", transport.calls[1][0])
        self.assertEqual(transport.calls[1][2], 2.75)

    def test_runtime_merges_system_context_for_template_and_chat(self):
        messages = [
            ChatMessage(role="system", content="Instructions"),
            ChatMessage(role="system", content="Relevant memory"),
            ChatMessage(role="user", content="Previous question"),
            ChatMessage(role="assistant", content="Previous answer"),
            ChatMessage(role="system", content="Conversation summary"),
            ChatMessage(role="user", content="Current question"),
        ]
        expected = [
            {
                "role": "system",
                "content": "Instructions\n\nRelevant memory\n\nConversation summary",
            },
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
            {"role": "user", "content": "Current question"},
        ]
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                transport = FakeTransport(
                    {"choices": [{"message": {"content": "Local answer."}}]}
                )
                runtime = create_local_model(Settings(), transport=transport)
                if streaming:
                    list(runtime.stream(
                        messages, max_tokens=64, timeout=3,
                        cancel_event=threading.Event(),
                    ))
                else:
                    runtime.complete(messages, max_tokens=64, timeout=3)
                self.assertEqual(transport.calls[0][1]["messages"], expected)
                self.assertEqual(transport.calls[-1][1]["messages"], expected)

    def test_runtime_rejects_formatted_prompt_over_input_budget(self):
        transport = FakeTransport({"choices": [{"message": {"content": "unused"}}]})
        runtime = create_local_model(Settings(), transport=transport)

        with self.assertRaisesRegex(RuntimeError, "input token budget"):
            runtime.complete(
                [ChatMessage(role="user", content="x" * 20_000)],
                max_tokens=64,
                timeout=3,
            )
        self.assertEqual(len(transport.calls), 2)

    def test_stream_uses_local_nonthinking_chat_completion_and_usage(self):
        transport = FakeTransport({})
        runtime = create_local_model(Settings(), transport=transport)

        events = list(
            runtime.stream(
                [ChatMessage(role="user", content="Explain Kafka")],
                max_tokens=512,
                timeout=3,
                cancel_event=threading.Event(),
            )
        )

        self.assertEqual(events[0].text, "Local streamed answer.")
        url, payload, _timeout = transport.calls[-1]
        self.assertEqual(url, "http://127.0.0.1:8080/v1/chat/completions")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_stream_rejects_answer_budget_overrun_before_transport(self):
        transport = FakeTransport({})
        runtime = create_local_model(Settings(), transport=transport)

        with self.assertRaisesRegex(ValueError, "answer token budget"):
            runtime.stream(
                [ChatMessage(role="user", content="Hello")],
                max_tokens=513,
                timeout=3,
                cancel_event=threading.Event(),
            )
        self.assertEqual(transport.calls, [])

    def test_ollama_backend_is_rejected(self):
        transport = FakeTransport({})
        settings = replace(
            Settings(),
            model_backend="ollama",
            model_base_url="http://127.0.0.1:11434",
        )

        with self.assertRaisesRegex(ValueError, "Unsupported local model backend"):
            create_local_model(settings, transport=transport)
        self.assertEqual(transport.calls, [])

    def test_runtime_rejects_remote_endpoint_before_transport(self):
        transport = FakeTransport({})
        settings = replace(Settings(), model_base_url="https://example.com/v1")

        with self.assertRaises(NetworkPolicyError):
            create_local_model(settings, transport=transport)
        self.assertEqual(transport.calls, [])


class LlamaServerTests(unittest.TestCase):
    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.terminated = False
            self.killed = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

    def test_reuses_ready_server_without_starting_process(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.touch()
            settings = replace(Settings(), model_path=model_path)
            launches = []
            server = LlamaCppServer(
                settings,
                health_probe=lambda _url, _timeout: True,
                popen=lambda *args, **kwargs: launches.append((args, kwargs)),
            )
            with patch(
                "orion.llm.runtime.LocalJsonTransport.get",
                return_value={
                    "model_path": str(model_path.resolve()),
                    "build_info": "b123-d812350",
                },
            ):
                server.start()

            self.assertEqual(launches, [])
            self.assertFalse(server.owned_process)

    def test_rejects_ready_server_with_wrong_model_or_build(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.touch()
            server = LlamaCppServer(
                replace(Settings(), model_path=model_path),
                health_probe=lambda _url, _timeout: True,
            )
            with patch(
                "orion.llm.runtime.LocalJsonTransport.get",
                return_value={"model_path": str(model_path), "build_info": "old-build"},
            ):
                with self.assertRaisesRegex(RuntimeError, "pinned llama.cpp build"):
                    server.start()

    def test_starts_local_model_and_stops_owned_process(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.touch()
            settings = replace(Settings(), model_path=model_path)
            process = self.FakeProcess()
            launches = []
            times = [0.0]
            health_results = iter([False, False, True])

            def popen(command, **kwargs):
                launches.append((command, kwargs))
                return process

            server = LlamaCppServer(
                settings,
                health_probe=lambda _url, _timeout: next(health_results),
                popen=popen,
                monotonic=lambda: times[0],
                sleep=lambda delay: times.__setitem__(0, times[0] + delay),
            )

            server.start()
            server.stop()

            command, kwargs = launches[0]
            self.assertIn("--host", command)
            self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
            self.assertEqual(command[command.index("--port") + 1], "8080")
            self.assertIn(str(model_path), command)
            self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
            self.assertTrue(process.terminated)
            self.assertFalse(server.owned_process)
            self.assertIn("--jinja", command)
            self.assertIn("--no-mmproj", command)
            self.assertIn("--reasoning-format", command)
            self.assertEqual(command[command.index("--reasoning-format") + 1], "deepseek")

    def test_startup_timeout_terminates_server(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.touch()
            settings = replace(
                Settings(), model_path=model_path, llama_startup_timeout=0.3
            )
            process = self.FakeProcess()
            times = [0.0]
            server = LlamaCppServer(
                settings,
                health_probe=lambda _url, _timeout: False,
                popen=lambda *_args, **_kwargs: process,
                monotonic=lambda: times[0],
                sleep=lambda delay: times.__setitem__(0, times[0] + delay),
            )

            with self.assertRaisesRegex(TimeoutError, "did not become ready"):
                server.start()

            self.assertTrue(process.terminated)
            self.assertFalse(server.owned_process)

    def test_does_not_launch_without_local_model_file(self):
        launches = []
        server = LlamaCppServer(
            Settings(),
            health_probe=lambda _url, _timeout: False,
            popen=lambda *args, **kwargs: launches.append((args, kwargs)),
        )

        with self.assertRaisesRegex(FileNotFoundError, "local GGUF"):
            server.start()

        self.assertEqual(launches, [])

    def test_runtime_rejects_invalid_token_limit_before_transport(self):
        transport = FakeTransport({})
        runtime = create_local_model(Settings(), transport=transport)

        with self.assertRaisesRegex(ValueError, "max_tokens must be positive"):
            runtime.complete([], max_tokens=0)
        self.assertEqual(transport.calls, [])


if __name__ == "__main__":
    unittest.main()
