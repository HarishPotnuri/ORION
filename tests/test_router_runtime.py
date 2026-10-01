import unittest
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from orion.config import Settings
from orion.core.contracts import ChatMessage
from orion.core.policy import NetworkPolicy, NetworkPolicyError
from orion.core.router import OrionRouter, RouteKind
from orion.llm.runtime import LlamaCppServer, create_local_model


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        return self.response


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

        answer = runtime.complete(
            [ChatMessage(role="user", content="Explain the issue")],
            max_tokens=64,
            timeout=3.0,
        )

        self.assertEqual(answer, "Local answer.")
        url, payload, timeout = transport.calls[0]
        self.assertEqual(url, "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(payload["messages"][0]["content"], "Explain the issue")
        self.assertEqual(payload["max_tokens"], 64)
        self.assertEqual(timeout, 3.0)

    def test_ollama_uses_local_endpoint_and_resource_options(self):
        transport = FakeTransport({"message": {"content": "Local Ollama answer."}})
        settings = replace(
            Settings(),
            model_backend="ollama",
            model_base_url="http://127.0.0.1:11434",
        )
        runtime = create_local_model(settings, transport=transport)

        answer = runtime.complete(
            [ChatMessage(role="user", content="Hello")], max_tokens=32
        )

        self.assertEqual(answer, "Local Ollama answer.")
        url, payload, _timeout = transport.calls[0]
        self.assertEqual(url, "http://127.0.0.1:11434/api/chat")
        self.assertFalse(payload["think"])
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["options"]["num_ctx"], settings.model_context_size)
        self.assertEqual(payload["options"]["num_thread"], settings.model_threads)
        self.assertFalse(payload["stream"])

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
        launches = []
        server = LlamaCppServer(
            Settings(),
            health_probe=lambda _url, _timeout: True,
            popen=lambda *args, **kwargs: launches.append((args, kwargs)),
        )

        server.start()

        self.assertEqual(launches, [])
        self.assertFalse(server.owned_process)

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
