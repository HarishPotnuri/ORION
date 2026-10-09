import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path

from orion.config import Settings
from orion.core.contracts import ChatMessage, ModelStreamEvent
from orion.llm.provider import LlamaCppProvider


class FakeServer:
    active = 0
    peak_active = 0
    started_models = []

    def __init__(self, settings):
        self.settings = settings
        self._process = None

    def start(self):
        type(self).active += 1
        type(self).peak_active = max(type(self).peak_active, type(self).active)
        type(self).started_models.append(self.settings.active_llm_model)

    def stop(self):
        type(self).active -= 1


class FakeRuntime:
    def complete(self, messages, *, max_tokens, timeout):
        return f"answer:{messages[-1].content}:{max_tokens}:{timeout > 0}"

    def stream(self, messages, *, max_tokens, timeout, cancel_event):
        yield ModelStreamEvent(kind="text", text="partial")
        yield ModelStreamEvent(kind="completed", finish_reason="stop")


class LlamaCppProviderTests(unittest.TestCase):
    def setUp(self):
        FakeServer.active = 0
        FakeServer.peak_active = 0
        FakeServer.started_models = []

    def test_load_generate_stream_unload_and_switch_keeps_one_model_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            q4 = root / "LLM" / "Qwen3.5-2B-Q4_K_M.gguf"
            q8 = root / "LLM" / "Qwen3.5-2B-Q8_0.gguf"
            q4.parent.mkdir()
            q4.touch()
            q8.touch()
            settings = replace(
                Settings(), model_root=root, model_path=q4,
                active_llm_model="qwen3.5-2b-q4",
            )
            provider = LlamaCppProvider(
                settings,
                server_factory=FakeServer,
                runtime_factory=lambda _settings: FakeRuntime(),
            )

            self.assertFalse(provider.is_loaded())
            provider.load()
            self.assertTrue(provider.is_loaded())
            result = provider.generate(
                [ChatMessage(role="user", content="hello")],
                max_tokens=8,
                timeout=1,
            )
            self.assertIn("answer:hello", result)
            events = list(provider.generate_stream(
                [ChatMessage(role="user", content="stream")],
                max_tokens=8,
                timeout=1,
                cancel_event=threading.Event(),
            ))
            self.assertEqual(events[0].text, "partial")

            provider.switch_model("qwen3.5-2b-q8")
            self.assertTrue(provider.is_loaded())
            self.assertEqual(provider.model_info()["id"], "qwen3.5-2b-q8")
            self.assertEqual(FakeServer.started_models, ["qwen3.5-2b-q4", "qwen3.5-2b-q8"])
            self.assertEqual(FakeServer.peak_active, 1)
            self.assertEqual(FakeServer.active, 1)
            provider.unload()
            self.assertFalse(provider.is_loaded())
            self.assertEqual(FakeServer.active, 0)

    def test_switch_refuses_missing_model_without_unloading_current(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "LLM" / "Qwen3.5-2B-Q4_K_M.gguf"
            model_path.parent.mkdir()
            model_path.touch()
            provider = LlamaCppProvider(
                replace(Settings(), model_root=root, model_path=model_path),
                server_factory=FakeServer,
                runtime_factory=lambda _settings: FakeRuntime(),
            )
            provider.load()

            with self.assertRaisesRegex(FileNotFoundError, "Model file not found"):
                provider.switch_model("qwen3.5-2b-q8")

            self.assertTrue(provider.is_loaded())
            self.assertEqual(FakeServer.active, 1)
            provider.unload()

    def test_unload_rejects_active_request(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.touch()
            provider = LlamaCppProvider(
                replace(Settings(), model_path=model_path),
                server_factory=FakeServer,
                runtime_factory=lambda _settings: FakeRuntime(),
            )
            with provider._using_runtime():
                with self.assertRaisesRegex(RuntimeError, "active LLM request"):
                    provider.unload()
            provider.unload()

    def test_failed_unload_retains_current_model_and_blocks_second_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            q4 = root / "LLM" / "Qwen3.5-2B-Q4_K_M.gguf"
            q8 = root / "LLM" / "Qwen3.5-2B-Q8_0.gguf"
            q4.parent.mkdir()
            q4.touch()
            q8.touch()

            class FailingStopServer(FakeServer):
                def stop(self):
                    raise RuntimeError("stop failed")

            provider = LlamaCppProvider(
                replace(
                    Settings(), model_root=root, model_path=q4,
                    active_llm_model="qwen3.5-2b-q4",
                ),
                server_factory=FailingStopServer,
                runtime_factory=lambda _settings: FakeRuntime(),
            )
            provider.load()

            with self.assertRaisesRegex(RuntimeError, "stop failed"):
                provider.switch_model("qwen3.5-2b-q8")

            self.assertTrue(provider.is_loaded())
            self.assertEqual(provider.model_info()["id"], "qwen3.5-2b-q4")
            self.assertEqual(FailingStopServer.started_models, ["qwen3.5-2b-q4"])


if __name__ == "__main__":
    unittest.main()
