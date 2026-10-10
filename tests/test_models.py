import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from orion.config import Settings
from orion.model_cli import _benchmark, _benchmark_one, _current_selection, _select_model
from orion.model_cli import validate_model
from orion.models import (
    MODEL_REGISTRY,
    ModelSelectionStore,
    resolve_model_path,
    validate_model_file,
)


class ModelRegistryTests(unittest.TestCase):
    def test_registry_contains_requested_offline_inventory_ids(self):
        self.assertEqual(
            {key for key, spec in MODEL_REGISTRY.items() if spec.kind == "llm"},
            {"qwen3.5-2b-q4", "minicpm5-2b-q4", "qwen3.5-2b-q8", "qwen3-1.7b-q8", "phi4-mini-q4", "gemma4-e2b-q8"},
        )
        self.assertEqual(
            resolve_model_path(Path("/depot"), "minicpm5-2b-q4"),
            Path("/depot/LLM/MiniCPM5-2B-Q4_K_M.gguf"),
        )
        self.assertEqual(
            resolve_model_path(Path("/depot"), "qwen3.5-2b-q4"),
            Path("/depot/LLM/Qwen3.5-2B-Q4_K_M.gguf"),
        )
        self.assertEqual(
            resolve_model_path(Path("/depot"), "whisper-base-multilingual"),
            Path("/depot/STT/ggml-base.bin"),
        )

    def test_missing_model_and_missing_piper_sidecar_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ok, message, size = validate_model_file(root, "qwen3.5-2b-q4")
            self.assertFalse(ok)
            self.assertIn("Model file not found", message)
            self.assertIsNone(size)

            voice = resolve_model_path(root, "piper-lessac-medium")
            voice.parent.mkdir()
            voice.touch()
            ok, message, _size = validate_model_file(root, "piper-lessac-medium")
            self.assertFalse(ok)
            self.assertIn("Piper voice config not found", message)
            Path(f"{voice}.json").touch()
            self.assertTrue(validate_model_file(root, "piper-lessac-medium")[0])

    def test_selection_store_validates_roles_and_writes_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ModelSelectionStore(Path(directory) / "selection.json")
            store.write({"llm": "qwen3.5-2b-q8", "tts": "piper-hfc-female-medium"})
            self.assertEqual(store.read(), {
                "llm": "qwen3.5-2b-q8", "tts": "piper-hfc-female-medium"
            })
            with self.assertRaisesRegex(ValueError, "not a llm model"):
                store.write({"llm": "piper-lessac-medium"})

    def test_selection_file_overrides_legacy_model_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ORION-MODELS"
            state = Path(directory) / "state"
            selection_path = state / "model-selection.json"
            ModelSelectionStore(selection_path).write({"llm": "qwen3.5-2b-q8"})
            values = {
                "ORION_DATABASE_PATH": str(state / "orion.sqlite3"),
                "ORION_MODEL_ROOT": str(root),
                "ORION_MODEL_SELECTION_PATH": str(selection_path),
                "ORION_MODEL_PATH": str(Path(directory) / "old-model.gguf"),
            }
            with patch.dict("os.environ", values, clear=True):
                settings = Settings.from_environment()

            self.assertEqual(settings.active_llm_model, "qwen3.5-2b-q8")
            self.assertEqual(
                settings.model_path,
                root / "LLM" / "Qwen3.5-2B-Q8_0.gguf",
            )

    def test_environment_model_ids_resolve_stt_and_tts_without_manual_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "depot"
            values = {
                "ORION_DATABASE_PATH": str(Path(directory) / "state.sqlite3"),
                "ORION_MODEL_ROOT": str(root),
                "ORION_STT_MODEL_ID": "whisper-tiny-en",
                "ORION_TTS_MODEL_ID": "piper-hfc-female-medium",
            }
            with patch.dict("os.environ", values, clear=True):
                settings = Settings.from_environment()

        self.assertEqual(settings.active_stt_model, "whisper-tiny-en")
        self.assertEqual(settings.stt_backend, "whisper.cpp")
        self.assertEqual(settings.stt_model_path, root / "STT" / "ggml-tiny.en.bin")
        self.assertEqual(settings.active_tts_model, "piper-hfc-female-medium")
        self.assertEqual(settings.tts_backend, "piper")
        self.assertEqual(
            settings.tts_model_path,
            root / "TTS" / "en_US-hfc_female-medium.onnx",
        )

    def test_use_command_persists_id_and_current_settings_reads_it(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            values = {
                "ORION_DATABASE_PATH": str(state / "orion.sqlite3"),
                "ORION_MODEL_ROOT": str(state / "depot"),
                "ORION_MODEL_SELECTION_PATH": str(state / "selection.json"),
            }
            with patch.dict("os.environ", values, clear=True):
                settings = Settings.from_environment()
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(_select_model(settings, "qwen3.5-2b-q8"), 0)
                selected = Settings.from_environment()

            self.assertIn("Restart ORION service", output.getvalue())
            self.assertEqual(selected.active_llm_model, "qwen3.5-2b-q8")
            self.assertEqual(_current_selection(selected)["llm"], "qwen3.5-2b-q8")

    def test_benchmark_writes_machine_readable_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = replace(
                Settings(),
                model_root=root,
                model_selection_path=root / "selection.json",
            )
            expected = {
                "id": "qwen3.5-2b-q4",
                "model_name": "Qwen3.5 2B Q4_K_M",
                "success": True,
                "prompts": [],
            }
            with patch("orion.model_cli._benchmark_one", return_value=expected):
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(_benchmark(settings, None, True, None), 0)
            report_path = root / "model-benchmarks.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(len(report["prompts"]), 4)
            self.assertTrue(report["results"][0]["success"])

    def test_benchmark_refuses_external_server(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = resolve_model_path(root, "qwen3.5-2b-q4")
            model_path.parent.mkdir()
            model_path.touch()
            provider = SimpleNamespace(
                _server=SimpleNamespace(owned_process=False),
                load=Mock(),
                unload=Mock(),
            )
            settings = replace(Settings(), model_root=root)
            with patch("orion.model_cli.LlamaCppProvider", return_value=provider):
                with self.assertRaisesRegex(RuntimeError, "requires an ORION-owned"):
                    _benchmark_one(settings, "qwen3.5-2b-q4")

            provider.unload.assert_called_once()


if __name__ == "__main__":
    unittest.main()


def test_real_local_model_smoke_when_explicitly_enabled():
    import pytest

    model_id = os.getenv("ORION_TEST_REAL_MODEL_ID")
    if not model_id:
        pytest.skip("Set ORION_TEST_REAL_MODEL_ID to opt into a real local artifact check")
    settings = Settings.from_environment()
    result = validate_model(settings, model_id)
    assert result["success"], result
