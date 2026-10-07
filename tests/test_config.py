import os
import unittest
from pathlib import Path
from unittest.mock import patch

from orion.app import main
from orion.config import Settings


class SettingsTests(unittest.TestCase):
    def clean_environment(self, values: dict[str, str] | None = None):
        environment = {"ORION_DATABASE_PATH": str(Path.cwd() / "orion-test.sqlite3")}
        environment.update(values or {})
        return patch.dict(os.environ, environment, clear=True)

    def test_defaults_are_resource_bounded_and_local(self) -> None:
        with self.clean_environment():
            settings = Settings.from_environment()

        self.assertEqual(settings.model_context_size, 8192)
        self.assertEqual(settings.max_context_tokens, 6000)
        self.assertEqual(settings.model_backend, "llama.cpp")
        self.assertEqual(settings.model_base_url, "http://127.0.0.1:8080/v1")
        self.assertEqual(settings.memory_recent_message_limit, 16)
        self.assertEqual(settings.memory_search_limit, 5)
        self.assertEqual(settings.reminder_timezone, "Asia/Kolkata")
        self.assertEqual(settings.tts_interrupt_pin, 16)
        self.assertEqual(settings.audio_device, "default")
        self.assertEqual(settings.status_led_red_pin, 27)
        self.assertEqual(settings.status_led_pin, 25)
        self.assertEqual(settings.status_led_blue_pin, 17)
        self.assertEqual(settings.voice_profile, "legacy")
        self.assertEqual(settings.stt_backend, "vosk")
        self.assertEqual(
            settings.stt_model_path,
            Path("models/vosk-model-en-us-0.22-lgraph"),
        )
        self.assertIsNone(settings.wake_word_model_path)

    def test_audio_device_can_be_overridden_for_input_and_output(self) -> None:
        with self.clean_environment({"ORION_AUDIO_DEVICE": "USB fallback"}):
            settings = Settings.from_environment()

        self.assertEqual(settings.audio_device, "USB fallback")

    def test_local_tts_model_path_can_be_configured(self) -> None:
        model_path = Path.cwd() / "kitten-model"
        with self.clean_environment({"ORION_TTS_MODEL_PATH": str(model_path)}):
            settings = Settings.from_environment()

        self.assertEqual(settings.tts_model_path, model_path)

    def test_legacy_wake_word_uses_configurable_local_jarvis_assets(self) -> None:
        with self.clean_environment({
            "ORION_VOICE_PROFILE": "legacy",
            "ORION_WAKE_WORD_ENABLED": "true",
        }):
            settings = Settings.from_environment()

        self.assertEqual(settings.wake_word_model_path, Path("models/hey_jarvis.onnx"))
        self.assertEqual(
            settings.wake_word_melspectrogram_path,
            Path("artifacts/wakeword-staging/melspectrogram.onnx"),
        )
        self.assertEqual(
            settings.wake_word_embedding_path,
            Path("artifacts/wakeword-staging/embedding_model.onnx"),
        )
        self.assertIsNone(settings.wake_word_manifest_path)
        self.assertEqual(settings.wake_word_cooldown_seconds, 3.0)

    def test_local_jarvis_assets_threshold_and_cooldown_can_be_overridden(self) -> None:
        values = {
            "ORION_WAKE_WORD_ENABLED": "true",
            "ORION_WAKE_WORD_MODEL_PATH": "local/hey_jarvis.onnx",
            "ORION_WAKE_WORD_THRESHOLD": "0.72",
            "ORION_WAKE_WORD_COOLDOWN_SECONDS": "3.5",
        }
        with self.clean_environment(values):
            settings = Settings.from_environment()

        self.assertEqual(settings.wake_word_model_path, Path("local/hey_jarvis.onnx"))
        self.assertEqual(settings.wake_word_threshold, 0.72)
        self.assertEqual(settings.wake_word_cooldown_seconds, 3.5)

    def test_explicit_legacy_profile_keeps_vosk_defaults(self) -> None:
        with self.clean_environment({"ORION_VOICE_PROFILE": "legacy"}):
            settings = Settings.from_environment()

        self.assertEqual(settings.voice_profile, "legacy")
        self.assertEqual(settings.stt_backend, "vosk")
        self.assertIsNone(settings.wake_word_model_path)

    def test_explicit_legacy_vosk_is_active_by_default(self) -> None:
        with self.clean_environment():
            settings = Settings.from_environment()

        self.assertEqual(settings.voice_profile, "legacy")
        self.assertEqual(settings.stt_backend, "vosk")
        self.assertFalse(settings.wake_word_enabled)

    def test_rejects_non_local_model_endpoint(self) -> None:
        with self.clean_environment(
            {"ORION_MODEL_BASE_URL": "https://example.com/v1"}
        ):
            with self.assertRaisesRegex(ValueError, "must point to localhost"):
                Settings.from_environment()

    def test_rejects_invalid_context_budget(self) -> None:
        with self.clean_environment({"ORION_MAX_CONTEXT_TOKENS": "8192"}):
            with self.assertRaisesRegex(ValueError, "below model context size"):
                Settings.from_environment()

    def test_rejects_reused_gpio_pins(self) -> None:
        with self.clean_environment({"ORION_STATUS_LED_PIN": "23"}):
            with self.assertRaisesRegex(ValueError, "must be unique"):
                Settings.from_environment()

    def test_rejects_interrupt_button_gpio_conflict(self) -> None:
        with self.clean_environment({"ORION_TTS_INTERRUPT_PIN": "23"}):
            with self.assertRaisesRegex(ValueError, "must be unique"):
                Settings.from_environment()

    def test_rejects_rgb_led_gpio_conflict(self) -> None:
        with self.clean_environment({"ORION_STATUS_LED_BLUE_PIN": "27"}):
            with self.assertRaisesRegex(ValueError, "must be unique"):
                Settings.from_environment()

    def test_rejects_nonpositive_memory_limits(self) -> None:
        with self.clean_environment({"ORION_MEMORY_SEARCH_LIMIT": "0"}):
            with self.assertRaisesRegex(ValueError, "memory limits must be positive"):
                Settings.from_environment()

    def test_rejects_nonfinite_voice_limits(self) -> None:
        for key, value in (
            ("ORION_STT_TIMEOUT", "nan"),
            ("ORION_MICROPHONE_BUTTON_DEBOUNCE", "inf"),
        ):
            with self.subTest(key=key), self.clean_environment({key: value}):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    Settings.from_environment()

    def test_rejects_invalid_reminder_timezone(self) -> None:
        with self.clean_environment({"ORION_REMINDER_TIMEZONE": "Mars/Olympus"}):
            with self.assertRaisesRegex(ValueError, "valid IANA timezone"):
                Settings.from_environment()

    def test_accepts_configured_searxng_only_with_matching_allowlist(self) -> None:
        values = {
            "ORION_SEARXNG_BASE_URL": "https://search.example.test",
            "ORION_SEARXNG_ALLOWED_HOSTS": "search.example.test",
        }
        with self.clean_environment(values):
            settings = Settings.from_environment()

        self.assertEqual(settings.searxng_allowed_hosts, ("search.example.test",))

    def test_rejects_searxng_host_missing_from_allowlist(self) -> None:
        values = {
            "ORION_SEARXNG_BASE_URL": "https://search.example.test",
            "ORION_SEARXNG_ALLOWED_HOSTS": "other.example.test",
        }
        with self.clean_environment(values):
            with self.assertRaisesRegex(ValueError, "allowlisted HTTPS host"):
                Settings.from_environment()

    def test_cli_reports_configuration_errors(self) -> None:
        with self.clean_environment(
            {"ORION_MODEL_BASE_URL": "https://example.com/v1"}
        ):
            self.assertEqual(main(["--check-config"]), 2)

    def test_cli_forces_hugging_face_offline_before_runtime_initialization(self) -> None:
        with self.clean_environment({"HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "0"}):
            self.assertEqual(main(["--check-config"]), 0)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["HF_DATASETS_OFFLINE"], "1")

    def test_cli_validates_wake_assets_without_starting_voice_service(self) -> None:
        values = {
            "ORION_VOICE_PROFILE": "wakeword",
            "ORION_WAKE_WORD_MODEL_PATH": "/models/hey_orion.onnx",
            "ORION_WAKE_WORD_MELSPECTROGRAM_PATH": "/models/melspectrogram.onnx",
            "ORION_WAKE_WORD_EMBEDDING_PATH": "/models/embedding.onnx",
            "ORION_WAKE_WORD_MANIFEST_PATH": "/models/phase13-artifacts.json",
        }
        with self.clean_environment(values), patch(
            "orion.app.validate_wakeword_settings"
        ) as validate:
            self.assertEqual(main(["--validate-wakeword-assets"]), 0)

        validate.assert_called_once()
        self.assertEqual(validate.call_args.args[0].voice_profile, "wakeword")

    def test_wake_word_cooldown_must_not_be_negative(self) -> None:
        with self.clean_environment({"ORION_WAKE_WORD_COOLDOWN_SECONDS": "-0.1"}):
            with self.assertRaisesRegex(ValueError, "wake cooldown cannot be negative"):
                Settings.from_environment()


if __name__ == "__main__":
    unittest.main()