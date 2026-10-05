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
        self.assertEqual(settings.status_led_red_pin, 27)
        self.assertEqual(settings.status_led_pin, 25)
        self.assertEqual(settings.status_led_blue_pin, 17)

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


if __name__ == "__main__":
    unittest.main()