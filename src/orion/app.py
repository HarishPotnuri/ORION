"""ORION application bootstrap."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence

from .config import Settings
from .logging_config import configure_logging
from .audio.capture import AudioGate
from .audio.stt import PushToTalkInput, WakeWordInput
from .audio.tts import KittenSpeechOutput
from .audio.wakeword import OpenWakeWordDetector
from .core.router import OrionRouter
from .core.orchestrator import RequestOrchestrator
from .memory.manager import MemoryManager
from .reminders.manager import ReminderManager
from .reminders.scheduler import ReminderNotifier, ReminderScheduler
from .tools.factory import create_default_tools

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orion")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate settings and exit",
    )
    parser.add_argument(
        "--voice-check",
        action="store_true",
        help="transcribe one utterance and verify local speech output",
    )
    parser.add_argument(
        "--route",
        metavar="REQUEST",
        help="show deterministic request classification without model calls",
    )
    parser.add_argument(
        "--run-reminders",
        action="store_true",
        help="run the persistent reminder scheduler",
    )
    parser.add_argument(
        "--request",
        metavar="TEXT",
        help="handle one typed request using local tools and the local model",
    )
    args = parser.parse_args(argv)

    try:
        settings = Settings.from_environment()
    except (ValueError, OSError) as error:
        print(f"ORION configuration error: {error}", file=sys.stderr)
        return 2

    configure_logging(settings)
    if args.check_config:
        logger.info("Configuration is valid; runtime adapters are not initialized")
    elif args.route is not None:
        decision = OrionRouter().route(args.route)
        action = f" {decision.action}" if decision.action else ""
        print(f"{decision.kind.value}{action}")
    elif args.voice_check:
        try:
            _run_voice_check(settings)
        except (OSError, RuntimeError, ValueError) as error:
            logger.error("Voice check failed: %s", error)
            return 1
    elif args.run_reminders:
        try:
            _run_reminder_service(settings)
        except (OSError, RuntimeError, ValueError) as error:
            logger.error("Reminder service failed: %s", error)
            return 1
    elif args.request is not None:
        try:
            _run_request(settings, args.request)
        except (OSError, RuntimeError, ValueError) as error:
            logger.error("Request failed: %s", error)
            return 1
    else:
        logger.info("ORION foundation loaded; use --voice-check to test local audio")
    return 0


def _run_voice_check(settings: Settings) -> None:
    try:
        from gpiozero import Button
    except ImportError as error:
        raise RuntimeError("gpiozero is required for push-to-talk input") from error

    gate = AudioGate()
    if settings.wake_word_enabled:
        detector = OpenWakeWordDetector(
            settings.wake_word, settings.wake_word_threshold
        )
        audio_input = WakeWordInput(settings, detector, gate=gate)
        logger.info("Waiting for wake word '%s'", settings.wake_word)
    else:
        button = Button(settings.push_to_talk_pin)
        audio_input = PushToTalkInput(settings, button, gate=gate)
        logger.info("Press GPIO %d to speak", settings.push_to_talk_pin)

    speech_output = KittenSpeechOutput(settings, gate)
    transcript = audio_input.next_utterance()
    if not transcript:
        logger.info("No speech detected")
        return
    logger.info("Transcript: %s", transcript)
    speech_output.speak("Local voice pipeline is ready.")


def _run_reminder_service(settings: Settings) -> None:
    try:
        from gpiozero import PWMLED
    except ImportError as error:
        raise RuntimeError("gpiozero is required for reminder LED output") from error

    led = PWMLED(settings.status_led_pin)
    manager = ReminderManager.from_settings(settings)
    speech_output = KittenSpeechOutput(settings, AudioGate())

    def set_status(status: str) -> None:
        if status == "SPEAKING":
            led.on()
        else:
            led.off()

    scheduler = ReminderScheduler(
        manager.store,
        ReminderNotifier(speech_output.speak, set_status),
        poll_interval=settings.reminder_poll_interval,
    )
    try:
        scheduler.start()
        logger.info("Reminder scheduler started")
        while True:
            time.sleep(settings.reminder_poll_interval)
    except KeyboardInterrupt:
        logger.info("Stopping reminder scheduler")
    finally:
        scheduler.stop()
        manager.close()
        led.close()


class _LazyLocalModel:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._runtime = None
        self._server = None

    def complete(self, messages, *, max_tokens: int, timeout: float) -> str:
        if self._runtime is None:
            from .llm.runtime import LlamaCppServer, create_local_model

            if self.settings.model_backend == "llama.cpp":
                self._server = LlamaCppServer(self.settings)
                self._server.start()
            self._runtime = create_local_model(self.settings)
        return self._runtime.complete(
            messages, max_tokens=max_tokens, timeout=timeout
        )

    def close(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None


def _run_request(settings: Settings, request: str) -> None:
    model = _LazyLocalModel(settings)
    memory = MemoryManager.from_settings(settings)
    reminders = ReminderManager.from_settings(settings)
    try:
        tools = create_default_tools(settings)
        orchestrator = RequestOrchestrator(
            settings,
            model,
            tools,
            memory=memory,
            reminders=reminders,
        )
        result = orchestrator.handle(request)
        print(result.response)
    finally:
        model.close()
        memory.close()
        reminders.close()