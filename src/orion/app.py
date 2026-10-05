"""ORION application bootstrap."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from collections.abc import Sequence
from threading import Event

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


class _StatusSpeechOutput:
    def __init__(self, speech_output, set_status):
        self.speech_output = speech_output
        self.set_status = set_status

    def warmup(self) -> None:
        self.speech_output.warmup()

    def speak(self, text: str) -> None:
        self.set_status("SPEAKING")
        try:
            self.speech_output.speak(text)
        finally:
            self.set_status("IDLE")


def _confirm_spoken_action(prompt: str, audio_input, speech_output) -> bool:
    speech_output.speak(prompt)
    answer = audio_input.next_utterance(timeout=15.0).strip().casefold().strip(" .!?\t")
    return answer in {"yes", "yes confirm", "confirm", "do it"}


def _set_led_status(led, status: str) -> None:
    color = {
        "IDLE": (0.0, 0.0, 0.0),
        "LISTENING": (0.0, 0.2, 0.0),
        "PROCESSING": (0.0, 0.0, 0.45),
        "SPEAKING": (0.0, 0.7, 0.7),
        "MUTED": (0.05, 0.025, 0.0),
        "ERROR": (1.0, 0.0, 0.0),
    }.get(status, (0.0, 0.0, 0.0))
    led.color = color


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
        "--run-voice",
        action="store_true",
        help="run the continuous local voice assistant and reminder scheduler",
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
    elif args.run_voice:
        try:
            _run_voice_service(settings)
        except (OSError, RuntimeError, ValueError) as error:
            logger.error("Voice service failed: %s", error)
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


def _create_audio_input(settings: Settings, gate: AudioGate, stop_event: Event):
    try:
        from gpiozero import Button
    except ImportError as error:
        raise RuntimeError("gpiozero is required for voice input") from error

    if settings.wake_word_enabled:
        detector = OpenWakeWordDetector(
            settings.wake_word, settings.wake_word_threshold
        )
        audio_input = WakeWordInput(settings, detector, gate=gate, stop_event=stop_event)
        audio_input.warmup()
        logger.info("Waiting for wake word '%s'", settings.wake_word)
        return audio_input, None
    button = Button(settings.push_to_talk_pin)
    audio_input = PushToTalkInput(settings, button, gate=gate, stop_event=stop_event)
    try:
        audio_input.warmup()
    except Exception:
        button.close()
        raise
    logger.info("Press GPIO %d to speak", settings.push_to_talk_pin)
    return audio_input, button


def _create_interrupt_button(settings: Settings, gate: AudioGate, push_to_talk_button=None):
    try:
        from gpiozero import Button
    except ImportError as error:
        raise RuntimeError("gpiozero is required for voice interruption") from error

    interrupt_button = Button(settings.tts_interrupt_pin)
    interrupt_button.when_pressed = gate.request_interrupt
    if push_to_talk_button is not None:
        push_to_talk_button.when_pressed = gate.request_interrupt
    return interrupt_button


def _run_voice_check(settings: Settings) -> None:
    gate = AudioGate()
    stop_event = Event()
    speech_output = KittenSpeechOutput(settings, gate)
    speech_output.warmup()
    audio_input, button = _create_audio_input(settings, gate, stop_event)
    try:
        transcript = audio_input.next_utterance()
        if not transcript:
            logger.info("No speech detected")
            return
        logger.info("Transcript: %s", transcript)
        speech_output.speak("Local voice pipeline is ready.")
    finally:
        stop_event.set()
        if button is not None:
            button.close()


def _run_voice_loop(audio_input, orchestrator, speech_output, scheduler, stop_event, set_status):
    try:
        scheduler.start()
        logger.info("Voice assistant started")
        turn_number = 0
        while not stop_event.is_set():
            turn_number += 1
            stage_name = "input"
            stage_started = time.monotonic()
            try:
                set_status("LISTENING")
                transcript = audio_input.next_utterance()
                logger.info(
                    "Voice stage=input_total turn=%d elapsed_ms=%.1f outcome=%s",
                    turn_number,
                    (time.monotonic() - stage_started) * 1000,
                    "received" if transcript.strip() else "empty",
                )
                if stop_event.is_set():
                    break
                if not transcript.strip():
                    speech_output.speak("I didn't catch that. Please try again.")
                    continue
                logger.info("Transcript received (%d characters)", len(transcript))
                set_status("PROCESSING")
                stage_name = "orchestration"
                stage_started = time.monotonic()
                result = orchestrator.handle(transcript)
                logger.info(
                    "Voice stage=orchestration turn=%d elapsed_ms=%.1f route=%s completed=%s",
                    turn_number,
                    (time.monotonic() - stage_started) * 1000,
                    result.route.kind.value,
                    result.completed,
                )
                set_status("SPEAKING")
                stage_name = "tts"
                stage_started = time.monotonic()
                speech_output.speak(result.response)
                logger.info(
                    "Voice stage=tts turn=%d elapsed_ms=%.1f outcome=returned",
                    turn_number,
                    (time.monotonic() - stage_started) * 1000,
                )
            except Exception as error:
                logger.warning(
                    "Voice turn failed turn=%d stage=%s elapsed_ms=%.1f (%s)",
                    turn_number,
                    stage_name,
                    (time.monotonic() - stage_started) * 1000,
                    type(error).__name__,
                )
                set_status("ERROR")
                if not stop_event.is_set():
                    try:
                        speech_output.speak(
                            "I couldn't process that turn. Please try again."
                        )
                    except Exception as speech_error:
                        logger.warning(
                            "Voice error feedback failed (%s)",
                            type(speech_error).__name__,
                        )
                if not stop_event.wait(0.5):
                    set_status("LISTENING")
    finally:
        stop_event.set()
        try:
            scheduler.stop()
        finally:
            set_status("IDLE")


def _run_voice_service(settings: Settings) -> None:
    try:
        from gpiozero import RGBLED
    except ImportError as error:
        raise RuntimeError("gpiozero is required for voice status RGB LED output") from error

    stop_event = Event()
    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    gate = AudioGate()
    led = None
    audio_input = button = interrupt_button = None
    model = memory = reminders = scheduler = speech_output = None

    def set_status(status: str) -> None:
        if led is not None:
            _set_led_status(led, status)

    def speak(text: str) -> None:
        speech_output.speak(text)

    try:
        led = RGBLED(
            red=settings.status_led_red_pin,
            green=settings.status_led_pin,
            blue=settings.status_led_blue_pin,
        )
        model = _LazyLocalModel(settings)
        memory = MemoryManager.from_settings(settings)
        reminders = ReminderManager.from_settings(settings)
        speech_output = _StatusSpeechOutput(
            KittenSpeechOutput(settings, gate), set_status
        )
        speech_output.warmup()
        scheduler = ReminderScheduler(
            reminders.store,
            ReminderNotifier(speak, set_status),
            poll_interval=settings.reminder_poll_interval,
        )

        def stop_service(_signum, _frame) -> None:
            stop_event.set()
            gate.request_interrupt()

        signal.signal(signal.SIGINT, stop_service)
        signal.signal(signal.SIGTERM, stop_service)
        audio_input, button = _create_audio_input(settings, gate, stop_event)

        interrupt_button = _create_interrupt_button(settings, gate, button)

        def confirm_spoken(prompt: str) -> bool:
            return _confirm_spoken_action(prompt, audio_input, speech_output)

        orchestrator = RequestOrchestrator(
            settings,
            model,
            create_default_tools(settings),
            memory=memory,
            reminders=reminders,
            confirm_action=confirm_spoken,
        )
        _run_voice_loop(
            audio_input, orchestrator, speech_output, scheduler, stop_event, set_status
        )
    finally:
        stop_event.set()
        if audio_input is not None:
            audio_input.stop_event.set()
        cleanup_actions = (
            ("scheduler", scheduler.stop if scheduler is not None else None),
            ("model", model.close if model is not None else None),
            ("memory", memory.close if memory is not None else None),
            ("reminders", reminders.close if reminders is not None else None),
            ("interrupt button", interrupt_button.close if interrupt_button is not None else None),
            (
                "push-to-talk button",
                button.close
                if button is not None and button is not interrupt_button
                else None,
            ),
            ("status LED", led.close if led is not None else None),
            ("SIGINT handler", lambda: signal.signal(signal.SIGINT, old_sigint)),
            ("SIGTERM handler", lambda: signal.signal(signal.SIGTERM, old_sigterm)),
        )
        for name, cleanup in cleanup_actions:
            if cleanup is None:
                continue
            try:
                cleanup()
            except Exception as error:
                logger.warning("Could not clean up %s (%s)", name, type(error).__name__)


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
