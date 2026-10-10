"""ORION application bootstrap."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from threading import Event, RLock, Thread
from pathlib import Path

from .config import Settings
from .logging_config import configure_logging
from .audio.capture import AudioGate
from .audio.control import MicrophoneControl
from .audio.factory import create_speech_output, create_transcriber
from .audio.listener import WakeListenerInput
from .audio.stt import PushToTalkInput, WakeWordInput, WhisperPushToTalkInput
from .audio.streaming import SpeechPhraseBuffer
from .audio.wakeword import OpenWakeWordDetector, validate_settings as validate_wakeword_settings
from .core.router import OrionRouter
from .core.orchestrator import RequestOrchestrator
from .llm.provider import LlamaCppProvider
from .memory.manager import MemoryManager
from .models import (
    WAKEWORD_EMBEDDING_FILENAME,
    WAKEWORD_MELSPECTROGRAM_FILENAME,
    resolve_model_path,
)
from .reminders.manager import ReminderManager
from .reminders.scheduler import ReminderNotifier, ReminderScheduler
from .tools.factory import create_default_tools

logger = logging.getLogger(__name__)


class _StatusSpeechOutput:
    def __init__(self, speech_output, set_status, gate: AudioGate | None = None):
        self.speech_output = speech_output
        self.set_status = set_status
        self.gate = gate
        self._lock = RLock()
        self._session_depth = 0

    def warmup(self) -> None:
        self.speech_output.warmup()

    def speak(self, text: str) -> bool:
        with self._lock:
            outer_call = self._session_depth == 0
            self.set_status("SPEAKING")
            try:
                completed = self.speech_output.speak(text)
                interrupt_event = self.interrupt_event
                return completed is not False and not (
                    interrupt_event is not None and interrupt_event.is_set()
                )
            finally:
                if outer_call:
                    self.set_status("IDLE")

    @contextmanager
    def session(self):
        gate_context = self.gate.speaking() if self.gate is not None else nullcontext()
        with self._lock, gate_context:
            self._session_depth += 1
            try:
                yield self
            finally:
                self._session_depth -= 1
                if self._session_depth == 0:
                    self.set_status("IDLE")

    @property
    def interrupt_event(self):
        return self.gate.interrupt_event if self.gate is not None else None

    @contextmanager
    def turn(self):
        if self.gate is None:
            yield Event()
            return
        with self.gate.interruptible_turn() as cancellation:
            yield cancellation

    def evict_if_idle(self, timeout: float) -> bool:
        evict = getattr(self.speech_output, "evict_if_idle", None)
        return bool(evict and evict(timeout))


class _VoiceStatusLED:
    def __init__(self):
        self.led = None
        self._status = "IDLE"
        self._muted = False

    def attach(self, led) -> None:
        self.led = led
        self._render()

    def set_status(self, status: str) -> None:
        self._status = status
        self._render()

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        self._render()

    def _render(self) -> None:
        if self.led is not None:
            _set_led_status(self.led, "MUTED" if self._muted else self._status)


def _confirm_spoken_action(prompt: str, audio_input, speech_output) -> bool:
    speech_output.speak(prompt)
    answer = audio_input.next_utterance(timeout=15.0).strip().casefold().strip(" .!?\t")
    if hasattr(audio_input, "claim_utterance") and not audio_input.claim_utterance():
        return False
    return answer in {"yes", "yes confirm", "confirm", "do it"}


def _transcription_confidence(audio_input) -> float | None:
    confidence = getattr(audio_input, "transcription_confidence", None)
    if confidence is None:
        transcriber = getattr(audio_input, "transcriber", audio_input)
        confidence = getattr(transcriber, "last_confidence", None)
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        return None
    return float(confidence)


def _capture_transcript_confirmation(audio_input, *, followup_active: bool) -> str:
    next_followup = getattr(audio_input, "next_followup_utterance", None)
    if followup_active and next_followup is not None:
        answer = next_followup(timeout=10.0)
    else:
        answer = audio_input.next_utterance(timeout=15.0)
    if hasattr(audio_input, "claim_utterance") and not audio_input.claim_utterance():
        return ""
    return answer.strip()


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
        "--validate-wakeword-assets",
        action="store_true",
        help="validate local wake-word ONNX files and manifest, then exit",
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
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    if args.check_config:
        logger.info("Configuration is valid; runtime adapters are not initialized")
    elif args.validate_wakeword_assets:
        try:
            validate_wakeword_settings(settings)
        except (OSError, RuntimeError, ValueError) as error:
            logger.error("Wake-word asset validation failed: %s", error)
            return 1
        logger.info("Wake-word ONNX assets, hashes, metadata, and runtime compatibility are valid")
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


def _create_audio_input(
    settings: Settings, gate: AudioGate, stop_event: Event, set_status=None,
    set_microphone_muted=None,
):
    try:
        from gpiozero import Button
    except ImportError as error:
        raise RuntimeError("gpiozero is required for voice input") from error

    if settings.voice_profile == "wakeword" or settings.wake_word_enabled:
        try:
            detector = OpenWakeWordDetector.from_settings(settings)
        except FileNotFoundError as hey_orion_error:
            if (
                settings.active_wakeword_model != "hey-orion"
                or (settings.voice_profile == "legacy" and settings.wake_word_enabled)
            ):
                raise
            jarvis_model = resolve_model_path(settings.model_root, "hey-jarvis")
            if not jarvis_model.is_file():
                jarvis_model = Path("models/hey_jarvis.onnx")
            mel_path = settings.model_root / "WAKEWORD" / WAKEWORD_MELSPECTROGRAM_FILENAME
            embedding_path = settings.model_root / "WAKEWORD" / WAKEWORD_EMBEDDING_FILENAME
            if not mel_path.is_file():
                mel_path = Path("artifacts/wakeword-staging/melspectrogram.onnx")
            if not embedding_path.is_file():
                embedding_path = Path("artifacts/wakeword-staging/embedding_model.onnx")
            jarvis_settings = replace(
                settings,
                active_wakeword_model="hey-jarvis",
                wake_word="hey_jarvis",
                wake_word_model_path=jarvis_model,
                wake_word_melspectrogram_path=mel_path,
                wake_word_embedding_path=embedding_path,
                wake_word_manifest_path=None,
            )
            try:
                detector = OpenWakeWordDetector.from_settings(jarvis_settings)
                settings = jarvis_settings
                logger.warning("Hey Orion assets unavailable; using Hey Jarvis staging model")
            except FileNotFoundError as jarvis_error:
                fallback = replace(
                    settings,
                    voice_profile="legacy",
                    wake_word_enabled=False,
                    stt_backend="vosk",
                    active_stt_model="vosk-en-small",
                    stt_model_path=settings.vosk_model_path,
                )
                if not fallback.vosk_model_path.expanduser().is_dir():
                    raise RuntimeError(
                        "Hey Orion and Hey Jarvis assets are incomplete; "
                        f"legacy VOSK model is unavailable: {fallback.vosk_model_path}"
                    ) from jarvis_error
                gate.allow_wakeword_during_speech = False
                logger.warning(
                    "Wakeword assets incomplete (Hey Orion: %s; Hey Jarvis: %s); "
                    "starting legacy VOSK/PTT",
                    hey_orion_error,
                    jarvis_error,
                )
                return _create_audio_input(
                    fallback, gate, stop_event, set_status, set_microphone_muted
                )
        control = MicrophoneControl(settings.database_path.expanduser())
        transcriber = None
        audio_input = None
        button = None
        try:
            transcriber = create_transcriber(settings)
            audio_input = WakeListenerInput(
                settings, detector, transcriber, control, gate=gate,
                stop_event=stop_event,
                on_wake=(lambda: set_status("LISTENING")) if set_status is not None else None,
            )
            button = Button(
                settings.microphone_control_pin, bounce_time=settings.microphone_button_debounce
            )

            def toggle_microphone():
                muted = control.toggle()
                if set_microphone_muted is not None:
                    set_microphone_muted(muted)
                elif set_status is not None:
                    set_status("MUTED" if muted else "IDLE")
                logger.info("Microphone state=%s", "muted" if muted else "unmuted")

            button.when_pressed = toggle_microphone
            audio_input.start()
            if set_microphone_muted is not None:
                set_microphone_muted(control.muted)
            if set_status is not None:
                set_status("IDLE")
        except Exception:
            if button is not None:
                try:
                    button.close()
                except Exception as error:
                    logger.warning("Could not clean up microphone button (%s)", type(error).__name__)
            if audio_input is not None:
                try:
                    audio_input.close()
                except Exception as error:
                    logger.warning("Could not clean up wake listener (%s)", type(error).__name__)
            else:
                if transcriber is not None:
                    close_transcriber = getattr(transcriber, "close", None)
                    if close_transcriber is not None:
                        try:
                            close_transcriber()
                        except Exception as error:
                            logger.warning("Could not clean up STT adapter (%s)", type(error).__name__)
                control.close()
            raise
        logger.info(
            "Waiting for wake word '%s'; GPIO %d toggles microphone mute",
            settings.wake_word,
            settings.microphone_control_pin,
        )
        return audio_input, button
    button = Button(settings.push_to_talk_pin)
    if settings.stt_backend == "whisper.cpp":
        transcriber = create_transcriber(settings)
        audio_input = WhisperPushToTalkInput(
            settings, button, transcriber, gate=gate, stop_event=stop_event
        )
    else:
        audio_input = PushToTalkInput(settings, button, gate=gate, stop_event=stop_event)
    logger.info("Press GPIO %d to speak", settings.push_to_talk_pin)
    return audio_input, button


def _create_interrupt_button(settings: Settings, gate: AudioGate, push_to_talk_button=None):
    try:
        from gpiozero import Button
    except ImportError as error:
        raise RuntimeError("gpiozero is required for voice interruption") from error

    interrupt_button = Button(settings.tts_interrupt_pin)
    interrupt_button.when_pressed = gate.request_interrupt
    if (
        push_to_talk_button is not None
        and settings.voice_profile == "legacy"
        and not settings.wake_word_enabled
    ):
        push_to_talk_button.when_pressed = gate.request_interrupt
    return interrupt_button


def _run_voice_check(settings: Settings) -> None:
    gate = AudioGate(
        allow_wakeword_during_speech=(
            settings.voice_profile == "wakeword" or settings.wake_word_enabled
        )
    )
    stop_event = Event()
    speech_output = create_speech_output(settings, gate)
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
        close_input = getattr(audio_input, "close", None)
        if close_input is not None:
            close_input()
        if button is not None:
            button.close()


def _run_voice_loop(audio_input, orchestrator, speech_output, scheduler, stop_event, set_status):
    try:
        scheduler.start()
        logger.info("Voice assistant started")
        turn_number = 0
        followup_active = False
        while not stop_event.is_set():
            turn_number += 1
            stage_name = "input"
            stage_started = time.monotonic()
            spoken = False
            generated_text: list[str] = []
            delivered_text: list[str] = []
            try:
                set_status(
                    "IDLE"
                    if isinstance(audio_input, (WakeListenerInput, WakeWordInput))
                    else "LISTENING"
                )
                logger.info("Voice stage=capture turn=%d outcome=started", turn_number)
                next_followup = getattr(audio_input, "next_followup_utterance", None)
                if followup_active and next_followup is not None:
                    transcript = next_followup(timeout=10.0)
                else:
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
                    if isinstance(audio_input, WakeListenerInput):
                        followup_active = False
                        continue
                    speech_output.speak("I didn't catch that. Please try again.")
                    continue
                if hasattr(audio_input, "claim_utterance") and not audio_input.claim_utterance():
                    continue
                logger.info("Voice transcript turn=%d text=%r", turn_number, transcript)
                confidence = _transcription_confidence(audio_input)
                threshold = getattr(
                    getattr(orchestrator, "settings", None),
                    "stt_confidence_threshold",
                    0.55,
                )
                if confidence is not None:
                    logger.info(
                        "Voice stage=transcript_quality turn=%d min_word_confidence=%.3f outcome=%s",
                        turn_number,
                        confidence,
                        "uncertain" if confidence < threshold else "accepted",
                    )
                if confidence is not None and confidence < threshold:
                    set_status("PROCESSING")
                    stage_name = "transcript_recovery"
                    recovery_started = time.monotonic()
                    suggest_correction = getattr(
                        orchestrator, "suggest_transcript_correction", None
                    )
                    suggestion = None
                    if suggest_correction is not None:
                        try:
                            suggestion = suggest_correction(transcript)
                        except Exception as error:
                            logger.warning(
                                "Transcript correction suggestion failed (%s)",
                                type(error).__name__,
                            )
                    logger.info(
                        "Voice stage=llm_generation turn=%d purpose=transcript_recovery elapsed_ms=%.1f outcome=%s",
                        turn_number,
                        (time.monotonic() - recovery_started) * 1000,
                        "suggested" if suggestion else "uncertain",
                    )
                    original_transcript = transcript
                    accepted_correction = False
                    if suggestion:
                        logger.info(
                            "Voice transcript suggestion turn=%d original=%r suggested=%r",
                            turn_number,
                            original_transcript,
                            suggestion,
                        )
                        speech_output.speak(
                            f"I may have misheard you. Did you mean: {suggestion}? Say yes to use it, or no to repeat."
                        )
                        confirmation = _capture_transcript_confirmation(
                            audio_input, followup_active=followup_active
                        )
                        logger.info(
                            "Voice transcript confirmation turn=%d text=%r",
                            turn_number,
                            confirmation,
                        )
                        affirmative = confirmation.casefold().strip(" .!?\t") in {
                            "yes", "yes confirm", "confirm", "correct", "do it",
                        }
                        if affirmative:
                            transcript = suggestion
                            accepted_correction = True
                            logger.info(
                                "Voice transcript interpretation turn=%d original=%r corrected=%r confirmed=true",
                                turn_number,
                                original_transcript,
                                transcript,
                            )

                    if not accepted_correction:
                        speech_output.speak(
                            "Please repeat the full request."
                            if suggestion
                            else "I couldn't understand that. Please repeat your request."
                        )
                        retry_started = time.monotonic()
                        repeated_transcript = _capture_transcript_confirmation(
                            audio_input, followup_active=followup_active
                        )
                        retry_confidence = _transcription_confidence(audio_input)
                        logger.info(
                            "Voice transcript retry turn=%d elapsed_ms=%.1f text=%r min_word_confidence=%s",
                            turn_number,
                            (time.monotonic() - retry_started) * 1000,
                            repeated_transcript,
                            f"{retry_confidence:.3f}" if retry_confidence is not None else "unavailable",
                        )
                        if repeated_transcript and (
                            retry_confidence is None or retry_confidence >= threshold
                        ):
                            transcript = repeated_transcript
                        else:
                            logger.warning(
                                "Voice transcript rejected turn=%d reason=unclear_after_retry",
                                turn_number,
                            )
                            speech_output.speak(
                                "I still couldn't understand. Please say the wake phrase to try again."
                            )
                            followup_active = False
                            continue

                router = getattr(orchestrator, "router", None)
                decision = router.route(transcript) if router is not None else None
                if decision is not None:
                    logger.info(
                        "Voice stage=routing turn=%d outcome=completed route=%s action=%s",
                        turn_number,
                        decision.kind.value,
                        decision.action or "none",
                    )
                model_route = decision is None or decision.kind.value in {
                    "LLM_REASONING", "ONLINE_FACTUAL_LOOKUP"
                } or (
                    decision.kind.value == "REMINDER"
                    and decision.action in {"create", "update"}
                )
                set_status("PROCESSING")
                stage_name = "orchestration"
                stage_started = time.monotonic()
                stream_response = getattr(orchestrator, "stream", None)
                if stream_response is None:
                    generation_started = time.monotonic()
                    if model_route:
                        logger.info(
                            "Voice stage=llm_generation turn=%d outcome=started",
                            turn_number,
                        )
                    result = orchestrator.handle(transcript)
                    generated_text.append(result.response)
                    logger.info(
                        "Voice response turn=%d generated_text=%r outcome=%s",
                        turn_number,
                        result.response,
                        "completed" if result.completed else "failed",
                    )
                    if model_route:
                        logger.info(
                            "Voice stage=llm_generation turn=%d elapsed_ms=%.1f outcome=%s",
                            turn_number,
                            (time.monotonic() - generation_started) * 1000,
                            "completed" if result.completed else "failed",
                        )
                    if stop_event.is_set():
                        break
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
                    speech_completed = speech_output.speak(result.response)
                    if speech_completed is not False and not stop_event.is_set():
                        delivered_text.append(result.response)
                    logger.info(
                        "Voice stage=speech_delivery turn=%d elapsed_ms=%.1f outcome=%s",
                        turn_number,
                        (time.monotonic() - stage_started) * 1000,
                        "completed" if speech_completed is not False else "interrupted",
                    )
                    logger.info(
                        "Voice response delivery turn=%d delivered_text=%r",
                        turn_number,
                        " ".join(delivered_text),
                    )
                    followup_active = (
                        next_followup is not None
                        and (speech_completed is False or result.completed)
                    )
                else:
                    phrase_buffer = SpeechPhraseBuffer()
                    spoken = False
                    completed = False
                    failed = False
                    interrupted = False
                    speech_delivery_started = None
                    generation_started = time.monotonic()
                    if model_route:
                        logger.info(
                            "Voice stage=llm_generation turn=%d outcome=started",
                            turn_number,
                        )
                    session = getattr(speech_output, "session", None)
                    turn = getattr(speech_output, "turn", None)
                    with ExitStack() as active_turn:
                        cancellation = (
                            active_turn.enter_context(turn())
                            if turn is not None else Event()
                        )
                        session_started = False
                        if isinstance(orchestrator, RequestOrchestrator):
                            stream = iter(
                                stream_response(
                                    transcript,
                                    cancel_event=cancellation,
                                    delivered_text=delivered_text,
                                )
                            )
                        else:
                            stream = iter(
                                stream_response(transcript, cancel_event=cancellation)
                            )

                        def speak_phrases(phrases):
                            nonlocal session_started, spoken, speech_delivery_started, stage_name
                            if phrases and speech_delivery_started is None:
                                speech_delivery_started = time.monotonic()
                            for phrase in phrases:
                                if stop_event.is_set() or cancellation.is_set():
                                    break
                                stage_name = "tts"
                                if session is not None and not session_started:
                                    active_turn.enter_context(session())
                                    session_started = True
                                spoken = True
                                speech_completed = speech_output.speak(phrase)
                                if (
                                    speech_completed is not False
                                    and not cancellation.is_set()
                                    and not stop_event.is_set()
                                ):
                                    delivered_text.append(phrase)

                        try:
                            for event in stream:
                                if stop_event.is_set() or cancellation.is_set():
                                    break
                                if event.kind == "text":
                                    generated_text.append(event.text)
                                    speak_phrases(phrase_buffer.feed(event.text))
                                elif event.kind == "completed":
                                    completed = True
                                    speak_phrases(phrase_buffer.feed("", final=True))
                                elif event.kind == "failed":
                                    failed = True
                                    speak_phrases(phrase_buffer.feed("", final=True))
                                    break
                        finally:
                            close_stream = getattr(stream, "close", None)
                            if close_stream is not None:
                                close_stream()
                        interrupted = cancellation.is_set()
                    if stop_event.is_set():
                        break
                    if failed:
                        logger.warning("Voice stream failed completed=%s", completed)
                    if failed and not spoken and not interrupted:
                        error_response = "I couldn't process that turn. Please try again."
                        if speech_delivery_started is None:
                            speech_delivery_started = time.monotonic()
                        speech_completed = speech_output.speak(error_response)
                        if speech_completed is not False and not stop_event.is_set():
                            delivered_text.append(error_response)
                        spoken = True
                    generation_outcome = (
                        "interrupted" if interrupted else
                        "failed" if failed else
                        "completed" if completed else "incomplete"
                    )
                    if model_route:
                        logger.info(
                            "Voice stage=llm_generation turn=%d elapsed_ms=%.1f outcome=%s",
                            turn_number,
                            (time.monotonic() - generation_started) * 1000,
                            generation_outcome,
                        )
                    logger.info(
                        "Voice response turn=%d generated_text=%r delivered_text=%r outcome=%s",
                        turn_number,
                        "".join(generated_text),
                        " ".join(delivered_text),
                        generation_outcome,
                    )
                    logger.info(
                        "Voice stage=speech_delivery turn=%d elapsed_ms=%.1f outcome=%s",
                        turn_number,
                        (time.monotonic() - speech_delivery_started) * 1000
                        if speech_delivery_started is not None else 0.0,
                        "interrupted" if interrupted else "completed" if delivered_text else "no_output",
                    )
                    followup_active = next_followup is not None and (
                        completed or interrupted
                    )
                    logger.info(
                        "Voice stage=orchestration turn=%d elapsed_ms=%.1f route=stream completed=%s failed=%s",
                        turn_number,
                        (time.monotonic() - stage_started) * 1000,
                        completed,
                        failed,
                    )
                    logger.info(
                        "Voice stage=tts turn=%d outcome=%s",
                        turn_number,
                        "streamed" if spoken else "no_output",
                    )
            except Exception as error:
                if isinstance(audio_input, WakeListenerInput):
                    followup_active = False
                if generated_text:
                    logger.info(
                        "Voice response turn=%d generated_text=%r delivered_text=%r outcome=failed",
                        turn_number,
                        "".join(generated_text),
                        " ".join(delivered_text),
                    )
                logger.warning(
                    "Voice turn failed turn=%d stage=%s elapsed_ms=%.1f (%s)",
                    turn_number,
                    stage_name,
                    (time.monotonic() - stage_started) * 1000,
                    type(error).__name__,
                )
                set_status("ERROR")
                if not stop_event.is_set() and not spoken:
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


def _idle_resource_monitor(stop_event: Event, resources, timeout: float) -> None:
    interval = min(1.0, max(0.1, timeout / 4))
    while not stop_event.wait(interval):
        for resource in resources:
            evict = getattr(resource, "evict_if_idle", None)
            if evict is not None:
                try:
                    evict(timeout)
                except Exception as error:
                    logger.warning("Idle resource eviction failed (%s)", type(error).__name__)


def _run_voice_service(settings: Settings) -> None:
    try:
        from gpiozero import RGBLED
    except ImportError as error:
        raise RuntimeError("gpiozero is required for voice status RGB LED output") from error

    stop_event = Event()
    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    gate = AudioGate(
        allow_wakeword_during_speech=(
            settings.voice_profile == "wakeword" or settings.wake_word_enabled
        )
    )
    led = None
    voice_status = _VoiceStatusLED()
    audio_input = button = interrupt_button = None
    model = memory = reminders = scheduler = speech_output = idle_thread = notifier = None

    def set_status(status: str) -> None:
        voice_status.set_status(status)

    def speak(text: str) -> None:
        speech_output.speak(text)

    try:
        led = RGBLED(
            red=settings.status_led_red_pin,
            green=settings.status_led_pin,
            blue=settings.status_led_blue_pin,
        )
        voice_status.attach(led)
        model = _LazyLocalModel(settings)
        memory = MemoryManager.from_settings(settings)
        reminders = ReminderManager.from_settings(settings)
        speech_output = _StatusSpeechOutput(
            create_speech_output(settings, gate), set_status, gate
        )
        idle_thread = Thread(
            target=_idle_resource_monitor,
            args=(stop_event, (model, speech_output), settings.runtime_idle_timeout),
            name="orion-idle-resource-monitor",
            daemon=True,
        )
        idle_thread.start()
        notifier = ReminderNotifier(speak, set_status)
        scheduler = ReminderScheduler(
            reminders.store, notifier, poll_interval=settings.reminder_poll_interval
        )

        def stop_service(_signum, _frame) -> None:
            stop_event.set()
            gate.request_interrupt()

        signal.signal(signal.SIGINT, stop_service)
        signal.signal(signal.SIGTERM, stop_service)
        if settings.voice_profile == "wakeword" or settings.wake_word_enabled:
            audio_input, button = _create_audio_input(
                settings, gate, stop_event, set_status, voice_status.set_muted
            )
        else:
            audio_input, button = _create_audio_input(settings, gate, stop_event)
        if isinstance(audio_input, WakeListenerInput):
            notifier.on_speech_complete = audio_input.open_followup_window

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
        if idle_thread is not None:
            idle_thread.join(timeout=2.0)
        if audio_input is not None:
            audio_input.stop_event.set()
        cleanup_actions = (
            ("audio input", getattr(audio_input, "close", None)),
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
    gate = AudioGate()
    speech_output = create_speech_output(settings, gate)
    stop_event = Event()
    idle_thread = Thread(
        target=_idle_resource_monitor,
        args=(stop_event, (speech_output,), settings.runtime_idle_timeout),
        name="orion-reminder-idle-monitor",
        daemon=True,
    )

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
        idle_thread.start()
        scheduler.start()
        logger.info("Reminder scheduler started")
        while True:
            time.sleep(settings.reminder_poll_interval)
    except KeyboardInterrupt:
        logger.info("Stopping reminder scheduler")
    finally:
        stop_event.set()
        gate.request_interrupt()
        idle_thread.join(timeout=2.0)
        scheduler.stop()
        manager.close()
        led.close()


class _LazyLocalModel(LlamaCppProvider):
    """Keep the historical application name for the provider-backed adapter."""


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
