import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from orion.app import (
    _confirm_spoken_action,
    _create_audio_input,
    _create_interrupt_button,
    _run_voice_loop,
    _set_led_status,
    _VoiceStatusLED,
    main,
)
from orion.audio.capture import AudioGate
from orion.audio.stt import PushToTalkInput, WakeWordInput
from orion.config import Settings


class FakeRecognizer:
    def __init__(self, _model, _sample_rate):
        self.chunks = []

    def AcceptWaveform(self, chunk):
        self.chunks.append(chunk)

    def FinalResult(self):
        return '{"text":"hello orion"}'


class FakeSource:
    def __init__(self, chunks):
        self.values = chunks
        self.seen_stop_event = None

    def chunks(self, stop_event=None):
        self.seen_stop_event = stop_event
        yield from self.values


class FakeButton:
    def __init__(self):
        self.is_pressed = False
        self.closed = False

    def wait_for_press(self, timeout=None):
        self.is_pressed = True

    def close(self):
        self.closed = True


class FakeScheduler:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class VoiceServiceTests(unittest.TestCase):
    def test_led_status_maps_to_distinct_rgb_colors(self):
        class FakeLED:
            color = None

        led = FakeLED()
        states = {
            "IDLE": (0.0, 0.0, 0.0),
            "LISTENING": (0.0, 0.2, 0.0),
            "PROCESSING": (0.0, 0.0, 0.45),
            "SPEAKING": (0.0, 0.7, 0.7),
            "MUTED": (0.05, 0.025, 0.0),
            "ERROR": (1.0, 0.0, 0.0),
        }
        for status, expected in states.items():
            with self.subTest(status=status):
                _set_led_status(led, status)
                self.assertEqual(led.color, expected)
        _set_led_status(led, "UNKNOWN")
        self.assertEqual(led.color, (0.0, 0.0, 0.0))

    def test_muted_led_override_restores_current_voice_status(self):
        class FakeLED:
            color = None

        led = FakeLED()
        status = _VoiceStatusLED()
        status.attach(led)
        status.set_status("SPEAKING")
        status.set_muted(True)
        self.assertEqual(led.color, (0.05, 0.025, 0.0))

        status.set_muted(False)
        self.assertEqual(led.color, (0.0, 0.7, 0.7))

        status.set_muted(True)
        status.set_status("PROCESSING")
        self.assertEqual(led.color, (0.05, 0.025, 0.0))
        status.set_muted(False)
        self.assertEqual(led.color, (0.0, 0.0, 0.45))

    def test_legacy_jarvis_profile_uses_muted_led_callback(self):
        settings = replace(
            Settings(), voice_profile="legacy", stt_backend="vosk",
            wake_word_enabled=True,
        )
        status = _VoiceStatusLED()
        led = SimpleNamespace(color=None)
        status.attach(led)

        self.assertTrue(settings.wake_word_enabled)
        status.set_muted(True)
        self.assertEqual(led.color, (0.05, 0.025, 0.0))
        status.set_muted(False)
        self.assertEqual(led.color, (0.0, 0.0, 0.0))

    def test_spoken_confirmation_accepts_only_clear_yes(self):
        class AudioInput:
            def __init__(self, answer):
                self.answer = answer

            def next_utterance(self, timeout=None):
                self.timeout = timeout
                return self.answer

        class SpeechOutput:
            def __init__(self):
                self.spoken = []

            def speak(self, text):
                self.spoken.append(text)

        for answer, expected in (("yes", True), ("no", False), ("maybe", False), ("", False)):
            with self.subTest(answer=answer):
                speech = SpeechOutput()
                result = _confirm_spoken_action("Confirm action?", AudioInput(answer), speech)
                self.assertEqual(result, expected)
                self.assertEqual(speech.spoken, ["Confirm action?"])

    def test_spoken_confirmation_requires_current_microphone_claim(self):
        class AudioInput:
            def next_utterance(self, timeout=None):
                self.timeout = timeout
                return "yes"

            def claim_utterance(self):
                return False

        class SpeechOutput:
            def __init__(self):
                self.spoken = []

            def speak(self, text):
                self.spoken.append(text)

        speech = SpeechOutput()
        result = _confirm_spoken_action("Confirm action?", AudioInput(), speech)

        self.assertFalse(result)
        self.assertEqual(speech.spoken, ["Confirm action?"])

    def test_cli_starts_persistent_voice_service_mode(self):
        with patch("orion.app._run_voice_service") as run_service:
            self.assertEqual(main(["--run-voice"]), 0)
        run_service.assert_called_once()

    def test_push_to_talk_warms_stt_before_announcing_readiness(self):
        events = []
        button = FakeButton()

        class AudioInput:
            def __init__(self, *_args, **_kwargs):
                pass

            def warmup(self):
                events.append("warmup")

        with (
            patch.dict("sys.modules", {"gpiozero": SimpleNamespace(Button=lambda _pin: button)}),
            patch("orion.app.PushToTalkInput", AudioInput),
            patch(
                "orion.app.logger.info",
                side_effect=lambda message, *args: events.append(message % args if args else message),
            ),
        ):
            audio_input, actual_button = _create_audio_input(
                replace(Settings(), voice_profile="legacy", stt_backend="vosk"),
                AudioGate(),
                threading.Event(),
            )

        self.assertIsInstance(audio_input, AudioInput)
        self.assertIs(actual_button, button)
        self.assertEqual(events, ["warmup", "Press GPIO 23 to speak"])

    def test_push_to_talk_closes_button_when_stt_warmup_fails(self):
        button = FakeButton()

        class AudioInput:
            def __init__(self, *_args, **_kwargs):
                pass

            def warmup(self):
                raise RuntimeError("model load failed")

        with (
            patch.dict("sys.modules", {"gpiozero": SimpleNamespace(Button=lambda _pin: button)}),
            patch("orion.app.PushToTalkInput", AudioInput),
        ):
            with self.assertRaisesRegex(RuntimeError, "model load failed"):
                _create_audio_input(
                    replace(Settings(), voice_profile="legacy", stt_backend="vosk"),
                    AudioGate(),
                    threading.Event(),
                )

        self.assertTrue(button.closed)

    def test_legacy_wake_word_never_constructs_downloadable_builtin_model(self):
        settings = replace(
            Settings(), voice_profile="legacy", stt_backend="vosk", wake_word_enabled=True
        )
        with (
            patch.dict("sys.modules", {"gpiozero": SimpleNamespace(Button=Mock())}),
            patch("orion.app.OpenWakeWordDetector") as detector_class,
        ):
            detector_class.from_settings.side_effect = FileNotFoundError(
                "local ONNX assets missing"
            )
            with self.assertRaisesRegex(FileNotFoundError, "local ONNX assets missing"):
                _create_audio_input(settings, AudioGate(), threading.Event())

        detector_class.from_settings.assert_called_once_with(settings)
        detector_class.assert_not_called()

    def test_legacy_vosk_wake_gpio_toggles_persistent_microphone_control(self):
        settings = replace(
            Settings(), voice_profile="legacy", stt_backend="vosk",
            wake_word_enabled=True,
        )
        statuses = []
        buttons = []
        led = SimpleNamespace(color=None)
        led_status = _VoiceStatusLED()
        led_status.attach(led)

        def set_status(status):
            statuses.append(status)
            led_status.set_status(status)

        class Control:
            def __init__(self, _path):
                self.muted = True

            def toggle(self):
                self.muted = not self.muted
                return self.muted

            def close(self):
                pass

        class WakeListener:
            def __init__(self, _settings, _detector, _transcriber, control, **_kwargs):
                self.control = control

            def start(self):
                pass

            def close(self):
                pass

        def make_button(pin, **kwargs):
            button = SimpleNamespace(
                pin=pin, options=kwargs, when_pressed=None, closed=False,
                close=lambda: setattr(button, "closed", True),
            )
            buttons.append(button)
            return button

        with (
            patch.dict("sys.modules", {"gpiozero": SimpleNamespace(Button=make_button)}),
            patch("orion.app.OpenWakeWordDetector") as detector_factory,
            patch("orion.app.MicrophoneControl", Control),
            patch("orion.app.create_transcriber", return_value=object()),
            patch("orion.app.WakeListenerInput", WakeListener),
        ):
            detector_factory.from_settings.return_value = object()
            audio_input, button = _create_audio_input(
                settings,
                AudioGate(),
                threading.Event(),
                set_status,
                led_status.set_muted,
            )
            self.assertIsInstance(audio_input, WakeListener)
            self.assertIs(button, buttons[0])
            self.assertEqual(button.pin, settings.microphone_control_pin)
            self.assertEqual(button.options["bounce_time"], settings.microphone_button_debounce)
            self.assertEqual(statuses, ["IDLE"])
            self.assertEqual(led.color, (0.05, 0.025, 0.0))

            button.when_pressed()
            self.assertEqual(led.color, (0.0, 0.0, 0.0))
            button.when_pressed()
            self.assertEqual(led.color, (0.05, 0.025, 0.0))

        self.assertEqual(statuses, ["IDLE"])

    def test_wake_profile_start_failure_closes_button_and_control(self):
        settings = replace(Settings(), voice_profile="wakeword")
        button = FakeButton()
        controls = []

        class Control:
            def __init__(self, _path):
                self.closed = False
                controls.append(self)

            def close(self):
                self.closed = True

        class WakeListener:
            def __init__(self, _settings, _detector, _transcriber, control, **_kwargs):
                self.control = control
                self.closed = False

            def start(self):
                raise RuntimeError("wake listener failed")

            def close(self):
                self.closed = True
                self.control.close()

        with (
            patch.dict("sys.modules", {"gpiozero": SimpleNamespace(Button=lambda *_args, **_kwargs: button)}),
            patch("orion.app.OpenWakeWordDetector") as detector_factory,
            patch("orion.app.MicrophoneControl", Control),
            patch("orion.app.create_transcriber", return_value=object()),
            patch("orion.app.WakeListenerInput", WakeListener),
        ):
            detector_factory.from_settings.return_value = object()
            with self.assertRaisesRegex(RuntimeError, "wake listener failed"):
                _create_audio_input(settings, AudioGate(), threading.Event())

        self.assertTrue(button.closed)
        self.assertTrue(controls[0].closed)

    def test_separate_interrupt_button_and_ptt_both_request_playback_cutoff(self):
        created = []

        class Button:
            def __init__(self, pin):
                self.pin = pin
                self.when_pressed = None
                created.append(self)

        ptt_button = Button(23)
        gate = AudioGate()
        with patch.dict("sys.modules", {"gpiozero": SimpleNamespace(Button=Button)}):
            interrupt_button = _create_interrupt_button(
                replace(Settings(), voice_profile="legacy", stt_backend="vosk"),
                gate,
                ptt_button,
            )

        self.assertEqual(interrupt_button.pin, 16)
        self.assertIsNot(interrupt_button, ptt_button)
        with gate.speaking():
            interrupt_button.when_pressed()
            self.assertTrue(gate.wait_for_interrupt(0))
        with gate.speaking():
            ptt_button.when_pressed()
            self.assertTrue(gate.wait_for_interrupt(0))

    def test_legacy_wake_mute_button_is_not_rebound_as_tts_interrupt(self):
        settings = replace(
            Settings(), voice_profile="legacy", stt_backend="vosk",
            wake_word_enabled=True,
        )
        activations = []
        button = SimpleNamespace(when_pressed=lambda: activations.append("mute"))
        mute_callback = button.when_pressed

        class InterruptButton:
            def __init__(self, pin):
                self.pin = pin

        with patch.dict("sys.modules", {"gpiozero": SimpleNamespace(Button=InterruptButton)}):
            interrupt_button = _create_interrupt_button(settings, AudioGate(), button)

        button.when_pressed()
        self.assertEqual(interrupt_button.pin, settings.tts_interrupt_pin)
        self.assertIs(button.when_pressed, mute_callback)
        self.assertEqual(activations, ["mute"])

    def test_voice_loop_handles_multiple_turns_and_stops_scheduler(self):
        stop_event = threading.Event()
        statuses = []
        spoken = []
        handled = []
        scheduler = FakeScheduler()

        class AudioInput:
            turns = iter(["What time is it?", "Remind me tomorrow", ""])

            def next_utterance(self):
                return next(self.turns)

        class Orchestrator:
            def handle(self, request):
                handled.append(request)
                return SimpleNamespace(
                    response=f"reply: {request}",
                    route=SimpleNamespace(kind=SimpleNamespace(value="test")),
                    completed=True,
                )

        class SpeechOutput:
            def speak(self, text):
                spoken.append(text)
                if len(spoken) == 2:
                    stop_event.set()

        with self.assertLogs("orion.app", level="INFO") as logs:
            _run_voice_loop(
                AudioInput(),
                Orchestrator(),
                SpeechOutput(),
                scheduler,
                stop_event,
                statuses.append,
            )

        self.assertTrue(scheduler.started)
        self.assertTrue(scheduler.stopped)
        self.assertEqual(handled, ["What time is it?", "Remind me tomorrow"])
        self.assertEqual(spoken, [
            "reply: What time is it?",
            "reply: Remind me tomorrow",
        ])
        self.assertEqual(statuses[-1], "IDLE")
        joined_logs = "\n".join(logs.output)
        self.assertIn("stage=input_total", joined_logs)
        self.assertIn("stage=orchestration", joined_logs)
        self.assertIn("stage=tts", joined_logs)
        self.assertNotIn("What time is it?", joined_logs)
        self.assertNotIn("Remind me tomorrow", joined_logs)

    def test_voice_loop_speaks_short_error_after_turn_failure(self):
        stop_event = threading.Event()
        spoken = []
        scheduler = FakeScheduler()

        class AudioInput:
            def next_utterance(self):
                raise RuntimeError("device failed")

        class Orchestrator:
            def handle(self, _request):
                self.fail("orchestrator must not run")

        class SpeechOutput:
            def speak(self, text):
                spoken.append(text)
                stop_event.set()

        _run_voice_loop(
            AudioInput(),
            Orchestrator(),
            SpeechOutput(),
            scheduler,
            stop_event,
            lambda _status: None,
        )

        self.assertEqual(
            spoken,
            ["I couldn't process that turn. Please try again."],
        )
        self.assertTrue(scheduler.stopped)

    def test_voice_loop_never_dispatches_unclaimed_wake_transcript(self):
        stop_event = threading.Event()
        scheduler = FakeScheduler()
        handled = []

        class AudioInput:
            calls = 0

            def next_utterance(self):
                self.calls += 1
                if self.calls > 1:
                    stop_event.set()
                    return ""
                return "delete my reminder"

            def claim_utterance(self):
                return False

        class Orchestrator:
            def handle(self, request):
                handled.append(request)
                raise AssertionError("unclaimed transcript reached orchestration")

        class SpeechOutput:
            def speak(self, _text):
                raise AssertionError("unclaimed transcript must not produce speech")

        _run_voice_loop(
            AudioInput(), Orchestrator(), SpeechOutput(), scheduler,
            stop_event, lambda _status: None,
        )

        self.assertEqual(handled, [])
        self.assertTrue(scheduler.stopped)

    def test_voice_loop_does_not_start_speech_after_shutdown_during_orchestration(self):
        stop_event = threading.Event()
        spoken = []
        scheduler = FakeScheduler()

        class AudioInput:
            def next_utterance(self):
                return "What time is it?"

        class Orchestrator:
            def handle(self, _request):
                stop_event.set()
                return SimpleNamespace(
                    response="must not be spoken",
                    route=SimpleNamespace(kind=SimpleNamespace(value="test")),
                    completed=True,
                )

        class SpeechOutput:
            def speak(self, text):
                spoken.append(text)

        _run_voice_loop(
            AudioInput(), Orchestrator(), SpeechOutput(), scheduler, stop_event, lambda _status: None
        )

        self.assertEqual(spoken, [])
        self.assertTrue(scheduler.stopped)

    def test_wake_profile_startup_closes_listener_if_button_cleanup_fails(self):
        from orion.app import _create_audio_input

        class Button:
            when_pressed = None

            def __init__(self, *_args, **_kwargs):
                self.closed = False

            def close(self):
                self.closed = True
                raise RuntimeError("button close failed")

        class Control:
            muted = True
            on_change = None

            def toggle(self):
                return True

        class Listener:
            def __init__(self, *_args, **_kwargs):
                self.closed = False

            def start(self):
                raise RuntimeError("listener start failed")

            def close(self):
                self.closed = True

        button = Button()
        with (
            patch.dict("sys.modules", {"gpiozero": SimpleNamespace(Button=lambda *_args, **_kwargs: button)}),
            patch("orion.app.OpenWakeWordDetector.from_settings", return_value=object()),
            patch("orion.app.MicrophoneControl", return_value=Control()),
            patch("orion.app.create_transcriber", return_value=object()),
            patch("orion.app.WakeListenerInput", Listener),
        ):
            with self.assertRaisesRegex(RuntimeError, "listener start failed"):
                _create_audio_input(
                    replace(Settings(), voice_profile="wakeword"), AudioGate(), threading.Event()
                )

        self.assertTrue(button.closed)

    def test_wake_profile_closes_control_store_if_transcriber_creation_fails(self):
        control = SimpleNamespace(closed=False)
        control.close = lambda: setattr(control, "closed", True)

        with (
            patch.dict("sys.modules", {"gpiozero": SimpleNamespace(Button=object)}),
            patch("orion.app.OpenWakeWordDetector.from_settings", return_value=object()),
            patch("orion.app.MicrophoneControl", return_value=control),
            patch("orion.app.create_transcriber", side_effect=RuntimeError("STT setup failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "STT setup failed"):
                _create_audio_input(
                    replace(Settings(), voice_profile="wakeword"), AudioGate(), threading.Event()
                )

        self.assertTrue(control.closed)

    def test_push_to_talk_uses_interruptible_button_and_streaming_transcription(self):
        speech = b"\x10\x00" * 10
        silence = b"\x00\x00" * 10
        source = FakeSource([speech, silence, silence, silence])
        button = FakeButton()
        settings = replace(
            Settings(),
            audio_sample_rate=1000,
            audio_block_frames=10,
            vad_rms_threshold=10,
            vad_silence_ms=20,
            vad_preroll_ms=0,
            stt_model_path=Path("."),
        )
        audio_input = PushToTalkInput(
            settings,
            button,
            source=source,
            model_factory=lambda _path: object(),
            recognizer_factory=FakeRecognizer,
        )

        self.assertEqual(audio_input.next_utterance(), "hello orion")
        self.assertTrue(button.is_pressed)
        self.assertIsNotNone(source.seen_stop_event)

    def test_wake_word_consumes_trigger_audio_then_transcribes_speech(self):
        wake = b"\x01\x00" * 10
        speech = b"\x10\x00" * 10
        silence = b"\x00\x00" * 10
        source = FakeSource([wake, speech, silence, silence, silence])
        settings = replace(
            Settings(),
            audio_sample_rate=1000,
            audio_block_frames=10,
            vad_rms_threshold=10,
            vad_silence_ms=20,
            vad_preroll_ms=0,
            stt_model_path=Path("."),
        )

        class Detector:
            def detected(self, _chunk):
                return True

        statuses = []
        recognizers = []

        def create_recognizer(model, sample_rate):
            recognizer = FakeRecognizer(model, sample_rate)
            recognizers.append(recognizer)
            return recognizer

        audio_input = WakeWordInput(
            settings,
            Detector(),
            on_wake=lambda: statuses.append("LISTENING"),
            source=source,
            model_factory=lambda _path: object(),
            recognizer_factory=create_recognizer,
        )

        self.assertEqual(audio_input.next_utterance(), "hello orion")
        self.assertEqual(statuses, ["LISTENING"])
        self.assertNotIn(wake, recognizers[0].chunks)
        self.assertIn(speech, recognizers[0].chunks)

    def test_voice_loop_wake_turn_returns_to_idle(self):
        stop_event = threading.Event()
        statuses = []
        scheduler = FakeScheduler()

        class AudioInput(WakeWordInput):
            def __init__(self):
                self.calls = 0

            def next_utterance(self):
                self.calls += 1
                if self.calls == 1:
                    statuses.append("LISTENING")
                    return "What time is it?"
                stop_event.set()
                return ""

        class Orchestrator:
            def handle(self, _request):
                return SimpleNamespace(
                    response="It is noon.",
                    route=SimpleNamespace(kind=SimpleNamespace(value="test")),
                    completed=True,
                )

        class SpeechOutput:
            def speak(self, _text):
                pass

        _run_voice_loop(
            AudioInput(), Orchestrator(), SpeechOutput(), scheduler,
            stop_event, statuses.append,
        )

        self.assertEqual(
            statuses,
            ["IDLE", "LISTENING", "PROCESSING", "SPEAKING", "IDLE", "IDLE"],
        )

    def test_voice_gate_is_shared_by_input_and_speech(self):
        gate = AudioGate()
        self.assertTrue(gate.is_open)
        with gate.speaking():
            self.assertFalse(gate.is_open)
        self.assertTrue(gate.is_open)

    def test_status_speech_output_warms_wrapped_engine(self):
        from orion.app import _StatusSpeechOutput

        events = []
        speech = _StatusSpeechOutput(
            type("Speech", (), {"warmup": lambda _self: events.append("warm")})(),
            lambda status: events.append(status),
        )

        speech.warmup()

        self.assertEqual(events, ["warm"])


if __name__ == "__main__":
    unittest.main()
