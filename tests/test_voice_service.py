import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from orion.app import (
    _confirm_spoken_action,
    _create_audio_input,
    _create_interrupt_button,
    _run_voice_loop,
    _set_led_status,
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
                Settings(), AudioGate(), threading.Event()
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
                _create_audio_input(Settings(), AudioGate(), threading.Event())

        self.assertTrue(button.closed)

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
                Settings(), gate, ptt_button
            )

        self.assertEqual(interrupt_button.pin, 16)
        self.assertIsNot(interrupt_button, ptt_button)
        with gate.speaking():
            interrupt_button.when_pressed()
            self.assertTrue(gate.wait_for_interrupt(0))
        with gate.speaking():
            ptt_button.when_pressed()
            self.assertTrue(gate.wait_for_interrupt(0))

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

        recognizers = []

        def create_recognizer(model, sample_rate):
            recognizer = FakeRecognizer(model, sample_rate)
            recognizers.append(recognizer)
            return recognizer

        audio_input = WakeWordInput(
            settings,
            Detector(),
            source=source,
            model_factory=lambda _path: object(),
            recognizer_factory=create_recognizer,
        )

        self.assertEqual(audio_input.next_utterance(), "hello orion")
        self.assertNotIn(wake, recognizers[0].chunks)
        self.assertIn(speech, recognizers[0].chunks)

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
