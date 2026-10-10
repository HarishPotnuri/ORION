import array
import json
import threading
import tempfile
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from orion.audio.capture import AudioBufferOverflow, AudioGate, SoundDeviceChunkSource
from orion.audio.stt import PushToTalkInput, VoskTranscriber, WhisperPushToTalkInput
from orion.audio.tts import KittenSpeechOutput
from orion.audio.vad import VoiceActivityDetector
from orion.audio.wakeword import OpenWakeWordDetector
from orion.config import Settings


def pcm(value: int, sample_count: int) -> bytes:
    return array.array("h", [value] * sample_count).tobytes()


class FakeRecognizer:
    def __init__(self, _model, _sample_rate):
        self.chunks = []

    def AcceptWaveform(self, chunk):
        self.chunks.append(chunk)
        return False

    def FinalResult(self):
        return '{"text": "recognized words"}'


class AudioTests(unittest.TestCase):
    def small_settings(self, **changes) -> Settings:
        values = {
            "audio_sample_rate": 1000,
            "audio_block_frames": 10,
            "audio_queue_chunks": 2,
            "max_utterance_seconds": 1,
            "vad_rms_threshold": 100,
            "vad_silence_ms": 30,
            "vad_preroll_ms": 20,
        }
        values.update(changes)
        return replace(Settings(), **values)

    def test_vad_yields_bounded_utterance_with_preroll(self):
        settings = self.small_settings()
        chunks = [pcm(0, 10), pcm(0, 10), pcm(1000, 10), pcm(1000, 10)]
        chunks.extend([pcm(0, 10)] * 3)
        chunks.append(pcm(1000, 10))

        result = list(VoiceActivityDetector(settings).utterance_chunks(chunks))

        self.assertEqual(result, chunks[1:7])

    def test_vad_logs_end_of_speech_timing_without_audio_content(self):
        settings = self.small_settings()
        chunks = [pcm(1000, 10), pcm(0, 10), pcm(0, 10), pcm(0, 10)]

        with self.assertLogs("orion.audio.vad", level="INFO") as logs:
            list(VoiceActivityDetector(settings).utterance_chunks(chunks))

        self.assertIn("stage=vad", logs.output[0])
        self.assertIn("outcome=end_of_speech", logs.output[0])
        self.assertNotIn("1000", logs.output[0])

    def test_vad_enforces_maximum_utterance_length(self):
        detector = VoiceActivityDetector(self.small_settings(max_utterance_seconds=0.03))
        chunks = [pcm(1000, 10)] * 10

        result = list(detector.utterance_chunks(chunks))

        self.assertEqual(len(result), 3)

    def test_vad_returns_after_silence_when_no_speech_starts(self):
        detector = VoiceActivityDetector(
            self.small_settings(vad_silence_ms=30, vad_preroll_ms=20)
        )
        silence = [pcm(0, 10)] * 10

        with self.assertLogs("orion.audio.vad", level="INFO") as logs:
            self.assertEqual(list(detector.utterance_chunks(silence)), [])

        self.assertIn("outcome=no_speech", logs.output[-1])

    def test_vosk_transcribes_chunks_without_audio_files(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            model_path.mkdir()
            settings = self.small_settings(stt_model_path=model_path)
            recognizers = []

            def recognizer_factory(model, sample_rate):
                recognizer = FakeRecognizer(model, sample_rate)
                recognizers.append(recognizer)
                return recognizer

            transcriber = VoskTranscriber(
                settings,
                model_factory=lambda _path: object(),
                recognizer_factory=recognizer_factory,
            )
            chunks = iter([pcm(0, 10), pcm(0, 10), pcm(1000, 10), pcm(0, 10), pcm(0, 10), pcm(0, 10)])

            transcript = transcriber.transcribe_chunks(chunks)

            self.assertEqual(transcript, "recognized words")
            self.assertEqual(len(recognizers[0].chunks), 5)
            self.assertEqual(list(model_path.iterdir()), [])

    def test_vosk_exposes_minimum_word_confidence_when_available(self):
        class ConfidenceRecognizer(FakeRecognizer):
            words_enabled = False

            def SetWords(self, enabled):
                self.words_enabled = enabled

            def FinalResult(self):
                return json.dumps({
                    "text": "five files",
                    "result": [
                        {"word": "five", "conf": 0.32},
                        {"word": "files", "conf": 0.94},
                    ],
                })

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            model_path.mkdir()
            recognizers = []

            def create_recognizer(model, sample_rate):
                recognizer = ConfidenceRecognizer(model, sample_rate)
                recognizers.append(recognizer)
                return recognizer

            transcriber = VoskTranscriber(
                self.small_settings(stt_model_path=model_path),
                model_factory=lambda _path: object(),
                recognizer_factory=create_recognizer,
            )

            transcript = transcriber.transcribe(
                pcm(1000, 20), cancel_event=threading.Event()
            )

        self.assertEqual(transcript, "five files")
        self.assertTrue(recognizers[0].words_enabled)
        self.assertEqual(transcriber.last_confidence, 0.32)

    def test_vosk_logs_stt_duration_and_outcome_without_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            model_path.mkdir()
            transcriber = VoskTranscriber(
                self.small_settings(stt_model_path=model_path),
                model_factory=lambda _path: object(),
                recognizer_factory=FakeRecognizer,
            )
            chunks = iter(
                [pcm(1000, 10), pcm(0, 10), pcm(0, 10), pcm(0, 10)]
            )

            with self.assertLogs("orion.audio.stt", level="INFO") as logs:
                transcriber.transcribe_chunks(chunks)

        self.assertIn("stage=stt", logs.output[-1])
        self.assertIn("outcome=transcribed", logs.output[-1])
        self.assertNotIn("recognized words", logs.output[-1])

    def test_whisper_ptt_uses_vad_pcm_in_memory(self):
        settings = self.small_settings()
        audio = [pcm(1000, 10), pcm(1000, 10), pcm(0, 10), pcm(0, 10), pcm(0, 10)]
        captured = []

        class FakeTranscriber:
            def transcribe(self, pcm_data, *, cancel_event, timeout):
                captured.append((pcm_data, cancel_event.is_set(), timeout))
                return "recognized locally"

            def close(self):
                pass

        class FakeSource:
            def chunks(self, _stop_event=None):
                yield from audio

        whisper_input = WhisperPushToTalkInput(
            settings,
            object(),
            FakeTranscriber(),
            source=FakeSource(),
        )
        self.assertEqual(whisper_input.transcribe_chunks(iter(audio)), "recognized locally")
        self.assertEqual(captured[0][0], b"".join(audio))
        self.assertFalse(captured[0][1])
        self.assertEqual(captured[0][2], settings.stt_timeout)

    def test_push_to_talk_confirmation_timeout_does_not_stop_service(self):
        service_stop = threading.Event()

        class FakeButton:
            is_pressed = False

            def wait_for_press(self, timeout=None):
                return None

        class FakeSource:
            def chunks(self, stop_event=None):
                yield from ()

        audio_input = PushToTalkInput(
            self.small_settings(),
            FakeButton(),
            source=FakeSource(),
            model_factory=lambda _path: object(),
            recognizer_factory=FakeRecognizer,
            stop_event=service_stop,
        )

        self.assertEqual(audio_input.next_utterance(timeout=0.01), "")
        self.assertFalse(service_stop.is_set())

    def test_push_to_talk_interrupt_during_capture_wait_returns_empty(self):
        gate = AudioGate()

        class InterruptingButton:
            is_pressed = False

            def wait_for_press(self, timeout=None):
                gate.request_interrupt()

        audio_input = PushToTalkInput(
            self.small_settings(), InterruptingButton(), gate=gate
        )

        self.assertEqual(audio_input.next_utterance(timeout=1), "")
        self.assertTrue(gate.capture_cancel_event.is_set())

    def test_capture_overflow_discards_utterance(self):
        settings = self.small_settings(audio_queue_chunks=1)

        class FakeStream:
            def __init__(self, callback):
                self.callback = callback

            def __enter__(self):
                self.callback(pcm(1, 10), 10, None, None)
                self.callback(pcm(2, 10), 10, None, None)
                return self

            def __exit__(self, *_args):
                return False

        source = SoundDeviceChunkSource(
            settings, stream_factory=lambda **kwargs: FakeStream(kwargs["callback"])
        )
        chunks = source.chunks()

        with self.assertRaises(AudioBufferOverflow):
            next(chunks)
        chunks.close()

    def test_wakeword_accepts_model_output_key_alias(self):
        class FakeWakeModel:
            def __init__(self, **_kwargs):
                pass

            def predict(self, _samples):
                return {"hey_jarvis_v0.1": 0.8}

        detector = OpenWakeWordDetector(
            "hey_jarvis",
            0.5,
            model_factory=FakeWakeModel,
            samples_factory=lambda chunk: array.array("h", chunk),
        )

        self.assertTrue(detector.detected(pcm(0, 10)))

    def test_wakeword_detection_is_debounced_and_respects_cooldown(self):
        now = [10.0]

        class FakeWakeModel:
            score = 0.8

            def __init__(self, **_kwargs):
                pass

            def predict(self, _samples):
                return {"hey_jarvis": self.score}

        model = FakeWakeModel()
        detector = OpenWakeWordDetector(
            "hey_jarvis", 0.5, model_factory=lambda **_kwargs: model,
            samples_factory=lambda chunk: chunk, cooldown_seconds=2.0,
            clock=lambda: now[0],
        )

        assert detector.detected(pcm(0, 10))
        assert not detector.detected(pcm(0, 10))
        model.score = 0.1
        assert not detector.detected(pcm(0, 10))
        model.score = 0.8
        now[0] = 11.9
        assert not detector.detected(pcm(0, 10))
        model.score = 0.1
        assert not detector.detected(pcm(0, 10))
        model.score = 0.8
        now[0] = 12.0
        assert detector.detected(pcm(0, 10))

    def test_wakeword_rejects_invalid_threshold_and_cooldown(self):
        with self.assertRaisesRegex(ValueError, "threshold must be between 0 and 1"):
            OpenWakeWordDetector("hey_jarvis", 1.1, model_factory=lambda **_kwargs: None)
        with self.assertRaisesRegex(ValueError, "cooldown cannot be negative"):
            OpenWakeWordDetector("hey_jarvis", 0.5, model_factory=lambda **_kwargs: None, cooldown_seconds=-1)

    def test_tts_uses_memory_and_reopens_capture_gate_on_error(self):
        gate = AudioGate()
        played = []

        class FakeTTS:
            def __init__(self, _model_name):
                pass

            def generate(self, _text, *, voice):
                self.voice = voice
                return [0.1, -0.1]

        def fail_play(audio, *, samplerate, device=None):
            played.append((list(audio), samplerate, device))
            raise RuntimeError("playback failed")

        speech = KittenSpeechOutput(
            self.small_settings(), gate, tts_factory=FakeTTS, play=fail_play, wait=lambda: None
        )

        with self.assertRaisesRegex(RuntimeError, "playback failed"):
            speech.speak("test")
        gate.wait_until_open()
        self.assertEqual(played, [([0.1, -0.1], 24000, "default")])

    def test_tts_warmup_loads_engine_and_runs_discarded_synthesis(self):
        generated = []
        played = []

        class FakeTTS:
            def __init__(self, _model_name):
                generated.append("initialized")

            def generate(self, text, *, voice):
                generated.append((text, voice))
                return [0.1, -0.1]

        speech = KittenSpeechOutput(
            self.small_settings(tts_voice="test-voice"),
            AudioGate(),
            tts_factory=FakeTTS,
            play=lambda audio, **_kwargs: played.append(audio),
            wait=lambda: None,
            stop=lambda: None,
        )

        speech.warmup()
        self.assertEqual(played, [])
        speech.speak("A real response")

        self.assertEqual(generated, ["initialized", ("Ready.", "test-voice"), ("A real response", "test-voice")])
        self.assertEqual(played, [[0.1, -0.1]])

    def test_tts_loads_local_onnx_artifacts_without_hub_model_factory(self):
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            (model_directory / "config.json").write_text(
                json.dumps({"type": "ONNX1", "model_file": "model.onnx", "voices": "voices.npz"}),
                encoding="utf-8",
            )
            (model_directory / "model.onnx").touch()
            (model_directory / "voices.npz").touch()
            loaded = []

            def local_factory(**kwargs):
                loaded.append(kwargs)
                return object()

            speech = KittenSpeechOutput(
                replace(self.small_settings(), tts_model_path=model_directory),
                AudioGate(),
                tts_factory=lambda _model_name: self.fail("Hub loader must not run"),
                local_tts_factory=local_factory,
            )

            speech._load_engine()

            self.assertEqual(loaded, [{
                "model_path": str((model_directory / "model.onnx").resolve()),
                "voices_path": str((model_directory / "voices.npz").resolve()),
            }])

    def test_tts_missing_local_asset_fails_without_hub_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            (model_directory / "config.json").write_text(
                json.dumps({"type": "ONNX1", "model_file": "missing.onnx", "voices": "voices.npz"}),
                encoding="utf-8",
            )
            (model_directory / "voices.npz").touch()
            speech = KittenSpeechOutput(
                replace(self.small_settings(), tts_model_path=model_directory),
                AudioGate(),
                tts_factory=lambda _model_name: self.fail("Hub loader must not run"),
                local_tts_factory=lambda **_kwargs: self.fail("Local loader must not run"),
            )

            with self.assertRaisesRegex(FileNotFoundError, "Local KittenTTS asset"):
                speech._load_engine()

    def test_tts_resamples_when_output_device_rejects_generated_rate(self):
        gate = AudioGate()
        played = []

        class FakeTTS:
            def __init__(self, _model_name):
                pass

            def generate(self, _text, *, voice):
                return [0.0, 1.0]

        class FakeSoundDevice:
            class PortAudioError(Exception):
                pass

            @staticmethod
            def check_output_settings(*, samplerate, **_kwargs):
                if samplerate == 24000:
                    raise FakeSoundDevice.PortAudioError("unsupported rate")

            @staticmethod
            def query_devices(_device, _kind):
                return {"default_samplerate": 44100}

            @staticmethod
            def play(audio, *, samplerate):
                played.append((audio, samplerate))

            @staticmethod
            def wait():
                pass

            @staticmethod
            def stop():
                pass

            class RawOutputStream:
                def __init__(self, *, samplerate, **_kwargs):
                    self.samplerate = samplerate

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def write(self, audio):
                    played.append((audio, self.samplerate))

                def abort(self):
                    pass

        speech = KittenSpeechOutput(
            self.small_settings(), gate, tts_factory=FakeTTS
        )
        with patch.dict(sys.modules, {"sounddevice": FakeSoundDevice}):
            speech.speak("test")

        self.assertEqual(played[0][1], 44100)
        resampled = array.array("f")
        resampled.frombytes(played[0][0])
        self.assertEqual(len(resampled), 4)
        self.assertAlmostEqual(resampled[1], 24000 / 44100)

    def test_tts_aborts_owned_output_stream_on_interrupt(self):
        gate = AudioGate()
        playback = []

        class FakeTTS:
            def __init__(self, _model_name):
                pass

            def generate(self, _text, *, voice):
                return [0.1] * 100

        class FakeStream:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                playback.append("closed")
                return False

            def write(self, _audio):
                playback.append("write")
                gate.request_interrupt()

            def abort(self):
                playback.append("abort")

        class FakeSoundDevice:
            class PortAudioError(Exception):
                pass

            @staticmethod
            def check_output_settings(**_kwargs):
                pass

        speech = KittenSpeechOutput(
            self.small_settings(),
            gate,
            tts_factory=FakeTTS,
            output_stream_factory=FakeStream,
        )
        with patch.dict(sys.modules, {"sounddevice": FakeSoundDevice}):
            with self.assertLogs("orion.audio.tts", level="INFO") as logs:
                speech.speak("test")

        self.assertEqual(playback, ["write", "abort", "closed"])
        self.assertTrue(gate.is_open)
        self.assertIn("TTS playback outcome=interrupted", "\n".join(logs.output))

    def test_tts_stops_playback_when_gate_receives_interrupt(self):
        gate = AudioGate()
        playback_done = threading.Event()
        stopped = threading.Event()
        playback_started = threading.Event()

        class FakeTTS:
            def __init__(self, _model_name):
                pass

            def generate(self, _text, *, voice):
                return [0.1]

        def wait_for_playback():
            playback_done.wait(1)

        def start_playback(*_args, **_kwargs):
            playback_started.set()

        def stop_playback():
            stopped.set()
            playback_done.set()

        speech = KittenSpeechOutput(
            self.small_settings(),
            gate,
            tts_factory=FakeTTS,
            play=start_playback,
            wait=wait_for_playback,
            stop=stop_playback,
        )
        thread = threading.Thread(target=speech.speak, args=("test",))
        thread.start()
        self.assertTrue(playback_started.wait(1))
        self.assertFalse(gate.is_open)
        gate.request_interrupt()
        thread.join(1)

        self.assertFalse(thread.is_alive())
        self.assertTrue(stopped.is_set())
        self.assertTrue(gate.is_open)


if __name__ == "__main__":
    unittest.main()
