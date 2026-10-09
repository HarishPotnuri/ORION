import array
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from orion.audio.capture import AudioGate
from orion.audio.tts import PiperSpeechOutput
from orion.config import Settings


class PiperSpeechOutputTests(unittest.TestCase):
    def test_loads_local_voice_with_configured_threads_and_plays_metadata_rate_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "en_US-lessac-medium.onnx"
            config_path = Path(f"{model_path}.json")
            model_path.touch()
            config_path.write_text("{}", encoding="utf-8")
            loaded = []
            played = []

            class Voice:
                config = SimpleNamespace(sample_rate=16000)

                def synthesize(self, text):
                    self.text = text
                    yield SimpleNamespace(
                        audio_int16_bytes=array.array("h", [16384, -16384]).tobytes()
                    )

            class FakeStream:
                def __init__(self, *, samplerate, **_kwargs):
                    self.samplerate = samplerate

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def write(self, audio):
                    played.append((self.samplerate, array.array("f", audio)))

                def abort(self):
                    self.fail("unexpected interrupt")

            fake_sounddevice = SimpleNamespace(
                PortAudioError=type("PortAudioError", (Exception,), {}),
                check_output_settings=lambda **_kwargs: None,
                RawOutputStream=FakeStream,
            )
            voice = Voice()
            speech = PiperSpeechOutput(
                replace(Settings(), tts_backend="piper", tts_model_path=model_path),
                AudioGate(),
                voice_factory=lambda **kwargs: loaded.append(kwargs) or voice,
            )
            with patch.dict(sys.modules, {"sounddevice": fake_sounddevice}):
                speech.speak("Hello, Orion.")

            self.assertEqual(loaded, [{
                "model_path": model_path.resolve(),
                "config_path": config_path.resolve(),
                "threads": 2,
            }])
            self.assertEqual(voice.text, "Hello, Orion.")
            self.assertEqual(played[0][0], 24000)
            self.assertEqual(len(played[0][1]), 3)

    def test_missing_adjacent_voice_config_fails_without_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "voice.onnx"
            model_path.touch()
            speech = PiperSpeechOutput(
                replace(Settings(), tts_backend="piper", tts_model_path=model_path),
                AudioGate(),
                voice_factory=lambda **_kwargs: self.fail("voice loader must not run"),
            )

            with self.assertRaisesRegex(FileNotFoundError, "Piper voice config"):
                speech._load_voice()

    def test_interrupt_aborts_playback_and_discards_later_audio_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "voice.onnx"
            Path(f"{model_path}.json").touch()
            model_path.touch()
            gate = AudioGate()
            produced = []
            playback = []

            class Voice:
                config = SimpleNamespace(sample_rate=24000)

                def synthesize(self, _text):
                    yield SimpleNamespace(
                        audio_int16_bytes=array.array("h", [1000] * 4).tobytes()
                    )
                    produced.append("later chunk")
                    yield SimpleNamespace(
                        audio_int16_bytes=array.array("h", [1000] * 4).tobytes()
                    )

            class FakeStream:
                def __init__(self, **_kwargs):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def write(self, _audio):
                    playback.append("write")
                    gate.request_interrupt()

                def abort(self):
                    playback.append("abort")

            fake_sounddevice = SimpleNamespace(
                PortAudioError=type("PortAudioError", (Exception,), {}),
                check_output_settings=lambda **_kwargs: None,
                RawOutputStream=FakeStream,
            )
            speech = PiperSpeechOutput(
                replace(Settings(), tts_backend="piper", tts_model_path=model_path),
                gate,
                voice_factory=lambda **_kwargs: Voice(),
            )
            with patch.dict(sys.modules, {"sounddevice": fake_sounddevice}):
                speech.speak("Stop now.")

            self.assertEqual(playback, ["write", "abort"])
            self.assertEqual(produced, [])

    def test_idle_voice_can_be_evicted(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "voice.onnx"
            Path(f"{model_path}.json").touch()
            model_path.touch()
            speech = PiperSpeechOutput(
                replace(Settings(), tts_backend="piper", tts_model_path=model_path),
                AudioGate(),
                voice_factory=lambda **_kwargs: object(),
            )
            speech._load_voice()
            speech._last_used -= 61

            self.assertTrue(speech.evict_if_idle(60))
            self.assertIsNone(speech._voice)


if __name__ == "__main__":
    unittest.main()