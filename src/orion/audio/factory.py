"""Explicit local audio provider selection without model initialization."""

from ..config import Settings
from .capture import AudioGate
from .stt import VoskTranscriber
from .tts import KittenSpeechOutput, PiperSpeechOutput
from .whisper import WhisperCppTranscriber


def create_transcriber(settings: Settings):
    if settings.stt_backend == "whisper.cpp":
        return WhisperCppTranscriber(settings)
    if settings.stt_backend == "vosk":
        return VoskTranscriber(settings)
    raise ValueError(f"Unsupported STT backend: {settings.stt_backend}")


def create_speech_output(settings: Settings, gate: AudioGate):
    if settings.tts_backend == "kittentts":
        return KittenSpeechOutput(settings, gate)
    if settings.tts_backend == "piper":
        return PiperSpeechOutput(settings, gate)
    raise ValueError(f"Unsupported TTS backend: {settings.tts_backend}")