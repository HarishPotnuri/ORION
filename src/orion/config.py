"""Central configuration and validation for ORION."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .core.policy import NetworkPolicy, NetworkPolicyError


def _env_int(name: str, default: int) -> int:
    value = os.getenv(f"ORION_{name}")
    try:
        return default if value is None else int(value)
    except ValueError as error:
        raise ValueError(f"ORION_{name} must be an integer") from error


def _env_float(name: str, default: float) -> float:
    value = os.getenv(f"ORION_{name}")
    try:
        return default if value is None else float(value)
    except ValueError as error:
        raise ValueError(f"ORION_{name} must be a number") from error


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(f"ORION_{name}", default)).expanduser()


@dataclass(frozen=True, slots=True)
class Settings:
    model_backend: str = "llama.cpp"
    model_path: Path | None = None
    model_name: str = "qwen3:1.7b"
    model_base_url: str = "http://127.0.0.1:8080/v1"
    model_context_size: int = 8192
    model_threads: int = 4
    model_temperature: float = 0.2
    llama_server_path: str = "llama-server"
    llama_startup_timeout: float = 60.0
    max_agent_steps: int = 5
    max_tool_execution_time: float = 15.0
    max_total_task_time: float = 120.0
    max_context_tokens: int = 6000
    database_path: Path = Path("~/.local/share/orion/orion.sqlite3")
    memory_recent_message_limit: int = 16
    memory_search_limit: int = 5
    reminder_timezone: str = "Asia/Kolkata"
    reminder_poll_interval: float = 5.0
    searxng_base_url: str = ""
    searxng_allowed_hosts: tuple[str, ...] = ()
    audio_device: str | None = None
    audio_sample_rate: int = 16000
    audio_block_frames: int = 320
    audio_queue_chunks: int = 8
    max_utterance_seconds: int = 120
    vad_rms_threshold: int = 500
    vad_silence_ms: int = 800
    vad_preroll_ms: int = 200
    wake_word: str = "hey_jarvis"
    wake_word_enabled: bool = False
    wake_word_threshold: float = 0.5
    stt_model_path: Path = Path("models/vosk-model-en-us-0.22-lgraph")
    tts_voice: str = "expr-voice-3-f"
    tts_sample_rate: int = 24000
    push_to_talk_pin: int = 23
    tts_interrupt_pin: int = 16
    status_led_red_pin: int = 27
    status_led_pin: int = 25
    status_led_blue_pin: int = 17
    microphone_mute_pin: int | None = None
    speaker_mute_pin: int | None = None
    log_level: str = "INFO"

    @classmethod
    def from_environment(cls) -> Settings:
        model_backend = os.getenv("ORION_MODEL_BACKEND", "llama.cpp").lower()
        model_path = os.getenv("ORION_MODEL_PATH")
        audio_device = os.getenv("ORION_AUDIO_DEVICE")
        microphone_mute_pin = os.getenv("ORION_MICROPHONE_MUTE_PIN")
        speaker_mute_pin = os.getenv("ORION_SPEAKER_MUTE_PIN")

        settings = cls(
            model_backend=model_backend,
            model_path=Path(model_path).expanduser() if model_path else None,
            model_name=os.getenv("ORION_MODEL_NAME", "qwen3:1.7b"),
            model_base_url=os.getenv(
                "ORION_MODEL_BASE_URL",
                "http://127.0.0.1:11434"
                if model_backend == "ollama"
                else "http://127.0.0.1:8080/v1",
            ),
            model_context_size=_env_int("MODEL_CONTEXT_SIZE", 8192),
            model_threads=_env_int("MODEL_THREADS", 4),
            model_temperature=_env_float("MODEL_TEMPERATURE", 0.2),
            llama_server_path=os.getenv(
                "ORION_LLAMA_SERVER_PATH", "llama-server"
            ),
            llama_startup_timeout=_env_float("LLAMA_STARTUP_TIMEOUT", 60.0),
            max_agent_steps=_env_int("MAX_AGENT_STEPS", 5),
            max_tool_execution_time=_env_float("MAX_TOOL_EXECUTION_TIME", 15.0),
            max_total_task_time=_env_float("MAX_TOTAL_TASK_TIME", 120.0),
            max_context_tokens=_env_int("MAX_CONTEXT_TOKENS", 6000),
            database_path=_env_path(
                "DATABASE_PATH", "~/.local/share/orion/orion.sqlite3"
            ),
            memory_recent_message_limit=_env_int("MEMORY_RECENT_MESSAGE_LIMIT", 16),
            memory_search_limit=_env_int("MEMORY_SEARCH_LIMIT", 5),
            reminder_timezone=os.getenv("ORION_REMINDER_TIMEZONE", "Asia/Kolkata"),
            reminder_poll_interval=_env_float("REMINDER_POLL_INTERVAL", 5.0),
            searxng_base_url=os.getenv("ORION_SEARXNG_BASE_URL", "").rstrip("/"),
            searxng_allowed_hosts=tuple(
                host.strip().lower()
                for host in os.getenv("ORION_SEARXNG_ALLOWED_HOSTS", "").split(",")
                if host.strip()
            ),
            audio_device=audio_device,
            audio_sample_rate=_env_int("AUDIO_SAMPLE_RATE", 16000),
            audio_block_frames=_env_int("AUDIO_BLOCK_FRAMES", 320),
            audio_queue_chunks=_env_int("AUDIO_QUEUE_CHUNKS", 8),
            max_utterance_seconds=_env_int("MAX_UTTERANCE_SECONDS", 120),
            vad_rms_threshold=_env_int("VAD_RMS_THRESHOLD", 500),
            vad_silence_ms=_env_int("VAD_SILENCE_MS", 800),
            vad_preroll_ms=_env_int("VAD_PREROLL_MS", 200),
            wake_word=os.getenv("ORION_WAKE_WORD", "hey_jarvis"),
            wake_word_enabled=os.getenv("ORION_WAKE_WORD_ENABLED", "false").lower()
            in {"1", "true", "yes", "on"},
            wake_word_threshold=_env_float("WAKE_WORD_THRESHOLD", 0.5),
            stt_model_path=_env_path(
                "STT_MODEL_PATH", "models/vosk-model-en-us-0.22-lgraph"
            ),
            tts_voice=os.getenv("ORION_TTS_VOICE", "expr-voice-3-f"),
            tts_sample_rate=_env_int("TTS_SAMPLE_RATE", 24000),
            push_to_talk_pin=_env_int("PUSH_TO_TALK_PIN", 23),
            tts_interrupt_pin=_env_int("TTS_INTERRUPT_PIN", 16),
            status_led_red_pin=_env_int("STATUS_LED_RED_PIN", 27),
            status_led_pin=_env_int("STATUS_LED_PIN", 25),
            status_led_blue_pin=_env_int("STATUS_LED_BLUE_PIN", 17),
            microphone_mute_pin=(
                int(microphone_mute_pin) if microphone_mute_pin else None
            ),
            speaker_mute_pin=int(speaker_mute_pin) if speaker_mute_pin else None,
            log_level=os.getenv("ORION_LOG_LEVEL", "INFO").upper(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.model_backend not in {"llama.cpp", "ollama"}:
            raise ValueError("ORION_MODEL_BACKEND must be 'llama.cpp' or 'ollama'")
        if self.model_context_size < 512:
            raise ValueError("ORION_MODEL_CONTEXT_SIZE must be at least 512")
        if self.model_threads < 1:
            raise ValueError("ORION_MODEL_THREADS must be at least 1")
        if not 0 <= self.model_temperature <= 2:
            raise ValueError("ORION_MODEL_TEMPERATURE must be between 0 and 2")
        if self.llama_startup_timeout <= 0:
            raise ValueError("ORION_LLAMA_STARTUP_TIMEOUT must be greater than zero")
        if self.max_agent_steps < 1:
            raise ValueError("ORION_MAX_AGENT_STEPS must be at least 1")
        if self.max_tool_execution_time <= 0 or self.max_total_task_time <= 0:
            raise ValueError("ORION task time limits must be greater than zero")
        if not 1 <= self.max_context_tokens < self.model_context_size:
            raise ValueError(
                "ORION_MAX_CONTEXT_TOKENS must be positive and below model context size"
            )
        if self.memory_recent_message_limit < 1 or self.memory_search_limit < 1:
            raise ValueError("ORION memory limits must be positive")
        try:
            ZoneInfo(self.reminder_timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError("ORION_REMINDER_TIMEZONE must be a valid IANA timezone") from error
        if self.reminder_poll_interval <= 0:
            raise ValueError("ORION_REMINDER_POLL_INTERVAL must be positive")
        if self.searxng_base_url:
            try:
                hostname = urlparse(self.searxng_base_url).hostname
                NetworkPolicy().validate_factual_url(
                    self.searxng_base_url, set(self.searxng_allowed_hosts)
                )
            except (ValueError, NetworkPolicyError) as error:
                raise ValueError("ORION_SEARXNG_BASE_URL must use an allowlisted HTTPS host") from error
            if hostname is None or hostname.lower() not in self.searxng_allowed_hosts:
                raise ValueError("ORION_SEARXNG_ALLOWED_HOSTS must include the configured host")
        if min(
            self.audio_sample_rate,
            self.audio_block_frames,
            self.audio_queue_chunks,
            self.max_utterance_seconds,
            self.vad_silence_ms,
        ) < 1:
            raise ValueError("ORION audio limits must be positive")
        if self.vad_preroll_ms < 0 or self.vad_rms_threshold < 0:
            raise ValueError("ORION VAD threshold and pre-roll must be non-negative")
        if not 0 <= self.wake_word_threshold <= 1:
            raise ValueError("ORION_WAKE_WORD_THRESHOLD must be between 0 and 1")
        if self.tts_sample_rate < 1:
            raise ValueError("ORION_TTS_SAMPLE_RATE must be positive")
        self._validate_local_model_url()
        self._validate_gpio_pins()
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("ORION_LOG_LEVEL is not a supported logging level")

    def _validate_local_model_url(self) -> None:
        try:
            NetworkPolicy().validate_model_endpoint(self.model_base_url)
        except NetworkPolicyError as error:
            raise ValueError(str(error)) from error

    def _validate_gpio_pins(self) -> None:
        configured_pins = [
            self.push_to_talk_pin,
            self.tts_interrupt_pin,
            self.status_led_red_pin,
            self.status_led_pin,
            self.status_led_blue_pin,
        ]
        configured_pins.extend(
            pin
            for pin in (self.microphone_mute_pin, self.speaker_mute_pin)
            if pin is not None
        )
        if any(pin < 0 for pin in configured_pins):
            raise ValueError("ORION GPIO pin numbers must be non-negative")
        if len(configured_pins) != len(set(configured_pins)):
            raise ValueError("ORION GPIO pin numbers must be unique")