"""Central configuration and validation for ORION."""

from __future__ import annotations

import os
import re
import math
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .core.policy import NetworkPolicy, NetworkPolicyError
from .models import (
    ModelSelectionStore,
    WAKEWORD_EMBEDDING_FILENAME,
    WAKEWORD_MANIFEST_FILENAME,
    WAKEWORD_MELSPECTROGRAM_FILENAME,
    default_model_root,
    get_model,
    resolve_model_path,
)


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
    model_root: Path = Path("/opt/orion/models")
    model_selection_path: Path = Path("~/.local/share/orion/model-selection.json")
    active_llm_model: str = "qwen3.5-2b-q4"
    active_stt_model: str = "vosk-en-small"
    active_tts_model: str = "kittentts-nano"
    active_embedding_model: str = "bge-small-en-v1.5"
    active_wakeword_model: str = "hey-orion"
    model_backend: str = "llama.cpp"
    model_path: Path | None = None
    model_base_url: str = "http://127.0.0.1:8080/v1"
    model_context_size: int = 4096
    model_template_overhead_tokens: int = 512
    model_answer_tokens: int = 512
    model_planner_tokens: int = 128
    model_threads: int = 4
    model_temperature: float = 0.2
    llama_server_path: str = "llama-server"
    llama_startup_timeout: float = 60.0
    max_agent_steps: int = 5
    max_tool_execution_time: float = 15.0
    max_total_task_time: float = 120.0
    max_context_tokens: int = 3072
    database_path: Path = Path("~/.local/share/orion/orion.sqlite3")
    memory_recent_message_limit: int = 16
    memory_search_limit: int = 5
    reminder_timezone: str = "Asia/Kolkata"
    reminder_poll_interval: float = 5.0
    searxng_base_url: str = ""
    searxng_allowed_hosts: tuple[str, ...] = ()
    online_tools_enabled: bool = True
    voice_profile: str = "legacy"
    stt_backend: str = "vosk"
    stt_language: str = "en"
    stt_threads: int = 2
    stt_timeout: float = 30.0
    stt_confidence_threshold: float = 0.55
    whisper_cli_path: str = "whisper-cli"
    tts_backend: str = "kittentts"
    tts_threads: int = 2
    runtime_idle_timeout: float = 60.0
    wake_word_model_path: Path | None = None
    wake_word_melspectrogram_path: Path | None = None
    wake_word_embedding_path: Path | None = None
    wake_word_manifest_path: Path | None = None
    wake_word_threads: int = 1
    wake_word_cooldown_seconds: float = 3.0
    microphone_button_debounce: float = 0.01
    tts_model_path: Path | None = None
    audio_device: str | None = "default"
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
    vosk_model_path: Path = Path("models/vosk-model-en-us-0.22-lgraph")
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
        voice_profile = os.getenv("ORION_VOICE_PROFILE", "legacy").lower()
        wake_word_enabled = os.getenv("ORION_WAKE_WORD_ENABLED", "false").lower() in {
            "1", "true", "yes", "on"
        }
        database_path = _env_path(
            "DATABASE_PATH", "~/.local/share/orion/orion.sqlite3"
        )
        model_root = Path(
            os.getenv("ORION_MODEL_ROOT", str(default_model_root()))
        ).expanduser().resolve()
        selection_path = Path(
            os.getenv(
                "ORION_MODEL_SELECTION_PATH",
                str(database_path.parent / "model-selection.json"),
            )
        ).expanduser()
        selections = ModelSelectionStore(selection_path).read()
        llm_model_id = os.getenv("ORION_LLM_MODEL_ID") or selections.get(
            "llm", "qwen3.5-2b-q4"
        )
        default_stt_backend = "whisper.cpp" if voice_profile == "wakeword" else "vosk"
        stt_env_id = os.getenv("ORION_STT_MODEL_ID")
        explicit_stt_backend = os.getenv("ORION_STT_BACKEND")
        stt_model_id = stt_env_id or selections.get(
            "stt",
            "whisper-base-en"
            if explicit_stt_backend == "whisper.cpp" or default_stt_backend == "whisper.cpp"
            else "vosk-en-small",
        )
        stt_backend = (
            get_model(stt_model_id).runtime
            if stt_env_id or "stt" in selections
            else (explicit_stt_backend or default_stt_backend).lower()
        )
        tts_env_id = os.getenv("ORION_TTS_MODEL_ID")
        explicit_tts_backend = os.getenv("ORION_TTS_BACKEND")
        if (
            explicit_tts_backend == "piper"
            and not os.getenv("ORION_TTS_MODEL_PATH")
            and not tts_env_id
            and "tts" not in selections
        ):
            raise ValueError("Piper requires ORION_TTS_MODEL_PATH or ORION_TTS_MODEL_ID")
        default_tts_id = (
            "piper-lessac-medium" if explicit_tts_backend == "piper" else "kittentts-nano"
        )
        tts_model_id = tts_env_id or selections.get("tts", default_tts_id)
        tts_backend = (
            get_model(tts_model_id).runtime
            if tts_env_id or "tts" in selections
            else explicit_tts_backend.lower()
            if explicit_tts_backend
            else get_model(tts_model_id).runtime
        )
        embedding_model_id = os.getenv("ORION_EMBEDDING_MODEL_ID") or selections.get(
            "embedding", "bge-small-en-v1.5"
        )
        wakeword_model_id = os.getenv("ORION_WAKEWORD_MODEL_ID") or selections.get(
            "wakeword",
            "hey-jarvis"
            if voice_profile == "legacy" and wake_word_enabled
            else "hey-orion",
        )
        llm_spec = get_model(llm_model_id)
        stt_spec = get_model(stt_model_id)
        tts_spec = get_model(tts_model_id)
        embedding_spec = get_model(embedding_model_id)
        wakeword_spec = get_model(wakeword_model_id)
        if any(spec.kind != expected for spec, expected in (
            (llm_spec, "llm"), (stt_spec, "stt"), (tts_spec, "tts"),
            (embedding_spec, "embedding"), (wakeword_spec, "wakeword"),
        )):
            raise ValueError("Selected model ID does not match its model role")
        wakeword_profile = voice_profile == "wakeword" or wake_word_enabled
        model_path = os.getenv("ORION_MODEL_PATH")
        stt_model_path = os.getenv("ORION_STT_MODEL_PATH")
        vosk_model_path = Path(
            os.getenv(
                "ORION_VOSK_MODEL_PATH",
                stt_model_path or "models/vosk-model-en-us-0.22-lgraph",
            )
        ).expanduser()
        audio_device = os.getenv("ORION_AUDIO_DEVICE", "default")
        microphone_mute_pin = os.getenv("ORION_MICROPHONE_MUTE_PIN")
        speaker_mute_pin = os.getenv("ORION_SPEAKER_MUTE_PIN")

        settings = cls(
            model_root=model_root,
            model_selection_path=selection_path,
            active_llm_model=llm_model_id,
            active_stt_model=stt_model_id,
            active_tts_model=tts_model_id,
            active_embedding_model=embedding_model_id,
            active_wakeword_model=wakeword_model_id,
            model_backend=model_backend,
            model_path=(
                resolve_model_path(model_root, llm_model_id)
                if os.getenv("ORION_LLM_MODEL_ID") or "llm" in selections
                else Path(model_path).expanduser()
                if model_path else resolve_model_path(model_root, llm_model_id)
            ),
            model_base_url=os.getenv(
                "ORION_MODEL_BASE_URL",
                "http://127.0.0.1:8080/v1",
            ),
            model_context_size=_env_int(
                "MODEL_CONTEXT_SIZE", llm_spec.context_size or 4096
            ),
            model_template_overhead_tokens=_env_int("MODEL_TEMPLATE_OVERHEAD_TOKENS", 512),
            model_answer_tokens=_env_int("MODEL_ANSWER_TOKENS", 512),
            model_planner_tokens=_env_int("MODEL_PLANNER_TOKENS", 128),
            model_threads=_env_int("MODEL_THREADS", 4),
            model_temperature=_env_float("MODEL_TEMPERATURE", 0.2),
            llama_server_path=os.getenv(
                "ORION_LLAMA_SERVER_PATH", "llama-server"
            ),
            llama_startup_timeout=_env_float("LLAMA_STARTUP_TIMEOUT", 60.0),
            max_agent_steps=_env_int("MAX_AGENT_STEPS", 5),
            max_tool_execution_time=_env_float("MAX_TOOL_EXECUTION_TIME", 15.0),
            max_total_task_time=_env_float("MAX_TOTAL_TASK_TIME", 120.0),
            max_context_tokens=_env_int("MAX_CONTEXT_TOKENS", 3072),
            database_path=database_path,
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
            online_tools_enabled=os.getenv("ORION_ONLINE_TOOLS_ENABLED", "true").lower()
            in {"1", "true", "yes", "on"},
            voice_profile=voice_profile,
            stt_backend=stt_backend,
            stt_language=os.getenv("ORION_STT_LANGUAGE", "en").lower(),
            stt_threads=_env_int("STT_THREADS", 2),
            stt_timeout=_env_float("STT_TIMEOUT", 30.0),
            stt_confidence_threshold=_env_float("STT_CONFIDENCE_THRESHOLD", 0.55),
            whisper_cli_path=os.getenv("ORION_WHISPER_CLI_PATH", "whisper-cli"),
            tts_backend=tts_backend,
            tts_threads=_env_int("TTS_THREADS", 2),
            runtime_idle_timeout=_env_float("RUNTIME_IDLE_TIMEOUT", 60.0),
            tts_model_path=(
                _env_path("TTS_MODEL_PATH", "")
                if os.getenv("ORION_TTS_MODEL_PATH") and not (
                    tts_env_id or "tts" in selections
                )
                else resolve_model_path(model_root, tts_model_id)
                if tts_spec.runtime == "piper" else None
            ),
            wake_word_model_path=(
                _env_path("WAKE_WORD_MODEL_PATH", "")
                if os.getenv("ORION_WAKE_WORD_MODEL_PATH")
                else Path("models/hey_jarvis.onnx")
                if wakeword_profile and wakeword_model_id == "hey-jarvis" and voice_profile == "legacy"
                else resolve_model_path(model_root, wakeword_model_id)
                if wakeword_profile else None
            ),
            wake_word_melspectrogram_path=(
                _env_path("WAKE_WORD_MELSPECTROGRAM_PATH", "")
                if os.getenv("ORION_WAKE_WORD_MELSPECTROGRAM_PATH")
                else Path("artifacts/wakeword-staging/melspectrogram.onnx")
                if wakeword_profile and wakeword_model_id == "hey-jarvis" and voice_profile == "legacy"
                else model_root / "WAKEWORD" / WAKEWORD_MELSPECTROGRAM_FILENAME
                if wakeword_profile else None
            ),
            wake_word_embedding_path=(
                _env_path("WAKE_WORD_EMBEDDING_PATH", "")
                if os.getenv("ORION_WAKE_WORD_EMBEDDING_PATH")
                else Path("artifacts/wakeword-staging/embedding_model.onnx")
                if wakeword_profile and wakeword_model_id == "hey-jarvis" and voice_profile == "legacy"
                else model_root / "WAKEWORD" / WAKEWORD_EMBEDDING_FILENAME
                if wakeword_profile else None
            ),
            wake_word_manifest_path=(
                _env_path("WAKE_WORD_MANIFEST_PATH", "")
                if os.getenv("ORION_WAKE_WORD_MANIFEST_PATH")
                else model_root / "WAKEWORD" / WAKEWORD_MANIFEST_FILENAME
                if wakeword_profile and wakeword_model_id == "hey-orion"
                else None
            ),
            wake_word_threads=_env_int("WAKE_WORD_THREADS", 1),
            wake_word_cooldown_seconds=_env_float("WAKE_WORD_COOLDOWN_SECONDS", 3.0),
            microphone_button_debounce=_env_float("MICROPHONE_BUTTON_DEBOUNCE", 0.01),
            audio_sample_rate=_env_int("AUDIO_SAMPLE_RATE", 16000),
            audio_block_frames=_env_int("AUDIO_BLOCK_FRAMES", 320),
            audio_queue_chunks=_env_int("AUDIO_QUEUE_CHUNKS", 8),
            max_utterance_seconds=_env_int("MAX_UTTERANCE_SECONDS", 120),
            vad_rms_threshold=_env_int("VAD_RMS_THRESHOLD", 500),
            vad_silence_ms=_env_int("VAD_SILENCE_MS", 800),
            vad_preroll_ms=_env_int("VAD_PREROLL_MS", 200),
            wake_word=os.getenv("ORION_WAKE_WORD", "hey_jarvis"),
            wake_word_enabled=wake_word_enabled,
            wake_word_threshold=_env_float("WAKE_WORD_THRESHOLD", 0.5),
            stt_model_path=(
                _env_path("STT_MODEL_PATH", "")
                if stt_model_path and not (stt_env_id or "stt" in selections)
                else resolve_model_path(model_root, stt_model_id)
                if stt_spec.runtime == "whisper.cpp" else vosk_model_path
            ),
            vosk_model_path=vosk_model_path,
            tts_voice=os.getenv(
                "ORION_TTS_VOICE",
                Path(tts_spec.filename).stem
                if tts_spec.runtime == "piper" else "expr-voice-3-f",
            ),
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
        if self.voice_profile not in {"legacy", "wakeword"}:
            raise ValueError("ORION_VOICE_PROFILE must be 'legacy' or 'wakeword'")
        if self.stt_backend not in {"vosk", "whisper.cpp"}:
            raise ValueError("ORION_STT_BACKEND must be 'vosk' or 'whisper.cpp'")
        if self.tts_backend not in {"kittentts", "piper"}:
            raise ValueError("ORION_TTS_BACKEND must be 'kittentts' or 'piper'")
        if self.tts_backend == "piper" and (
            self.tts_model_path is None or self.tts_model_path.suffix.lower() != ".onnx"
        ):
            raise ValueError("Piper requires ORION_TTS_MODEL_PATH to name a local .onnx voice")
        if self.voice_profile == "wakeword" or self.wake_word_enabled:
            if self.voice_profile == "wakeword" and self.audio_sample_rate != 16000:
                raise ValueError("Wake-word input requires 16000 Hz mono audio")
            if any(path is None for path in (
                self.wake_word_model_path,
                self.wake_word_melspectrogram_path,
                self.wake_word_embedding_path,
            )):
                raise ValueError("Wake-word input requires a local classifier and feature ONNX paths")
        if not re.fullmatch(r"(?:[a-z]{2,3}|auto)", self.stt_language):
            raise ValueError("ORION_STT_LANGUAGE must be a language code or 'auto'")
        if self.stt_threads < 1 or self.wake_word_threads < 1:
            raise ValueError("ORION STT and wake-word threads must be positive")
        if (
            not math.isfinite(self.stt_confidence_threshold)
            or not 0 <= self.stt_confidence_threshold <= 1
        ):
            raise ValueError("ORION_STT_CONFIDENCE_THRESHOLD must be between 0 and 1")
        if (
            not math.isfinite(self.stt_timeout)
            or not math.isfinite(self.microphone_button_debounce)
            or not math.isfinite(self.wake_word_cooldown_seconds)
            or self.stt_timeout <= 0
            or self.microphone_button_debounce <= 0
            or self.wake_word_cooldown_seconds < 0
        ):
            raise ValueError("ORION STT timeout and microphone debounce must be positive; wake cooldown cannot be negative")
        if self.model_backend != "llama.cpp":
            raise ValueError("ORION_MODEL_BACKEND currently supports only 'llama.cpp'")
        if self.model_context_size < 512:
            raise ValueError("ORION_MODEL_CONTEXT_SIZE must be at least 512")
        if self.model_context_size > 4096:
            raise ValueError("ORION_MODEL_CONTEXT_SIZE cannot exceed 4096 before larger-context validation")
        if min(
            self.model_template_overhead_tokens,
            self.model_answer_tokens,
            self.model_planner_tokens,
        ) < 1:
            raise ValueError("ORION model token budgets must be positive")
        if self.model_planner_tokens > self.model_answer_tokens:
            raise ValueError("ORION_MODEL_PLANNER_TOKENS cannot exceed answer token budget")
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
        if (
            self.max_context_tokens
            + self.model_template_overhead_tokens
            + self.model_answer_tokens
            > self.model_context_size
        ):
            raise ValueError("ORION prompt and output budgets exceed model context size")
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
        if self.tts_sample_rate < 1 or self.tts_threads < 1:
            raise ValueError("ORION TTS sample rate and thread count must be positive")
        if not math.isfinite(self.runtime_idle_timeout) or self.runtime_idle_timeout <= 0:
            raise ValueError("ORION_RUNTIME_IDLE_TIMEOUT must be finite and positive")
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
            self.microphone_control_pin if self.voice_profile == "wakeword" else self.push_to_talk_pin,
            self.tts_interrupt_pin,
            self.status_led_red_pin,
            self.status_led_pin,
            self.status_led_blue_pin,
        ]
        configured_pins.extend(
            pin
            for pin in (
                self.microphone_mute_pin if self.voice_profile == "legacy" else None,
                self.speaker_mute_pin,
            )
            if pin is not None
        )
        if any(pin < 0 for pin in configured_pins):
            raise ValueError("ORION GPIO pin numbers must be non-negative")
        if len(configured_pins) != len(set(configured_pins)):
            raise ValueError("ORION GPIO pin numbers must be unique")

    @property
    def microphone_control_pin(self) -> int:
        return self.microphone_mute_pin if self.microphone_mute_pin is not None else self.push_to_talk_pin