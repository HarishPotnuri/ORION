"""Offline model inventory, path resolution, and selection persistence."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ModelKind = Literal["llm", "stt", "tts", "embedding", "wakeword"]
WAKEWORD_MELSPECTROGRAM_FILENAME = "melspectrogram.onnx"
WAKEWORD_EMBEDDING_FILENAME = "embedding_model.onnx"
WAKEWORD_MANIFEST_FILENAME = "phase13-artifacts.json"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    display_name: str
    kind: ModelKind
    depot_directory: str
    filename: str
    runtime: str
    quantization: str | None = None
    context_size: int | None = None
    expected_ram_gb: float | None = None
    notes: str = ""


MODEL_SPECS = (
    ModelSpec("qwen3.5-2b-q4", "Qwen3.5 2B Q4_K_M", "llm", "LLM", "Qwen3.5-2B-Q4_K_M.gguf", "llama.cpp", "Q4_K_M", 4096, notes="Current benchmark target"),
    ModelSpec("minicpm5-2b-q4", "MiniCPM5 2B Q4_K_M", "llm", "LLM", "MiniCPM5-2B-Q4_K_M.gguf", "llama.cpp", "Q4_K_M", 4096),
    ModelSpec("qwen3.5-2b-q8", "Qwen3.5 2B Q8_0", "llm", "LLM", "Qwen3.5-2B-Q8_0.gguf", "llama.cpp", "Q8_0", 4096),
    ModelSpec("qwen3-1.7b-q8", "Qwen3 1.7B Q8_0", "llm", "LLM", "Qwen3-1.7B-Q8_0.gguf", "llama.cpp", "Q8_0", 4096),
    ModelSpec("phi4-mini-q4", "Phi-4 mini instruct Q4_K_M", "llm", "LLM", "Phi-4-mini-instruct-Q4_K_M.gguf", "llama.cpp", "Q4_K_M", 4096),
    ModelSpec("gemma4-e2b-q8", "Gemma 4 E2B Q8_0", "llm", "LLM", "gemma-4-E2B-Q8_0.gguf", "llama.cpp", "Q8_0", 4096),
    ModelSpec("whisper-tiny-en", "Whisper Tiny English", "stt", "STT", "ggml-tiny.en.bin", "whisper.cpp"),
    ModelSpec("whisper-base-en", "Whisper Base English", "stt", "STT", "ggml-base.en.bin", "whisper.cpp"),
    ModelSpec("whisper-base-multilingual", "Whisper Base Multilingual", "stt", "STT", "ggml-base.bin", "whisper.cpp"),
    ModelSpec("vosk-en-small", "VOSK Small English", "stt", "", "", "vosk", notes="Externally provisioned directory; configure ORION_VOSK_MODEL_PATH"),
    ModelSpec("piper-lessac-medium", "Piper Lessac Medium", "tts", "TTS", "en_US-lessac-medium.onnx", "piper", notes="Requires adjacent .onnx.json"),
    ModelSpec("piper-hfc-female-medium", "Piper HFC Female Medium", "tts", "TTS", "en_US-hfc_female-medium.onnx", "piper", notes="Requires adjacent .onnx.json"),
    ModelSpec("kittentts-nano", "KittenTTS Nano", "tts", "", "", "kittentts", notes="Uses configured local directory or offline pre-provisioned cache"),
    ModelSpec("bge-small-en-v1.5", "BGE Small English v1.5", "embedding", "EMBEDDING", "BGE-small-en-v1.5-model.onnx", "onnxruntime", notes="Tokenizer/input contract must be verified before vector generation"),
    ModelSpec("hey-orion", "Hey Orion", "wakeword", "WAKEWORD", "hey_orion.onnx", "openWakeWord", notes="Trained local candidate; requires provenance manifest and Pi acoustic acceptance before activation"),
    ModelSpec("hey-jarvis", "Hey Jarvis", "wakeword", "WAKEWORD", "hey_jarvis.onnx", "openWakeWord", notes="Development/staging fallback only"),
)
MODEL_REGISTRY = {spec.model_id: spec for spec in MODEL_SPECS}


def default_model_root() -> Path:
    configured = os.getenv("ORION_MODEL_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        source_root = Path(__file__).resolve().parents[2]
        sibling_depot = source_root.parent / "ORION-MODELS"
        if source_root.name.casefold() == "orion" or sibling_depot.exists():
            return sibling_depot.resolve()
        return (Path.cwd().resolve().parent / "ORION-MODELS").resolve()
    return Path("/opt/orion/models")


def resolve_model_path(root: Path, model_id: str) -> Path:
    spec = get_model(model_id)
    if not spec.filename or not spec.depot_directory:
        raise ValueError(f"Model {model_id!r} uses a separately configured asset path")
    return root.expanduser() / spec.depot_directory / spec.filename


def get_model(model_id: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError:
        raise ValueError(f"Unknown model ID: {model_id}") from None


class ModelSelectionStore:
    def __init__(self, path: Path):
        self.path = path.expanduser()

    def read(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Model selection file is unreadable: {self.path}") from error
        if not isinstance(value, dict) or any(
            key not in {"llm", "stt", "tts", "embedding", "wakeword"}
            or not isinstance(model_id, str)
            for key, model_id in value.items()
        ):
            raise ValueError(f"Model selection file has invalid entries: {self.path}")
        for role, model_id in value.items():
            if get_model(model_id).kind != role:
                raise ValueError(f"Model {model_id!r} is not valid for role {role!r}")
        return value

    def write(self, selections: dict[str, str]) -> None:
        for role, model_id in selections.items():
            if role not in {"llm", "stt", "tts", "embedding", "wakeword"}:
                raise ValueError(f"Unknown model role: {role}")
            if get_model(model_id).kind != role:
                raise ValueError(f"Model {model_id!r} is not a {role} model")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
                json.dump(selections, output, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_name, self.path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise


def validate_model_file(root: Path, model_id: str) -> tuple[bool, str, int | None]:
    spec = get_model(model_id)
    if not spec.filename:
        return False, spec.notes, None
    path = resolve_model_path(root, model_id)
    if not path.is_file():
        return False, f"Model file not found: {path}", None
    try:
        with path.open("rb") as model_file:
            model_file.read(1)
    except OSError as error:
        return False, f"Model file is not readable: {path} ({type(error).__name__})", None
    if spec.kind == "llm" and path.suffix.lower() != ".gguf":
        return False, f"Expected a .gguf model file: {path}", None
    if spec.kind == "stt" and spec.runtime == "whisper.cpp" and path.suffix.lower() != ".bin":
        return False, f"Expected a Whisper .bin model file: {path}", None
    if spec.kind in {"tts", "embedding", "wakeword"} and path.suffix.lower() != ".onnx":
        return False, f"Expected an .onnx model file: {path}", None
    if spec.kind == "tts" and not Path(f"{path}.json").is_file():
        return False, f"Piper voice config not found: {path}.json", None
    return True, "file present and readable", path.stat().st_size
