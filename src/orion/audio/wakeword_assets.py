"""Validation for the local openWakeWord ONNX asset set."""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any, Callable


_ASSETS = ("wakeword", "melspectrogram", "embedding")
_EXPECTED_RUNTIME_VERSION = "0.6.0"
_EXPECTED_COMPATIBILITY = {
    "sample_rate_hz": 16000,
    "melspectrogram_bins": 32,
    "embedding_window_frames": 76,
    "embedding_dimension": 96,
    "wake_output_classes": 1,
}
_PLACEHOLDER_METADATA = {"unknown", "n/a", "na", "tbd", "todo", "unverified"}


def _metadata_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Wake-word manifest field '{field}' must be a non-empty string")
    if value.strip().casefold() in _PLACEHOLDER_METADATA:
        raise ValueError(f"Wake-word manifest field '{field}' cannot be placeholder metadata")
    return value.strip()


def _axis_matches(actual: Any, expected: int | None) -> bool:
    if expected is None:
        return actual is None or isinstance(actual, str) or (isinstance(actual, int) and actual > 0)
    return actual == expected


def _check_tensor(value: Any, *, name: str | None, rank: int, shape: tuple[int | None, ...], label: str) -> None:
    if name is not None and value.name != name:
        raise ValueError(f"{label} input tensor must be named '{name}'")
    if value.type != "tensor(float)":
        raise ValueError(f"{label} tensor must use float32")
    actual_shape = value.shape
    if len(actual_shape) != rank or len(shape) != rank:
        raise ValueError(f"{label} tensor must have rank {rank}")
    if not all(_axis_matches(actual, expected) for actual, expected in zip(actual_shape, shape)):
        raise ValueError(f"{label} tensor shape is incompatible: {actual_shape}")


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read wake-word manifest ({type(error).__name__})") from None
    if not isinstance(manifest, dict):
        raise ValueError("Wake-word manifest root must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise ValueError("Wake-word manifest schema_version must be 1")
    if manifest.get("wake_phrase") != "Hey Orion" or manifest.get("sample_rate_hz") != 16000:
        raise ValueError("Wake-word manifest must declare Hey Orion at 16000 Hz")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("name") != "openwakeword":
        raise ValueError("Wake-word manifest must identify openwakeword runtime")
    if runtime.get("version") != _EXPECTED_RUNTIME_VERSION or runtime.get("framework") != "onnx":
        raise ValueError("Wake-word manifest runtime must be openwakeword 0.6.0 with ONNX")
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, dict):
        raise ValueError("Wake-word manifest must declare compatibility metadata")
    for field, expected in _EXPECTED_COMPATIBILITY.items():
        if compatibility.get(field) != expected:
            raise ValueError(f"Wake-word compatibility field '{field}' must equal {expected}")
    assets = manifest.get("assets")
    if not isinstance(assets, dict) or set(assets) != set(_ASSETS):
        raise ValueError("Wake-word manifest must describe wakeword, melspectrogram, and embedding assets")
    feature_hashes = compatibility.get("feature_model_sha256")
    if not isinstance(feature_hashes, dict):
        raise ValueError("Wake-word manifest must bind the feature-model checksums")
    for key in ("melspectrogram", "embedding"):
        entry = assets[key]
        if not isinstance(entry, dict) or feature_hashes.get(key) != entry.get("sha256"):
            raise ValueError(f"Wake-word manifest compatibility does not bind {key} checksum")
    wake_entry = assets["wakeword"]
    if not isinstance(wake_entry, dict):
        raise ValueError("Wake-word model metadata must be an object")
    _metadata_text(wake_entry.get("training_recipe"), "assets.wakeword.training_recipe")
    _metadata_text(wake_entry.get("training_data"), "assets.wakeword.training_data")
    for key in _ASSETS:
        entry = assets[key]
        if not isinstance(entry, dict):
            raise ValueError(f"Wake-word metadata for {key} must be an object")
        for field in ("file", "source", "revision", "license"):
            _metadata_text(entry.get(field), f"assets.{key}.{field}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
            raise ValueError(f"Wake-word SHA-256 for {key} must contain 64 lowercase hexadecimal characters")
        if Path(entry["file"]).name != entry["file"] or entry["file"] in {".", ".."}:
            raise ValueError(f"Wake-word asset filename for {key} must be a basename")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_wakeword_assets(
    *,
    wakeword_path: Path | None,
    melspectrogram_path: Path | None,
    embedding_path: Path | None,
    manifest_path: Path | None,
    session_factory: Callable[..., Any] | None = None,
    runtime_version: str | None = None,
    numpy_module: Any | None = None,
    require_manifest: bool = True,
) -> None:
    """Verify metadata, hashes, ONNX contracts, and end-to-end feature shapes."""
    paths = {
        "wakeword": wakeword_path,
        "melspectrogram": melspectrogram_path,
        "embedding": embedding_path,
    }
    if any(path is None for path in paths.values()):
        raise FileNotFoundError("Wake-word profile requires three local ONNX model files")
    if require_manifest and (manifest_path is None or not manifest_path.is_file()):
        raise FileNotFoundError("Wake-word profile requires a verified local asset manifest")
    for key, path in paths.items():
        if not path.is_file() or path.suffix.lower() != ".onnx":
            raise FileNotFoundError(f"Wake-word {key} asset must be a local ONNX file")

    if manifest_path is not None:
        manifest = _load_manifest(manifest_path)
        for key, path in paths.items():
            entry = manifest["assets"][key]
            if entry["file"] != path.name:
                raise ValueError(f"Wake-word manifest filename does not match configured {key} path")
            if not hmac.compare_digest(_sha256(path), entry["sha256"]):
                raise ValueError(f"Wake-word SHA-256 mismatch for {key} asset")

    if runtime_version is None:
        try:
            runtime_version = importlib.metadata.version("openwakeword")
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError("openwakeword 0.6.0 must be installed to validate wake assets") from None
    if runtime_version != _EXPECTED_RUNTIME_VERSION:
        raise RuntimeError(
            f"Wake-word assets require openwakeword {_EXPECTED_RUNTIME_VERSION}; found {runtime_version}"
        )

    if session_factory is None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError("onnxruntime is required to validate wake-word ONNX assets") from None

        def session_factory(path: str, **kwargs):
            options = ort.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            return ort.InferenceSession(path, sess_options=options, **kwargs)

    sessions: dict[str, Any] = {}
    for key, path in paths.items():
        try:
            sessions[key] = session_factory(str(path), providers=["CPUExecutionProvider"])
        except Exception as error:
            raise ValueError(f"Invalid ONNX model for {key} asset ({type(error).__name__})") from None

    wake, mel, embedding = (sessions[key] for key in _ASSETS)
    wake_inputs, wake_outputs = wake.get_inputs(), wake.get_outputs()
    mel_inputs, mel_outputs = mel.get_inputs(), mel.get_outputs()
    embedding_inputs, embedding_outputs = embedding.get_inputs(), embedding.get_outputs()
    for values, label in (
        (wake_inputs, "wakeword inputs"), (wake_outputs, "wakeword outputs"),
        (mel_inputs, "melspectrogram inputs"), (mel_outputs, "melspectrogram outputs"),
        (embedding_inputs, "embedding inputs"), (embedding_outputs, "embedding outputs"),
    ):
        if len(values) != 1:
            raise ValueError(f"Wake-word ONNX contract requires exactly one {label}")

    _check_tensor(mel_inputs[0], name="input", rank=2, shape=(None, None), label="Melspectrogram input")
    _check_tensor(
        mel_outputs[0], name=None, rank=4, shape=(None, 1, None, 32),
        label="Melspectrogram output",
    )
    _check_tensor(embedding_inputs[0], name="input_1", rank=4, shape=(None, 76, 32, 1), label="Embedding input")
    _check_tensor(
        embedding_outputs[0], name=None, rank=4, shape=(None, 1, 1, 96),
        label="Embedding output",
    )
    if require_manifest:
        _check_tensor(wake_inputs[0], name=None, rank=3, shape=(None, None, 96), label="Wakeword input")
        _check_tensor(wake_outputs[0], name=None, rank=3, shape=(None, 1, 1), label="Wakeword output")
        wake_frames = wake_inputs[0].shape[1]
        if not isinstance(wake_frames, int) or wake_frames < 1:
            raise ValueError("Wakeword input must declare a positive fixed feature-frame count")
        wake_input_shape = (1, wake_frames, 96)
    else:
        wake_input = wake_inputs[0]
        if wake_input.type != "tensor(float)" or len(wake_input.shape) != 3:
            raise ValueError("Pretrained wakeword input must be a rank-3 float32 tensor")
        if not _axis_matches(wake_input.shape[-1], 96):
            raise ValueError("Pretrained wakeword input must have 96 embedding features")
        wake_output = wake_outputs[0]
        if (
            wake_output.type != "tensor(float)"
            or len(wake_output.shape) not in {2, 3}
            or not all(_axis_matches(axis, 1) for axis in wake_output.shape[1:])
        ):
            raise ValueError("Pretrained wakeword output must contain one float32 score")
        wake_input_shape = tuple(
            dimension if isinstance(dimension, int) and dimension > 0 else default
            for dimension, default in zip(wake_input.shape, (1, 16, 96))
        )

    try:
        if numpy_module is None:
            import numpy as np
        else:
            np = numpy_module

        mel_result = mel.run(None, {mel_inputs[0].name: np.zeros((1, 64000), dtype=np.float32)})[0]
        if (
            len(mel_result.shape) != 4
            or mel_result.shape[0] != 1
            or mel_result.shape[1] != 1
            or mel_result.shape[2] < 76
            or mel_result.shape[3] != 32
        ):
            raise ValueError("Melspectrogram output is incompatible with the embedding model")
        mel_feature_shape = tuple(axis for axis in mel_result.shape[1:] if axis != 1)
        if len(mel_feature_shape) != 2 or mel_feature_shape[1] != 32:
            raise ValueError("Melspectrogram output cannot be normalized to [frames, 32]")
        embedding_result = embedding.run(
            None, {embedding_inputs[0].name: np.zeros((1, 76, 32, 1), dtype=np.float32)}
        )[0]
        if tuple(embedding_result.shape) != (1, 1, 1, 96):
            raise ValueError("Embedding output must have shape [1, 1, 1, 96]")
        embedding_feature_shape = tuple(axis for axis in embedding_result.shape[1:] if axis != 1)
        if embedding_feature_shape != (96,):
            raise ValueError("Embedding output cannot be normalized to a 96-dimensional feature vector")
        wake_result = wake.run(
            None,
            {wake_inputs[0].name: np.zeros(wake_input_shape, dtype=np.float32)},
        )[0]
        if require_manifest and tuple(wake_result.shape) != (1, 1, 1):
            raise ValueError("Wakeword output must have shape [1, 1, 1]")
        if not require_manifest and (
            len(wake_result.shape) not in {2, 3}
            or wake_result.shape[0] != 1
            or any(axis != 1 for axis in wake_result.shape[1:])
        ):
            raise ValueError("Pretrained wakeword output must return one score")
        for key, result in (("melspectrogram", mel_result), ("embedding", embedding_result), ("wakeword", wake_result)):
            if not np.isfinite(result).all():
                raise ValueError(f"Wake-word {key} model returned non-finite values")
        if (wake_result < 0).any() or (wake_result > 1).any():
            raise ValueError("Wakeword model output must be a confidence score in [0, 1]")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Wake-word ONNX compatibility check failed ({type(error).__name__})") from None