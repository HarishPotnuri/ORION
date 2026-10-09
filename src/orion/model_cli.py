"""Offline model inventory, selection, validation, and benchmark commands."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .audio.capture import AudioGate
from .audio.tts import PiperSpeechOutput
from .config import Settings
from .core.contracts import ChatMessage
from .llm.provider import LlamaCppProvider
from .memory.embedding import create_embedding_provider
from .models import (
    MODEL_SPECS,
    ModelSelectionStore,
    WAKEWORD_EMBEDDING_FILENAME,
    WAKEWORD_MANIFEST_FILENAME,
    WAKEWORD_MELSPECTROGRAM_FILENAME,
    get_model,
    resolve_model_path,
    validate_model_file,
)

_BENCHMARK_PROMPTS = (
    ("conversation", "What is one simple way to start a calm morning?"),
    ("reasoning", "A reminder is set for 10:00. It is now 09:40. How many minutes remain? Answer briefly."),
    ("instruction", "Give exactly three short bullet points about keeping a desk organized."),
    ("structured", 'Return only JSON with keys "action" and "reason" for a request to set a reminder.'),
)


def model_main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="orion-model")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list supported models and local asset status")
    commands.add_parser("current", help="show selected models")
    use_parser = commands.add_parser("use", help="select a model for its role")
    use_parser.add_argument("model_id")
    validate_parser = commands.add_parser("validate", help="validate local assets and runtime")
    validate_parser.add_argument("model_id")
    benchmark_parser = commands.add_parser("benchmark", help="run fixed local LLM benchmark prompts")
    benchmark_group = benchmark_parser.add_mutually_exclusive_group(required=True)
    benchmark_group.add_argument("model_id", nargs="?")
    benchmark_group.add_argument("--all", action="store_true")
    benchmark_parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        settings = Settings.from_environment()
        if args.command == "list":
            return _list_models(settings)
        if args.command == "current":
            _print_json(_current_selection(settings))
            return 0
        if args.command == "use":
            return _select_model(settings, args.model_id)
        if args.command == "validate":
            result = validate_model(settings, args.model_id)
            _print_json(result)
            return 0 if result["success"] else 1
        return _benchmark(settings, args.model_id, args.all, args.output)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Model command failed: {error}", file=sys.stderr)
        return 2


def _current_selection(settings: Settings) -> dict[str, str]:
    return {
        "llm": settings.active_llm_model,
        "stt": settings.active_stt_model,
        "tts": settings.active_tts_model,
        "embedding": settings.active_embedding_model,
        "wakeword": settings.active_wakeword_model,
    }


def _list_models(settings: Settings) -> int:
    active = _current_selection(settings)
    rows = []
    for spec in MODEL_SPECS:
        ok, detail, size = validate_model_file(settings.model_root, spec.model_id)
        if spec.model_id == "vosk-en-small":
            path = settings.vosk_model_path.expanduser()
            ok = path.is_dir()
            detail = f"directory present: {path}" if ok else f"VOSK model directory not found: {path}"
        elif spec.model_id == "kittentts-nano":
            ok, detail = True, "uses configured offline KittenTTS cache or local directory"
        elif spec.kind == "wakeword":
            files = (
                resolve_model_path(settings.model_root, spec.model_id),
                settings.model_root / "WAKEWORD" / WAKEWORD_MELSPECTROGRAM_FILENAME,
                settings.model_root / "WAKEWORD" / WAKEWORD_EMBEDDING_FILENAME,
                settings.model_root / "WAKEWORD" / WAKEWORD_MANIFEST_FILENAME,
            )
            missing = [str(path) for path in files if not path.is_file()]
            ok = not missing
            detail = "all local ONNX files present" if ok else "wakeword assets incomplete: " + "; ".join(missing)
        rows.append({
            "id": spec.model_id,
            "name": spec.display_name,
            "type": spec.kind,
            "runtime": spec.runtime,
            "path": str(resolve_model_path(settings.model_root, spec.model_id))
            if spec.filename else None,
            "available": ok,
            "size_bytes": size,
            "selected": active.get(spec.kind) == spec.model_id,
            "detail": detail,
        })
    _print_json({"model_root": str(settings.model_root), "models": rows})
    return 0


def _select_model(settings: Settings, model_id: str) -> int:
    spec = get_model(model_id)
    selections = ModelSelectionStore(settings.model_selection_path).read()
    selections[spec.kind] = spec.model_id
    ModelSelectionStore(settings.model_selection_path).write(selections)
    print(f"Selected {spec.model_id} for {spec.kind}. Restart ORION service to apply.")
    return 0


def validate_model(settings: Settings, model_id: str) -> dict[str, object]:
    spec = get_model(model_id)
    if spec.kind == "llm":
        return _validate_llm(settings, model_id)
    if spec.model_id == "vosk-en-small":
        path = settings.vosk_model_path.expanduser()
        success = path.is_dir()
        return {"id": model_id, "success": success,
                "detail": str(path) if success else f"VOSK model directory not found: {path}"}
    if spec.kind == "stt":
        success, detail, _size = validate_model_file(settings.model_root, model_id)
        if success:
            import shutil
            if shutil.which(settings.whisper_cli_path) is None:
                success, detail = False, f"whisper-cli executable not found: {settings.whisper_cli_path}"
        return {"id": model_id, "success": success, "detail": detail}
    if spec.kind == "tts":
        success, detail, _size = validate_model_file(settings.model_root, model_id)
        if success:
            candidate = replace(
                settings,
                active_tts_model=model_id,
                tts_backend="piper",
                tts_model_path=resolve_model_path(settings.model_root, model_id),
                tts_voice=Path(spec.filename).stem,
            )
            PiperSpeechOutput(candidate, AudioGate())._load_voice()
            detail = "Piper voice and adjacent config loaded locally"
        return {"id": model_id, "success": success, "detail": detail}
    if spec.kind == "embedding":
        success, detail, _size = validate_model_file(settings.model_root, model_id)
        if success:
            provider = create_embedding_provider(
                replace(settings, active_embedding_model=model_id)
            )
            provider._load_session()
            detail = "ONNX runtime initialized; text input contract verified"
        return {"id": model_id, "success": success, "detail": detail}
    if spec.kind == "wakeword":
        success, detail, _size = validate_model_file(settings.model_root, model_id)
        if success:
            from .audio.wakeword import validate_settings
            candidate = replace(
                settings,
                voice_profile="wakeword",
                wake_word_enabled=True,
                active_wakeword_model=model_id,
                wake_word_model_path=resolve_model_path(settings.model_root, model_id),
                wake_word_melspectrogram_path=settings.model_root / "WAKEWORD" / WAKEWORD_MELSPECTROGRAM_FILENAME,
                wake_word_embedding_path=settings.model_root / "WAKEWORD" / WAKEWORD_EMBEDDING_FILENAME,
                wake_word_manifest_path=(
                    settings.wake_word_manifest_path
                    or settings.model_root / "WAKEWORD" / WAKEWORD_MANIFEST_FILENAME
                ),
            )
            validate_settings(candidate)
            detail = "wakeword assets validated locally"
        return {"id": model_id, "success": success, "detail": detail}
    return {"id": model_id, "success": False, "detail": "unsupported model type"}


def _validate_llm(settings: Settings, model_id: str) -> dict[str, object]:
    spec = get_model(model_id)
    path = resolve_model_path(settings.model_root, model_id).resolve()
    if not path.is_file():
        return {"id": model_id, "success": False, "detail": f"Model file not found: {path}"}
    if path.suffix.lower() != ".gguf":
        return {"id": model_id, "success": False, "detail": f"Expected a .gguf model file: {path}"}
    candidate = replace(
        settings,
        active_llm_model=model_id,
        model_path=path,
        model_context_size=spec.context_size or settings.model_context_size,
    )
    provider = LlamaCppProvider(candidate)
    started = time.perf_counter()
    try:
        provider.load()
        load_seconds = time.perf_counter() - started
        response = provider.generate(
            [
                ChatMessage(role="system", content="Follow the user's exact output instruction."),
                ChatMessage(role="user", content="Reply with exactly: ORION_OK"),
            ],
            max_tokens=16,
            timeout=min(30.0, settings.max_total_task_time),
        )
        success = "ORION_OK" in response.strip()
        detail = "local model loaded and deterministic smoke prompt passed" if success else "local model loaded but smoke output did not contain ORION_OK"
        return {"id": model_id, "success": success, "load_seconds": load_seconds, "detail": detail}
    finally:
        provider.unload()


def _benchmark(settings: Settings, model_id: str | None, all_models: bool, output: Path | None) -> int:
    ids = [spec.model_id for spec in MODEL_SPECS if spec.kind == "llm"] if all_models else [model_id]
    results = []
    for candidate_id in ids:
        try:
            results.append(_benchmark_one(settings, candidate_id))
        except (OSError, RuntimeError, ValueError) as error:
            results.append({"id": candidate_id, "success": False, "error": f"{type(error).__name__}: {error}"})
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompts": [{"name": name, "text": text} for name, text in _BENCHMARK_PROMPTS],
        "results": results,
    }
    destination = output or settings.model_selection_path.parent / "model-benchmarks.json"
    destination = destination.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_json({"result_file": str(destination), "results": results})
    return 0 if all(item.get("success") for item in results) else 1


def _benchmark_one(settings: Settings, model_id: str) -> dict[str, object]:
    spec = get_model(model_id)
    if spec.kind != "llm":
        raise ValueError(f"Benchmark requires an LLM model ID: {model_id}")
    path = resolve_model_path(settings.model_root, model_id).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    candidate = replace(
        settings,
        active_llm_model=model_id,
        model_path=path,
        model_context_size=spec.context_size or settings.model_context_size,
    )
    provider = LlamaCppProvider(candidate)
    load_started = time.perf_counter()
    memory_before = _process_snapshot(provider)
    provider.load()
    if not getattr(provider._server, "owned_process", False):
        provider.unload()
        raise RuntimeError(
            "Benchmark requires an ORION-owned llama-server; stop the active service or external server first"
        )
    load_seconds = time.perf_counter() - load_started
    memory_after_load = _process_snapshot(provider)
    cpu_started_at = time.monotonic()
    cpu_ticks_before = memory_after_load.get("cpu_ticks")
    prompts = []
    try:
        for name, prompt in _BENCHMARK_PROMPTS:
            started = time.perf_counter()
            first_token = None
            usage = None
            generated = []
            finish = None
            reported_tps = None
            for event in provider.generate_stream(
                [
                    ChatMessage(role="system", content="Answer the request directly. Follow its output format."),
                    ChatMessage(role="user", content=prompt),
                ],
                max_tokens=96,
                timeout=settings.max_total_task_time,
                cancel_event=__import__("threading").Event(),
            ):
                if event.kind == "text":
                    if first_token is None:
                        first_token = time.perf_counter()
                    generated.append(event.text)
                elif event.kind == "usage":
                    usage = event.usage
                elif event.kind == "completed":
                    finish = event.finish_reason
                    reported_tps = event.tokens_per_second
                elif event.kind == "failed":
                    raise RuntimeError(event.error or "stream failed")
            finished = time.perf_counter()
            elapsed = finished - started
            output_tokens = usage.completion_tokens if usage else None
            speech_duration = finished - first_token if first_token else elapsed
            tps = (
                reported_tps
                if reported_tps is not None
                else output_tokens / speech_duration
                if output_tokens is not None and speech_duration > 0
                else None
            )
            prompts.append({
                "name": name,
                "time_to_first_token_seconds": first_token - started if first_token else None,
                "generation_seconds": elapsed,
                "generated_tokens": output_tokens,
                "tokens_per_second": tps,
                "finish_reason": finish,
                "output": "".join(generated),
            })
        observed = _process_snapshot(provider)
        cpu_elapsed = time.monotonic() - cpu_started_at
        cpu_ticks_after = observed.get("cpu_ticks")
        cpu_percent = None
        if cpu_ticks_before is not None and cpu_ticks_after is not None and cpu_elapsed > 0:
            tick_rate = os.sysconf("SC_CLK_TCK")
            cpu_percent = (cpu_ticks_after - cpu_ticks_before) / tick_rate / cpu_elapsed * 100
        return {
            "id": model_id,
            "model_name": spec.display_name,
            "path": str(path),
            "file_size_bytes": path.stat().st_size,
            "load_seconds": load_seconds,
            "observed_rss_bytes": max(
                (value for value in (
                    memory_before.get("rss_bytes"),
                    memory_after_load.get("rss_bytes"),
                    observed.get("rss_bytes"),
                ) if value is not None),
                default=None,
            ),
            "cpu_percent_observed": cpu_percent,
            "temperature_c_before": memory_before.get("temperature_c"),
            "temperature_c_after": observed.get("temperature_c"),
            "success": all(prompt["generated_tokens"] is not None for prompt in prompts),
            "prompts": prompts,
        }
    finally:
        provider.unload()


def _process_snapshot(provider: LlamaCppProvider) -> dict[str, object]:
    result: dict[str, object] = {
        "rss_bytes": None,
        "cpu_ticks": None,
        "temperature_c": None,
    }
    server = provider._server
    process = getattr(server, "_process", None)
    pid = getattr(process, "pid", None)
    if os.name == "posix":
        try:
            temperature_path = Path("/sys/class/thermal/thermal_zone0/temp")
            if temperature_path.is_file():
                result["temperature_c"] = int(temperature_path.read_text().strip()) / 1000
        except (OSError, ValueError):
            pass
    if os.name == "posix" and pid is not None:
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    result["rss_bytes"] = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            pass
        try:
            process_stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").split()
            result["cpu_ticks"] = int(process_stat[13]) + int(process_stat[14])
        except (OSError, ValueError, IndexError):
            pass
    return result


def _print_json(value) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))
