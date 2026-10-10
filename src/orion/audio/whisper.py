"""Lazy, bounded whisper.cpp transcription using stdin WAV transport."""

from __future__ import annotations

import io
import math
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import BinaryIO

from ..config import Settings


class WhisperCppTranscriber:
    """Own one CLI at a time; cancellation/close discard the current transcript.

    The configured CLI must support WAV on stdin with ``-f -``. No model is
    loaded by construction or asset validation. Timeouts raise TimeoutError;
    process, pipe, and output-limit failures raise RuntimeError.
    """

    _OUTPUT_LIMIT = 16 * 1024
    _POLL_SECONDS = 0.01

    def __init__(self, settings: Settings):
        self.settings = settings
        self._run_lock = threading.Lock()
        self._closed = threading.Event()

    def _assets(self) -> tuple[str, Path]:
        model = Path(self.settings.stt_model_path).expanduser().resolve()
        if not model.is_file():
            raise FileNotFoundError(f"Whisper model file not found: {model}")
        executable = shutil.which(str(Path(self.settings.whisper_cli_path).expanduser()))
        if executable is None or (
            os.name == "nt" and Path(executable).suffix.lower() not in {".exe", ".com"}
        ):
            raise FileNotFoundError("whisper-cli executable not found")
        return str(Path(executable).resolve()), model

    def validate_assets(self) -> None:
        """Check local regular model file and executable without running either."""
        self._assets()

    def transcribe(
        self,
        pcm: bytes,
        *,
        cancel_event: threading.Event,
        timeout: float | None = None,
    ) -> str:
        if not pcm or cancel_event.is_set() or self._closed.is_set():
            return ""
        budget = self.settings.stt_timeout if timeout is None else timeout
        if not math.isfinite(budget):
            raise ValueError("Whisper timeout must be finite")
        deadline = time.monotonic() + budget
        if not isinstance(pcm, bytes) or len(pcm) % 2:
            raise ValueError("Whisper requires complete mono int16 PCM sample bytes")
        if self.settings.audio_sample_rate != 16000:
            raise ValueError("Whisper requires 16000 Hz mono audio")
        maximum = self.settings.max_utterance_seconds
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("Whisper utterance duration must be positive and finite")
        if len(pcm) > maximum * 16000 * 2:
            raise ValueError("Whisper audio exceeds the utterance size limit")

        while True:
            if cancel_event.is_set() or self._closed.is_set():
                return ""
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Whisper transcription timed out")
            if self._run_lock.acquire(timeout=min(self._POLL_SECONDS, remaining)):
                break
        try:
            if cancel_event.is_set() or self._closed.is_set():
                return ""
            executable, model = self._assets()
            with io.BytesIO() as buffer:
                with wave.open(buffer, "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(16000)
                    audio.writeframes(pcm)
                wav = buffer.getvalue()
            if cancel_event.is_set() or self._closed.is_set():
                return ""
            if time.monotonic() >= deadline:
                raise TimeoutError("Whisper transcription timed out")
            with tempfile.TemporaryDirectory(prefix="orion-whisper-") as output_directory:
                argv = [
                    executable, "-m", str(model), "-f", "-",
                    "-of", str(Path(output_directory) / "transcript"),
                    "-t", str(self.settings.stt_threads),
                    "-l", self.settings.stt_language, "-ng", "-nt", "-np", "-otxt",
                ]
                return self._run(argv, wav, cancel_event, deadline)
        finally:
            self._run_lock.release()

    def _run(
        self, argv: list[str], wav: bytes, cancel_event: threading.Event, deadline: float
    ) -> str:
        try:
            process = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, shell=False, bufsize=0,
                start_new_session=(os.name == "posix"),
            )
        except OSError:
            raise RuntimeError("Unable to start whisper-cli") from None

        stdout = bytearray()
        overflow = threading.Event()
        pipe_error = threading.Event()
        workers: list[threading.Thread] = []
        stop_reason = "finished"

        def drain(pipe: BinaryIO, keep: bool) -> None:
            count = 0
            try:
                while chunk := pipe.read(4096):
                    count = min(self._OUTPUT_LIMIT + 1, count + len(chunk))
                    if count > self._OUTPUT_LIMIT:
                        overflow.set()
                    elif keep:
                        stdout.extend(chunk)
            except (OSError, ValueError):
                pipe_error.set()

        def feed(pipe: BinaryIO) -> None:
            try:
                pending = memoryview(wav)
                while pending:
                    written = pipe.write(pending[:4096])
                    if written is None or written <= 0:
                        pipe_error.set()
                        break
                    pending = pending[written:]
            except (OSError, ValueError):
                pipe_error.set()
            finally:
                try:
                    pipe.close()
                except OSError:
                    pipe_error.set()

        try:
            assert process.stdin is not None
            assert process.stdout is not None
            assert process.stderr is not None
            jobs = (
                (drain, (process.stdout, True)),
                (drain, (process.stderr, False)),
                (feed, (process.stdin,)),
            )
            for target, arguments in jobs:
                worker = threading.Thread(
                    target=target, args=arguments, name="orion-whisper-pipe", daemon=True
                )
                worker.start()
                workers.append(worker)
            while True:
                if cancel_event.is_set() or self._closed.is_set():
                    return ""
                if overflow.is_set():
                    break
                if time.monotonic() >= deadline:
                    stop_reason = "timeout"
                    break
                if process.poll() is not None and not any(worker.is_alive() for worker in workers):
                    break
                if pipe_error.is_set():
                    stop_reason = "pipe"
                    break
                self._closed.wait(self._POLL_SECONDS)
        finally:
            try:
                workers_active = any(worker.is_alive() for worker in workers)
                if process.poll() is None or workers_active:
                    self._kill_owned_process_tree(process)
                process.wait()
            finally:
                for pipe in (process.stdin, process.stdout, process.stderr):
                    if pipe is not None:
                        try:
                            pipe.close()
                        except (OSError, ValueError):
                            pass
                for worker in workers:
                    worker.join(timeout=0.5)

        if cancel_event.is_set() or self._closed.is_set():
            return ""
        if overflow.is_set():
            raise RuntimeError("Whisper output exceeds the size limit")
        if stop_reason == "timeout" or time.monotonic() >= deadline:
            raise TimeoutError("Whisper transcription timed out")
        if stop_reason == "pipe":
            raise RuntimeError("Whisper pipe I/O failed")
        if process.returncode != 0:
            raise RuntimeError(f"whisper-cli exited with status {process.returncode}")
        if pipe_error.is_set():
            raise RuntimeError("Whisper pipe I/O failed")
        try:
            return stdout.decode("utf-8").strip()
        except UnicodeDecodeError:
            raise RuntimeError("Whisper returned invalid UTF-8") from None

    @staticmethod
    def _kill_owned_process_tree(process) -> None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
                return
            except ProcessLookupError:
                pass
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass

    def close(self) -> None:
        """Cancel active/queued work, and wait for the owned CLI and pipe workers."""
        self._closed.set()
        with self._run_lock:
            pass