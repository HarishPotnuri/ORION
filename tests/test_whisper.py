import io
import json
import os
import subprocess
import sys
import threading
import time
import unittest
import wave
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from orion.audio.whisper import WhisperCppTranscriber
from orion.config import Settings


class WhisperTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = Path(self.directory.name) / "local model.bin"
        self.model.touch()
        self.settings = replace(
            Settings(), stt_model_path=self.model, whisper_cli_path=sys.executable,
            max_utterance_seconds=1, stt_timeout=5,
        )
        self.adapter = WhisperCppTranscriber(self.settings)
        self.addCleanup(self.adapter.close)
        self.cancel = threading.Event()
        self.processes = []
        self.commands = []
        self.real_popen = subprocess.Popen
        guard = patch.object(
            subprocess.Popen, "communicate", side_effect=AssertionError("Unbounded communicate")
        )
        guard.start()
        self.addCleanup(guard.stop)

    def child(self, code, spawned=None, configure=None):
        def spawn(argv, **kwargs):
            self.assertFalse(any(process.poll() is None for process in self.processes))
            self.commands.append((argv, kwargs))
            source = code(len(self.processes)) if callable(code) else code
            process = self.real_popen([sys.executable, "-u", "-c", source], **kwargs)
            self.processes.append(process)
            if configure is not None:
                configure(process)
            if spawned is not None:
                spawned.set()
            return process

        return patch("orion.audio.whisper.subprocess.Popen", side_effect=spawn)

    def background(self, *, pcm=b"\x00\x00", cancel=None):
        outcomes = []

        def transcribe():
            try:
                outcomes.append(self.adapter.transcribe(
                    pcm, cancel_event=cancel if cancel is not None else self.cancel
                ))
            except Exception as error:
                outcomes.append(error)

        worker = threading.Thread(target=transcribe, daemon=True)
        worker.start()
        return worker, outcomes

    def finish(self, worker, outcomes):
        worker.join(5)
        self.assertFalse(worker.is_alive(), "Transcription did not finish cleanup")
        self.assertEqual(outcomes, [""])
        self.assert_clean()

    def assert_clean(self):
        for process in self.processes:
            self.assertIsNotNone(process.poll())
            self.assertTrue(process.stdin.closed)
            self.assertTrue(process.stdout.closed)
            self.assertTrue(process.stderr.closed)
        self.assertFalse(any(
            worker.name == "orion-whisper-pipe" for worker in threading.enumerate()
        ))

    def test_wav_and_cpu_arguments(self):
        code = (
            "import io,json,sys,wave; "
            "audio=wave.open(io.BytesIO(sys.stdin.buffer.read()),'rb'); "
            "print(json.dumps([audio.getnchannels(),audio.getframerate(),"
            "audio.getsampwidth(),audio.getnframes(),audio.readframes(2).hex()]))"
        )
        with self.child(code):
            result = self.adapter.transcribe(b"\x01\x00\xff\x7f", cancel_event=self.cancel)
        self.assertEqual(json.loads(result), [1, 16000, 2, 2, "0100ff7f"])
        argv, kwargs = self.commands[0]
        self.assertEqual(argv, [
            str(Path(sys.executable).resolve()), "-m", str(self.model.resolve()),
            "-f", "-", "-t", "2", "-l", "en", "-ng", "-nt", "-np", "-otxt",
        ])
        self.assertFalse(kwargs["shell"])
        self.assertEqual(set(Path(self.directory.name).iterdir()), {self.model})
        self.assert_clean()

    def test_lazy_constructor_and_asset_validation_do_not_spawn_or_load(self):
        with patch("orion.audio.whisper.subprocess.Popen") as spawn:
            adapter = WhisperCppTranscriber(self.settings)
            adapter.validate_assets()
            adapter.close()
        spawn.assert_not_called()
        self.assertEqual(self.model.stat().st_size, 0)

    def test_missing_model_and_directory_are_rejected_without_spawn(self):
        for model in (self.model.parent / "missing.bin", self.model.parent):
            with self.subTest(model=model):
                adapter = WhisperCppTranscriber(replace(self.settings, stt_model_path=model))
                with patch("orion.audio.whisper.subprocess.Popen") as spawn:
                    with self.assertRaises(FileNotFoundError):
                        adapter.validate_assets()
                    with self.assertRaises(FileNotFoundError):
                        adapter.transcribe(b"\x00\x00", cancel_event=self.cancel)
                spawn.assert_not_called()
                adapter.close()

    def test_missing_executable_is_rejected_without_spawn(self):
        self.adapter.settings = replace(
            self.settings, whisper_cli_path=str(self.model.parent / "missing-cli")
        )
        with patch("orion.audio.whisper.subprocess.Popen") as spawn:
            with self.assertRaisesRegex(FileNotFoundError, "executable"):
                self.adapter.validate_assets()
        spawn.assert_not_called()

    def test_regular_nonexecutable_file_is_not_a_cli(self):
        self.adapter.settings = replace(self.settings, whisper_cli_path=str(self.model))
        with self.assertRaisesRegex(FileNotFoundError, "executable"):
            self.adapter.validate_assets()

    def test_executable_is_resolved_from_path(self):
        self.adapter.settings = replace(self.settings, whisper_cli_path="whisper-cli")
        with patch("orion.audio.whisper.shutil.which", return_value=sys.executable) as which:
            self.adapter.validate_assets()
        which.assert_called_once_with("whisper-cli")

    def test_empty_and_precancelled_pcm_bypass_assets_and_process(self):
        self.adapter.settings = replace(self.settings, stt_model_path=Path("missing.bin"))
        with patch("orion.audio.whisper.subprocess.Popen") as spawn:
            self.assertEqual(self.adapter.transcribe(b"", cancel_event=self.cancel), "")
            self.cancel.set()
            self.assertEqual(self.adapter.transcribe(b"odd", cancel_event=self.cancel), "")
        spawn.assert_not_called()

    def test_pcm_bounds_and_sample_rate_fail_before_spawn(self):
        with patch("orion.audio.whisper.subprocess.Popen") as spawn:
            for pcm in (b"odd", b"\x00\x00" * 16001, bytearray(b"\x00\x00")):
                with self.subTest(pcm_type=type(pcm), length=len(pcm)):
                    with self.assertRaises(ValueError):
                        self.adapter.transcribe(pcm, cancel_event=self.cancel)
            self.adapter.settings = replace(self.settings, audio_sample_rate=8000)
            with self.assertRaisesRegex(ValueError, "16000"):
                self.adapter.transcribe(b"\x00\x00", cancel_event=self.cancel)
        spawn.assert_not_called()

    def test_exact_maximum_audio_and_configured_language_and_threads(self):
        self.adapter.settings = replace(self.settings, stt_language="auto", stt_threads=3)
        with self.child("import sys; sys.stdin.buffer.read(); print('  words  ')"):
            self.assertEqual(self.adapter.transcribe(
                b"\x00\x00" * 16000, cancel_event=self.cancel
            ), "words")
        argv = self.commands[0][0]
        self.assertEqual(argv[argv.index("-t") + 1], "3")
        self.assertEqual(argv[argv.index("-l") + 1], "auto")
        self.assert_clean()

    def test_exact_output_limits_are_accepted_and_stderr_is_not_transcript(self):
        limit = self.adapter._OUTPUT_LIMIT
        code = (
            "import sys; sys.stdin.buffer.read(); "
            f"sys.stderr.buffer.write(b'e'*{limit}); "
            f"sys.stdout.buffer.write(b'a'*{limit})"
        )
        with self.child(code):
            self.assertEqual(self.adapter.transcribe(
                b"\x00\x00", cancel_event=self.cancel
            ), "a" * limit)
        self.assert_clean()

    def test_output_overflow_is_deterministic_on_both_streams_and_exit_failure(self):
        for stream in ("stdout", "stderr"):
            for exitcode in (0, 7):
                with self.subTest(stream=stream, exitcode=exitcode):
                    code = (
                        "import sys; sys.stdin.buffer.read(); "
                        f"sys.{stream}.buffer.write(b'x'*{self.adapter._OUTPUT_LIMIT + 1}); "
                        f"sys.{stream}.buffer.flush(); sys.exit({exitcode})"
                    )
                    with self.child(code), self.assertRaisesRegex(
                        RuntimeError, "^Whisper output exceeds the size limit$"
                    ):
                        self.adapter.transcribe(b"\x00\x00", cancel_event=self.cancel)
                    self.assert_clean()

    def test_continuous_output_overflow_kills_and_reaps(self):
        code = "import os;\nwhile True: os.write(1,b'x'*4096); os.write(2,b'e'*4096)"
        with self.child(code), self.assertRaisesRegex(RuntimeError, "output exceeds"):
            self.adapter.transcribe(b"\x00\x00", cancel_event=self.cancel)
        self.assert_clean()

    def test_overflow_wins_over_broken_stdin_and_nonzero_exit(self):
        self.adapter.settings = replace(self.settings, max_utterance_seconds=120)
        code = (
            "import sys; "
            f"sys.stdout.buffer.write(b'x'*{self.adapter._OUTPUT_LIMIT + 1}); "
            "sys.stdout.buffer.flush(); sys.exit(7)"
        )
        for attempt in range(10):
            with self.subTest(attempt=attempt):
                with self.child(code), self.assertRaisesRegex(RuntimeError, "output exceeds"):
                    self.adapter.transcribe(
                        b"\x00\x00" * (120 * 16000), cancel_event=self.cancel
                    )
                self.assert_clean()

    def test_nonzero_exit_excludes_private_output_from_errors_and_logs(self):
        code = (
            "import sys; sys.stdin.buffer.read(); "
            "print('private transcript'); print('private diagnostics',file=sys.stderr); "
            "sys.exit(7)"
        )
        with self.child(code), self.assertNoLogs(), self.assertRaisesRegex(
            RuntimeError, "^whisper-cli exited with status 7$"
        ):
            self.adapter.transcribe(b"\x00\x00", cancel_event=self.cancel)
        self.assert_clean()

    def test_invalid_utf8_is_an_error_and_empty_stdout_is_valid(self):
        with self.child("import sys; sys.stdin.buffer.read(); sys.stdout.buffer.write(b'\\xff')"):
            with self.assertRaisesRegex(RuntimeError, "invalid UTF-8"):
                self.adapter.transcribe(b"\x00\x00", cancel_event=self.cancel)
        with self.child("import sys; sys.stdin.buffer.read()"):
            self.assertEqual(self.adapter.transcribe(b"\x00\x00", cancel_event=self.cancel), "")
        self.assert_clean()

    def test_launch_failure_is_sanitized_and_adapter_can_be_reused(self):
        with patch("orion.audio.whisper.subprocess.Popen", side_effect=OSError("private error")):
            with self.assertRaisesRegex(RuntimeError, "^Unable to start whisper-cli$"):
                self.adapter.transcribe(b"\x00\x00", cancel_event=self.cancel)
        with self.child("import sys; sys.stdin.buffer.read(); print('reused')"):
            self.assertEqual(self.adapter.transcribe(
                b"\x00\x00", cancel_event=self.cancel
            ), "reused")
        self.assert_clean()

    def test_reader_io_failure_kills_and_cleans_up(self):
        def configure(process):
            failure = patch.object(process.stdout, "read", side_effect=OSError("private I/O error"))
            failure.start()
            self.addCleanup(failure.stop)

        with self.child("import threading; threading.Event().wait(60)", configure=configure):
            with self.assertRaisesRegex(RuntimeError, "^Whisper pipe I/O failed$"):
                self.adapter.transcribe(b"\x00\x00", cancel_event=self.cancel)
        self.assert_clean()

    def test_writer_io_failure_kills_and_cleans_up(self):
        def configure(process):
            failure = patch.object(process.stdin, "write", side_effect=OSError("private I/O error"))
            failure.start()
            self.addCleanup(failure.stop)

        with self.child("import threading; threading.Event().wait(60)", configure=configure):
            with self.assertRaisesRegex(RuntimeError, "^Whisper pipe I/O failed$"):
                self.adapter.transcribe(b"\x00\x00", cancel_event=self.cancel)
        self.assert_clean()

    def test_partial_stdin_writes_preserve_complete_wav(self):
        def configure(process):
            write = process.stdin.write
            partial = patch.object(process.stdin, "write", side_effect=lambda data: write(data[:1]))
            partial.start()
            self.addCleanup(partial.stop)

        code = (
            "import io,sys,wave; "
            "audio=wave.open(io.BytesIO(sys.stdin.buffer.read()),'rb'); "
            "print(audio.readframes(2).hex())"
        )
        with self.child(code, configure=configure):
            self.assertEqual(self.adapter.transcribe(
                b"\x01\x00\xff\x7f", cancel_event=self.cancel
            ), "0100ff7f")
        self.assert_clean()

    def test_worker_start_failure_reaps_child_and_joins_started_workers(self):
        real_start = threading.Thread.start
        starts = []

        def start(worker):
            starts.append(worker)
            if len(starts) == 3:
                raise RuntimeError("Cannot start worker")
            real_start(worker)

        with self.child("import threading; threading.Event().wait(60)"):
            with patch.object(threading.Thread, "start", autospec=True, side_effect=start):
                with self.assertRaisesRegex(RuntimeError, "Cannot start worker"):
                    self.adapter.transcribe(b"\x00\x00", cancel_event=self.cancel)
        self.assert_clean()

    def test_deadline_does_not_reset_when_output_arrives(self):
        code = "import os,threading;\nwhile True: os.write(1,b'x'); threading.Event().wait(0.01)"
        with self.child(code), self.assertRaises(TimeoutError):
            self.adapter.transcribe(b"\x00\x00", cancel_event=self.cancel, timeout=0.1)
        self.assert_clean()

    @unittest.skipUnless(os.name == "posix", "Process-group cleanup is POSIX-specific")
    def test_timeout_kills_descendant_holding_inherited_pipes(self):
        code = (
            "import subprocess,sys; sys.stdin.buffer.read(); "
            "subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)'], "
            "stdin=subprocess.DEVNULL); print('done',flush=True)"
        )
        started = time.monotonic()
        with self.child(code), self.assertRaises(TimeoutError):
            self.adapter.transcribe(b"\x00\x00", cancel_event=self.cancel, timeout=0.15)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assert_clean()

    def test_default_and_override_timeouts_kill_blocked_stdin_and_join_workers(self):
        self.adapter.settings = replace(self.settings, max_utterance_seconds=120, stt_timeout=0.1)
        for timeout in (None, 0.05):
            with self.subTest(timeout=timeout):
                with self.child("import threading; threading.Event().wait(60)"):
                    with self.assertRaises(TimeoutError):
                        self.adapter.transcribe(
                            b"\x00\x00" * (120 * 16000),
                            cancel_event=self.cancel, timeout=timeout,
                        )
                self.assert_clean()

    def test_expired_and_nonfinite_timeouts_do_not_spawn(self):
        with patch("orion.audio.whisper.subprocess.Popen") as spawn:
            for timeout in (0, -1):
                with self.assertRaises(TimeoutError):
                    self.adapter.transcribe(b"\x00\x00", cancel_event=self.cancel, timeout=timeout)
            for timeout in (float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    self.adapter.transcribe(b"\x00\x00", cancel_event=self.cancel, timeout=timeout)
        spawn.assert_not_called()

    def test_cancel_active_process_discards_output_and_reaps(self):
        spawned = threading.Event()
        with self.child("import threading; print('partial',flush=True); threading.Event().wait(60)", spawned):
            worker, outcomes = self.background()
            self.assertTrue(spawned.wait(5))
            self.cancel.set()
            self.finish(worker, outcomes)

    def test_close_kills_blocked_writer_and_is_idempotent(self):
        self.adapter.settings = replace(self.settings, max_utterance_seconds=120)
        spawned = threading.Event()
        with self.child("import threading; threading.Event().wait(60)", spawned):
            worker, outcomes = self.background(pcm=b"\x00\x00" * (120 * 16000))
            self.assertTrue(spawned.wait(5))
            self.adapter.close()
            self.finish(worker, outcomes)
            self.adapter.close()
            self.assertEqual(self.adapter.transcribe(b"\x00\x00", cancel_event=self.cancel), "")
        self.assertEqual(len(self.processes), 1)

    def test_deadline_includes_serialization_wait(self):
        spawned = threading.Event()
        with self.child("import threading; threading.Event().wait(60)", spawned):
            worker, outcomes = self.background()
            self.assertTrue(spawned.wait(5))
            try:
                with self.assertRaises(TimeoutError):
                    self.adapter.transcribe(
                        b"\x00\x00", cancel_event=threading.Event(), timeout=0.05
                    )
                self.assertEqual(len(self.processes), 1)
            finally:
                self.cancel.set()
                self.finish(worker, outcomes)

    def test_concurrent_calls_serialize_and_reuse_after_cancel(self):
        spawned = threading.Event()
        def code(index):
            if index == 0:
                return "import threading; threading.Event().wait(60)"
            return "import sys; sys.stdin.buffer.read(); print('next turn')"

        with self.child(code, spawned):
            first, first_outcomes = self.background()
            self.assertTrue(spawned.wait(5))
            second, second_outcomes = self.background(cancel=threading.Event())
            self.cancel.set()
            first.join(5)
            second.join(5)
            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(first_outcomes, [""])
            self.assertEqual(second_outcomes, ["next turn"])
        self.assertEqual(len(self.processes), 2)
        self.assert_clean()

    def test_cancel_and_close_of_queued_work_never_spawn_another_child(self):
        for close in (False, True):
            with self.subTest(close=close):
                spawned = threading.Event()
                queued_cancel = threading.Event()
                with self.child("import threading; threading.Event().wait(60)", spawned):
                    first, first_outcomes = self.background()
                    self.assertTrue(spawned.wait(5))
                    second, second_outcomes = self.background(cancel=queued_cancel)
                    if close:
                        self.adapter.close()
                    else:
                        queued_cancel.set()
                        second.join(5)
                        self.cancel.set()
                    self.finish(first, first_outcomes)
                    self.finish(second, second_outcomes)
                self.assertEqual(len(self.processes), 1)
                self.processes.clear()
                self.cancel.clear()