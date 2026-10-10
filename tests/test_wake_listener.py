import array
import hashlib
import json
import queue
import threading
import time
from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

from orion.audio.capture import AudioGate
from orion.audio.control import MicrophoneControl
from orion.audio.factory import create_speech_output, create_transcriber
from orion.audio.listener import WakeListenerInput
from orion.audio.wakeword import OpenWakeWordDetector
from orion.config import Settings


WAIT = 2.0
WAKE_AMPLITUDE = 3000


class FakeTensor:
    def __init__(self, name, shape, tensor_type="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = tensor_type


class FakeArray:
    def __init__(self, shape, value=0.0):
        self.shape = shape
        self.value = value

    def __lt__(self, _other):
        return FakeMask()

    def __gt__(self, _other):
        return FakeMask()


class FakeMask:
    def any(self):
        return False

    def all(self):
        return True


class FakeNumpy:
    float32 = "float32"

    @staticmethod
    def zeros(shape, dtype=None):
        return FakeArray(shape)

    @staticmethod
    def isfinite(_array):
        return FakeMask()


def fake_wake_assets(tmp_path, *, tensor_overrides=None, invalid_asset=None):
    names = {
        "wakeword": "hey_orion.onnx",
        "melspectrogram": "melspectrogram.onnx",
        "embedding": "embedding.onnx",
    }
    paths = {key: tmp_path / name for key, name in names.items()}
    manifest_assets = {}
    for key, path in paths.items():
        path.write_bytes(f"fixture:{key}".encode())
        manifest_assets[key] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source": f"test-fixture:{key}",
            "revision": "test-revision",
            "license": "test-fixture-only",
        }
    manifest_assets["wakeword"].update(
        {"training_recipe": "synthetic unit-test fixture", "training_data": "synthetic unit-test fixture"}
    )
    manifest = {
        "schema_version": 1,
        "wake_phrase": "Hey Orion",
        "sample_rate_hz": 16000,
        "runtime": {"name": "openwakeword", "version": "0.6.0", "framework": "onnx"},
        "compatibility": {
            "sample_rate_hz": 16000,
            "melspectrogram_bins": 32,
            "embedding_window_frames": 76,
            "embedding_dimension": 96,
            "wake_output_classes": 1,
            "feature_model_sha256": {
                "melspectrogram": manifest_assets["melspectrogram"]["sha256"],
                "embedding": manifest_assets["embedding"]["sha256"],
            },
        },
        "assets": manifest_assets,
    }
    manifest_path = tmp_path / "phase13-artifacts.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    overrides = tensor_overrides or {}
    tensor_contracts = {
        "wakeword": (
            FakeTensor("features", overrides.get("wakeword_input_shape", ["batch", 16, 96])),
            FakeTensor("probability", overrides.get("wakeword_output_shape", ["batch", 1, 1])),
        ),
        "melspectrogram": (
            FakeTensor(overrides.get("melspectrogram_input_name", "input"), ["batch", "samples"]),
            FakeTensor("mel", overrides.get("melspectrogram_output_shape", ["time", 1, "frames", 32])),
        ),
        "embedding": (
            FakeTensor(overrides.get("embedding_input_name", "input_1"), ["batch", 76, 32, 1]),
            FakeTensor("embedding", overrides.get("embedding_output_shape", ["batch", 1, 1, 96])),
        ),
    }

    class Session:
        def __init__(self, key):
            self.key = key
            self.inputs = [tensor_contracts[key][0]]
            self.outputs = [tensor_contracts[key][1]]

        def get_inputs(self):
            return self.inputs

        def get_outputs(self):
            return self.outputs

        def run(self, _names, _feed):
            if self.key == "melspectrogram":
                return [FakeArray(overrides.get("melspectrogram_runtime_shape", (1, 1, 397, 32)))]
            if self.key == "embedding":
                return [FakeArray(overrides.get("embedding_runtime_shape", (1, 1, 1, 96)))]
            return [FakeArray(overrides.get("wakeword_runtime_shape", (1, 1, 1)))]

    def session_factory(path, **_kwargs):
        key = next(key for key, asset_path in paths.items() if str(asset_path) == path)
        if key == invalid_asset:
            raise RuntimeError("invalid model bytes")
        return Session(key)

    return paths, manifest_path, session_factory


def wake_settings(paths, manifest_path):
    return replace(
        Settings(), voice_profile="wakeword", stt_backend="whisper.cpp",
        wake_word_model_path=paths["wakeword"],
        wake_word_melspectrogram_path=paths["melspectrogram"],
        wake_word_embedding_path=paths["embedding"],
        wake_word_manifest_path=manifest_path,
    )


def pcm(amplitude):
    return array.array("h", [amplitude] * 320).tobytes()


class CaptureSession:
    def __init__(self, cancelled):
        self.cancelled = cancelled
        self.pending = queue.Queue()
        self.closed = threading.Event()

    def feed(self, *blocks):
        for block in blocks:
            self.pending.put(block)

    def drain(self):
        consumed = threading.Event()
        self.pending.put(consumed)
        assert consumed.wait(WAIT), "Capture worker did not consume queued blocks"


class QueueSource:
    def __init__(self):
        self.sessions = queue.Queue()
        self.opened = []

    def chunks(self, stop_event):
        session = CaptureSession(stop_event)
        self.opened.append(session)
        self.sessions.put(session)
        try:
            while not stop_event.is_set():
                try:
                    item = session.pending.get(timeout=0.02)
                except queue.Empty:
                    continue
                if isinstance(item, threading.Event):
                    item.set()
                elif isinstance(item, Exception):
                    raise item
                else:
                    yield item
        finally:
            session.closed.set()

    def next_session(self):
        return self.sessions.get(timeout=WAIT)


class MarkerDetector:
    def __init__(self):
        self.frames = []
        self.resets = queue.Queue()

    def reset(self):
        self.resets.put(True)

    def detected(self, frame):
        assert len(frame) == 2560
        self.frames.append(frame)
        return WAKE_AMPLITUDE in memoryview(frame).cast("h")


class Transcriber:
    def __init__(self):
        self.validations = 0
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.closed = False

    def validate_assets(self):
        self.validations += 1

    def transcribe(self, raw_pcm, *, cancel_event, timeout):
        self.calls.append((raw_pcm, cancel_event, timeout))
        self.entered.set()
        assert self.release.wait(WAIT), "Test did not release transcription"
        return "recognized words"

    def close(self):
        self.closed = True
        self.release.set()


class BackgroundCall:
    def __init__(self, function):
        self.outcome = queue.Queue()
        self.thread = threading.Thread(target=self._run, args=(function,))
        self.thread.start()

    def _run(self, function):
        try:
            self.outcome.put((True, function()))
        except BaseException as error:
            self.outcome.put((False, error))

    def result(self):
        succeeded, result = self.outcome.get(timeout=WAIT)
        self.thread.join(WAIT)
        assert not self.thread.is_alive()
        if not succeeded:
            raise result
        return result


class ListenerHarness:
    def __init__(self, directory, *, muted=False, allow_wakeword_during_speech=False):
        self.settings = replace(
            Settings(), audio_sample_rate=16000, audio_block_frames=320,
            vad_rms_threshold=500, vad_silence_ms=40, vad_preroll_ms=20,
            max_utterance_seconds=1, stt_timeout=1,
        )
        self.database = directory / "microphone.sqlite3"
        if muted:
            seed = MicrophoneControl(self.database)
            try:
                assert seed.set_muted(True)
            finally:
                seed.close()
        self.control = MicrophoneControl(self.database)
        self.source = QueueSource()
        self.detector = MarkerDetector()
        self.transcriber = Transcriber()
        self.gate = AudioGate(
            allow_wakeword_during_speech=allow_wakeword_during_speech
        )
        self.stop = threading.Event()
        self.listener = WakeListenerInput(
            self.settings, self.detector, self.transcriber, self.control,
            gate=self.gate, stop_event=self.stop, source=self.source,
            on_wake=lambda: self.wake_events.append("LISTENING"),
        )
        self.wake_events = []
        self.calls = []

    def request(self, timeout=1, *, followup=False):
        call = BackgroundCall(
            lambda: self.listener.next_utterance(
                timeout=timeout, followup=followup
            )
        )
        self.calls.append(call)
        assert self.listener._armed.wait(WAIT), "Consumer did not arm listener"
        return call

    def close(self):
        self.transcriber.release.set()
        self.listener.close()
        for call in self.calls:
            call.thread.join(WAIT)
            assert not call.thread.is_alive(), "Consumer outlived listener shutdown"


@pytest.fixture
def harness(tmp_path):
    instance = ListenerHarness(tmp_path)
    try:
        yield instance
    finally:
        instance.close()


def wake(session):
    session.feed(*([pcm(WAKE_AMPLITUDE)] * 4))


def voiced_turn(session):
    wake(session)
    blocks = [pcm(1000), pcm(1200), pcm(0), pcm(0)]
    session.feed(*blocks)
    return b"".join(blocks)


def test_start_validates_assets_once_without_loading_stt(harness):
    harness.listener.start()
    harness.listener.start()
    session = harness.source.next_session()
    session.feed(*([pcm(0)] * 4))
    session.drain()
    assert harness.transcriber.validations == 1
    assert harness.transcriber.calls == []
    assert len(harness.detector.frames) == 1
    assert not harness.listener._armed.is_set()


def test_no_wake_never_transcribes_even_voiced_pcm(harness):
    call = harness.request(timeout=0.2)
    session = harness.source.next_session()
    session.feed(*([pcm(1200)] * 8))
    session.drain()
    assert len(harness.detector.frames) == 2
    assert call.result() == ""
    assert harness.transcriber.calls == []
    assert not harness.listener.claim_utterance()
    assert not harness.stop.is_set()


def test_wake_followed_by_silence_does_not_transcribe(harness):
    call = harness.request()
    session = harness.source.next_session()
    wake(session)
    session.feed(pcm(0), pcm(0))
    session.drain()
    assert call.result() == ""
    assert harness.transcriber.calls == []
    assert not harness.listener.claim_utterance()


def test_wake_voiced_turn_transcribes_raw_pcm_once_and_claims_once(harness):
    call = harness.request()
    session = harness.source.next_session()
    expected = voiced_turn(session)
    assert call.result() == "recognized words"
    assert len(harness.transcriber.calls) == 1
    raw_pcm, cancelled, timeout = harness.transcriber.calls[0]
    assert raw_pcm == expected
    assert cancelled.events[1] is harness.control.cancel_event
    assert not cancelled.events[0].is_set()
    assert not cancelled.events[1].is_set()
    assert 0 < timeout <= harness.settings.stt_timeout
    assert harness.listener.claim_utterance()
    assert harness.wake_events == ["LISTENING"]
    assert not harness.listener.claim_utterance()


def test_followup_speech_uses_vad_without_wakeword(harness):
    call = harness.request(followup=True)
    session = harness.source.next_session()
    blocks = [pcm(1000), pcm(1200), pcm(0), pcm(0)]
    session.feed(*blocks)

    assert call.result() == "recognized words"
    assert harness.transcriber.calls[0][0] == b"".join(blocks)
    assert harness.detector.frames == []
    assert harness.listener.claim_utterance()


def test_reminder_followup_window_switches_active_listener(harness):
    call = harness.request(timeout=1)
    session = harness.source.next_session()
    harness.listener.open_followup_window(timeout=1)
    blocks = [pcm(1000), pcm(1200), pcm(0), pcm(0)]
    session.feed(*blocks)

    assert call.result() == "recognized words"
    assert harness.transcriber.calls[0][0] == b"".join(blocks)
    assert harness.detector.frames == []
    assert harness.listener.claim_utterance()


def test_followup_start_deadline_does_not_bound_utterance_capture(harness):
    harness.transcriber.release.clear()
    started = time.monotonic()
    call = harness.request(timeout=0.1, followup=True)
    session = harness.source.next_session()
    session.feed(pcm(1000), pcm(1200), pcm(0), pcm(0))

    assert harness.transcriber.entered.wait(WAIT)
    assert harness.listener._turn_deadline > started + 0.5
    harness.transcriber.release.set()
    assert call.result() == "recognized words"
    assert harness.listener.claim_utterance()


def test_worker_detects_while_transcription_and_model_consumer_are_busy(harness):
    harness.transcriber.release.clear()
    call = harness.request()
    session = harness.source.next_session()
    voiced_turn(session)
    assert harness.transcriber.entered.wait(WAIT)
    initial_frames = len(harness.detector.frames)
    voiced_turn(session)
    session.drain()
    assert len(harness.detector.frames) == initial_frames + 2
    assert harness.listener._results.empty()
    assert harness.listener._turn_phase == "transcribing"
    assert len(harness.transcriber.calls) == 1
    harness.transcriber.release.set()
    assert call.result() == "recognized words"
    assert harness.listener.claim_utterance()
    initial_frames = len(harness.detector.frames)
    voiced_turn(session)
    session.drain()
    assert len(harness.detector.frames) == initial_frames + 2
    assert harness.listener._results.empty()
    assert len(harness.transcriber.calls) == 1
    next_call = harness.request()
    session.feed(*([pcm(1000)] * 4))
    session.drain()
    assert len(harness.transcriber.calls) == 1
    voiced_turn(session)
    assert next_call.result() == "recognized words"
    assert len(harness.transcriber.calls) == 2


def test_detection_started_before_timeout_cannot_activate_next_turn(harness):
    entered = threading.Event()
    release = threading.Event()
    original = harness.detector.detected

    def delayed_detection(frame):
        entered.set()
        assert release.wait(WAIT)
        return original(frame)

    harness.detector.detected = delayed_detection
    first = harness.request(timeout=0.1)
    session = harness.source.next_session()
    session.feed(*([pcm(WAKE_AMPLITUDE)] * 4))
    assert entered.wait(WAIT)
    assert first.result() == ""

    second = harness.request(timeout=0.5)
    release.set()
    session.feed(pcm(1200), pcm(1200), pcm(0), pcm(0))
    session.drain()

    assert second.result() == ""
    assert harness.transcriber.calls == []
    assert not harness.listener.claim_utterance()


def test_expired_vad_cannot_disarm_next_turn(harness):
    entered = threading.Event()
    release = threading.Event()
    original = harness.listener.vad.utterance_chunks

    def delayed_vad(chunks):
        entered.set()
        assert release.wait(WAIT)
        yield from original(chunks)

    harness.listener.vad.utterance_chunks = delayed_vad
    first = harness.request(timeout=0.1)
    session = harness.source.next_session()
    voiced_turn(session)
    assert entered.wait(WAIT)
    assert first.result() == ""

    second = harness.request(timeout=0.5)
    release.set()
    session.drain()
    voiced_turn(session)

    assert second.result() == "recognized words"
    assert len(harness.transcriber.calls) == 1
    assert harness.listener.claim_utterance()


def test_mute_closes_capture_resets_and_discards_blocked_stale_transcript(harness):
    harness.transcriber.release.clear()
    old_generation = harness.control.generation
    call = harness.request()
    session = harness.source.next_session()
    voiced_turn(session)
    assert harness.transcriber.entered.wait(WAIT)
    old_cancel = harness.transcriber.calls[0][1]
    assert harness.control.set_muted(True)
    assert session.closed.is_set()
    assert old_cancel.is_set()
    assert harness.control.generation > old_generation
    assert harness.listener._results.empty()
    assert not harness.control.set_muted(False)
    fresh_session = harness.source.next_session()
    assert fresh_session is not session
    assert harness.detector.resets.qsize() >= 3
    harness.transcriber.release.set()
    assert call.result() == ""
    assert not harness.listener.claim_utterance()
    next_call = harness.request()
    voiced_turn(fresh_session)
    assert next_call.result() == "recognized words"
    assert harness.listener.claim_utterance()
    assert harness.transcriber.calls[1][1] is not old_cancel


def test_mute_invalidates_transcript_already_returned_but_not_claimed(harness):
    call = harness.request()
    session = harness.source.next_session()
    voiced_turn(session)
    assert call.result() == "recognized words"
    assert harness.control.set_muted(True)
    assert session.closed.is_set()
    assert not harness.control.set_muted(False)
    harness.source.next_session()
    assert not harness.listener.claim_utterance()


def test_persisted_mute_starts_without_capture_timeout_does_not_stop_service(tmp_path):
    instance = ListenerHarness(tmp_path, muted=True)
    try:
        assert instance.control.muted
        assert instance.listener.next_utterance(timeout=0.1) == ""
        assert instance.source.opened == []
        assert instance.detector.frames == []
        assert instance.transcriber.calls == []
        assert not instance.stop.is_set()
        assert instance.listener._worker.is_alive()
        assert not instance.control.set_muted(False)
        session = instance.source.next_session()
        call = instance.request()
        voiced_turn(session)
        assert call.result() == "recognized words"
        assert instance.listener.claim_utterance()
    finally:
        instance.close()
    restored = MicrophoneControl(instance.database)
    try:
        assert not restored.muted
    finally:
        restored.close()


def test_playback_gate_prevents_capture_and_close_joins_while_gate_closed(harness):
    with harness.gate.speaking():
        assert harness.listener.next_utterance(timeout=0.1) == ""
        assert harness.source.opened == []
        assert harness.detector.frames == []
        assert harness.transcriber.calls == []
        closing = BackgroundCall(harness.listener.close)
        assert closing.result() is None
        assert not harness.listener._worker.is_alive()
        assert harness.transcriber.closed
        assert harness.control.on_change is None


def test_playback_started_during_capture_suppresses_wake_and_reopens(harness):
    harness.listener.start()
    session = harness.source.next_session()
    with harness.gate.speaking():
        wake(session)
        assert session.closed.wait(WAIT)
        assert harness.detector.frames == []
        assert harness.transcriber.calls == []
    fresh_session = harness.source.next_session()
    call = harness.request()
    voiced_turn(fresh_session)
    assert call.result() == "recognized words"


def test_wakeword_barge_in_interrupts_playback_then_captures_fresh_command(tmp_path):
    harness = ListenerHarness(tmp_path, allow_wakeword_during_speech=True)
    try:
        harness.listener.start()
        session = harness.source.next_session()
        with harness.gate.interruptible_turn() as cancellation:
            with harness.gate.speaking():
                session.feed(*([pcm(WAKE_AMPLITUDE)] * 8))
                assert harness.gate.interrupt_event.wait(WAIT)
                assert cancellation.is_set()
                assert harness.listener._turn_phase == "capturing"
                assert harness.transcriber.calls == []

            blocks = [pcm(1000), pcm(1200), pcm(0), pcm(0)]
            session.feed(*blocks)
            session.drain()

        assert harness.listener._barge_in_ready.wait(WAIT)
        assert harness.transcriber.calls == []
        assert harness.listener.next_utterance() == "recognized words"
        assert harness.listener.claim_utterance()
        assert harness.transcriber.calls[0][0] == b"".join(blocks)
        assert harness.wake_events == ["LISTENING"]
    finally:
        harness.close()


def test_close_joins_active_capture_and_waiting_consumer(harness):
    call = harness.request()
    session = harness.source.next_session()
    harness.listener.close()
    assert call.result() == ""
    assert session.closed.is_set()
    assert not harness.listener._worker.is_alive()
    assert harness.transcriber.closed


def test_transcriber_close_failure_still_closes_control_store(harness):
    harness.transcriber.close = Mock(
        side_effect=[RuntimeError("STT cleanup failed"), None]
    )

    with pytest.raises(RuntimeError, match="STT cleanup failed"):
        harness.listener.close()

    assert harness.stop.is_set()
    assert harness.listener._worker is None or not harness.listener._worker.is_alive()
    assert harness.control.store is None
    assert harness.control.on_change is None


@pytest.mark.parametrize("failure", [b"invalid block", RuntimeError("capture failed")])
def test_worker_errors_surface_without_transcription_or_recording(harness, tmp_path, failure):
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    call = harness.request()
    session = harness.source.next_session()
    session.feed(failure)
    expected_error = RuntimeError if isinstance(failure, Exception) else ValueError
    with pytest.raises(expected_error, match="capture failed|invalid mono PCM block"):
        call.result()
    assert session.closed.wait(WAIT)
    assert harness.transcriber.calls == []
    assert not harness.listener.claim_utterance()
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == before


def test_iterator_close_failure_stops_worker_and_releases_waiting_turn(harness):
    class BadCloseSource:
        def chunks(self, _stop_event):
            class BadIterator:
                def __iter__(self):
                    return self

                def __next__(self):
                    return b"invalid block"

                def close(self):
                    raise RuntimeError("capture close failed")

            return BadIterator()

    harness.listener.source = BadCloseSource()
    call = harness.request()
    assert call.result() == ""
    assert harness.stop.is_set()
    assert harness.listener._session_closed.is_set()
    harness.listener._worker.join(WAIT)
    assert not harness.listener._worker.is_alive()


def test_valid_custom_wake_assets_use_local_onnx_and_configured_threads(tmp_path):
    paths, manifest_path, session_factory = fake_wake_assets(tmp_path)
    settings = replace(
        Settings(), voice_profile="wakeword", stt_backend="whisper.cpp",
        wake_word_model_path=paths["wakeword"],
        wake_word_melspectrogram_path=paths["melspectrogram"],
        wake_word_embedding_path=paths["embedding"],
        wake_word_manifest_path=manifest_path,
        wake_word_threads=3, wake_word_threshold=0.7,
    )
    model = Mock()
    model.predict.return_value = {"hey_orion": 0.8}
    model_factory = Mock(return_value=model)

    detector = OpenWakeWordDetector.from_settings(
        settings, model_factory=model_factory, samples_factory=lambda block: block,
        session_factory=session_factory, runtime_version="0.6.0", numpy_module=FakeNumpy,
    )

    model_factory.assert_called_once_with(
        wakeword_models=[str(paths["wakeword"])],
        inference_framework="onnx",
        melspec_model_path=str(paths["melspectrogram"]),
        embedding_model_path=str(paths["embedding"]),
        ncpu=3,
    )
    assert detector.detected(pcm(0))


def test_pretrained_classifier_without_manifest_accepts_rank_two_score(tmp_path):
    paths, manifest_path, session_factory = fake_wake_assets(
        tmp_path,
        tensor_overrides={
            "wakeword_output_shape": ["batch", 1],
            "wakeword_runtime_shape": (1, 1),
        },
    )
    settings = replace(
        wake_settings(paths, manifest_path), wake_word_manifest_path=None
    )
    model_factory = Mock()

    OpenWakeWordDetector.from_settings(
        settings,
        model_factory=model_factory,
        session_factory=session_factory,
        runtime_version="0.6.0",
        numpy_module=FakeNumpy,
    )

    model_factory.assert_called_once_with(
        wakeword_models=[str(paths["wakeword"])],
        inference_framework="onnx",
        melspec_model_path=str(paths["melspectrogram"]),
        embedding_model_path=str(paths["embedding"]),
        ncpu=settings.wake_word_threads,
    )


@pytest.mark.parametrize("asset_index", [0, 1, 2])
@pytest.mark.parametrize("invalid_kind", ["unset", "missing", "directory", "wrong_suffix"])
def test_wake_assets_rejected_before_factory_call(tmp_path, asset_index, invalid_kind):
    paths_by_key, manifest_path, session_factory = fake_wake_assets(tmp_path)
    paths = [paths_by_key[key] for key in ("wakeword", "melspectrogram", "embedding")]
    if invalid_kind == "unset":
        paths[asset_index] = None
    elif invalid_kind == "missing":
        paths[asset_index].unlink()
    elif invalid_kind == "directory":
        paths[asset_index].unlink()
        paths[asset_index].mkdir()
    else:
        paths[asset_index] = tmp_path / "invalid.tflite"
        paths[asset_index].touch()
    settings = replace(
        Settings(), voice_profile="wakeword", stt_backend="whisper.cpp",
        wake_word_model_path=paths[0],
        wake_word_melspectrogram_path=paths[1],
        wake_word_embedding_path=paths[2],
        wake_word_manifest_path=manifest_path,
    )
    model_factory = Mock()
    samples_factory = Mock()
    expected = "three local ONNX model files" if invalid_kind == "unset" else "must be a local ONNX file"
    with pytest.raises(FileNotFoundError, match=expected):
        OpenWakeWordDetector.from_settings(
            settings, model_factory=model_factory, samples_factory=samples_factory,
            session_factory=session_factory, runtime_version="0.6.0", numpy_module=FakeNumpy,
        )
    model_factory.assert_not_called()
    samples_factory.assert_not_called()


def test_local_wake_assets_select_exact_path_onnx_threads_and_reset(tmp_path):
    paths, manifest_path, session_factory = fake_wake_assets(tmp_path)
    wake_path = paths["wakeword"]
    melspec_path = paths["melspectrogram"]
    embedding_path = paths["embedding"]
    settings = replace(
        Settings(), voice_profile="wakeword", stt_backend="whisper.cpp",
        wake_word_model_path=wake_path,
        wake_word_melspectrogram_path=melspec_path,
        wake_word_embedding_path=embedding_path,
        wake_word_manifest_path=manifest_path,
        wake_word_threshold=0.6, wake_word_threads=2,
    )
    model = Mock()
    model_factory = Mock(return_value=model)
    samples = object()
    samples_factory = Mock(return_value=samples)
    detector = OpenWakeWordDetector.from_settings(
        settings, model_factory=model_factory, samples_factory=samples_factory,
        session_factory=session_factory, runtime_version="0.6.0", numpy_module=FakeNumpy,
    )
    model_factory.assert_called_once_with(
        wakeword_models=[str(wake_path)], inference_framework="onnx",
        melspec_model_path=str(melspec_path), embedding_model_path=str(embedding_path), ncpu=2,
    )
    samples_factory.assert_not_called()
    frame = pcm(0) * 4
    model.predict.return_value = {str(wake_path): 0.59, "unselected": 1.0}
    assert not detector.detected(frame)
    model.predict.return_value = {str(wake_path): 0.6}
    assert detector.detected(frame)
    model.predict.assert_called_with(samples)
    samples_factory.assert_called_with(frame)
    detector.reset()
    model.reset.assert_called_once_with()


def test_invalid_onnx_bytes_fail_before_wake_model_construction(tmp_path):
    paths, manifest_path, session_factory = fake_wake_assets(tmp_path, invalid_asset="embedding")
    model_factory = Mock()

    with pytest.raises(ValueError, match="Invalid ONNX model for embedding"):
        OpenWakeWordDetector.from_settings(
            wake_settings(paths, manifest_path), model_factory=model_factory,
            session_factory=session_factory, runtime_version="0.6.0", numpy_module=FakeNumpy,
        )

    model_factory.assert_not_called()


def test_checksum_mismatch_fails_before_onnx_session_creation(tmp_path):
    paths, manifest_path, _session_factory = fake_wake_assets(tmp_path)
    paths["wakeword"].write_bytes(b"changed after manifest hashing")
    session_factory = Mock()

    with pytest.raises(ValueError, match="SHA-256 mismatch for wakeword"):
        OpenWakeWordDetector.from_settings(
            wake_settings(paths, manifest_path), session_factory=session_factory,
            runtime_version="0.6.0", numpy_module=FakeNumpy,
        )

    session_factory.assert_not_called()


def test_missing_manifest_fails_before_onnx_session_creation(tmp_path):
    paths, manifest_path, _session_factory = fake_wake_assets(tmp_path)
    manifest_path.unlink()
    session_factory = Mock()

    with pytest.raises(FileNotFoundError, match="verified local asset manifest"):
        OpenWakeWordDetector.from_settings(
            wake_settings(paths, manifest_path), session_factory=session_factory,
            runtime_version="0.6.0", numpy_module=FakeNumpy,
        )

    session_factory.assert_not_called()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"melspectrogram_input_name": "audio"}, "Melspectrogram input input tensor must be named"),
        ({"embedding_output_shape": ["batch", 1, 1, 64]}, "Embedding output tensor shape is incompatible"),
        ({"wakeword_input_shape": ["batch", 16, 64]}, "Wakeword input tensor shape is incompatible"),
        ({"wakeword_runtime_shape": (1, 2, 1)}, "Wakeword output must have shape"),
    ],
)
def test_incompatible_onnx_tensor_contracts_are_rejected(tmp_path, overrides, message):
    paths, manifest_path, session_factory = fake_wake_assets(tmp_path, tensor_overrides=overrides)
    with pytest.raises(ValueError, match=message):
        OpenWakeWordDetector.from_settings(
            wake_settings(paths, manifest_path), session_factory=session_factory,
            runtime_version="0.6.0", numpy_module=FakeNumpy,
        )


def test_manifest_requires_real_provenance_and_license_metadata(tmp_path):
    paths, manifest_path, session_factory = fake_wake_assets(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"]["wakeword"]["license"] = "TBD"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be placeholder metadata"):
        OpenWakeWordDetector.from_settings(
            wake_settings(paths, manifest_path), session_factory=session_factory,
            runtime_version="0.6.0", numpy_module=FakeNumpy,
        )


def test_manifest_requires_matching_openwakeword_runtime_version(tmp_path):
    paths, manifest_path, session_factory = fake_wake_assets(tmp_path)
    with pytest.raises(RuntimeError, match="require openwakeword 0.6.0"):
        OpenWakeWordDetector.from_settings(
            wake_settings(paths, manifest_path), session_factory=session_factory,
            runtime_version="0.7.0", numpy_module=FakeNumpy,
        )


@pytest.mark.parametrize(
    ("backend", "selected", "unselected"),
    [("whisper.cpp", "WhisperCppTranscriber", "VoskTranscriber"),
     ("vosk", "VoskTranscriber", "WhisperCppTranscriber")],
)
def test_transcriber_factory_selects_only_requested_adapter(backend, selected, unselected):
    settings = replace(Settings(), stt_backend=backend)
    with patch(f"orion.audio.factory.{selected}") as constructor, patch(
        f"orion.audio.factory.{unselected}"
    ) as other:
        assert create_transcriber(settings) is constructor.return_value
        constructor.assert_called_once_with(settings)
        other.assert_not_called()
        constructor.return_value.validate_assets.assert_not_called()
        constructor.return_value.transcribe.assert_not_called()


def test_speech_factory_selects_kitten_without_synthesis():
    settings = replace(Settings(), tts_backend="kittentts")
    gate = AudioGate()
    with patch("orion.audio.factory.KittenSpeechOutput") as constructor:
        assert create_speech_output(settings, gate) is constructor.return_value
        constructor.assert_called_once_with(settings, gate)
        constructor.return_value.speak.assert_not_called()


def test_factories_reject_unsupported_backends_without_construction():
    with patch("orion.audio.factory.WhisperCppTranscriber") as whisper, patch(
        "orion.audio.factory.VoskTranscriber"
    ) as vosk, patch("orion.audio.factory.KittenSpeechOutput") as kitten:
        with pytest.raises(ValueError, match="Unsupported STT backend"):
            create_transcriber(replace(Settings(), stt_backend="unknown"))
        with pytest.raises(ValueError, match="Unsupported TTS backend"):
            create_speech_output(replace(Settings(), tts_backend="unknown"), AudioGate())
        whisper.assert_not_called()
        vosk.assert_not_called()
        kitten.assert_not_called()