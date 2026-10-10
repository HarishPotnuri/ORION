"""Wake-only capture independent of transcription and model generation."""

from __future__ import annotations

import logging
import queue
import threading
import time

from ..config import Settings
from .capture import AudioGate, SoundDeviceChunkSource
from .control import MicrophoneControl
from .vad import VoiceActivityDetector

logger = logging.getLogger(__name__)


class _Cancelled:
    def __init__(self, *events, deadline=float("inf")):
        self.events = events
        self.deadline = deadline

    def is_set(self):
        return time.monotonic() >= self.deadline or any(event.is_set() for event in self.events)

    def wait(self, timeout):
        return self.events[0].wait(timeout) or self.is_set()


class WakeListenerInput:
    def __init__(
        self, settings: Settings, detector, transcriber, control: MicrophoneControl,
        *, gate: AudioGate, stop_event: threading.Event, source=None, on_wake=None,
    ):
        self.settings = settings
        self.detector = detector
        self.transcriber = transcriber
        self.control = control
        self.gate = gate
        self.stop_event = stop_event
        self.source = source or SoundDeviceChunkSource(settings, gate=gate)
        self.on_wake = on_wake
        self.vad = VoiceActivityDetector(settings)
        self._results = queue.Queue(maxsize=1)
        self._armed = threading.Event()
        self._session_stop = threading.Event()
        self._session_closed = threading.Event()
        self._session_closed.set()
        self._turn_lock = threading.RLock()
        self._turn_sequence = 0
        self._active_turn = None
        self._turn_phase = "idle"
        self._turn_deadline = float("inf")
        self._turn_start_deadline = float("inf")
        self._turn_followup = False
        self._followup_until = 0.0
        self._turn_cancel = threading.Event()
        self._last_transcription_confidence: float | None = None
        self._control_cancel = self.control.cancel_event
        self._turn_token = None
        self._pending_barge_in = None
        self._barge_in_ready = threading.Event()
        self._call_lock = threading.Lock()
        self._worker = None
        self._last_capture_error = None
        self.control.on_change = self._changed

    def start(self) -> None:
        if self._worker is not None:
            return
        self.transcriber.validate_assets()
        self._worker = threading.Thread(target=self._listen, name="orion-wake-listener")
        self._worker.start()

    def _clear_results(self):
        while True:
            try:
                self._results.get_nowait()
            except queue.Empty:
                return

    def _changed(self):
        self._session_stop.set()
        with self._turn_lock:
            self._turn_cancel.set()
            self._active_turn = None
            self._turn_phase = "idle"
            self._turn_deadline = float("inf")
            self._turn_start_deadline = float("inf")
            self._turn_followup = False
            self._followup_until = 0.0
            self._armed.clear()
        self._clear_results()
        self._pending_barge_in = None
        self._barge_in_ready.clear()
        if self._worker is not None:
            if not self._session_closed.wait(1.0):
                logger.error("Microphone capture did not close within its shutdown budget")

    def _listen(self):
        while not self.stop_event.is_set():
            if self.control.muted or not self.gate.wakeword_capture_allowed:
                self.stop_event.wait(0.05)
                continue
            token = self.control.generation
            cancelled = self.control.cancel_event
            gate_generation = self.gate.generation
            self._session_stop.clear()
            self._session_closed.clear()
            chunks = None
            terminal_error = None
            capture_failed = False
            try:
                if not self.control.allows(token) or not self.gate.wakeword_capture_allowed:
                    continue
                self.detector.reset()
                session = _Cancelled(
                    self.stop_event, cancelled, self._session_stop,
                    self.gate.capture_cancel_event,
                )
                chunks = self.source.chunks(session)
                frame = bytearray()
                while not session.is_set():
                    with self._turn_lock:
                        active_turn = self._active_turn
                        deadline = (
                            self._turn_start_deadline
                            if self._turn_phase == "listening"
                            else self._turn_deadline
                        )
                        followup_turn = (
                            active_turn is not None
                            and self._turn_phase == "listening"
                            and self._turn_followup
                        )
                    if active_turn is not None and time.monotonic() >= deadline:
                        self._finish_turn(active_turn)
                        continue
                    chunk = next(chunks, None)
                    if chunk is None:
                        break
                    self._last_capture_error = None
                    if not self.gate.wakeword_capture_allowed:
                        self._session_stop.set()
                        break
                    current_gate_generation = self.gate.generation
                    if current_gate_generation != gate_generation:
                        frame.clear()
                        self.detector.reset()
                        gate_generation = current_gate_generation
                        continue
                    if len(chunk) != self.settings.audio_block_frames * 2:
                        raise ValueError("Capture returned an invalid mono PCM block")
                    frame.extend(chunk)
                    activated = False
                    while len(frame) >= 2560:
                        wake_frame = bytes(frame[:2560])
                        del frame[:2560]
                        started = time.monotonic()
                        detected = (
                            self.vad._rms(wake_frame) >= self.vad.threshold
                            if followup_turn
                            else self.detector.detected(wake_frame)
                        )
                        with self._turn_lock:
                            still_armed = (
                                active_turn is not None
                                and self._active_turn == active_turn
                                and self._turn_phase == "listening"
                                and time.monotonic() < self._turn_deadline
                            )
                            barge_in = (
                                active_turn is None
                                and self.gate.is_speaking
                                and self.gate.allow_wakeword_during_speech
                            )
                        if self.gate.generation != gate_generation:
                            frame.clear()
                            self.detector.reset()
                            gate_generation = self.gate.generation
                            continue
                        if detected:
                            activated = still_armed or barge_in
                            if activated:
                                logger.info(
                                    "Follow-up speech detected"
                                    if followup_turn else "Wake word detected"
                                )
                                if barge_in:
                                    self.gate.interrupt_playback()
                                if not followup_turn and self.on_wake is not None:
                                    self.on_wake()
                            break
                    if not activated:
                        continue
                    with self._turn_lock:
                        if barge_in:
                            if not self.gate.is_speaking or self._active_turn is not None:
                                continue
                            self._turn_sequence += 1
                            active_turn = self._turn_sequence
                            self._active_turn = active_turn
                            self._turn_phase = "capturing"
                            turn_deadline = time.monotonic() + self.settings.max_utterance_seconds + 2
                            turn_cancel = threading.Event()
                            self._turn_deadline = turn_deadline
                            self._armed.set()
                        else:
                            if self._active_turn != active_turn or self._turn_phase != "listening":
                                continue
                            self._turn_phase = "capturing"
                            turn_deadline = (
                                time.monotonic()
                                + self.settings.max_utterance_seconds
                                + 2
                                if followup_turn else self._turn_deadline
                            )
                            self._turn_deadline = turn_deadline
                            turn_cancel = self._turn_cancel
                    frame.clear()
                    turn_stop = _Cancelled(
                        self.stop_event, cancelled, self._session_stop, turn_cancel,
                        self.gate.capture_cancel_event, deadline=turn_deadline,
                    )

                    def speech_chunks():
                        if followup_turn:
                            block_bytes = self.settings.audio_block_frames * 2
                            for offset in range(0, len(wake_frame), block_bytes):
                                if turn_stop.is_set():
                                    return
                                yield wake_frame[offset:offset + block_bytes]
                        for speech in chunks:
                            if turn_stop.is_set():
                                break
                            if not self.gate.is_open:
                                continue
                            if len(speech) != self.settings.audio_block_frames * 2:
                                raise ValueError("Capture returned an invalid mono PCM block")
                            yield speech

                    pcm = b"".join(self.vad.utterance_chunks(speech_chunks()))
                    with self._turn_lock:
                        owns_turn = (
                            self._active_turn == active_turn
                            and self._turn_phase == "capturing"
                            and time.monotonic() < self._turn_deadline
                        )
                        if owns_turn and not turn_stop.is_set() and self.control.allows(token):
                            if barge_in:
                                self._pending_barge_in = (token, cancelled, pcm)
                                self._barge_in_ready.set()
                                self._finish_turn(active_turn)
                            else:
                                self._turn_phase = "transcribing"
                                self._results.put_nowait((active_turn, token, cancelled, turn_cancel, pcm))
                        elif self._active_turn == active_turn:
                            self._finish_turn(active_turn)
                    self.detector.reset()
                if not session.is_set():
                    self.stop_event.wait(0.05)
            except Exception as error:
                if self._armed.is_set() and self.control.allows(token):
                    try:
                        with self._turn_lock:
                            active_turn = self._active_turn
                            if active_turn is not None:
                                self._results.put_nowait((active_turn, error))
                    except queue.Full:
                        pass
                error_name = type(error).__name__
                if error_name != self._last_capture_error:
                    logger.warning("Wake capture failed (%s); retrying in 1s", error_name)
                self._last_capture_error = error_name
                capture_failed = True
            finally:
                try:
                    if chunks is not None:
                        close = getattr(chunks, "close", None)
                        if close is not None:
                            close()
                except Exception as error:
                    terminal_error = error
                    logger.error("Wake capture close failed (%s)", type(error).__name__)
                finally:
                    self._session_closed.set()
                if terminal_error is not None:
                    with self._turn_lock:
                        self._turn_cancel.set()
                        self._active_turn = None
                        self._turn_phase = "idle"
                        self._armed.clear()
                    self.stop_event.set()
            if capture_failed and not self.stop_event.is_set():
                self.stop_event.wait(1.0)

    def _finish_turn(self, turn_id):
        with self._turn_lock:
            if self._active_turn == turn_id:
                self._turn_cancel.set()
                self._active_turn = None
                self._turn_phase = "idle"
                self._turn_deadline = float("inf")
                self._turn_start_deadline = float("inf")
                self._turn_followup = False
                self._armed.clear()

    def next_utterance(self, timeout=None, *, followup: bool = False) -> str:
        with self._call_lock:
            if self.stop_event.is_set():
                return ""
            self.gate.begin_capture()
            self.start()
            self._clear_results()
            self._turn_token = None
            self._last_transcription_confidence = None
            deadline = float("inf") if timeout is None else time.monotonic() + timeout
            pending_barge_in = self._pending_barge_in
            self._pending_barge_in = None
            self._barge_in_ready.clear()
            if pending_barge_in is not None:
                token, captured_cancel, pcm = pending_barge_in
                if not self.control.allows(token) or not pcm:
                    return ""
                remaining = min(self.settings.stt_timeout, deadline - time.monotonic())
                if remaining <= 0:
                    return ""
                text = self._transcribe(
                    pcm,
                    cancel_event=_Cancelled(
                        self.stop_event, captured_cancel,
                        self.gate.turn_cancel_event,
                        self.gate.capture_cancel_event,
                        deadline=deadline,
                    ),
                    timeout=remaining,
                )
                if (
                    self.control.allows(token)
                    and not self.stop_event.is_set()
                    and not self.gate.turn_cancel_event.is_set()
                    and not self.gate.capture_cancel_event.is_set()
                ):
                    self._turn_token = token
                    return text
                return ""
            with self._turn_lock:
                self._turn_sequence += 1
                turn_id = self._turn_sequence
                self._active_turn = turn_id
                self._turn_phase = "listening"
                pending_followup = self._followup_until > time.monotonic()
                followup_turn = followup or pending_followup
                if pending_followup:
                    deadline = min(deadline, self._followup_until)
                    self._followup_until = 0.0
                self._turn_followup = followup_turn
                self._turn_deadline = deadline
                self._turn_start_deadline = deadline
                self._turn_cancel = threading.Event()
                self._control_cancel = self.control.cancel_event
                turn_cancel = self._turn_cancel
                control_cancel = self._control_cancel
                control_token = self.control.generation
                self._armed.set()
            try:
                while not self.stop_event.is_set():
                    if (
                        turn_cancel.is_set()
                        or self.gate.turn_cancel_event.is_set()
                        or self.gate.capture_cancel_event.is_set()
                        or control_cancel.is_set()
                        or not self.control.allows(control_token)
                    ):
                        return ""
                    with self._turn_lock:
                        deadline = self._turn_deadline
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return ""
                    try:
                        result = self._results.get(timeout=min(0.05, remaining))
                    except queue.Empty:
                        continue
                    result_turn, *payload = result
                    if result_turn != turn_id:
                        continue
                    if len(payload) == 1 and isinstance(payload[0], Exception):
                        raise payload[0]
                    token, captured_cancel, captured_turn_cancel, pcm = payload
                    if not self.control.allows(token):
                        continue
                    if not pcm:
                        return ""
                    remaining = min(self.settings.stt_timeout, deadline - time.monotonic())
                    if remaining <= 0:
                        return ""
                    started = time.monotonic()
                    text = self._transcribe(
                        pcm,
                        cancel_event=_Cancelled(
                            self.stop_event, captured_cancel, captured_turn_cancel,
                            self.gate.capture_cancel_event, deadline=deadline,
                        ),
                        timeout=remaining,
                    )
                    logger.info("Voice stage=stt elapsed_ms=%.1f outcome=%s", (time.monotonic() - started) * 1000, "transcribed" if text else "empty")
                    if (
                        self.control.allows(token)
                        and not self.stop_event.is_set()
                        and not turn_cancel.is_set()
                        and not self.gate.turn_cancel_event.is_set()
                        and not self.gate.capture_cancel_event.is_set()
                    ):
                        self._turn_token = token
                        return text
                    return ""
                return ""
            finally:
                self._finish_turn(turn_id)

    def next_followup_utterance(self, timeout: float = 10.0) -> str:
        return self.next_utterance(timeout=timeout, followup=True)

    def wait_until_unmuted(self) -> bool:
        while not self.stop_event.is_set() and self.control.muted:
            self.stop_event.wait(0.1)
        return not self.stop_event.is_set()

    def _transcribe(self, pcm, *, cancel_event, timeout) -> str:
        text = self.transcriber.transcribe(
            pcm, cancel_event=cancel_event, timeout=timeout
        )
        self._last_transcription_confidence = getattr(
            self.transcriber, "last_confidence", None
        )
        return text

    @property
    def transcription_confidence(self) -> float | None:
        return self._last_transcription_confidence

    def open_followup_window(self, timeout: float = 10.0) -> None:
        if timeout <= 0 or self.stop_event.is_set() or self.control.muted:
            return
        deadline = time.monotonic() + timeout
        with self._turn_lock:
            if self._active_turn is not None and self._turn_phase == "listening":
                self._turn_followup = True
                self._turn_deadline = deadline
                self._turn_start_deadline = deadline
                self._followup_until = 0.0
            else:
                self._followup_until = deadline

    def claim_utterance(self) -> bool:
        token = self._turn_token
        self._turn_token = None
        return token is not None and self.control.claim(token)

    def close(self) -> None:
        self.stop_event.set()
        self._session_stop.set()
        failure = None
        try:
            self.transcriber.close()
        except Exception as error:
            failure = error
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            if self._worker.is_alive():
                failure = failure or RuntimeError("Wake listener failed to stop")
        self.control.on_change = None
        try:
            self.control.close()
        except Exception as error:
            failure = failure or error
        if failure is not None:
            raise failure