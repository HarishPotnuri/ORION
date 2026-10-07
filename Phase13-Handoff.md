# Phase 13 Handoff

Phase 13 local implementation is complete; release acceptance remains open for real model artifacts and Pi validation. Source and deployed Pi remain on legacy/VOSK. Candidate wakeword startup requires a complete manifest-backed compatible asset set. Provision and verify custom wake assets, then follow the documented backup, validation, smoke-test, and rollback procedure before candidate deployment. Do not promote Phase 13 until all remaining gates pass.

## Saved Implementation

- Central settings add explicit legacy/wakeword profiles, VOSK/whisper.cpp selection, language, local wake/STT/TTS model paths, manifest path, thread limits, timeout, GPIO debounce, and an online-tools flag defaulting false. Source defaults to legacy/VOSK. Explicit wakeword selection requires verified manifest-backed ONNX assets. Piper is not implemented; TTS selection currently accepts KittenTTS only.
- Weather and search are gated in both adapters and orchestration. Offline planner authorization excludes factual tools; existing online tests explicitly opt in.
- `audio/whisper.py` implements lazy per-utterance CLI transcription with in-memory stdin WAV, bounded output, explicit CPU threads, cancellation, and process ownership. It now requests CLI text output explicitly. The pinned CLI/model passed a service-account stdin transcription on Pi; see `docs/phase13-artifacts.md` for revisions and hash.
- `audio/control.py` adds a narrow SQLite microphone-control record, fail-closed state, generation cancellation, persistence, callbacks, and dispatch permission checks. Reminders and speaker are not muted.
- `audio/listener.py` adds an independent bounded wake listener, 80 ms frames, VAD before transcription, result bounds, and mute cancellation. Wake/VAD completion owns turn IDs, deadlines, and cancellation state; stale wake frames and prior turns cannot claim or clear later turns. AudioGate generations discard buffered state across speaking transitions. Capture-close errors stop the service.
- `audio/factory.py` selects transcribers and speech output without loading models. VOSK has a buffered PCM entry point for the new listener.
- KittenTTS 0.1 loads a local ONNX1 directory through its local ONNX class when `ORION_TTS_MODEL_PATH` is set; config and referenced model/voice files are validated, and failures do not fall back to Hub access. When unset, the existing identifier uses the pre-provisioned cache with application-forced Hugging Face offline mode.
- Custom wake loading requires three local ONNX assets and explicit feature inference threads. No custom "Hey Orion" model has been provisioned, trained, or acoustically validated.
- Application wiring supports GPIO23 microphone mute for an explicitly selected wakeword profile, keeps GPIO16 interruption, checks transcript permission before dispatch/confirmation, preserves legacy warmup, and independently attempts startup cleanup. It does not start speech after service shutdown during orchestration.
- ORION forces `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` before runtime initialization. Assets must be provisioned before use; runtime cache misses fail offline.
- Settings and systemd example now default to ALSA `default`, which accepts the Pi's configured 16 kHz capture and 24 kHz playback rates. Direct AIRHUG PortAudio accepts only 48 kHz and rejects both ORION rates. After selecting `ORION_AUDIO_DEVICE="default"` in the live environment, the user reports voice interaction appears to work. The current local changes have not been redeployed. Extended capture/playback, interruption, and soak validation remain open.
- Whisper now owns a POSIX process group and bounds pipe-worker cleanup. POSIX descendant test is skipped on this Windows laptop and still needs Linux/Pi execution.

## Validation Already Run

Use `C:/Users/LAPTOP/repos/ORION/.venv-local/Scripts/python.exe` (Python 3.11.9).

- Config tests after initial settings: 11 passed.
- Offline gating plus neighboring agent/router tests: 78 tests and 35 subtests passed.
- Whisper/audio/voice/config tests: 65 tests and 33 subtests passed.
- Microphone-control plus adjacent audio/service tests: 41 tests and 15 subtests passed.
- Wake-listener test file: 30 passed, including five repeat runs. Combined listener/audio checks: 87 tests and 28 subtests passed.
- After resume, focused checks passed: config/listener 47 tests and 2 subtests; audio/config/voice/listener/Whisper 103 tests, 35 subtests, 1 POSIX-only skipped test.
- Previous full suite before manifest-backed validation changes: 248 tests passed, 78 subtests passed, and 1 POSIX-only test skipped on Windows. Rerun after current changes.
- Local results are not Pi proof. ALSA rate negotiation and one pinned Whisper CLI/model stdin transcription were checked on Pi; the user reports voice interaction appears to work. Actual device routing, sustained audio, POSIX descendant cleanup, wake acoustics, full ORION integration, and custom assets remain unverified.

## Superseded Review Findings (Resolved Locally)
The numbered review reproductions below are historical. Their fixes and regression tests are present locally; see Remaining Gates for current outstanding work.

1. **Pre-arm audio can become a fresh request.** In `WakeListenerInput`, wake inference begun while the consumer is busy can finish after a new call arms input. Audio queued before arming can then transcribe and pass `claim_utterance()`. Add per-turn identity/capture ownership and a regression that delays detection across arming.
2. **Expired VAD can disarm a new turn.** An earlier timed-out capture can finish after a later call sets `_armed`, then clear that flag. Cancel expired work and permit only its owning turn to clear/publish results. Test timeout followed by resumed capture and a second wake.
3. **Capture-close exceptions can kill the worker silently.** Ensure `_session_closed.set()` runs even if iterator close raises. Surface terminal worker failure or safely restart it; callers must not wait forever on a dead worker.
4. **Playback reset is incomplete.** The listener checks `AudioGate.is_open`, but buffered detector/VAD state can survive a closed/reopened playback gate while capture is blocked. Add playback generation/reset semantics, invalidate affected input, and test no stale activation after speech. Remove the unused `_reset` mechanism or integrate it correctly.
5. **Service cancellation must reach transcription.** Pass a combined service-stop and captured microphone-generation cancellation event into transcription, not a newly fetched event from a possibly changed control generation.
6. **Shutdown during orchestration still starts speech.** Recheck the service stop event immediately after `orchestrator.handle()` and before playback. Existing gate interruption while idle does not cancel later speech.
7. **Whisper cleanup can exceed its deadline.** A spawned helper inheriting stdout/stderr can keep drain threads blocked after the direct child exits. Review reproduced a 150 ms budget taking about 920 ms with an 800 ms helper. Add bounded cleanup and owned process-group/descendant handling; test inherited pipes without leaving workers behind.
8. **Startup cleanup can skip the listener.** If listener startup fails and `button.close()` also raises, input cleanup is skipped before the outer service receives the object. Run cleanup actions independently and preserve the initial exception. Also protect resources created before listener construction.
9. **Complete offline TTS/provider validation.** Local KittenTTS ONNX loading and no-Hub-fallback checks are implemented. Verify the local directory and offline-cache modes on Pi across voice-check, voice service, and reminder paths; cache misses must fail without download. Piper remains Phase 15.
10. **Finish integration/provisioning documentation and tests.** Add config/env/profile tests, app-level GPIO/LED/confirmation/shutdown tests, local custom wake provisioning instructions with artifact hashes/licenses, and candidate configuration documentation. No native model assets exist in this worktree. Do not claim "Hey Orion" is ready from mocked tests.

## Remaining Gates

1. Pi audio checks: PortAudio lists AIRHUG as `AIRHUG 01: USB Audio (hw:0,0)`, but direct-device 16 kHz input and 24 kHz output are rejected; both are accepted through ALSA `default`. Live `/etc/orion/orion.env` was changed to `ORION_AUDIO_DEVICE="default"`, and the user reports interaction appears to work. The local code default matches it but has not been redeployed. Verify sustained capture/playback, actual AIRHUG routing, interruption, and restart behavior before closing the gate.
2. **Verified 2026-10-06:** Built pinned `whisper.cpp` commit `927cfce34f31707e17f2bff35c349632fb9e2c3a` on Pi; installed the model from pinned repository revision `5359861c739e955e79d9a303bcbc70fb988958b1`; verified its hash, service-account CLI startup, and `-f - -otxt` transcription without output files. ORION adapter integration needs source deployment after wake assets pass manifest-backed validation.
3. Provision/checksum custom "Hey Orion" classifier and compatible openWakeWord feature graphs. Verify per-file licenses, training provenance, false accepts, and false rejects. No custom wake assets exist; missing files block activation.
4. On Pi, verify both local `ORION_TTS_MODEL_PATH` loading and cache-backed mode with Hugging Face offline across voice-check, voice service, and reminder paths. A missing local file/cache must fail without download. Piper remains Phase 15.
5. Local app tests cover GPIO23 mute/unmute wiring, startup cleanup, LED color transitions and mute precedence, fresh-claim confirmation rejection, and shutdown during orchestration. The mute LED override restores the current voice status after unmute. Keep these regressions; Pi status behavior and real confirmation/shutdown flows remain part of hardware acceptance.
6. Local suite, editor diagnostics, and `git diff --check` pass for the current implementation; rerun them after further code changes.
7. Validate mute persistence across restart/reboot, reminders while muted, interruption, AIRHUG real audio, wake acoustics, disconnected boot, and systemd recovery on Pi. These checks require a controlled candidate deployment; do not promote or claim Phase 13 complete until they pass.

## Resume Sequence

1. Provision and checksum custom wake assets; the pinned Whisper model and CLI are already installed and tested on Pi.
2. Deploy validator-capable source while preserving legacy/VOSK environment; validate candidate assets.
3. Back up and update `/etc/orion/orion.env`, run smoke test, and retain rollback until acceptance passes.