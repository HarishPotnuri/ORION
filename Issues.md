# Outstanding Issues

Last reviewed: 2026-10-02

This document tracks known open issues and unverified acceptance gates. A passing unit test or a successful isolated component check is not evidence that the complete voice service is reliable on the Raspberry Pi. Do not mark an item resolved until its closure criteria are met and the result is recorded here and, where appropriate, in `Plan.md`.

## Current Target

- Hardware: Raspberry Pi 5, 8 GB RAM, aarch64; Debian 13 (trixie), kernel `6.18.50+rpt-pi-2712`.
- Voice runtime: uv 0.12.22, Python 3.12.15, VOSK small English model `0.15`, KittenTTS 0.1.0 with nano 0.2 assets.
- Audio/control: Jabra EVOLVE 20 MS USB audio; GPIO23 push-to-talk; GPIO16 TTS cutoff; RGB status LED on GPIO27 red, GPIO25 green, and GPIO17 blue.
- Reasoning: Ollama 0.35.0 on loopback with `qwen3:1.7b`, digest `8f68893c685c`.
- Dependency profile: versions are recorded in `uv.lock`; scoped uv exclusion avoids installing Misaki 0.9.4's unused curated-transformer extra and its incompatible CUDA dependency tree.

## P0: Complete Voice Turn Is Not Reliably Verified

**Status:** User confirmed multi-turn reasoning works through the deployed systemd service on 2026-10-05. Keep a longer interruption/reliability soak open until repeated sessions and deliberate barge-in are separately recorded.

**Observed behavior:** Before the playback change, two live Pi sessions aborted with `double free or corruption (!prev)` immediately after a phonemizer warning. Python faulthandler on a deterministic reproduction located the abort in `sounddevice.wait()` closing the playback stream while another thread called `sounddevice.stop()`, which also stopped and closed that stream. The owned `RawOutputStream` change passed 12 real-device interruption cycles. After deploying the audio/planner fixes and systemd service, the user confirmed multi-turn interaction works.


**Impact:** Previous builds could abort or stall during repeated turns. The latest deployed build is reported to work across multiple turns; a longer reliability/barge-in soak is still needed before calling the path robust.

**Next actions:**

1. Run a longer supervised session with at least five reasoning turns and a context-dependent follow-up; record stage timing separately from PTT wait.
2. Speak a long reply and press the separate cutoff button during playback. Confirm playback stops, process survives, and the next PTT press captures a fresh utterance; investigate the earlier empty-capture observation.
3. Preserve full service logs and any `PYTHONFAULTHANDLER=1` abort trace during the soak.

**Close when:** The multi-turn session completes repeatedly without native abort, hung model calls, stuck capture/playback, or missed return to listening; interruption also completes without closing a stream from a second thread.

## P1: Planner Schema Text Can Leak Into Spoken Replies

**Status:** Prompt/agent fix is deployed to the Pi and the user reports multi-turn operation working. A targeted spoken placeholder regression remains useful because the earlier “user-facing…” output was directly observed.

**Observed behavior:** In a patched live run, the Pi spoke text beginning with “user-facing…” and stopped by itself; the user confirmed that no button press caused the cutoff. The original system prompt included the literal JSON example value `"user-facing answer"`. The agent accepted any non-empty string as a final response, so a model can return the schema description itself and ORION will pass it directly to TTS.

The deployed `src/orion/core/context.py` instructions omit placeholder-valued response examples and explicitly prohibit schema labels in spoken output. `src/orion/core/agent.py` rejects known schema placeholder phrases so the malformed-response retry/fallback path is used. Tests in `tests/test_agent_context.py` cover prompt content, retrying placeholder output, and failing closed on repeated placeholder output. The deployed Pi suite passed, and the user reports multi-turn voice use works; the exact earlier placeholder request has not been separately repeated and verified.

**Impact:** ORION can speak prompt-template text instead of an answer. Even if that utterance completes, it is not a successful reasoning turn and can sound like a cutoff or internal error.

**Next actions:**

1. Run the placeholder regression suite in the installed service environment.
2. Repeat the exact spoken request that previously caused “user-facing…” output and verify the spoken answer is valid.
3. Confirm invalid output is retried once and, if repeated, produces the concise safe fallback rather than schema text.

**Close when:** The deployed Pi never speaks schema placeholders, both local and Pi regression suites pass, and a valid concise response is spoken for the repeated request.

## P1: Voice Planner Latency and Deadline Behavior

**Status:** Partially mitigated. The 128-token cap and prompt changes are deployed, and multi-turn use is user-confirmed; end-to-end latency distribution and worst-case deadline behavior remain unmeasured.

**Observed behavior:** A Pi voice turn transcribed a 14-character request but produced no audible response while the Ollama runner stayed at 100% CPU. Another third turn was still consuming model CPU several minutes after its transcript. Earlier, the user-facing turn appeared to speak an incomplete phrase before the next model request stalled. ORION's configured `max_total_task_time` is 120 seconds, but a late observed run remained active past that interval; the service was then stopped with SIGTERM. No final turn result was captured from that run.

A diagnostic direct Ollama request using the planner's task state and a 264-token estimated prompt returned valid JSON in 7.75 seconds on the Pi; a typed “Explain Kafka briefly” request completed in 15 seconds. The deployed agent caps each planner completion at 128 output tokens (previously up to 512), and the user reports multi-turn operation working. Python worker timeouts remain cooperative only at the caller boundary: a timed-out worker thread cannot forcibly stop an adapter that ignores its timeout.

**Impact:** The assistant can appear unresponsive, fail to speak before the task deadline, consume CPU, and potentially leave a background inference thread running after the foreground turn times out.

**Next actions:**

1. Time direct typed and spoken reasoning requests with the same model and prompt; confirm normal turns finish within the configured budget.
2. Log safe stage timings (transcription end, model start/finish, TTS start/finish) and agent completion reason. Do not log private full conversation content.
3. Confirm the configured HTTP/model timeout bounds a live request and that a timeout does not leave an unbounded worker or stale request competing with subsequent turns.
4. If the turn still exceeds the deadline, inspect Ollama request duration and agent retry count; tune context, output budget, or model options only from those measurements.

**Close when:** Multiple spoken reasoning turns finish and speak inside a documented task budget, and a deliberately slow model returns a concise spoken timeout while the service remains responsive for the next PTT turn.

## P1: Clean Raspberry Pi Image Install Is Not Proven

**Status:** Open.

**Evidence:** A fresh Python 3.12 virtual environment was created on the existing Pi and installed the locked `voice`, `pi`, and `test` profile. `uv pip check` reported all 98 packages compatible; offline KittenTTS synthesis succeeded without Torch/CUDA packages; focused tests passed. This proves a fresh environment on the existing configured OS, not a clean-image installation. Existing system packages, downloaded models, Hub caches, Ollama setup, permissions, and other machine state may have supplied undocumented prerequisites.

**Impact:** Installation on a newly imaged supported Pi may fail or require manual fixes that are not in the runbook.

**Next actions:**

1. On a freshly imaged 64-bit supported Raspberry Pi OS installation, follow only the documented prerequisites and locked uv setup in `README.md`.
2. Record image/version, apt prerequisites, account, permissions/groups, disk requirements, and every command required to complete setup.
3. Verify model downloads/cache placement, database path permissions, Jabra input/output selection, GPIO access, offline startup after first asset download, and loopback Ollama operation.
4. Update the runbook with any missing reproducible prerequisites; repeat on a clean environment before closing.

**Close when:** A fresh supported image installs the locked profile with no undocumented manual changes and the documented command reaches working devices and local Ollama.

## Resolved: Pi Reminder Regression

**Status:** Resolved on the deployed Pi source.

**Evidence:** `tests/test_tools_orchestration.py::OrchestrationTests::test_typed_reminder_reschedule_and_notes_update` previously failed on an earlier Pi revision. The current deployed runtime's full test suite passed 140 tests and 34 subtests on 2026-10-05.

No further action is currently required. Reopen if the test fails on a later deployed revision.

## P2: Pi Full-Suite Verification After Current Changes

**Status:** Resolved for the current deployed revision.

**Evidence:** The Pi's deployed dedicated environment passed the full suite (140 tests, 34 subtests), following focused audio/agent/reminder/voice tests (70 tests, 13 subtests). The user confirmed multi-turn voice use after deployment.

No further action is currently required for this revision; rerun after the next source deployment.

## P2: Live Online Factual Providers Are Unverified

**Status:** Open validation item from Phase 7.

**Evidence:** Weather and search integrations are implemented with provider allowlisting and request boundaries, but live provider availability and Pi networking have not been validated. SearXNG remains disabled until an HTTPS instance and exact allowed host are configured.

**Impact:** Online weather/search may be unavailable or misconfigured in actual Pi deployment. Local-only model behavior must remain unaffected when providers are down.

**Next actions:** Test Open-Meteo from the Pi; only test SearXNG after a specific HTTPS service/host is configured. Verify bounded timeouts, error messages, allowlist rejection, and that only the explicit query/location is sent.

**Close when:** Each enabled provider passes the documented live request and policy checks, or is explicitly documented as not configured and unavailable by design.

## P2: Service Installation and Operational Acceptance (Phase 10)

**Status:** Deployed on the Pi on 2026-10-05; reboot and full hardware acceptance remain open.

**Outstanding work:**

- Test systemd across a Pi reboot, including reminder catch-up; test failed TTS retry and recurring reminder delivery under the unit. Restart must not produce concurrent scheduler owners or duplicate deliveries.
- Run a longer PTT soak and verify barge-in does not crash the service.
- Verify upgrade/rollback and journal inspection against the actual installed unit.
- Keep wake-word disabled by default until verified. Do not add mute controls without known wiring and end-to-end gating behavior.

**Implemented safeguards and evidence:** `--run-voice` owns the reminder scheduler as part of the same process. A per-database OS lock rejects a second scheduler process; interrupted `delivering` rows are recovered only after the lock is obtained. The deployed unit runs as dedicated `orion-service`, with `audio`/`gpio` groups, read-only application/Python/model paths, writable cache/state paths, and a 150-second graceful-stop limit. The migrated database passes SQLite integrity check and retains one reminder. The service is enabled, reached readiness, survived a manual systemd restart, and the user confirmed multi-turn voice interaction. The installed runtime's full test suite passed 140 tests and 34 subtests. Reboot, live reminder catch-up/retry/recurrence under systemd, and upgrade/rollback remain unverified.

**Close when:** Reboot starts the service, voice and reminder delivery work, restart does not duplicate reminders, shutdown is clean, and permissions/operations are documented and tested.

## P2: Persistent Runtime Configuration

**Status:** Deployed in `/etc/orion/orion.env`; the service reached readiness with the intended audio, VOSK, Ollama, database, and cache configuration.

**Evidence:** Missing shell-local exports previously caused the Pi to fall back to the larger default VOSK model and queue size 8. The working profile requires the small VOSK model path, `ORION_AUDIO_QUEUE_CHUNKS=100`, exact Jabra device name, Ollama backend/base URL/model, and offline Hugging Face mode after asset caching.

**Impact:** A new terminal, reboot, or service launch can silently run with different models/devices or fail to reach the intended local backend.

**Next actions:** For interactive use, follow the runbook's persistent-shell guidance. Recheck effective values after any unit/environment change.

**Close when:** The supported launch mechanism persists and reports the intended settings across shell sessions and reboot without relying on undocumented exports.

## P2: Raspberry Pi Performance, Sustained Operation, and Acoustics (Phase 11)

**Status:** Partially measured. Corrected idle and interactive captures exist, but cold-start, peak-concurrency, sustained thermal, and acoustic gates remain open.

**Outstanding measurements:**

- Idle and peak RSS/CPU during cold model load, warm reasoning, STT, TTS, and combined voice/reminder activity.
- End-to-end and stage latency, including button-to-capture, end-of-speech, transcription, model, and TTS.
- Repeated-turn and sustained-session temperature, throttling, swap, crashes, audio dropouts, and memory growth.
- Power supply, cooling, ambient conditions, exact model/runtime/context details, and memory headroom.
- VAD behavior, Jabra microphone placement, playback feedback/leakage, expected room noise, and wake-word false accepts/rejects if wake word is evaluated.

The deployed source logs PTT wait, VAD/end-of-speech, STT, orchestration, planner model-call, and TTS durations without transcript content. `scripts/phase11-capture.sh` records one-second process/system resource samples and service logs. A corrected 30-second idle capture (27 samples) measured ORION RSS 487,808 KB, unloaded Ollama RSS 36,864 KB, minimum available RAM 7.69 GB, zero swap, CPU temperature 46.1-48.3 C, 0-2% process CPU, and throttle flags `0x0`. A 120-second PTT/time/reasoning capture (115 samples) measured max ORION RSS 498,672 KB, max Ollama RSS 2,393,264 KB, minimum available RAM 5.29 GB, zero swap, max temperature 58.7 C, and `0x0` throttle flags. Planner calls took 10.90 s and 8.11 s; STT took 2.49 s and 2.85 s; TTS took 6.11 s and 14.74 s. The deterministic request measured VAD 1.49 s, STT 1.63 s, and TTS 11.47 s. PTT wait (including 75.3 s of user wait) is distinct from STT. Aggregate CPU peaks were 357% ORION and 427% Ollama across cores. These preliminary short captures do not satisfy cold-start, peak-concurrency, sustained thermal, acoustic, or reliability gates.

A single typed inference, an isolated TTS run, or the 12-cycle interruption stress test is not a sustained-performance benchmark. Multi-turn voice is user-confirmed working; continue supervised workload soaks to establish the reliability gate.

**Close when:** Repeated interactions remain bounded and stable under documented conditions, memory/thermal headroom meet an explicit budget, and all advertised performance claims are tied to those measurements.

## P3: Conservative Hybrid Routing Coverage

**Status:** Implemented initial slice; known coverage limitation.

**Evidence:** Compound request detection is intentionally conservative and may miss natural-language combinations. Calendar/device actions and multi-step memory tools remain out of scope.

**Impact:** Some compound or incomplete requests may be routed to deterministic error responses instead of the planner, or may need the user to restate intent.

**Next actions:** Collect real failed requests and add focused routing examples only when they represent a demonstrated use case. Preserve deterministic fast paths, grounding, privacy, tool allowlists, and task bounds.

**Close when:** No broad rewrite is required; track each accepted routing limitation with an example and regression test, and keep unsupported actions explicitly out of scope.

## Resolved or Mitigated Items (Retain for Context)

- **ARM64 CUDA wheel conflict:** uv's scoped exclusion removes Misaki 0.9.4's unused `spacy-curated-transformers` extra, avoiding Torch 2.14.1 and the SBSA-only `nvidia-cusparselt-cu13` wheel. The locked fresh-venv Pi check reported all 98 packages compatible, and offline TTS synthesis passed. Reopen if dependency versions or resolver behavior changes.
- **Interrupted playback double-close:** Root cause was confirmed with faulthandler: concurrent `sounddevice.wait()` and global `sounddevice.stop()` both closed the same stream. The current owned `RawOutputStream` implementation passed local focused tests, 12 real Pi interruption cycles, and two user-reported integrated turns. Keep under P0 follow-up until the complete repeated reasoning service gate passes.
- **Deterministic local request without model:** A Pi time request succeeded while the configured Ollama base URL pointed to an unreachable local port. Retain as verified behavior unless routing/config changes.
