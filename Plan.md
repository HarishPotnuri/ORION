# ORION Implementation Plan

## Goal

Build ORION as a privacy-first, offline-by-default personal voice assistant for Raspberry Pi 5 with 8 GB RAM and Raspberry Pi OS Lite 64-bit. The target uses wakeword-only activation, local speech recognition, streamed local responses, persistent reminders, and user-controlled memory.

The foundation and Phase 12 planner are implemented. Phase 14 source and a verified Qwen3.5-2B GGUF are local. The owner confirms Phase 14 is deployed and validated on Pi as of 2026-10-08. This session could not independently recheck the live host because SSH timed out. Phases 9–13 hardware acceptance and remaining Phase 16 gates stay separate; Phase 15 and broader memory/semantic retrieval remain planned or deferred.

## Product and Engineering Principles

- Keep model inference local; no cloud fallback or second routing model.
- Never persist raw microphone audio; bound audio, context, tools, and task time.
- Keep factual network tools opt-in, allowlisted, and scoped to the explicit query.
- Keep authorization and side effects deterministic; route clear local actions directly and compound requests through the bounded planner.
- Treat Pi compatibility and performance as unverified until measured on target hardware.

## Current Status (2026-10-08)

Foundation Phases 1–8 and Phase 12 are implemented. Phase 14 source, pinned Qwen3.5-2B artifact, deployment, and Pi validation are reported complete by the owner. The deploy script uses Phase 14 by default. This workstation could not independently recheck the live Pi because SSH timed out. Phase 13 wake assets and Pi acceptance remain open. Phase 15 and the remaining Phase 16 gates remain planned.

Status terms in this plan:

- **Implemented:** code exists and has local test coverage where applicable.
- **Integrated:** components are connected in the intended application flow.
- **Pi-verified:** exercised on the stated Raspberry Pi, OS image, peripherals, and software versions.
- **Measured:** resource or latency numbers were collected on the Pi under a documented workload.

Do not promote a feature to Pi-verified or measured based on laptop tests or code inspection.

The supported runtime is the `src/orion` package and `orion` command. Keep changes within existing module boundaries unless a concrete ownership need requires an abstraction.

## Requirements Through Phase 13

Use this precedence when older phase text, the deployed Pi, and the agreed target differ:

- **Phases 1–8: foundation retained.** Keep the package architecture, privacy rules, bounded local planner, SQLite memory/reminders, deterministic authorization, single service, confirmations, and resource limits. Historical implementation details are not extra requirements.
- **Phases 9–11: validation retained.** Clean install, service/reboot/recovery, audio/GPIO operation, reliability, acoustics, and measured performance remain release gates. Prior hardware or PTT measurements do not certify the new wake profile.
- **Phase 12: behavior retained.** Keep the deterministic fast paths and bounded compound-request planner, including grounding, confirmation, and tool policy. Do not broaden tools as part of the voice migration.
- **Phase 13: target supersedes old activation behavior.** The intended post-acceptance target has wakeword-only activation; GPIO23 is persistent microphone mute/unmute, and GPIO16 interrupts speech. Use custom local “Hey Orion” assets, VAD before selected local STT, and explicit local TTS/STT adapters. Keep online factual tools disabled by default. Until the candidate assets and Pi acceptance gates pass, retain legacy/VOSK as the supported staging and rollback profile; do not remove it during the migration.
- **Temporary migration compatibility.** Do not preserve PTT mode, old wake phrases, automatic provider fallback, or duplicate runtime choices as permanent product requirements. Keep the legacy/VOSK path only until wakeword acceptance and rollback are complete, then remove it only through a separately accepted change. Preserve user data and the deterministic/privacy contracts above.

“Implemented” means local code/tests; it does not mean deployed or Pi-verified. Promote the Phase 13 profile only after its real artifacts are provisioned and the Pi gates pass.

## Foundation (Phases 1–8)

The implemented foundation provides a modular Python package, validated settings, local model routing, bounded tool planning, SQLite memory/reminders, privacy-safe audio capture, VAD, local STT/TTS, and one persistent voice/reminder service. Deterministic requests bypass the model; external factual tools are policy-gated; raw microphone audio is not persisted. Local tests cover orchestration, confirmations, cancellation, and resource cleanup. Older phase-by-phase design history is intentionally omitted; remaining target acceptance is tracked below.

## Pi Acceptance Gates (Phases 9–11)

- **Installation:** prove a clean Raspberry Pi OS Lite 64-bit installation with the locked Python 3.12 voice environment, service account, GPIO/audio permissions, provisioned models, and writable SQLite. Existing-Pi dependency checks do not prove a clean-image install.
- **Deployed pre-migration baseline:** Raspberry Pi 5 8 GB, Debian 13/aarch64, AIRHUG 01 via ALSA `default`, Python 3.12 via uv, Ollama `qwen3:1.7b`, VOSK, and KittenTTS. Direct AIRHUG accepts 48 kHz only; ORION requires 16 kHz capture and 24 kHz playback. GPIO16 is speech interruption.
- **Service/recovery:** systemd and the dedicated account are installed. Verify reboot startup, reminder recovery/retry/recurrence, clean shutdown, and exclusive scheduler ownership.
- **Reliability/performance:** initial captures exist. Still verify cold start, repeated turns, a 30-minute soak, thermal/throttle/swap behavior, acoustics, playback leakage, and latency/resource limits. Publish only Pi-measured results.

### Phase 12: Bounded Hybrid Routing and Tool Planning - Implemented; Pi Acceptance Pending

Deterministic routes handle clear local requests; the bounded planner handles compound or incomplete requests. Tool authorization, confirmation, argument grounding, online policy, and resource limits remain deterministic. Regression coverage includes compound reminders, clarification, rejected actions, and grounded factual queries. Avoid expanding planner scope without a demonstrated need.

## Offline Stack Migration: Phases 13-18

### Current Behavior Versus Target Requirements

- Reuse the local model, deterministic router/planner, SQLite, reminders, network policy, and bounded 16 kHz audio interfaces. Phase 13 separates wake activation, VAD, and transcription; Phases 14–15 add streaming and Piper.
- Recent-turn memory and reminders already exist. Summaries, general tasks, broader device state, and embeddings are deferred to Phases 17–18.
- Factual network tools are default-off and remain constrained by exact-host HTTPS policy.

### Confirmed Target Profile and Controls

- Preserve Phase 12. Implement the runtime migration as Phases 13-16; promote code and deployment defaults together after Pi validation.
- Use Qwen3.5-2B post-trained weights converted to GGUF `Q4_K_M` with a pinned compatible `llama.cpp` build. Run text-only without a vision projector. Use non-thinking mode initially; do not assume Qwen3 `/nothink` or Ollama-specific flags apply to this runtime.
- Use `whisper.cpp` with local Whisper `base.en` for the initial English profile. Allow `base` selection for a separately validated multilingual profile.
- Use Piper with local medium-quality English ONNX/config assets. `en_US-lessac-medium` is an initial candidate, subject to voice selection and license verification. Use the voice's metadata sample rate and preserve hardware playback resampling.
- Use openWakeWord with custom "Hey Orion" assets. Provision or train the model on a development machine, not the Pi. Package the custom model and feature-extractor assets locally, record hashes/licenses, and verify acoustic behavior. Missing assets block release; never silently substitute another phrase or download models at startup.
- **Interim development target (2026-10):** use the official upstream "Hey Jarvis" model only for an explicitly selected, development-only wakeword trial while custom "Hey Orion" training is deferred. "Hey Jarvis" is not equivalent to "Hey Orion" and is not an automatic fallback; record its own source, license, revision, and hashes. The current validator requires the manifest phrase to be exactly "Hey Orion", so it must be updated to validate an explicitly declared phrase before this interim model can pass. Keep production and the Pi on legacy/VOSK; resume the custom "Hey Orion" work before Phase 13 promotion.
- Remap the existing GPIO23 button to debounced microphone mute/unmute without rewiring. Wake word becomes the only voice activation mode; no push-to-talk fallback remains in the target profile. GPIO16 retains speech interruption and cancels remaining streamed generation/output.
- Persist microphone mute across restart/reboot. Mute stops capture, cancels/discards active input, clears audio/detector state, and prevents stale transcripts from dispatching. Speaker and reminders remain active; already-dispatched actions are not undone. Read mute state before opening capture. A new installation starts unmuted; unreadable/corrupt state fails closed. Report persistence errors without claiming a successful durable unmute.
- Preserve the existing RGB LED wiring. `MUTED` takes precedence over normal status transitions. Privacy mute and playback suppression are separate states. Listen continuously while unmuted, except intentional playback suppression to avoid self-activation; voice barge-in/echo cancellation is out of scope.
- The target profile uses the selected local wake, STT, LLM, and TTS artifacts only. Missing artifacts fail clearly; no provider or wake-phrase fallback is allowed.
- Continue using SQLite and compact relevant retrieval. Broader memory features remain deferred; the small persistent microphone-control record is the immediate device-state exception.

### Phase 13: Configurable Adapters and Wake-Word Input - Implemented Locally; Artifacts and Pi Acceptance Pending

Progress checkpoint (2026-10-06): Phase 13 adapters are implemented locally. Source and Pi remain on legacy/VOSK until a complete wake asset set is verified. Explicit wakeword selection requires local ONNX files, a provenance/license/checksum manifest, openWakeWord 0.6.0, exact tensor contracts, and compatibility smoke checks before capture starts. The separate `--validate-wakeword-assets` command validates candidate files without changing the service environment. Regression tests cover missing/invalid assets, checksum mismatches, metadata gaps, runtime mismatch, and incompatible tensors. KittenTTS supports a validated local ONNX directory or a pre-provisioned Hub cache with Hugging Face forced offline. The pinned Whisper CLI/model stdin check passed on Pi; custom Hey Orion assets and remaining Pi acceptance (offline TTS, wake/mute/audio/reboot/soak checks) remain open. See `Phase13-Handoff.md` and `docs/phase13-artifacts.md`.

Implemented scope: explicit STT/TTS providers and local artifact paths; default-off weather/search; persistent fail-closed GPIO23 mute; independent 80 ms custom wake detection; VAD-before-STT; and bounded, cancellable CPU Whisper subprocesses using in-memory WAV stdin. Spoken confirmations require a fresh claimed wake turn. Keep configuration/route checks free of heavy model initialization.

Acceptance criteria:

- No wake, silence, or microphone mute activates STT/LLM; deterministic local requests still bypass the LLM.
- GPIO23 toggles once per press, mute survives restart, stale audio never resumes, and reminders/speaker remain functional while muted.
- Missing local assets fail clearly without network access or alternate-provider/phrase fallback. Captured audio never reaches disk.
- Mocked wake-to-VAD-to-transcript integration, cancellation, error cleanup, and existing privacy/tool tests pass.

### Phase 14: Qwen Runtime and Safe Streaming Generation - Implemented, Deployed, and Pi-Validated

Progress checkpoint (2026-10-08): Removed Ollama backend support. Added pinned-model context budgets, llama.cpp template/tokenizer budget checks, local SSE decoding and cancellation, typed stream events, tool-free direct streams, buffered tool planning before final generation, incomplete-turn labeling, and delayed memory suggestions. Tests cover arbitrary SSE byte splits, Unicode, malformed/truncated streams, reasoning suppression, tool gating, budgets, cancellation, and partial-turn recording. The Q4_K_M artifact is pinned and SHA-256 verified on Windows and Pi. Built and installed the pinned llama.cpp commit on Pi; model load, health, `/props`, `/apply-template`, `/tokenize`, and streamed completion passed smoke tests. The owner confirms the production Phase 14 deployment and Pi validation are complete. The deploy script selects this runtime by default, verifies model/build pins before stopping the service, changes only managed model settings, and backs up/restores environment and unit files. This session could not independently recheck the host because SSH timed out. Broader Phase 16 acceptance remains open; see `docs/phase14-artifacts.md`.

1. Select and checksum the actual Qwen3.5-2B `Q4_K_M` GGUF, not the Base variant, and pin/smoke-test its llama.cpp build and chat template. Start with 4096 context tokens, a 3072-token input budget, up to 512 answer tokens, and 512 tokens reserved for template/safety overhead. Validate formatted prompt plus output capacity through the runtime tokenizer/template path. Allow 8192 only after validation. Keep the planner cap separately bounded, initially 128 tokens.
2. Add typed streaming events and policy-preserving SSE transport with incremental UTF-8/event decoding, response bounds, finish/`[DONE]` handling, usage/timings, absolute deadlines, cancellation, and connection cleanup. Retain `complete()` for buffered consumers. Reject remote endpoints, redirects, and hidden cloud fallback.
3. Buffer and schema-validate tool/control decisions before execution. For streaming requests with authorized tools, finish bounded planning using tool/respond decisions, then stream a tools-disabled final answer. Tool-free reasoning can stream directly without an extra planner call. Preserve existing argument grounding, confirmations, deadlines, step limits, and repeated-action protection. Never speak planner JSON, tool arguments, memory suggestions, or reasoning channels.
4. Reuse `ContextBuilder`, bounded recent turns, memory retrieval, and orchestrator recording. Commit successful streamed text once. For interruption/failure, retain only delivered text with explicit incomplete status, not unsaid continuation or fabricated success. Never retry after audible output begins or save aborted-response memory suggestions automatically.

Acceptance criteria:

- Local streams decode correctly across arbitrary byte splits, malformed/truncated events, Unicode boundaries, and cancellation.
- Partial tool JSON never causes a side effect or reaches speech/history. All existing tool/privacy limits remain covered.
- Prompt/output budgets are enforced, structured output works with the selected artifact, and reasoning content never reaches user output/history.

Dependency: Phase 13 contracts and artifact provisioning precede this phase. Streaming transport and Piper adapter preparation can proceed in parallel; streamed orchestration depends on the transport.

### Phase 15: Piper and Streamed Voice Delivery - Planned

1. Select Piper through the shared factory for voice-check, voice service, and reminder service. Load local ONNX/config only for first speech, set CPU inference threads explicitly, and play iterable synthesized audio chunks using the existing resampling/interruptible playback patterns.
2. Extend orchestration with user-facing text events and bounded sentence/phrase accumulation. Handle punctuation, decimals, abbreviations, long punctuation-free text, Unicode, and final tails. Start synthesis/playback on the first complete phrase before generation finishes. Bound phrase size and text/audio queues; use backpressure.
3. Serialize reminder and conversational speech through one playback coordinator. Keep capture/detection suppressed across the entire speaking session, including gaps between chunks. GPIO16 cancels remaining generation/playback and flushes queues. Shutdown remains responsive while muted or blocked and joins workers/stops owned processes.
4. Remove unconditional STT/TTS warmup. Add configurable owned LLM/Piper idle eviction, initially 60 seconds and subject to measurement. Never evict an active turn or terminate an externally managed server. Whisper exits after each utterance; wake detection remains resident. Keep one LLM backend active.

Acceptance criteria:

- Fake-stream/audio tests prove first playback occurs before generation completes, without duplicate speech or unbounded buffering.
- Interruptions discard queued output; no stale chunks resume. Playback suppression prevents self-transcription between phrases.
- Reminder speech remains serialized and recoverable; adapter failures and shutdown leave no owned processes/workers active.

Dependency: Piper adapter work can begin after Phase 13 contracts. Integrated streamed playback depends on Phase 14 and the Piper adapter.

### Phase 16: Metrics, Pi Validation, and Default Promotion - Planned

1. Add turn-correlated instrumentation while building Phases 13-15, then complete wake inference/detection timing, VAD end-of-speech, STT load/inference, LLM load/TTFT/tokens per second/total, first phrase, TTS synthesis/first playback, speech-end-to-first-audio, and total response completion metrics. Exclude indefinite wake waiting from processing latency. Use backend token usage/timings, not character counts as tokens.
2. Extend the Phase 11 capture script to identify selected runtimes and owned process trees. Sample RSS/CPU, available RAM, CPU temperature, throttling, and swap at bounded intervals. Unsupported sensors report unavailable, not zero. Performance logs exclude audio/transcript/reasoning content by default.
3. Pin builds, dependencies, model hashes, and licenses. Verify Piper distribution requirements and individual voice/wake-model licenses. Compare thread settings 1, 2, 3, and 4 under combined LLM/TTS load; cap runtime pools to avoid oversubscription. Record identical prompts/audio, context, quantization, cooling, power, and ambient conditions.
4. Complete disconnected cold boot, idle/cold/repeated-turn captures, a 30-minute soak, reboot, reminder recovery, wake false accepts/rejects, playback leakage, mute persistence, and output-rate compatibility on the Pi. Compare explicit baseline/candidate profiles one component at a time before combined load. Reuse Phase 9-11 gates rather than declaring them complete from laptop tests.
5. Prove installation on a fresh Raspberry Pi OS Lite 64-bit image with the unprivileged systemd account, correct GPIO/audio permissions, writable SQLite, and read-only provisioned models. No runtime downloads, GPU dependencies, or swap tuning are introduced.
6. Promote `Settings`, deployment environment, dependency profiles, README, this plan, and issue status together after acceptance. Preserve existing user data. If a gate fails, leave the unaccepted profile disabled and report the blocked criterion.

Acceptance criteria:

- The requested roughly 2-3 GB LLM RSS is a benchmark target, not a verified guarantee. Proposed release guard: at least 1 GiB available system RAM throughout peak/soak workloads, no swap growth or thermal throttling, bounded worker/process counts, no stuck audio, and no missed/duplicate reminder delivery.
- The complete provisioned workflow works without external network access. Weather/search refuse operation until explicitly opted in.
- Measured latency, answer quality, wake accuracy, native build compatibility, and release limits are published. No numerical latency promise is invented from model size or laptop mocks.

Dependency: instrumentation starts during adapter implementation; integrated measurements and promotion require Phases 13-15. Independent laptop implementation may proceed while Phase 9-11 hardware gates remain open. Default promotion is the final migration step.

### Phase 17: Broader Persistent Memory - Deferred

Define concrete use cases and acceptance before implementation. Reuse SQLite preferences/facts/reminders and FTS5. Add bounded conversation summaries, general task storage/tools, and allowlisted non-secret device/configuration state through additive versioned schema changes with rollback proof. Summaries obey Incognito/sensitive-data policy, retain provenance, and never override authoritative facts. Keep runtime configuration in `Settings` unless a specific feature needs persistence; never store secrets or the entire conversation/database in prompts.

Acceptance: existing data/readers remain usable, summaries/tasks/state have focused tests, retrieval stays bounded, and privacy/deletion rules survive migration. Do not start automatically after Phase 16.

### Phase 18: Optional Semantic Retrieval - Deferred

Require documented lexical-retrieval failures before adding embeddings. Select and pin a small local CPU-friendly model using Pi RAM/latency and retrieval-quality measurements. Store vectors and their model version/dimensions in SQLite initially. Generate only during memory save/search, bound input/results, honor deletion and Incognito, and retain lexical fallback. No separate vector service, continuous embedding activity, or unsolicited background re-embedding is required. Embeddings remain disabled by default.

### Migration Verification and Handoff

- Extend `config.py`, `core/contracts.py`, and `app.py` for provider selection, controls, streaming, and owned lifecycle. Reuse the audio capture/VAD/gate/playback code, `llm/runtime.py`, agent/orchestrator/context, SQLite managers, tools factory, and network policy. Add focused adapter/coordinator modules only where ownership requires them.
- Add focused regressions in the existing config/audio/voice-service tests and runtime/agent-context/tools/memory/reminder tests. Run `python -m pytest tests/test_config.py tests/test_audio.py tests/test_voice_service.py -q` for input/control work and `python -m pytest tests/test_router_runtime.py tests/test_agent_context.py tests/test_tools_orchestration.py -q` for generation/planning. Use the selected interpreter or `scripts/test-local.ps1`; run the full suite before deployment.
- Test offline gates, missing/corrupt artifacts, Whisper stdin without output files, subprocess cancellation, persistent mute failures, malformed SSE, tool authorization, Incognito, bounded streaming queues, reminder serialization, and shutdown cleanup. Inspect SQLite and prove existing-reader rollback without destructive migrations.
- Preserve recorded Phase 9-11 baseline evidence and update README, deployment files, dependencies/locks, capture scripts, and Issues.md only as the corresponding implementation/validation lands.
- Stop the migration after Phase 16 acceptance and documented promotion. Artifact, licensing, or hardware blockers leave the candidate opt-in; broader memory and semantic retrieval need separate approval.

## Verification Checklist

### Local development checks

- Simple local requests do not initialize or call the local model.
- Reasoning requests use only the configured local model endpoint.
- Cloud model endpoints are rejected; factual online tools are HTTPS and exact-host allowlisted.
- Audio buffers are bounded, and no microphone recording remains on disk after success or failure.
- Agent context, tool result, step count, per-tool time, and total task time stay within configured bounds.
- Failed, malformed, repeated, or timed-out model/tool actions do not cause unapproved side effects.
- Memory writes obey sensitive-data policy and Incognito behavior; episode deletion is unambiguous.
- Reminders persist across restart and handle timezone, recurrence, cancellation, missed events, and delivery retry.
- The complete fake-audio integration path exercises turn ordering, errors, and resource cleanup.

### Raspberry Pi release checks

- Clean installation succeeds on the documented OS/Python/dependency profile.
- Microphone, selected STT/TTS adapters, speaker, GPIO, local model runtime, SQLite, and systemd behavior are verified on the target setup; baseline evidence does not verify replacement runtimes.
- Speech playback cannot be transcribed as new user input; VAD works in the intended acoustic environment.
- Voice and reminder service run under the documented unprivileged account and recover across reboot/restart.
- Peak memory, CPU, latency, temperature, throttling, and sustained-session behavior are recorded.
- No Pi compatibility, performance, or model recommendation is described as verified without that evidence.

## Configuration and Deployment Decisions

Keep validated configuration centralized. Before promotion, record supported OS/runtime versions, artifact hashes/licenses, service permissions, and measured hardware limits. The tested Pi uses ALSA `default`; direct AIRHUG only accepts 48 kHz, while ORION requires 16 kHz capture and 24 kHz playback. Treat other performance claims as unverified until measured.

## Confirmed Product Decisions

- Raspberry Pi 5 8 GB is the release target. Run one local LLM backend; reject cloud endpoints.
- Keep raw audio off disk, recent context bounded, and persistent memory/reminders in SQLite.
- Keep factual network tools opt-in; weather/search receive only explicit user queries.
- Target interaction is custom “Hey Orion” wake activation, GPIO23 persistent microphone mute/unmute, and GPIO16 speech interruption. Mute leaves reminders and speaker output available.
- Phase 16 promotes the validated Qwen3.5-2B `Q4_K_M`/llama.cpp, Whisper `base.en`/whisper.cpp, and Piper profile. Do not introduce fallback providers.
- Defer summaries, general tasks, broader device state, and embeddings to Phases 17–18.

## Known Limits

- Custom “Hey Orion” assets are absent. The wake profile fails closed until they are provisioned; wake acoustics and mute persistence are not Pi-verified.
- A pinned Whisper CLI/model stdin check passed on Pi; full ORION integration, local/offline TTS, reboot recovery, and sustained performance remain unverified. Phase 14 is implemented, deployed, and Pi-validated; Phase 15 is not implemented.
- Incognito resets on process restart. SearXNG remains unavailable until an exact HTTPS host is configured.

## Immediate Milestone and Handoff

Phase 13 source defaults to legacy/VOSK. The pinned Whisper model and CLI are installed and stdin-tested on Pi. Next: provision licensed custom wake assets, deploy validator-capable source while preserving VOSK, and validate assets before backing up and changing `/etc/orion/orion.env`. Then run candidate acceptance gates. Implement Phases 14–16 in order; do not start Phases 17–18 until Phase 16 acceptance.
