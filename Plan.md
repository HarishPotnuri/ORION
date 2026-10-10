# ORION Implementation Plan

## Goal

Build ORION as a privacy-first, offline-by-default personal voice assistant for Raspberry Pi 5 with 8 GB RAM and Raspberry Pi OS Lite 64-bit. The target uses wakeword-only activation, local speech recognition, streamed local responses, persistent reminders, and user-controlled memory.

The foundation and Phase 12 planner are implemented. Phase 14 source and a verified Qwen3.5-2B GGUF are local. The owner confirms Phase 14 is deployed and validated on Pi as of 2026-10-08 and Phase 15 is completed and Pi-verified as of 2026-10-09. A trained Hey Orion ONNX candidate now exists locally and passes structural CPU inference checks; its provenance manifest and Pi acoustic acceptance remain open. Phase 16 covers tool and conversation fixes. Phases 9–13 hardware acceptance and Phase 17 release gates stay separate; broader memory/semantic retrieval remain deferred.

## Product and Engineering Principles

- Keep model inference local; no cloud fallback or second routing model.
- Never persist raw microphone audio; bound audio, context, tools, and task time.
- Keep factual network tools opt-in, allowlisted, and scoped to the explicit query.
- Keep authorization and side effects deterministic; route clear local actions directly and compound requests through the bounded planner.
- Treat Pi compatibility and performance as unverified until measured on target hardware.

## Current Status (2026-10-09)

Foundation Phases 1–8 and Phase 12 are implemented. Phase 14 source, pinned Qwen3.5-2B artifact, deployment, and Pi validation are reported complete by the owner. The owner confirms Phase 15 completed and Pi-verified. The deploy script uses Phase 14 by default pending Phase 17 promotion. This workstation has not independently rechecked the live Pi. Phase 13 wake assets, Phase 16 behavior fixes, and Phase 17 release gates remain open.

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

## Offline Stack Migration: Phases 13-19

### Current Behavior Versus Target Requirements

- Reuse the local model, deterministic router/planner, SQLite, reminders, network policy, and bounded 16 kHz audio interfaces. Phase 13 separates wake activation, VAD, and transcription; Phases 14–15 add streaming and Piper.
- Recent-turn memory and reminders already exist. Summaries, general tasks, broader device state, and embeddings are deferred to Phases 18–19.
- Factual network tools are default-off and remain constrained by exact-host HTTPS policy.

### Confirmed Target Profile and Controls

- Preserve Phase 12. Implement the runtime migration and behavior fixes as Phases 13-16; promote code and deployment defaults together after Phase 17 Pi validation.
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

Progress checkpoint (2026-10-09): Phase 13 adapters are implemented locally. A trained Hey Orion classifier bundle and shared features now exist locally; ONNX structure and CPU inference pass with openWakeWord 0.6.0. Source and Pi remain on legacy/VOSK until the training-produced provenance/license/checksum manifest is available and Pi acoustic acceptance passes. Explicit wakeword activation requires local ONNX files, manifest, exact tensor contracts, and compatibility smoke checks before capture starts. The separate `--validate-wakeword-assets` command validates candidate files without changing service environment. Regression tests cover missing/invalid assets, checksum mismatches, metadata gaps, runtime mismatch, and incompatible tensors. KittenTTS supports a validated local ONNX directory or a pre-provisioned Hub cache with Hugging Face forced offline. The pinned Whisper CLI/model stdin check passed on Pi; wake/mute/audio/reboot/soak checks remain open. See `Phase13-Handoff.md` and `docs/phase13-artifacts.md`.

Implemented scope: explicit STT/TTS providers and local artifact paths; default-off weather/search; persistent fail-closed GPIO23 mute; independent 80 ms custom wake detection; VAD-before-STT; and bounded, cancellable CPU Whisper subprocesses using in-memory WAV stdin. Spoken confirmations require a fresh claimed wake turn. Keep configuration/route checks free of heavy model initialization.

Acceptance criteria:

- No wake, silence, or microphone mute activates STT/LLM; deterministic local requests still bypass the LLM.
- GPIO23 toggles once per press, mute survives restart, stale audio never resumes, and reminders/speaker remain functional while muted.
- Missing local assets fail clearly without network access or alternate-provider/phrase fallback. Captured audio never reaches disk.
- Mocked wake-to-VAD-to-transcript integration, cancellation, error cleanup, and existing privacy/tool tests pass.

### Phase 14: Qwen Runtime and Safe Streaming Generation - Implemented, Deployed, and Pi-Validated

Progress checkpoint (2026-10-09): Removed Ollama backend support. Added pinned-model context budgets, llama.cpp template/tokenizer budget checks, local SSE decoding and cancellation, typed stream events, tool-free direct streams, buffered tool planning before final generation, incomplete-turn labeling, and delayed memory suggestions. Tests cover arbitrary SSE byte splits, Unicode, malformed/truncated streams, reasoning suppression, tool gating, budgets, cancellation, and partial-turn recording. The owner-approved operational Q4_K_M file matches between Pi and local depot at SHA-256 `a8b50de00b780c0f3c16f07cb041a1ecd678b6e22fd8cde91e57fe4b5031523b`; this differs from the documented immutable upstream revision fingerprint, and provenance difference remains unresolved. Built and installed the pinned llama.cpp commit on Pi; owner confirms model runtime works. The deploy script verifies the accepted operational checksum before stopping the service. Broader Phase 16 acceptance remains open; see `docs/phase14-artifacts.md`.

1. Select and checksum the actual Qwen3.5-2B `Q4_K_M` GGUF, not the Base variant, and pin/smoke-test its llama.cpp build and chat template. Start with 4096 context tokens, a 3072-token input budget, up to 512 answer tokens, and 512 tokens reserved for template/safety overhead. Validate formatted prompt plus output capacity through the runtime tokenizer/template path. Allow 8192 only after validation. Keep the planner cap separately bounded, initially 128 tokens.
2. Add typed streaming events and policy-preserving SSE transport with incremental UTF-8/event decoding, response bounds, finish/`[DONE]` handling, usage/timings, absolute deadlines, cancellation, and connection cleanup. Retain `complete()` for buffered consumers. Reject remote endpoints, redirects, and hidden cloud fallback.
3. Buffer and schema-validate tool/control decisions before execution. For streaming requests with authorized tools, finish bounded planning using tool/respond decisions, then stream a tools-disabled final answer. Tool-free reasoning can stream directly without an extra planner call. Preserve existing argument grounding, confirmations, deadlines, step limits, and repeated-action protection. Never speak planner JSON, tool arguments, memory suggestions, or reasoning channels.
4. Reuse `ContextBuilder`, bounded recent turns, memory retrieval, and orchestrator recording. Commit successful streamed text once. For interruption/failure, retain only delivered text with explicit incomplete status, not unsaid continuation or fabricated success. Never retry after audible output begins or save aborted-response memory suggestions automatically.

Acceptance criteria:

- Local streams decode correctly across arbitrary byte splits, malformed/truncated events, Unicode boundaries, and cancellation.
- Partial tool JSON never causes a side effect or reaches speech/history. All existing tool/privacy limits remain covered.
- Prompt/output budgets are enforced, structured output works with the selected artifact, and reasoning content never reaches user output/history.

Dependency: Phase 13 contracts and artifact provisioning precede this phase. Streaming transport and Piper adapter preparation can proceed in parallel; streamed orchestration depends on the transport.

### Phase 15: Offline Model Depot, Piper, and Streamed Voice Delivery - Completed and Pi-Verified

Progress checkpoint (2026-10-09): Added a local model registry for the adjacent Windows `ORION-MODELS` depot and configurable Pi model root. Stable IDs resolve the five configured GGUFs, three Whisper models, two Piper voices, BGE ONNX, and Hey Orion/Jarvis wake assets. `orion-model` lists/current/selects/validates models and runs fixed sequential LLM benchmarks; selection JSON is stored beside SQLite, and applying an LLM change requires service restart. The llama.cpp provider exposes load/unload/info/generate/stream and unload-before-load switching; only one owned LLM process is loaded at once. Whisper selection also works with legacy PTT. Missing Hey Orion assets try Hey Jarvis, then provisioned VOSK/PTT. In wakeword mode, detection remains active during playback; a barge-in wake cancels generation/playback, then captures the command after playback suppression ends. Legacy PTT remains suppressed. BGE has a lazy provider interface only; semantic retrieval remains deferred. Existing streamed Piper delivery and idle eviction remain. Local regression tests pass. The owner confirms Phase 15 deployment, artifact and license checks, and Pi hardware acceptance completed successfully. BGE semantic retrieval remains deferred; remaining Phase 13 and Phase 16 gates stay separate.

1. Select Piper through the shared factory for voice-check, voice service, and reminder service. Load local ONNX/config only for first speech, set CPU inference threads explicitly, and play iterable synthesized audio chunks using the existing resampling/interruptible playback patterns.
2. Extend orchestration with user-facing text events and bounded sentence/phrase accumulation. Handle punctuation, decimals, abbreviations, long punctuation-free text, Unicode, and final tails. Start synthesis/playback on the first complete phrase before generation finishes. Bound phrase size and synthesized audio chunks; synchronous consumption provides backpressure without queues.
3. Serialize reminder and conversational speech through one playback coordinator. Keep capture/detection suppressed across the entire speaking session, including gaps between chunks. GPIO16 cancels remaining generation/playback and discards unsaid text/audio chunks. Shutdown remains responsive while muted or blocked and joins workers/stops owned processes.
4. Remove unconditional STT/TTS warmup. Add configurable owned LLM/Piper idle eviction, initially 60 seconds and subject to measurement. Never evict an active turn or terminate an externally managed server. Whisper exits after each utterance; wake detection remains resident. Keep one LLM backend active.

Acceptance criteria:

- Fake-stream/audio tests prove first playback occurs before generation completes, without duplicate speech or unbounded buffering.
- GPIO16 and wakeword barge-in discard unsaid text/audio; no stale output resumes. Wakeword mode captures only after interrupted playback ends; legacy PTT remains suppressed.
- Playback acoustics do not trigger repeated false wake events on the Pi; verify microphone leakage and command capture after barge-in.
- Reminder speech remains serialized and recoverable; adapter failures and shutdown leave no owned processes/workers active.
- Stable IDs resolve only local depot paths. Selection, validation, model switching, and sequential benchmark output do not download artifacts or load multiple LLMs.
- The `orion-model` benchmark records fixed prompt results, model size, load time, TTFT, generation rate/time/tokens, and available resource observations in JSON.
- Missing Hey Orion assets try Hey Jarvis and then the separately provisioned VOSK/PTT fallback; incomplete VOSK provisioning reports a clear startup error.
- BGE stays behind a configurable provider interface; vector retrieval remains disabled until model input/tokenizer compatibility is confirmed.

Dependency: Local inventory/selection uses existing settings and adapters. Pi promotion still depends on Phase 13 artifacts, Phase 14 runtime pins, artifact licensing, Phase 16 behavior acceptance, and Phase 17 release gates.

### Phase 16: Tool Reliability, Conversation Flow, Logging, and Prompt Optimization - Partially Implemented Locally; Pi Validation Pending

Progress checkpoint (2026-10-10): Added deterministic routing for “Tell me what time it is,” fixed reminder updates that lost a cancellation race, and added voice-turn logs with full accepted transcripts and generated answers, including in Incognito. TTS adapters report playback completion; streamed and buffered voice history keeps only confirmed delivered phrases and marks interrupted replies incomplete. Wakeword follow-up uses VAD, a renewable 10-second speech-start window, no overall session cap, and the existing utterance limit. GPIO16 interruption keeps the session active; reminder speech opens follow-up capture. Low-confidence VOSK word scores trigger a one-step, tool-free local correction suggestion that requires spoken confirmation; one unclear retry then closes the session. The current Whisper adapter exposes no confidence, so this correction path is unavailable when Whisper is selected. Local routing, factual-tool, reminder, context, voice, and TTS regression suites pass. Open-Meteo and SearXNG remain the existing providers; live availability and latency are not measured. Prompt changes are deferred until the selected model can provide formatted-token and workload comparisons. STT false-acceptance, correction quality, and latency remain Pi validation gates under Phase 17.

#### New Goals (2026-10-09)

1. **Repair current time, weather, search, news, and reminders.** The reported symptom is that ORION says it cannot complete the request; current-time commands also receive a cannot-answer response. The causes are not yet verified. Diagnose the existing deterministic time route, timezone configuration, tool routing, adapters, and result handling before changing providers. Current-time queries must use the local system clock and configured timezone without network access or LLM inference; do not generate a guessed time. The owner delegates selection of fast, suitable open-source providers for weather, search, and news; evaluate licensing, reliability, setup requirements, and measured latency rather than assuming the fastest option. Fix weather and web search, and provide an explicit news query path using the existing factual-tool policy and adapter patterns. Verify reminder creation, listing, cancellation, timezone handling, recurrence, restart recovery, and delivery. Keep network tools opt-in, HTTPS-only, exact-host allowlisted, and bounded; report missing configuration, unavailable providers, malformed results, and timeouts without inventing facts or claiming failed actions succeeded.
2. **Make transcription and response logs useful.** Always show the full voice transcript and full final user-facing LLM answer in normal logs; do not require debug mode or opt-in and do not truncate that text. The owner confirms this content logging also applies in Incognito: Incognito prevents memory persistence but does not hide conversation text from logs. Add turn-correlated events for capture, transcription, routing, LLM generation, and speech delivery, including completion, cancellation, failure, and relevant timings. Distinguish generated text from text actually delivered when speech is interrupted. Never log raw audio, configured credentials, hidden reasoning, or planner JSON. Conversation logs can contain sensitive spoken information. Log destinations and retention remain unconfirmed; do not introduce a separate file sink or retention policy without agreement.
3. **Remove noisy or useless logs.** Audit the voice and tool paths for duplicate events, per-frame output, repeated idle messages, and misleading success messages. Retain actionable warnings, errors, lifecycle events, and bounded performance metrics. Put low-level diagnostics behind debug logging and verify normal operation remains readable.
4. **Optimize the system prompt.** Improve answer accuracy, verbosity, personality, tool selection, and latency. Use a neutral, concise response style, with details when requested. Consolidate redundant instructions and keep the prompt concise for the selected local model and chat template. Preserve tool authorization, factual grounding, confirmation requirements, privacy rules, and output constraints. Measure formatted prompt tokens and compare fixed conversational and tool workloads before and after the change; do not trade correctness for a smaller prompt.
5. **Repair conversational history and context interpretation.** The owner reports misunderstood context and excessive preference for earlier conversation, in addition to suspecting missing context. Verify prompt construction, retrieval, turn recording, and interpretation before assigning a cause. Cover both missing relevant history and overuse of irrelevant or mistaken history. Treat the current explicit request and user corrections as authoritative for current intent, subject to existing safety and tool policy; use earlier context only when relevant to a reference or an incomplete request. Detect topic changes and do not force a standalone query into the previous topic. Do not promote an assistant's earlier guess or a misrecognized transcript into a confirmed user fact. Clarify unresolved conflicting references instead of guessing. Reuse the existing context builder and turn recording. Preserve ordered, bounded recent turns across wakewords and follow-up timeouts; clear recent conversation context on explicit reset or service restart. This reset does not delete saved facts or reminders. Record each accepted turn once, exclude rejected noise and internal planner content, and mark interrupted replies incomplete with only delivered text retained. Cover streamed and buffered responses, context trimming, reset, Incognito, and failed turns without introducing broader persistent memory or summaries.
6. **Listen for follow-up queries after speech.** After a wake-initiated conversational response finishes playback, wait up to 10 seconds for speech to start without requiring another wakeword. If speech starts within the window, continue capture until VAD end-of-speech, subject to the existing bounded utterance limit. Renew the speech-start window after every conversational response. Start listening only after playback suppression ends and discard stale capture audio. Reject improper transcripts and noise before any LLM inference, using VAD and available STT quality signals; do not use an LLM as the rejection classifier. VAD alone does not prove user intent. Rejected input must not execute tools or enter history. Preserve deterministic authorization and confirmation requirements for accepted follow-ups. Confirm any overall session limit, GPIO16 behavior, reminder-triggered listening, and the response to unclear speech before implementation. Mute and shutdown must still stop capture and discard stale input.

7. **Evaluate general STT accuracy and interpretation recovery.** The owner reports multiple recognition mistakes; "five" transcribed as "file" is one example, not the scope of the fix. Investigate word substitutions, omissions, insertions, segmentation, names, numbers, times, and negation across representative voice requests. Verify capture, VAD boundaries, transcription, and interpretation before attributing failures to STT or reasoning. Evaluate available STT decoding controls and bounded interpretation against transcription accuracy, intent correctness, and latency. Use relevant conversational context as supporting evidence, not permission to rewrite clear current speech to match a previous topic. Preserve the original full transcript in logs and separately identify any interpreted correction. Avoid blanket word replacements and example-specific patches; legitimate uses of similar words must remain unchanged. Preserve negation and explicit user corrections. Do not silently guess names, reminder times, quantities, or action arguments when multiple interpretations are plausible; request clarification before a side effect. Keep noise and improper-transcript rejection before LLM inference. Using LLM reasoning to repair uncertain transcripts remains an open decision; clarify that boundary before adding such behavior.

Behavior acceptance criteria:

- Weather, search, and news have focused tests for successful grounded results, disabled policy, missing configuration, malformed responses, provider failure, and timeout. Reminder tests cover persistence, recurrence, cancellation, recovery, and no missed or duplicate delivery.
- Current-time tests use a controlled clock and configured timezone, cover common spoken phrasings and both buffered and streamed voice routes, and prove that no network or LLM call is needed.
- STT tests cover representative substitutions, omissions, insertions, utterance boundaries, names, numbers, times, and negation, including "five"/"file" as one regression. Include legitimate similar-word requests, short follow-up answers, user corrections, and ambiguous action arguments. Verify that interpreted corrections have supporting evidence, unrelated history does not bias current intent, rejected noise never reaches the LLM, and uncertain action arguments require clarification. Measure transcription accuracy, intent correctness, false correction rate, and Pi latency rather than assuming better reasoning fixes recognition.
- Normal logs show full voice transcripts and final user-facing LLM answers, including in Incognito, omit internal reasoning and configured credentials, and distinguish interrupted delivery from complete output. Normal idle operation does not produce repetitive log noise.
- Prompt changes preserve the selected runtime's single-leading-system-message template contract and input/output budgets. Fixed regression queries retain tool correctness and grounded responses, with before/after token counts recorded.
- Multi-turn tests cover relevant pronouns and references, standalone requests after unrelated conversation, topic changes, explicit corrections, conflicting references, and earlier assistant misunderstandings. Verify that current explicit intent is not overridden by stale history and that uncertain interpretations do not become confirmed facts. Also cover correct ordering, bounded trimming, duplicate prevention, interruption, failure, reset, and Incognito. Rejected background input never enters history.
- Fake-audio tests prove wake, first query, TTS completion, follow-up capture, query acceptance, and the agreed timeout behavior. Cover silence, improper transcripts, non-speech noise, background speech, playback leakage, short valid replies, mute, interruption, and shutdown. Rejected input causes no LLM inference, tool execution, or history writes. Measure false acceptance/rejection on the Pi; no filter is treated as infallible.

Dependency: reuse the existing tools, reminder manager, context builder, runtime, and voice coordinator from Phases 12-15. Resolve the open decisions above before implementing the affected behavior. Phase 16 does not perform default promotion; metrics, Pi validation, and promotion move to Phase 17.

### Phase 17: Metrics, Pi Validation, and Default Promotion - Planned

This phase retains the former Phase 16 release scope, separate from the new Phase 16 behavior work.

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

Dependency: instrumentation starts during adapter implementation; integrated measurements and promotion require Phases 13-16 and their behavior acceptance checks. Independent laptop implementation may proceed while Phase 9-11 hardware gates remain open. Default promotion is the final migration step.

### Phase 18: Broader Persistent Memory - Deferred

Define concrete use cases and acceptance before implementation. Reuse SQLite preferences/facts/reminders and FTS5. Add bounded conversation summaries, general task storage/tools, and allowlisted non-secret device/configuration state through additive versioned schema changes with rollback proof. Summaries obey Incognito/sensitive-data policy, retain provenance, and never override authoritative facts. Keep runtime configuration in `Settings` unless a specific feature needs persistence; never store secrets or the entire conversation/database in prompts.

Acceptance: existing data/readers remain usable, summaries/tasks/state have focused tests, retrieval stays bounded, and privacy/deletion rules survive migration. Do not start automatically after Phase 17.

### Phase 19: Optional Semantic Retrieval - Deferred

Require documented lexical-retrieval failures before adding embeddings. Select and pin a small local CPU-friendly model using Pi RAM/latency and retrieval-quality measurements. Store vectors and their model version/dimensions in SQLite initially. Generate only during memory save/search, bound input/results, honor deletion and Incognito, and retain lexical fallback. No separate vector service, continuous embedding activity, or unsolicited background re-embedding is required. Embeddings remain disabled by default.

### Migration Verification and Handoff

- Extend `config.py`, `core/contracts.py`, and `app.py` for provider selection, controls, streaming, and owned lifecycle. Reuse the audio capture/VAD/gate/playback code, `llm/runtime.py`, agent/orchestrator/context, SQLite managers, tools factory, and network policy. Add focused adapter/coordinator modules only where ownership requires them.
- Add focused regressions in the existing config/audio/voice-service tests and runtime/agent-context/tools/memory/reminder tests. Run `python -m pytest tests/test_config.py tests/test_audio.py tests/test_voice_service.py -q` for input/control work and `python -m pytest tests/test_router_runtime.py tests/test_agent_context.py tests/test_tools_orchestration.py -q` for generation/planning. Use the selected interpreter or `scripts/test-local.ps1`; run the full suite before deployment.
- Test offline gates, missing/corrupt artifacts, Whisper stdin without output files, subprocess cancellation, persistent mute failures, malformed SSE, tool authorization, Incognito, bounded streaming queues, reminder serialization, and shutdown cleanup. Inspect SQLite and prove existing-reader rollback without destructive migrations.
- Preserve recorded Phase 9-11 baseline evidence and update README, deployment files, dependencies/locks, capture scripts, and Issues.md only as the corresponding implementation/validation lands.
- Stop the migration after Phase 17 acceptance and documented promotion. Artifact, licensing, or hardware blockers leave the candidate opt-in; broader memory and semantic retrieval need separate approval.

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
- Target interaction is custom “Hey Orion” wake activation followed by a bounded conversational session with wakeword-free follow-up queries after speech. Outside an active session, voice input requires a wakeword. GPIO23 provides persistent microphone mute/unmute, and GPIO16 interrupts speech. Mute leaves reminders and speaker output available.
- Phase 17 promotes the validated Qwen3.5-2B `Q4_K_M`/llama.cpp, Whisper `base.en`/whisper.cpp, and Piper profile. Do not introduce fallback providers.
- Defer summaries, general tasks, broader device state, and embeddings to Phases 18–19.

## Known Limits

- Trained “Hey Orion” ONNX candidate and feature assets now exist locally. The wake profile remains gated on the training-produced manifest and Pi acoustic validation; wake acoustics and mute persistence are not Pi-verified.
- A pinned Whisper CLI/model stdin check passed on Pi. Phase 14 is implemented, deployed, and Pi-validated; the owner confirms Phase 15 is completed and Pi-verified. Remaining Phase 13 and Phase 16 gates are not implied complete.
- Incognito resets on process restart. SearXNG remains unavailable until an exact HTTPS host is configured.

## Immediate Milestone and Handoff

Phase 13 source defaults to legacy/VOSK. The pinned Whisper model and CLI are installed and stdin-tested on Pi. Phase 15 is completed and Pi-verified. Clarify and implement Phase 16 behavior fixes, then complete Phase 17 validation and promotion; keep production defaults unchanged until those release gates pass. Continue remaining Phase 13 gates independently. Do not start Phases 18–19 automatically after Phase 17 acceptance.
