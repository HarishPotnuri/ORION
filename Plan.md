# ORION Implementation Plan

## Goal

Build ORION as a privacy-first, local-first personal voice assistant for Raspberry Pi 5 with 8 GB RAM. The product should provide reliable push-to-talk interaction, local speech recognition, local model reasoning, local speech output, persistent reminders, and user-controlled memory.

The architecture foundation and voice-service integration are implemented in the repository. Remaining work is target validation: verify deployment and behavior on the Raspberry Pi, measure resource use, then decide whether further agent planning is warranted.

## Product and Engineering Principles

- Run all language-model inference locally. Do not add a cloud LLM fallback or a second model for routing.
- Keep deterministic routes for clear, low-risk tasks. Invoke the local model only when natural-language reasoning is needed.
- Never persist raw microphone audio. Keep short-term conversation context volatile and bounded.
- Keep online factual lookups opt-in, allowlisted, and limited to the requested query or location. Never send conversation history or model prompts to online providers.
- Bound audio queues, utterance length, model context, agent steps, tool time, and task time.
- Prefer one simple, long-running ORION service for voice interaction and reminder delivery; avoid unnecessary background services.
- Report hardware and performance support only for configurations actually verified on a Raspberry Pi.
- Use deterministic fast paths only for clear, complete actions. Send compound or incomplete requests to the bounded local planner.
- Keep all tool authorization, input validation, side effects, and online-data boundaries in deterministic code.
- Keep planner scope narrow. Add tool actions only when they solve a demonstrated user request.

## Current Status (September 2026)

Phases 1 through 8 are implemented in code. Typed CLI requests and the module-level tests provide a useful development path. Pi installation, audio/GPIO behavior, performance, and thermal behavior remain unverified.

Status terms in this plan:

- **Implemented:** code exists and has local test coverage where applicable.
- **Integrated:** components are connected in the intended application flow.
- **Pi-verified:** exercised on the stated Raspberry Pi, OS image, peripherals, and software versions.
- **Measured:** resource or latency numbers were collected on the Pi under a documented workload.

Do not promote a feature to Pi-verified or measured based on laptop tests or code inspection.

## Existing Architecture

```text
src/orion/
  app.py                 CLI entry point and current component lifecycle
  config.py              Central settings and validation
  core/
    router.py            Deterministic request classification
    orchestrator.py      Request dispatch and stateful command handling
    agent.py             Bounded model/tool controller
    context.py           Relevant prompt context and token budget
    policy.py            Local model and approved factual URL policy
    capabilities.py      Honest capability/degradation responses
    contracts.py         Interfaces between core and adapters
  audio/
    capture.py           Bounded microphone capture and audio gate
    wakeword.py          Optional openWakeWord adapter
    vad.py               Energy-based voice activity detection
    stt.py               Streaming VOSK transcription
    tts.py               Local KittenTTS generation and playback
  llm/
    runtime.py           Local llama.cpp and Ollama adapters
  memory/
    manager.py           Volatile turns, retrieval, and write policy
    policy.py            Durable-memory allowlist and privacy checks
    store.py              SQLite and FTS5 persistence
  reminders/
    manager.py           Reminder operations
    parser.py            Natural-language schedule parsing
    scheduler.py         Due and missed reminder delivery
    store.py              SQLite reminder persistence
  tools/
    factory.py           Default tool construction
    registry.py          Explicit tool registry
    time.py              Local time/date
    system.py            Local system status
    weather.py           Open-Meteo factual lookup
    web_search.py        Operator-configured SearXNG lookup
tests/                    Unit and orchestration regression tests
```

Adapt module boundaries when needed, but do not add empty modules or abstractions without a concrete integration need. The old `main.py`, `main_llama.py`, and `stt_lib.py` are historical/reference entry points; the supported runtime is the `orion` package and its console command.

## Implemented Phases

### Phase 1: Architecture Foundation — Implemented

- Modular `src/orion/` package, central validated settings, logging, contracts, and console entry point are in place.
- Optional dependency groups keep laptop tests independent from voice and Pi hardware packages.
- The older flat scripts are no longer the supported application architecture.

### Phase 2: Privacy-Safe Audio Components — Implemented in Code; Pi Verification Pending

- Microphone input uses bounded streaming chunks and does not write captured speech to a file.
- VOSK consumes chunks, with energy-based VAD, silence termination, pre-roll, and a maximum utterance duration.
- KittenTTS playback is in memory. `AudioGate` prevents microphone capture during speech output.
- Push-to-talk and optional wake-word input adapters exist. Wake-word mode defaults off.
- These adapters have local tests, but installability, audio-device selection, capture quality, wake-word behavior, and audio/TTS coexistence must be verified on the target Pi.

### Phase 3: Deterministic Router and Local Model — Implemented; Deployment Modes Need Validation

- The deterministic router classifies local commands, reminders, memory actions, factual lookups, and reasoning requests.
- One local model backend is selected through settings: llama.cpp or Ollama.
- Model endpoints are restricted to loopback. The managed llama.cpp server checks readiness and is configured for the local endpoint.
- Model startup is lazy for typed requests; simple deterministic commands do not need to load the model.
- Verify the intended Pi deployment mode for each supported backend, including whether ORION manages the process or connects to an already-running service.

### Phase 4: Bounded Agent and Context — Implemented

- `AgentController` accepts only registered tools, enforces step and elapsed-time limits, rejects repeated actions, and limits tool results.
- `ContextBuilder` includes the current request, bounded recent turns, selected memories, compact tool results, and task state while preserving a configured approximate token budget.
- The default token counter is a character-based estimate, not the selected model's tokenizer. Measure whether this is adequate on the target model; add exact token counting only if needed and practical.
- Tool adapters receive timeouts and cancellation events. Python cannot forcibly terminate an adapter that ignores them, so every adapter must honor its limits.
- Do not store or expose hidden chain-of-thought.

### Phase 5: SQLite Memory — Implemented for Typed Requests

- Recent turns are held in a bounded in-process buffer; selected episodes and durable facts/preferences/decisions are stored in SQLite with FTS5 retrieval.
- Explicit save requests are categorized deterministically. Model-suggested memories pass deterministic category, length, and sensitive-data checks.
- Incognito blocks durable writes while allowing retrieval. Its state is process-local and resets on restart; document and revisit this behavior when the persistent voice service is implemented.
- Episode deletion occurs only for an unambiguous match; ambiguous matches ask the user to choose.
- Raw transcripts and every-turn history are not persisted by default.
- Ensure the final voice-service behavior for Incognito and volatile context matches the product promise, including what happens to in-process turns and on service restart.

### Phase 6: Persistent Reminders — Implemented for Typed Requests and Standalone Scheduler

- Reminders share the configured SQLite database, store due times in UTC with IANA timezone metadata, and support one-time, daily, and weekly schedules.
- The parser supports relative durations, today/tomorrow, weekdays, and recurring schedules; it rejects missing or past times rather than guessing.
- The scheduler recovers interrupted deliveries, processes overdue reminders at startup, retries failures, and advances recurring reminders after successful delivery.
- The standalone `--run-reminders` mode connects notifications to local speech and a GPIO status LED.
- Integrate reminder scheduling into the final persistent assistant service so reminders can be created and delivered without launching separate app modes. Preserve missed-reminder catch-up and verify behavior across service restarts.

### Phase 7: Tools and Capability Policy — Implemented for Typed Requests

- Local time/date and system status tools are available.
- Weather uses Open-Meteo; web/news search uses SearXNG and is disabled until an HTTPS instance and exact allowed host are configured.
- Online transport disables proxies and redirects, validates HTTPS and exact allowlisted hostnames, applies timeouts, and caps response size.
- Online services receive only the explicitly requested query or location. Compact factual results are passed to the local model for interpretation.
- Capability policy refuses selected broad, long-horizon analysis with an honest narrower alternative.
- Live provider availability and Pi networking have not been validated. Keep online lookup failures bounded and clear to the user.

### Phase 8: End-to-End Voice Assistant — Implemented; Pi Verification Pending

Connect the existing adapters and services into one persistent voice interaction. Push-to-talk is the default; wake-word input is optional and selected through `ORION_WAKE_WORD_ENABLED`.

1. Add a long-running assistant lifecycle that waits for push-to-talk, captures and transcribes one utterance, passes the transcript to `RequestOrchestrator`, and speaks the final response through `KittenSpeechOutput`.
2. Keep the process alive across turns so the configured model, STT model, TTS engine, SQLite stores, and volatile conversation context can be reused instead of reloaded for every turn.
3. Preserve deterministic fast paths: simple commands, local tools, reminder actions, and memory actions must not initialize or call the LLM unless their defined handling requires it.
4. Route typed and spoken requests through the same orchestration logic. Keep the typed CLI available for diagnostics and laptop development.
5. Set explicit hardware status transitions for idle, listening, processing, speaking, muted, and error where supported by the connected hardware. Make cleanup reliable on normal exit, exceptions, and SIGTERM.
6. Handle empty/unclear transcripts, audio overflow, missing devices, STT/model/TTS errors, and user interruption without corrupting reminder or memory state. Retry a failed local model call once, then speak a concise error and return to listening.
7. Support barge-in during TTS through the configured push-to-talk button: stop playback on button press, then capture while the button remains pressed. In wake-word mode, button press stops playback and the next turn still requires the configured wake word. Keep microphone capture gated during playback. Ensure audio streams, button callbacks, worker threads, GPIO devices, model processes, and database connections close on shutdown.
8. Require spoken confirmation before cancelling a reminder or deleting a saved memory. Repeat the selected target, accept only a clear affirmative response within 15 seconds, and make no change on rejection, timeout, or unclear speech.
9. Add fake-based integration tests for the complete turn lifecycle, ensuring audio → transcript → orchestrator → response → speech ordering and testing interruption, confirmation, retry, failures, and cleanup.
10. Do not add a second routing model or multi-agent behavior in this phase.

Acceptance criteria:

- A simulated push-to-talk turn transcribes once, dispatches once, speaks once, supports barge-in, and releases resources after shutdown.
- A local deterministic command does not initialize the LLM.
- Audio overflow or an adapter failure discards the affected utterance and does not leave capture or playback stuck.
- Reminder cancellation and memory deletion require clear spoken confirmation within 15 seconds; unclear, rejected, or timed-out confirmation causes no mutation.
- A failed local model call is retried once; a second failure produces a short spoken error and returns the assistant to listening.
- The application can run multiple turns in one process without unbounded audio or conversation growth.
- Incognito, memory retrieval, reminder operations, and reminder delivery preserve their current persistence/privacy rules in the long-running process.

Implementation status: `orion --run-voice` runs repeated turns through shared audio input, orchestration, local speech output, and reminder scheduling. The service reuses model and SQLite managers across turns; Incognito resets on restart. Push-to-talk remains default; optional openWakeWord consumes its activation chunk before transcription. Push-to-talk interrupts TTS; reminder cancellation and episode deletion require a clear spoken confirmation in voice mode. The agent retries one failed or malformed model response within the task deadline. Error feedback, PWM LED states, signal handling, and cleanup have fake-based local tests. Raspberry Pi installation, real audio/GPIO behavior, acoustics, performance, and sustained operation remain unverified.

## Remaining Work

### Phase 9: Raspberry Pi Installation and Compatibility Gate

Prove the supported installation on a clean Raspberry Pi 5 8 GB before adding more runtime features.

1. Select and record the exact 64-bit Raspberry Pi OS image/release, Python version, package manager, hardware peripherals, and installation steps.
2. Validate Python-version compatibility for the core package and every optional dependency used on Pi: VOSK, sounddevice/PortAudio, KittenTTS and its ML dependencies, GPIO library, wake-word packages if enabled, and selected model runtime.
3. Resolve the current Python constraint (`>=3.11,<3.12`) based on successful dependency installation and testing on the chosen OS image. Do not widen it based only on laptop compatibility.
4. Provide reproducible Pi install instructions and separate dependency profiles for core typed use, voice, wake-word, and GPIO if that separation remains useful.
5. Verify audio-device enumeration/selection, microphone capture, speaker playback, GPIO access, local database directory permissions, and model file/service paths under the intended non-root service account.
6. Document model backend modes clearly: externally managed local service versus an ORION-managed llama.cpp server. Do not claim both modes work until each configured path is verified.
7. Verify the selected local model returns the required structured JSON reliably through the configured backend. Exercise invalid/truncated model output and give a clear failure response without performing unintended tool actions.
8. Keep the default inference endpoint loopback-only. Do not expose a model server on the LAN as a deployment shortcut.

Acceptance criteria:

- A fresh supported Pi image can install the documented profile without undocumented manual fixes.
- The documented command starts the assistant under the intended user account, opens the intended microphone/speaker and GPIO devices, and reaches a local model when reasoning is requested.
- Core deterministic requests still work if the model runtime is stopped or unavailable.
- All selected dependency and model versions are recorded so a later update can be compared reproducibly.

Current validation evidence: Laptop environment is Windows x64, Python 3.11.9; editable package installation and `pip check` pass; the full suite has 123 passing tests and 30 subtests. Configuration and a deterministic local-time request pass. Ollama 0.34.4 with `qwen3:1.7b` returned valid ORION responses in 3/3 consecutive laptop requests. Malformed/truncated model output has regression coverage proving no tool executes before a valid agent decision. This does not verify Linux ARM64 wheels, Raspberry Pi OS installation, audio devices, GPIO permissions, local model startup on Pi, or performance. The exact Pi OS image and release must be recorded from the image used for the first hardware run; do not infer it from the laptop.

### Phase 10: Integrated Service, Controls, and Operations

1. Provide one native systemd service for the persistent voice assistant and reminder scheduler. Avoid a separate reminder process/database unless testing demonstrates a concrete need.
2. Add service configuration with a dedicated unprivileged account, explicit working directory, restart policy, orderly shutdown, and useful journal logging. Do not run ORION as root merely to access GPIO/audio; document the required groups/permissions.
3. Reconcile the existing standalone `--run-reminders` mode with the integrated service. Keep it only if it has a documented operational use and does not create duplicate notification delivery.
4. Implement the configured push-to-talk button and status LED with clear ownership and cleanup. GPIO pins must match the user's wiring and avoid conflicts.
5. Implement microphone and speaker mute controls only if the hardware/wiring is known. A software mute must reliably gate the actual corresponding path and expose its state clearly.
6. Ensure one scheduler instance owns reminder delivery. Test systemd restart, reboot catch-up, failed TTS delivery/retry, and recurring reminder behavior.
7. Document service installation, status/log inspection, safe upgrade, rollback, shutdown, database backup, model placement, and troubleshooting.
8. Keep wake-word optional and disabled by default until validated in the actual environment. Preserve push-to-talk as a reliable fallback.

Acceptance criteria:

- The service starts after reboot, handles voice turns, delivers due reminders, and stops cleanly.
- Restart/reboot does not lose reminders or cause duplicate concurrent scheduler delivery.
- Service runs with only the permissions it needs; local model and online tool boundaries remain unchanged.

### Phase 11: Raspberry Pi Performance, Reliability, and Acoustic Validation

Collect evidence on the target hardware before recommending a model or enabling always-on features.

1. Record exact Pi model/RAM, OS image, kernel, Python, model runtime build, model filename/hash/quantization, context size, audio devices, power supply, cooling, and ambient conditions.
2. Measure baseline idle RSS/CPU and peak RSS/CPU during: cold model start, warm reasoning request, STT capture/transcription, TTS generation/playback, and concurrent reminder activity.
3. Measure interaction latency by stage: button-to-capture, end-of-speech detection, transcription completion, model time-to-first-token/response, TTS startup, and total end-of-turn time.
4. Run repeated-turn and sustained-session checks. Record temperature, throttling, crashes, swap activity, and any power/audio dropouts.
5. Benchmark model choices only with the same representative prompts, context, quantization, backend build, and thermal conditions. Select for acceptable answer quality, latency, and memory headroom—not model file size alone.
6. Confirm the 8 GB system retains documented memory headroom during peak voice + model + OS workload. Treat swap as a failure signal for the interactive target unless evidence shows otherwise.
7. Verify VAD thresholds, microphone placement, speaker feedback, playback leakage, wake-word false accepts/rejects if wake word is tested, and behavior in expected room noise.
8. Publish measurements with the test method and limits. Do not advertise unsupported performance numbers or generalize results to untested hardware/configurations.

Acceptance criteria:

- Peak memory and sustained thermal behavior remain within a documented operating budget with headroom.
- Repeated voice interactions complete without unbounded memory growth, stuck audio, model crashes, or missed reminder delivery.
- Model/backend choice is supported by recorded Pi measurements; unverified alternatives remain labeled unverified.

### Phase 12: Bounded Hybrid Routing and Tool Planning — Initial Slice Implemented

Use deterministic code for recognizable, complete commands. Use the local model as planner for compound requests and for supported operations that lack required details. The model selects registered tools; deterministic adapters validate every argument and perform every side effect.

1. Keep complete local time/status, simple weather/search, memory commands, and unambiguous reminder operations on existing deterministic routes.
2. Detect compound requests with action cues and connectors, then route them to the existing bounded local agent. This is a conservative recognizer, not a confidence score from a second model.
3. Send incomplete reminder creation/reschedule requests to the agent so it can ask for missing information instead of returning parser errors or guessing.
4. Expose one typed `reminder` tool with list, find, create, reschedule, add-notes, and cancel actions. Tool code checks user intent, validates fields, requires title/time/query/notes to be grounded in the user request, and refuses ambiguous targets.
5. Let the planner choose only relevant tools based on request cues. Bind agent-selected weather/search queries to text from the user request; preserve HTTPS allowlists and provider boundaries.
6. Preserve agent maximum steps, total deadline, per-tool timeout, repeated-action protection, context limits, and failure recovery. Require explicit user intent for reminder mutations; ask for clarification when selection is ambiguous.
7. Keep one-shot requests stateless. Do not claim earlier details are known when the current process has no such context.
8. Add regression tests for deterministic bypass, compound tool sequences, incomplete reminders, invalid/unrequested actions, query grounding, ambiguity, and privacy/local-only guarantees.
9. After the initial slice, use real request failures and Pi measurements to decide which additional routes/tools are worth adding. Do not add embeddings, external agent frameworks, a second routing LLM, or broad plugin systems without measured need.

Implementation status: The router sends recognized compound requests to the bounded agent. Simple complete commands retain deterministic fast paths. Incomplete reminder creation/reschedule falls back to local clarification/planning. The agent can sequence allowed weather/search/local tools with the request-scoped reminder tool. Reminder mutations require explicit matching user intent; reminder fields and factual lookup queries are checked against the original request. Tests cover weather-plus-reminder planning, incomplete reminders, rejected unrequested actions, grounded factual queries, and existing fast-path regressions. Initial operation-cue detection is intentionally conservative and may miss natural-language combinations; broaden it only with regression examples. Multi-step memory tools, calendar/device actions, and model-originated irreversible actions remain out of scope.

Acceptance criteria:

- Every enhancement addresses a documented failure/use case and has focused regression tests.
- Existing privacy, local-only model, network allowlist, deterministic fast path, and time/resource limits remain covered by tests.
- Pi memory/latency remain within the baseline budget after each feature is added.

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
- Microphone, VOSK, speaker, KittenTTS, GPIO, local model runtime, SQLite, and systemd behavior are verified on the target setup.
- Speech playback cannot be transcribed as new user input; VAD works in the intended acoustic environment.
- Voice and reminder service run under the documented unprivileged account and recover across reboot/restart.
- Peak memory, CPU, latency, temperature, throttling, and sustained-session behavior are recorded.
- No Pi compatibility, performance, or model recommendation is described as verified without that evidence.

## Configuration and Deployment Decisions

Keep settings centralized and validated. The current configuration includes model backend/URL/name/path, context and threading, agent limits, memory and reminder database, timezone, SearXNG endpoint/allowlist, audio input, VAD, wake word, TTS, GPIO, and logging.

Before Pi release, resolve and document:

- Supported Raspberry Pi OS image/release and Python version.
- Supported installation extras and verified versions of native/ML dependencies.
- Whether llama.cpp is managed by ORION or externally managed in the supported deployment; separately document the Ollama service mode.
- Selected model, quantization, context size, quality/latency tradeoff, and measured peak memory.
- Expected non-root account, audio/GPIO permissions, power supply, cooling, and model/database paths.
- Incognito scope and reset behavior in the long-running service.
- Physical button/LED/mute wiring and which controls are actually implemented.
- Whether online weather/search are enabled in the deployment and which provider hosts receive queries.

Use resource-conscious defaults, but label RAM and latency expectations as estimates until measured on the Pi. Keep wake-word detection disabled by default until its CPU use and acoustic performance have been measured.

## Confirmed Product Decisions

- Raspberry Pi 5 with 8 GB RAM is the primary target. Do not claim compatibility with other models/configurations without checking them.
- Keep local inference local; use only one configured LLM backend at a time and reject cloud endpoints.
- Use deterministic fast paths for clear local operations and call the model for reasoning.
- Store selected memories and reminders in SQLite; keep recent turns volatile and bounded; never persist raw microphone audio by default.
- Explicit user memory requests and policy-screened model suggestions are allowed, subject to Incognito and sensitive-data rules.
- Default reminder timezone is `Asia/Kolkata`; recover missed reminders after service downtime.
- Use Open-Meteo for weather and operator-configured SearXNG for factual web/news search. Search remains disabled until exact HTTPS host configuration is present.
- Start with push-to-talk. Wake-word detection is optional and disabled by default until tested on target hardware.
- No Docker, cloud LLM fallback, second routing LLM, or unbounded audio/conversation history.

## Known Limits

- End-to-end voice flow is implemented but not verified with microphone, speaker, and GPIO hardware.
- The Pi installation and the compatibility of the pinned Python/dependency set on the target OS are unverified.
- Audio hardware, VOSK/TTS coexistence, GPIO behavior, wake-word behavior, CPU, memory, thermal, latency, and long-running stability have not been measured on the Pi.
- The current README includes older setup/model guidance and must be reconciled with the verified packaged runtime before release.
- The current llama.cpp and Ollama deployment modes and structured-output reliability need validation on the chosen Pi configuration.
- The plan mentions a `systemd/orion.service`, but a production-ready integrated service and controls remain to be delivered.
- Incognito is process-local and resets on restart; its intended user-facing promise must be finalized for a persistent service.
- A trusted SearXNG instance must be configured before web search is available. Search queries and weather locations are sent to their configured providers.

## Immediate Milestone and Handoff

Phase 8 and the initial Phase 12 planner slice are implemented in code and covered by local tests. Next milestone is Phase 9: install the documented stack on a clean Raspberry Pi OS image and close dependency/runtime compatibility gaps. Then complete deployment and measured validation. Expand planner tools only for concrete, tested user needs and retain execution safeguards.

For an implementation agent, work phase by phase. Before editing, inspect the current code and tests; implement only the active phase, add or update focused tests, run the relevant local checks, and report which acceptance criteria passed. Do not mark Pi-specific criteria complete from laptop results. Preserve uncommitted user changes and do not make unrelated cleanup changes.
