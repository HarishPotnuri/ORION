# ORION Implementation Plan

## Goal

Build ORION as a privacy-first, offline-by-default personal voice assistant for Raspberry Pi 5 with 8 GB RAM and Raspberry Pi OS Lite 64-bit. The target product provides wake-word-only interaction, local speech recognition, streamed local model responses and speech output, persistent reminders, and user-controlled memory. The existing push-to-talk deployment remains the baseline until the replacement profile passes target validation.

The architecture foundation, voice-service integration, and initial Phase 12 hybrid planner are implemented. Remaining work includes the open Phase 9-11 validation gates and the agreed Phase 13-16 offline-stack migration. Broader persistent memory and optional semantic retrieval are deferred to Phases 17 and 18.

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

## Current Status (2026-10-06)

Phases 1 through 8 and the initial Phase 12 planner slice are implemented. The existing Pi deployment runs Ollama with `qwen3:1.7b`, VOSK, KittenTTS, GPIO23 push-to-talk, GPIO16 speech interruption, SQLite, and the integrated systemd service. The 2026-10-05 deployment update records working multi-turn voice interaction and passing installed-runtime tests. Initial resource captures exist. Fresh-image installation, reboot/reminder recovery, wake-word acoustics, and sustained thermal/performance acceptance remain open. Phases 13-18 below are planned, not implemented or Pi-verified.

The phase descriptions for Phases 1-12 document the baseline implementation and its original requirements. The Phase 13+ target decisions supersede the earlier push-to-talk/fallback and resident-model requirements when the new profile is promoted; they do not claim those changes are already deployed.

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

Status: **Partially implemented; acceptance incomplete.** The locked Python 3.12 profile on the existing Pi passes dependency checks; this is not a clean-image installation. The ARM64 CUDA warning is resolved for uv installs by excluding Misaki 0.9.4's unused curated-transformer extra. The owned-stream TTS change passed Pi interruption stress, the planner prompt/response guard and 128-token output cap are deployed, and the user confirmed multi-turn voice interaction works through the running systemd service. Fresh-image installation and sustained/performance measurements remain open, so Phase 9 remains open.

1. Record the actual target used: Raspberry Pi 5 8 GB, 64-bit Raspberry Pi OS, kernel `6.18.50+rpt-rpi-2712`, system Python 3.13.5/aarch64, Jabra EVOLVE 20 MS USB audio, GPIO23 push-to-talk, GPIO25 green status LED, uv-managed Python 3.12.15 voice environment, VOSK small-en-us 0.15, KittenTTS nano 0.2 assets, and Ollama `qwen3:1.7b`.
2. Keep core support at `>=3.11,<3.14`, but use Python 3.12 for the voice extra on this OS: KittenTTS 0.1.0 requires `misaki>=0.9.4`, which was unavailable for Python 3.13 from the configured indexes. Confirm this compatibility boundary again when dependency versions change.
3. Validate ARM64 dependencies and retain the proven OS prerequisites: `build-essential`, `python3-dev`, `swig`, `liblgpio-dev`, `libportaudio2`, `espeak-ng`, `unzip`, and `curl`. `lgpio` first failed without SWIG, then linked unsuccessfully without `liblgpio`; sounddevice needed PortAudio; TTS needed eSpeak NG.
4. Provide reproducible Pi install instructions and separate dependency profiles for core typed use and voice/GPIO, including a uv-managed Python 3.12 venv. Preserve the `src/orion` source layout during Windows-to-Pi SCP transfers; copying only `src/orion` flattened the tree and made editable install fail with `egg_base: 'src' does not exist`.
5. Persist/document the runtime exports for `ORION_STT_MODEL_PATH`, `ORION_AUDIO_QUEUE_CHUNKS=100`, `ORION_AUDIO_DEVICE`, and the Ollama backend/model settings; shell-local exports disappeared between sessions and caused later checks to silently use the large default VOSK model and queue size 8.
6. Verify device configuration, microphone capture, speaker playback/resampling, GPIO access, local database permissions, and model/service paths. Evidence so far: Jabra capture yielded peak 16687/RMS 664.5 at 16 kHz; GPIO test blinked green three times and detected the button; PortAudio supported 32/44.1/48 kHz but not TTS native 24 kHz, so resampling to output default is required.
7. Document and exercise the local-model backend. Voice mode initially used default managed llama.cpp without `ORION_MODEL_PATH` and failed reasoning with `FileNotFoundError`; installing Ollama and pulling `qwen3:1.7b` fixed the typed reasoning path. Still run a complete voice conversation through Ollama on Pi.
8. Keep the default inference endpoint loopback-only. Do not expose a model server on the LAN as a deployment shortcut.
9. Exclude Misaki 0.9.4's `spacy-curated-transformers` extra from uv resolution: it pulled Torch 2.14.1 and `nvidia-cusparselt-cu13`, whose SBSA wheel is incompatible with Raspberry Pi aarch64. The installed KittenTTS/Misaki path does not import curated transformers; verify the exclusion with a clean profile, `uv pip check`, and offline synthesis.

Acceptance criteria:

- A fresh supported Pi image can install the documented profile without undocumented manual fixes.
- The documented command starts the assistant under the intended user account, opens the intended microphone/speaker and GPIO devices, and reaches a local model when reasoning is requested.
- Core deterministic requests still work if the model runtime is stopped or unavailable.
- All selected dependency and model versions are recorded so a later update can be compared reproducibly.

Current validation evidence: Windows x64 Python 3.11.9 and 3.13.16 full suites each showed 123 passed, 1 known reminder orchestration failure, and 30 passing subtests; after the TTS warm-up tests were added, the local suite showed 127 passed, the same 1 failure, and 30 passing subtests. The Pi is Debian 13 (trixie), kernel `6.18.50+rpt-pi-2712`, aarch64, system Python 3.13.5; voice Python 3.12.15 under uv 0.12.22. Ollama 0.35.0 serves `qwen3:1.7b` (digest `8f68893c685c`, Q4_K_M, 1.4 GB) over loopback. Key voice/Pi packages are KittenTTS 0.1.0, Misaki 0.9.4, ONNX Runtime 1.30.0, VOSK 0.3.45, sounddevice 0.5.6, gpiozero 2.0.1.post3, and lgpio 0.2.2.0; `uv.lock` records the complete dependency resolution. Hardware checks passed for GPIO button/LED, USB Jabra enumeration, 16 kHz mic capture, speaker playback at supported rate, VOSK small-model transcription, and KittenTTS spoken voice-check response. TTS warm-up took 5.63 seconds online and 4.97 seconds with `HF_HUB_OFFLINE=1`; the clean locked profile took 3.35 seconds. On the existing Pi, a clean Python 3.12 env synced the locked voice/Pi/test profile; all 98 packages passed `uv pip check`, the deterministic time request worked, focused audio/voice tests passed (22 tests, 10 subtests), offline KittenTTS synthesis generated 46,200 samples without Torch/CUDA dependencies, and three repeated ORION playback calls completed. The full suite showed 127 passed, the same known reminder failure, and 30 passing subtests. Typed `Explain Kafka briefly` returned a local Ollama answer. However, two live voice runs (the second on the corrected profile) aborted with `double free or corruption` immediately after a phonemizer word-count warning. Isolated generation/playback tests passed and no core dump was available. A subsequent faulthandler-enabled run transcribed a 14-character request but produced no audible reply; the Ollama runner remained CPU-bound and the voice process was stopped cleanly after exceeding the 120-second task limit. Treat repeated-turn voice operation as a known blocker, not Pi-verified. A complete spoken Ollama reasoning turn and fresh-image install remain unverified. Phase 11 sustained/performance gates also remain open; no performance or service-deployment claims are made.

Deployment update (2026-10-05): the latest source and lock are installed under `/opt/orion`; the dedicated systemd service is enabled and active. `orion-service` runs the installed Python 3.12 profile with a dedicated TTS cache and migrated database. `uv pip check` passed for all 98 packages; the full installed-runtime test suite passed 140 tests and 34 subtests. Systemd startup and manual restart both reached readiness. The user confirmed multi-turn voice use through the service. This supersedes the earlier aborted/stalled voice attempts as current behavior evidence; longer barge-in soak, reboot, reminder catch-up/retry/recurrence, clean-image installation, and sustained/performance checks remain open.

### Phase 10: Integrated Service, Controls, and Operations

Status: **Deployed; lifecycle acceptance partially verified.** On 2026-10-05, the dedicated non-login `orion-service` account, `/opt/orion` runtime, locked Python 3.12 environment, `/var/cache/orion` TTS cache, migrated `/var/lib/orion/orion.sqlite3`, and root-managed `orion.service` were installed and enabled. The service reached “Voice assistant started” after launch and manual restart; the user confirmed multi-turn voice interaction through the service. The dedicated-runtime full suite passed 140 tests and 34 subtests. The process owns reminders, GPIO23 PTT, GPIO16 TTS interruption, and the RGB LED (GPIO27/25/17). A per-database OS lock prevents concurrent schedulers and stale-delivery recovery only occurs after lock acquisition. Mute pins remain unset because no mute hardware is wired. Reboot behavior and reminder catch-up/retry/recurrence under systemd remain unverified; keep Phase 10 open until those checks pass.

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

Status: **Initial idle and interactive captures recorded; measurement gate remains open.** The installed service logs PTT wait, VAD/end-of-speech, STT, orchestration, planner model calls, and TTS without transcript contents. A corrected 30-second idle-listening capture contained 27 samples: ORION RSS 487,808 KB, Ollama unloaded RSS 36,864 KB at 0% sampled CPU, minimum available RAM 7.69 GB, zero swap, temperature 46.1-48.3 C, process CPU 0-2%, and `vcgencmd get_throttled=0x0`. A separate 120-second labeled PTT/time/reasoning capture contained 115 samples across one deterministic time request and two reasoning turns: max ORION RSS 498,672 KB, max Ollama RSS 2,393,264 KB, minimum available RAM 5.29 GB, zero swap, max temperature 58.7 C, throttling `0x0`. Planner calls were 10.90 s and 8.11 s; STT 2.49 s and 2.85 s; TTS 6.11 s and 14.74 s. The deterministic request measured VAD 1.49 s, STT 1.63 s, and TTS 11.47 s. A 75.3 s PTT wait was user delay, not STT latency. The interactive sampler reported aggregate multi-core peak CPU of 357% for ORION and 427% for Ollama, not single-core utilization. These preliminary runs are not cold-start peak, concurrency, thermal-soak, or 30-minute sustained results. Power/cooling/ambient and acoustics remain manual checks; Phase 10 reboot/reminder gates remain open.

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

### Phase 12: Bounded Hybrid Routing and Tool Planning - Implemented Locally; Pi Validation Pending

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

Local implementation update (2026-10-06): Compound recognition now requires an operation clause after a connector instead of counting overlapping keywords. Reminder titles, saved facts, search topics, and location names containing conjunctions retain their deterministic handling. Compound local questions use the planner; request-scoped authorization recognizes plural reminders and the existing time/date/system-status query forms. Initial planner instructions and authorized-tool descriptions survive every tool step while only the latest step state replaces the previous state. Added regressions cover these failures, reschedule-plus-notes sequencing, and incomplete reschedule clarification without mutation. Focused Phase 12 checks passed 78 tests and 29 subtests; the full Windows Python 3.11.9 suite passed 150 tests and 42 subtests. This update has not been deployed or measured on the Pi. Phase 13 runtime/default changes remain planned, not implemented.

Acceptance criteria:

- Every enhancement addresses a documented failure/use case and has focused regression tests.
- Existing privacy, local-only model, network allowlist, deterministic fast path, and time/resource limits remain covered by tests.
- Pi memory/latency remain within the baseline budget after each feature is added.

## Offline Stack Migration: Phases 13-18

### Current Behavior Versus Target Requirements

- LLM settings default to `llama.cpp`, no local model path, `qwen3:1.7b` as the model label, 8192 context tokens, a 6000-token estimated input budget, and four threads. The deployed environment selects Ollama instead. Neither adapter currently streams; a model label alone does not verify a GGUF artifact.
- `LocalModel.complete`, `SpeechOutput.speak`, loopback-only transport, managed llama.cpp ownership, deterministic routes, and bounded structured tool decisions already exist. Extend these boundaries rather than adding an agent or plugin framework.
- Capture already uses bounded 16 kHz mono int16 PCM and RMS VAD. Activation and transcription are coupled through `WakeWordInput` inheriting `VoskTranscriber`; startup warms STT/TTS, and VOSK loads before consuming VAD. Separate utterance collection from transcription so silence never loads Whisper.
- openWakeWord is optional and defaults to `hey_jarvis`. The target requires a custom local "Hey Orion" model, wake-word-only activation, and independent lightweight listening. Existing capture blocks are 320 samples; assemble bounded 1280-sample frames for 80 ms wake inference at 16 kHz.
- Bounded recent turns, relevant SQLite/FTS5 retrieval, facts/preferences/decisions/episodes, reminders, sensitive-data checks, and Incognito already exist. Generated summaries, general tasks, broader device/configuration state, and embeddings are new, deferred requirements. Separate tables for existing facts/preferences are not required merely to rename their categories.
- KittenTTS currently waits for the complete response. Piper and early phrase playback require new adapters and a streamed orchestration path, not only a model-name change.
- Stage timers and `scripts/phase11-capture.sh` already exist. Wake timing, LLM TTFT/token rate, first audible output, turn correlation, and provider-independent resource attribution need extension.
- The current tool factory registers weather without a global online opt-in setting. Add an explicit default-off gate for both weather and search; existing HTTPS allowlists alone do not enforce offline-by-default behavior.

### Confirmed Target Profile and Controls

- Preserve Phase 12. Implement the runtime migration as Phases 13-16; promote code and deployment defaults together after Pi validation.
- Use Qwen3.5-2B post-trained weights converted to GGUF `Q4_K_M` with a pinned compatible `llama.cpp` build. Run text-only without a vision projector. Use non-thinking mode initially; do not assume Qwen3 `/nothink` or Ollama-specific flags apply to this runtime.
- Use `whisper.cpp` with local Whisper `base.en` for the initial English profile. Allow `base` selection for a separately validated multilingual profile.
- Use Piper with local medium-quality English ONNX/config assets. `en_US-lessac-medium` is an initial candidate, subject to voice selection and license verification. Use the voice's metadata sample rate and preserve hardware playback resampling.
- Use openWakeWord with custom "Hey Orion" assets. Provision or train the model on a development machine, not the Pi. Package the custom model and feature-extractor assets locally, record hashes/licenses, and verify acoustic behavior. Missing assets block release; never silently substitute another phrase or download models at startup.
- Remap the existing GPIO23 button to debounced microphone mute/unmute without rewiring. Wake word becomes the only voice activation mode; no push-to-talk fallback remains in the target profile. GPIO16 retains speech interruption and cancels remaining streamed generation/output.
- Persist microphone mute across restart/reboot. Mute stops capture, cancels/discards active input, clears audio/detector state, and prevents stale transcripts from dispatching. Speaker and reminders remain active; already-dispatched actions are not undone. Read mute state before opening capture. A new installation starts unmuted; unreadable/corrupt state fails closed. Report persistence errors without claiming a successful durable unmute.
- Preserve the existing RGB LED wiring. `MUTED` takes precedence over normal status transitions. Privacy mute and playback suppression are separate states. Listen continuously while unmuted, except intentional playback suppression to avoid self-activation; voice barge-in/echo cancellation is out of scope.
- Keep Ollama, VOSK, and KittenTTS selectable for explicit benchmarking/rollback. Do not silently change providers, re-enable PTT, add cloud fallback, or run duplicate LLM backends.
- Continue using SQLite and compact relevant retrieval. Broader memory features remain deferred; the small persistent microphone-control record is the immediate device-state exception.

### Phase 13: Configurable Adapters and Wake-Word Input - Planned

1. Extend existing contracts/factories with explicit STT/TTS providers, local artifact paths, language, per-provider threads, cancellation, and cleanup. Decouple activation/VAD from transcription. Keep settings validated and configuration/route checks free of heavy model initialization.
2. Add a default-off online-tools flag. Gate deterministic and planner weather/search consistently while preserving exact-host HTTPS policy when enabled. Downloads/builds are explicit installation operations only.
3. Add a small additive SQLite control-state record and GPIO23 mute handling with debounce, fail-closed reads, reliable capture close/restart, and stale-input rejection. Preserve GPIO16 and pin-conflict validation; an old PTT setting may serve only as a deprecated wiring alias, not an activation mode.
4. Add an independent bounded wake listener, 80 ms frame accumulation, reset behavior, and local "Hey Orion" artifact validation. Keep detection lightweight during model activity and suppress playback leakage. Spoken confirmations still require wake activation and a clear affirmative within the existing 15-second timeout; muted input cannot confirm a mutation.
5. Collect a bounded nonempty voiced utterance through VAD before starting Whisper. Prefer an owned per-utterance `whisper-cli` subprocess with an in-memory WAV over stdin, transcript stdout, explicit CPU threads/no GPU, bounded output, deadlines, cancellation, and guaranteed process cleanup. Pin and prove stdin support; enable no output-file options. Measure reload cost before considering a resident worker.

Acceptance criteria:

- No wake, silence, or microphone mute activates STT/LLM; deterministic local requests still bypass the LLM.
- GPIO23 toggles once per press, mute survives restart, stale audio never resumes, and reminders/speaker remain functional while muted.
- Missing local assets fail clearly without network access or alternate-provider/phrase fallback. Captured audio never reaches disk.
- Mocked wake-to-VAD-to-transcript integration, cancellation, error cleanup, and existing privacy/tool tests pass.

Dependency: adapter/capture separation precedes mute, wake, and Whisper integration. Offline gating and artifact preparation can proceed in parallel after the contracts are defined.

### Phase 14: Qwen Runtime and Safe Streaming Generation - Planned

1. Select and checksum the actual Qwen3.5-2B `Q4_K_M` GGUF, not the Base variant, and pin/smoke-test its llama.cpp build and chat template. Start with 4096 context tokens, a 3072-token input budget, up to 512 answer tokens, and 512 tokens reserved for template/safety overhead. Validate formatted prompt plus output capacity through the runtime tokenizer/template path. Allow 8192 only after validation. Keep the planner cap separately bounded, initially 128 tokens.
2. Add typed streaming events and policy-preserving SSE transport with incremental UTF-8/event decoding, response bounds, finish/`[DONE]` handling, usage/timings, absolute deadlines, cancellation, and connection cleanup. Retain `complete()` for buffered consumers and rollback adapters. Reject remote endpoints, redirects, and hidden cloud fallback.
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
6. Promote `Settings`, deployment environment, dependency profiles, README, this plan, and issue status together after acceptance. Keep existing data and explicit runtime rollback selectors. If a gate fails, leave current defaults unchanged and report the blocked criterion.

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

Use resource-conscious defaults, but label RAM and latency expectations as estimates until measured on the Pi. The current deployment keeps wake-word detection disabled. The Phase 13+ target uses wake-word-only activation after custom-model, privacy-control, CPU, and acoustic validation; promote it with the complete replacement profile in Phase 16.

## Confirmed Product Decisions

- Raspberry Pi 5 with 8 GB RAM is the primary target. Do not claim compatibility with other models/configurations without checking them.
- Keep local inference local; use only one configured LLM backend at a time and reject cloud endpoints.
- Use deterministic fast paths for clear local operations and call the model for reasoning.
- Store selected memories and reminders in SQLite; keep recent turns volatile and bounded; never persist raw microphone audio by default.
- Explicit user memory requests and policy-screened model suggestions are allowed, subject to Incognito and sensitive-data rules.
- Default reminder timezone is `Asia/Kolkata`; recover missed reminders after service downtime.
- Retain Open-Meteo and operator-configured SearXNG only as explicitly opted-in factual tools. Add a default-off gate for both during Phase 13; search also requires exact HTTPS host configuration.
- The current deployment uses push-to-talk. The agreed target uses only "Hey Orion" wake activation, remaps GPIO23 to persistent microphone mute/unmute without rewiring, and retains GPIO16 speech interruption. Microphone mute does not silence reminders or speaker output.
- Promote Qwen3.5-2B GGUF `Q4_K_M`/llama.cpp, Whisper `base.en`/whisper.cpp, and Piper medium-English as defaults only after Phase 16 validation. Keep existing runtime adapters selectable for explicit rollback, not automatic fallback.
- Defer summaries, general tasks, broader device state, and optional embeddings to Phases 17-18. Persist microphone mute in the initial migration as a narrow control-state exception.
- No Docker, cloud LLM fallback, second routing LLM, or unbounded audio/conversation history.

## Known Limits

- The existing Pi has passed basic audio/GPIO checks, installed-runtime tests, systemd launch/restart, and user-confirmed multi-turn voice use. Fresh-image installation and reboot/reminder recovery remain unverified.
- Initial idle/interactive resource and stage-latency captures exist; cold-start peaks, 30-minute soak, sustained thermal behavior, and acoustic acceptance remain open.
- Qwen3.5 GGUF/llama.cpp, Whisper, Piper, custom "Hey Orion", persistent GPIO23 mute, and streamed speech are planned replacements, not implemented or Pi-verified. The 2-3 GB LLM RAM target is unverified.
- The current README includes older setup/model guidance and must be reconciled with the verified packaged runtime before release.
- The current llama.cpp and Ollama deployment modes and structured-output reliability need validation on the chosen Pi configuration.
- The integrated systemd service and existing PTT/interrupt/RGB controls are deployed. The new mute mapping, wake-only activation, and replacement-runtime lifecycle still require implementation and target validation.
- Incognito is process-local and resets on restart; its intended user-facing promise must be finalized for a persistent service.
- A trusted SearXNG instance must be configured before web search is available. Search queries and weather locations are sent to their configured providers.

## Immediate Milestone and Handoff

Phase 8 and the initial Phase 12 planner slice remain implemented. The next implementation milestone is Phase 13: adapter/capture separation, explicit offline gating, custom wake activation, persistent microphone controls, and Whisper integration. Keep the current deployment unchanged while developing the candidate. Complete the open Phase 9-11 installation/reboot/soak gates alongside target validation, then promote the replacement defaults only at Phase 16. Do not start Phases 17-18 automatically or broaden planner tools without concrete tested needs.

For an implementation agent, work phase by phase. Before editing, inspect the current code and tests; implement only the active phase, add or update focused tests, run the relevant local checks, and report which acceptance criteria passed. Do not mark Pi-specific criteria complete from laptop results. Preserve uncommitted user changes and do not make unrelated cleanup changes.
