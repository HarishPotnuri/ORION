# ORION Implementation Plan

## Goal

Build ORION as a privacy-first, local-first personal voice assistant for Raspberry Pi 5 8GB. Reuse proven audio, STT, TTS, GPIO, and local-model components from the existing repository, but do not preserve its monolithic architecture as ORION's final design.

ORION's core will be a deterministic router, bounded agent controller, context manager, SQLite memory, and persistent reminder engine. No cloud LLM, second routing LLM, Docker, raw-audio persistence, or unnecessary background services.

## Existing Repository Findings

- `stt_lib.py` uses sounddevice and VOSK. It records audio into an unbounded in-memory frame list, writes `input.wav`, then transcribes the file.
- `main.py` uses Ollama, KittenTTS, GPIO button 23, and PWM LED 25 in one blocking loop.
- `main_llama.py` uses an OpenAI-compatible client against llama.cpp, starts the server with a fixed 30-second delay, and binds it to `0.0.0.0`.
- Both entry points write TTS audio to `output.wav` and keep unbounded in-memory chat history.
- `pyproject.toml` includes GPIO, VOSK, sounddevice, KittenTTS, Ollama, OpenAI compatibility, and other dependencies. Python is pinned to `>=3.11,<3.12`.
- There are no tests or modular application package yet. README describes the upstream voice-chat demo and systemd startup.

## Reusable Components

- VOSK recognizer and sounddevice input patterns, after replacing unbounded/file-based capture with bounded streaming.
- KittenTTS voice generation, after ensuring generated audio is temporary and removed after playback.
- gpiozero button and LED abstractions, with pins centralized in configuration.
- Existing local Ollama and llama.cpp integration ideas, behind one runtime interface. Only one local LLM is active at a time.
- Push-to-talk as a reliable fallback while wake-word and VAD support is introduced.

## Target Architecture

```text
src/orion/
  app.py                 Application lifecycle and orchestration
  config.py              Central settings and validation
  core/
    router.py            Deterministic intent routing
    agent.py             Bounded plan/action/observe controller
    context.py           Relevant context selection and token budget
    policy.py            Local-only LLM and online-tool policy
    capabilities.py      Honest capability and degradation responses
  audio/
    capture.py           Bounded streaming microphone input
    wakeword.py          Optional configurable openWakeWord adapter
    vad.py               Voice activity detection adapter
    stt.py               VOSK streaming transcription
    tts.py               Local speech generation and playback
  llm/
    runtime.py           Single configured local model interface
  memory/
    store.py             SQLite schema and persistence
    retrieval.py         FTS-based relevant memory retrieval
  reminders/
    manager.py           Create, list, query, edit, and cancel reminders
    scheduler.py         Persistent due-time and missed-reminder handling
  tools/
    registry.py          Explicit tool allowlist and execution limits
    time.py              Local date and time
    system.py            Local system status
    weather.py           Approved factual weather adapter
    web_search.py        Approved factual search adapter
  hardware/
    buttons.py           Configurable controls
    leds.py              Configurable status indicators
tests/
systemd/orion.service
```

Adapt package boundaries to repository needs; do not add empty modules or unnecessary abstractions.

## Implementation Phases

### Phase 1: Architecture Foundation

1. Add modular `src/orion/` package, central validated settings, logging, component interfaces, and one active entry point.
2. Treat `main.py` and `main_llama.py` as upstream reference material, not supported legacy applications. Clean them up or remove them as the ORION runtime replaces their reusable components.
3. Review dependency and Python-version compatibility for Raspberry Pi OS before changing constraints.

### Phase 2: Privacy-Safe Voice Path

1. Replace unlimited audio buffering and `input.wav` with bounded streaming capture and VOSK transcription.
2. Release captured chunks after transcription, including on errors. Never persist microphone audio.
3. Use a temporary TTS output or stream, then remove it after playback.
4. Disable listening during playback to reduce microphone feedback.
5. Keep wake-word detection configurable and disabled by default. Use push-to-talk until the Pi and microphone are available.

### Phase 3: Deterministic Router and Local Model

1. Route simple commands, local tools, reminders, memory intents, approved factual lookups, and reasoning using deterministic rules.
2. Call the model only when natural-language reasoning is needed.
3. Add one configurable local model runtime. Support llama.cpp and/or local Ollama through the same interface, with only one model active.
4. Reject cloud LLM endpoints. Bind llama.cpp to localhost, check server readiness, and shut it down cleanly.

### Phase 4: Bounded Agent and Context

1. Add explicit `AgentController` states for plan, action, observation, evaluation, and completion.
2. Enforce maximum steps, per-tool timeout, total task timeout, context token budget, and loop protection.
3. Build prompts from current request, relevant recent turns, selected memories, compact tool results, and active task state.
4. Trim irrelevant memories and old turns first. Preserve the current request and required task state.
5. Store or expose only concise state/action/result summaries. Never store or expose hidden chain-of-thought.

Implementation status: `AgentController` and `ContextBuilder` are implemented under `src/orion/core/`. The controller accepts only registered tools, limits steps and elapsed time, rejects repeated actions, and passes a deadline plus cancellation event to adapters. The default context token counter is an approximate size estimate; inject a model tokenizer for exact accounting. Tool adapters must honor their timeout and cancellation event because Python cannot forcibly stop a running thread.

### Phase 5: SQLite Memory

1. Keep short-term conversation turns in process memory only; discard them on shutdown.
2. Store selected concise episode summaries and durable facts, preferences, and decisions in SQLite.
3. Save explicit user memory requests. Also let the local model recommend stable, important memories for automatic saving, subject to deterministic category and privacy checks.
4. Add user-controlled Incognito On/Off commands. While Incognito is on, allow retrieval of existing memories but block every memory write, including explicit save requests, automatic durable-memory suggestions, and episode summaries. Keep active conversation context volatile and discard it when the process ends.
5. Support requests to find and delete a specific older episodic memory. Delete the matching episode when unambiguous; ask for clarification when multiple episodes match.
6. Do not save every message or raw transcript permanently by default.
7. Add lightweight FTS retrieval before local reasoning. Keep semantic embeddings deferred until measured need exists.

Implementation status: Complete. `MemoryManager` keeps a bounded recent-turn buffer only in RAM and stores selected episode/fact/preference/decision records in SQLite with FTS5 retrieval. Explicit “remember” requests use deterministic categorization; the local model can return optional durable-memory suggestions, which pass a deterministic allowlist, length, and sensitive-data check. Incognito allows reads but blocks all writes. Its toggle is process-local and resets to Off on restart. Episodic deletion removes only one unambiguous match and returns candidates for clarification otherwise. Recent-turn and search limits use `ORION_MEMORY_RECENT_MESSAGE_LIMIT` and `ORION_MEMORY_SEARCH_LIMIT`. The APIs are reachable through typed request orchestration; continuous voice-loop integration remains pending.

### Phase 6: Persistent Reminders

1. Store timezone-aware one-time and recurring reminders in SQLite with title, due time, timezone, notes, original request, status, and recurrence.
2. Implement create, list, query, modify, cancel, and note-update operations.
3. Run a lightweight scheduler inside the Orion service; do not add a separate database or scheduler service.
4. On boot, notify the user about every reminder missed while powered off. Use `Asia/Kolkata` as the default timezone, with configuration override.
5. Trigger local TTS and LED state through hardware interfaces.

Implementation status: Reminder records share the configured SQLite database and store due times in UTC with their IANA timezone, original request, notes, and recurrence. `ReminderParser` supports relative minutes/hours/days, today/tomorrow/weekday, and daily/weekly schedules; it rejects missing or past times instead of guessing. `ReminderScheduler` recovers interrupted deliveries, processes all overdue records at startup, retries failed notifications, and advances recurring schedules after delivery. `ReminderNotifier` speaks title/notes while using the `SPEAKING` LED state. `orion --run-reminders` runs the persistent scheduler and performs missed-reminder catch-up on startup. Typed `--request` dispatch supports reminder creation, listing, detail queries, and cancellation. Settings use `Asia/Kolkata` and `ORION_REMINDER_POLL_INTERVAL`; `tzdata` provides portable IANA timezone data. Continuous voice-loop integration remains pending.

### Phase 7: Tools and Capability Policy

1. Add local date/time and system status tools.
2. Add explicitly allowlisted online factual tools for weather and search. Return compact structured data only.
3. Pass factual results to the local model for interpretation. Never send prompts to online services or cloud LLMs.
4. Add capability checks and honest fallback responses when local reasoning is not reliable enough.

Implementation status: Complete for typed requests and deterministic tools. `RequestOrchestrator` dispatches local tools without loading the model, retrieves explicitly requested weather/search facts through `ToolRegistry`, and passes compact results only to the local model. Open-Meteo provides weather without an API key. Web search uses SearXNG and remains disabled until `ORION_SEARXNG_BASE_URL` and matching `ORION_SEARXNG_ALLOWED_HOSTS` are configured. The transport uses HTTPS, exact host allowlists, disabled proxies/redirects, timeouts, and a 1 MiB response cap. Online services receive only the requested factual query/location, never conversation history or model prompts. `CapabilityPolicy` refuses broad long-horizon analysis with an honest narrower alternative. `orion --request "..."` provides a typed single-turn interface; local model loading is lazy. Reminder creation/list/detail/cancel/reschedule/note-update and memory/Incognito commands are available through typed request dispatch. Continuous microphone-to-orchestrator runtime, online provider live tests, and Pi hardware validation remain pending.

### Phase 8: GPIO and Deployment

1. Centralize configurable pins and LED states: idle, listening, processing, speaking, muted, and error.
2. Preserve push-to-talk and add microphone mute and speaker mute controls where wiring supports them.
3. Add a native systemd service and Pi OS setup instructions. No Docker.
4. Document model installation, dependencies, GPIO wiring, privacy, resource expectations, startup/shutdown, and troubleshooting.

### Phase 9: Verification and Pi Measurement

1. Add unit tests for routing, memory classification/retrieval, context trimming, reminders, scheduler, agent limits, online-tool policy, and graceful degradation.
2. Add integration tests with fake audio, clock, model, tools, and GPIO.
3. Verify audio is not persisted after success or failure.
4. Verify memory and reminders survive process restart, including recurrence and missed-reminder behavior.
5. Run formatting, lint/type checks, and the full test suite.
6. Measure idle and peak RSS on Raspberry Pi 5 with the selected model loaded. Do not claim hardware verification before running on the target.

## Verification Checks

- Simple commands do not invoke the local model.
- Reasoning requests use only the configured local model.
- Cloud LLM endpoints are rejected, and online tools are explicitly allowlisted factual APIs.
- Audio buffering is bounded, and no microphone recording remains on disk.
- Context and tool results stay within configured limits.
- Agent stops at step/time limits and handles failed tool calls.
- SQLite memory and reminders persist across restart.
- Reminder schedule handles timezones, recurrence, cancellation, and missed events.
- TTS playback does not capture its own output as new user speech.
- Pi resource measurements leave meaningful headroom below 8GB.

## Configuration

Centralize at least:

- `MODEL_PATH`, `MODEL_CONTEXT_SIZE`, `MODEL_THREADS`, `MODEL_TEMPERATURE`
- `MAX_AGENT_STEPS`, `MAX_TOOL_EXECUTION_TIME`, `MAX_TOTAL_TASK_TIME`, `MAX_CONTEXT_TOKENS`
- `MEMORY_RECENT_MESSAGE_LIMIT`, `MEMORY_SEARCH_LIMIT`
- `ORION_REMINDER_TIMEZONE`, `ORION_REMINDER_POLL_INTERVAL`
- `ORION_SEARXNG_BASE_URL`, `ORION_SEARXNG_ALLOWED_HOSTS`
- audio device, wake word, VAD settings, STT model path, TTS voice
- SQLite path and memory retention settings
- reminder timezone and missed-reminder policy
- GPIO pins and online factual API configuration

Use resource-conscious defaults suitable for Raspberry Pi 5 8GB. Keep model and audio components loadable or unloadable where practical. Document RAM expectations as estimates until measured on-device.

## Confirmed Product Decisions

- The old flat scripts are reference material only. Reuse their proven pieces; clean them up as the ORION runtime replaces them.
- Select one local model suitable for Raspberry Pi 5 8GB and 64/128 GB storage. Compare model size, quantization, quality, and runtime behavior; verify RAM and latency on the Pi before making a final recommendation.
- Save memory on explicit user requests and allow the local model to recommend important facts, preferences, or decisions. Apply a deterministic policy before persistence; do not save every message.
- Keep conversation turns in process memory only. Store selected episode summaries and durable memories in SQLite. Incognito On allows reading existing memories but blocks every memory write; Incognito Off resumes normal save rules. Users can request deletion of a specific episodic item.
- Use `Asia/Kolkata` as the default reminder timezone. On boot, notify the user about all missed reminders.
- Use Open-Meteo for factual weather and operator-configured SearXNG for factual web/news search. Search is disabled until an HTTPS instance and exact allowed host are configured. Online services receive only user-requested query/location data; the local model performs interpretation.
- Keep wake-word detection configurable and disabled by default. Use push-to-talk until the Pi and microphone are available.

## Known Limits and Open Choices

- Wake-word and VAD behavior needs microphone testing. Push-to-talk remains fallback.
- GPIO assignments beyond current button 23 and LED 25 depend on the user's physical controls and wiring.
- A trusted SearXNG instance must be configured before web search is available. Public SearXNG instances can receive search queries, so choose an instance consistent with the user's privacy needs.
- The current Python constraint excludes Python 3.12+. Confirm dependency compatibility on target Raspberry Pi OS before changing it.
- Exact model RAM and response latency depend on quantization, context size, and llama.cpp build. Benchmark on the Pi before publishing definitive figures.

## Immediate First Milestone

Phases 1 through 7 establish the architecture, privacy-safe audio, deterministic routing/local model, bounded agent/context, SQLite memory, persistent reminders, factual tools, and capability limits. Next connect the continuous voice loop, integrate physical controls and native deployment, then measure behavior and resources on the Pi.