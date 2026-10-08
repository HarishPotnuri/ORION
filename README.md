# ORION

For a clean Windows development machine or freshly flashed Raspberry Pi, follow [EnvironmentSetup.md](EnvironmentSetup.md). It records pinned runtimes/artifacts, SSH setup, provisioning, service setup, backup, and recovery steps.

### Clear Conversation History

After wakeword activation, say "clear conversation history" to reset recent conversation context without restarting the voice service. "Clear chat history", "reset our conversation", and "wipe conversational memory" also work. The command runs without model inference and leaves saved facts, preferences, decisions, episodes, and reminders unchanged. Neither the command nor its acknowledgement enters the new conversation history. This does not correct speech recognition or remove misunderstandings already saved as durable memories.
Local voice assistant running on Raspberry Pi 5 with custom wakeword activation, Whisper STT, pinned local llama.cpp inference, and KittenTTS

## ORION local tools

### Local laptop testing

The default install targets typed local requests and tests. It does not install microphone, TTS, wake-word, or Raspberry Pi GPIO packages.

In PowerShell, create a separate Python 3.11 environment and install the test profile:

```powershell
py -3.11 -m venv .venv-local
& .\.venv-local\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-local\Scripts\python.exe -m pip install -e ".[test]"
& .\.venv-local\Scripts\python.exe -m pytest -q
& .\.venv-local\Scripts\orion.exe --check-config
& .\.venv-local\Scripts\orion.exe --request "What time is it?"
```

To test local reasoning, provision the pinned Qwen3.5-2B GGUF and a compatible `llama-server` build first. See [Phase 14 model and runtime pins](docs/phase14-artifacts.md). Set the local paths in the same PowerShell session:

```powershell
$env:ORION_MODEL_BACKEND = "llama.cpp"
$env:ORION_MODEL_BASE_URL = "http://127.0.0.1:8080/v1"
$env:ORION_MODEL_PATH = Join-Path $env:LOCALAPPDATA "ORION\models\Qwen3.5-2B-Q4_K_M.gguf"
$env:ORION_LLAMA_SERVER_PATH = "C:\path\to\llama-server.exe"
& .\.venv-local\Scripts\orion.exe --request "Explain Kafka briefly."
```

## Raspberry Pi Setup

ORION targets Raspberry Pi 5 with 8 GB RAM and 64-bit Raspberry Pi OS. Core currently declares Python `>=3.11,<3.14`; this Pi reported Python 3.13.5. The voice extra includes KittenTTS 0.1.0, whose `misaki>=0.9.4` dependency did not resolve from the Pi's indexes under Python 3.13. Use a separate Python 3.12 environment for the full voice profile. Keep the system Python unchanged.

### Copy the source tree from Windows

Run these from PowerShell at the repository root. Copy `src`, not only `src/orion`, so the Pi receives the layout expected by `pyproject.toml` (`~/orion/src/orion`).

```powershell
ssh orion@orion.local "mkdir -p ~/orion"
scp -r .\src .\tests .\pyproject.toml .\uv.lock .\README.md orion@orion.local:~/orion/
ssh orion@orion.local
```

If the source was copied as `~/orion/orion/`, it is at the wrong level. Re-copy using the command above; the project root must contain `src/`, `tests/`, and `pyproject.toml`.

### Install system prerequisites and Python environments

Install native build, GPIO, audio, and TTS prerequisites. `lgpio` needs SWIG, compiler/Python headers, and the `liblgpio` development library; `sounddevice` needs PortAudio; KittenTTS phonemization needs eSpeak NG.

```sh
sudo apt update
sudo apt install -y build-essential cmake python3-dev swig liblgpio-dev libportaudio2 espeak-ng unzip curl rsync
```

Use the OS Python 3.13 environment for core install/tests. Use uv-managed Python 3.12 for voice because KittenTTS's current dependency chain does not resolve on this Pi's Python 3.13.

```sh
cd ~/orion
python3 --version
uname -m
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest -q
python -m pip check
deactivate
```

Install uv and create the voice environment:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.12
uv venv --python 3.12 .venv-voice
UV_PROJECT_ENVIRONMENT=.venv-voice uv sync --locked --extra voice --extra wakeword --extra pi --extra test --python 3.12
source .venv-voice/bin/activate
```

`uv.lock` records the exact dependency resolution; use `uv sync --locked` so updates cannot silently change the tested profile. `uv venv` does not install pip by default. If `uv` is not on PATH, run it as `"$HOME/.local/bin/uv"`. The direct KittenTTS wheel URL in the `voice` extra is intentional: uv-only source configuration is ignored by pip and allowed an unrelated PyPI package to be installed previously. The uv config excludes only `spacy-curated-transformers` from Misaki 0.9.4: KittenTTS's path does not import it, and it pulls in Torch's CUDA 13 dependencies, including a wheel built for SBSA rather than Raspberry Pi aarch64. The clean Pi voice install and offline synthesis were verified without that package.

The default and deployed profile remains legacy/VOSK until verified wake-word assets are available. Install wake-word dependencies only when preparing a controlled candidate:

```sh
uv pip install -e '.[voice,pi,wakeword,test]'
```

### Phase 13 wake-word profile

The candidate profile uses custom `Hey Orion` activation and Whisper `base.en`. Startup requires one compatible wake classifier, its matching openWakeWord melspectrogram and embedding ONNX models, a local `whisper-cli` build, the Whisper GGML model, and a provenance/checksum manifest. ORION validates ONNX tensor contracts and runtime compatibility before capture. It does not download or substitute models. Keep Hugging Face offline. Deployment scripts intentionally exclude model directories. See [Phase 13 artifact requirements](docs/phase13-artifacts.md) for exact interfaces and controlled migration with rollback.

After provisioning all artifacts and validating the manifest on Pi, configure these in `/etc/orion/orion.env` for a controlled candidate test. Back up the file first and follow the documented rollback procedure:

```ini
ORION_VOICE_PROFILE=wakeword
ORION_STT_BACKEND=whisper.cpp
ORION_STT_MODEL_PATH=/opt/orion/models/whisper/ggml-base.en.bin
ORION_WHISPER_CLI_PATH=/usr/local/bin/whisper-cli
ORION_TTS_MODEL_PATH=/opt/orion/models/tts/kitten-tts-nano-0.2
ORION_WAKE_WORD_MODEL_PATH=/opt/orion/models/wakeword/hey_orion.onnx
ORION_WAKE_WORD_MELSPECTROGRAM_PATH=/opt/orion/models/wakeword/melspectrogram.onnx
ORION_WAKE_WORD_EMBEDDING_PATH=/opt/orion/models/wakeword/embedding_model.onnx
ORION_WAKE_WORD_MANIFEST_PATH=/opt/orion/models/wakeword/phase13-artifacts.json
```

In the candidate profile GPIO23 toggles persistent microphone mute/unmute; GPIO16 remains speech interruption. The microphone starts unmuted on a fresh database and retains its state across restarts. A muted or stale transcript cannot dispatch an action or confirm a mutation. Keep the service on `ORION_AUDIO_DEVICE="default"` unless another device has passed the configured sample-rate checks. Missing or incompatible assets fail validation before capture starts. VOSK remains active until the full set passes validation.

`ORION_TTS_MODEL_PATH` is optional and points to a local KittenTTS ONNX model directory. If unset, the existing KittenTTS model identifier uses the pre-provisioned Hugging Face cache while runtime offline mode is enforced. A configured local directory never falls back to Hub access when its config or model files are missing.

### Download models and configure runtime

Download the smaller VOSK model used successfully in Pi voice checks:

```sh
mkdir -p models
curl -L --fail -o /tmp/vosk-model.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip -q /tmp/vosk-model.zip -d models
rm /tmp/vosk-model.zip
```

The Phase 14 source selects `llama.cpp`; Ollama is not a supported backend. Provision the pinned Qwen GGUF and build `llama-server` from the exact commit in [Phase 14 model and runtime pins](docs/phase14-artifacts.md). Do not deploy this source to the Pi until Phase 16 promotion and acceptance. The Pi's deployed revision remains on its previous runtime until then. ORION never downloads a model at startup.

KittenTTS setup remains separate. Its first initialization uses pre-provisioned local assets or a Hub cache. Set `HF_HUB_OFFLINE=1` after provisioning to block runtime downloads. Voice startup initializes VOSK and KittenTTS before announcing readiness, including one short discarded TTS warm-up synthesis. On the tested Pi, warm-up took 5.63 seconds online and 4.97 seconds with Hub offline; this moves the cost before readiness and out of the first reply but does not eliminate cold-start synthesis time.

For each interactive shell, activate `.venv-voice` and configure ORION. These environment variables are required because settings revert to defaults in a new shell. Add the exports to the user's `~/.bashrc` after validating the setup if they should persist for interactive logins.

```sh
cd ~/orion
source "$HOME/.local/bin/env"
source .venv-voice/bin/activate
export ORION_STT_MODEL_PATH="$HOME/orion/models/vosk-model-small-en-us-0.15"
export ORION_AUDIO_QUEUE_CHUNKS=100
export ORION_AUDIO_DEVICE="default"
export ORION_MODEL_BACKEND=llama.cpp
export ORION_MODEL_BASE_URL=http://127.0.0.1:8080/v1
export ORION_MODEL_PATH=/opt/orion/models/Qwen3.5-2B-Q4_K_M.gguf
export ORION_LLAMA_SERVER_PATH=/usr/local/bin/llama-server
# Set after a successful online voice check has cached the KittenTTS assets.
export HF_HUB_OFFLINE=1
```

Keep llama.cpp bound to loopback. Verify the model hash before provisioning. A missing model or unreachable local server must fail clearly; ORION has no alternate-provider or cloud fallback.

### Verify hardware and voice

Use the ALSA `default` device for input and output unless a specific PortAudio device has been verified at ORION's configured rates. On the Pi, AIRHUG 01 appears as `AIRHUG 01: USB Audio (hw:0,0)` and accepts 48 kHz directly, but rejects ORION's 16 kHz capture and 24 kHz playback rates. ALSA `default` accepts both configured rates, and the user reports voice interaction appears to work with it. Confirm actual AIRHUG routing with real capture/playback before relying on the setup; earlier Jabra measurements do not validate AIRHUG 01.

Default controls are persistent microphone mute/unmute on BCM GPIO23 (physical pin 16), TTS cutoff on BCM GPIO16 (physical pin 36), and RGB status LED channels: red on BCM GPIO27 (physical pin 13), green on BCM GPIO25 (physical pin 22), and blue on BCM GPIO17 (physical pin 11). Explicit `ORION_VOICE_PROFILE=legacy` selects push-to-talk on GPIO23. Connect each normally-open button between its GPIO and ground; use physical pin 20 or 14 for ground. For the tested common-cathode RGB module with onboard resistors, connect `-` to ground, `R` to GPIO27, `G` to GPIO25, and `B` to GPIO17. Do not connect 5 V to a GPIO pin. Run `pinout` on the Pi if unsure about header numbering. Set `ORION_TTS_INTERRUPT_PIN`, `ORION_STATUS_LED_RED_PIN`, `ORION_STATUS_LED_PIN`, or `ORION_STATUS_LED_BLUE_PIN` to change these assignments. LED states: green means listening, blue means processing, cyan means speaking, amber means muted, and red means error. Mute color overrides other states.

Run checks separately before enabling the systemd service. Do not launch `orion --run-voice` or `orion --run-reminders` manually while `orion.service` is active; both modes use the same reminder database, and only one scheduler may own it:

```sh
uv pip check
python -m pytest -q
python -m sounddevice
python -c 'import gpiozero, lgpio; print("GPIO imports OK")'
orion --check-config
orion --request 'What time is it?'
orion --voice-check
```

`orion --voice-check` should log a short transcript and speak “Local voice pipeline is ready.” In the candidate wakeword profile, say “Hey Orion” and then speak clearly; GPIO23 mutes or unmutes the microphone. The default legacy profile uses GPIO23 for push-to-talk and VOSK. Run `orion --validate-wakeword-assets` before starting the candidate service. `orion --run-voice` also needs a running local model backend; otherwise reasoning turns retry and fail with `FileNotFoundError` when no GGUF is configured for managed llama.cpp.

The target is Debian 13 (trixie), kernel `6.18.50+rpt-pi-2712`, aarch64, Python 3.12.15 for voice, uv 0.12.22, Qwen3.5-2B GGUF, and pinned llama.cpp. Phase 14 is implemented, deployed, and validated on the Pi. The locked Python profile passes `uv pip check` (98 packages), and the installed dedicated runtime passes 140 tests and 34 subtests. The owned-stream TTS fix passed 12 interruption stress cycles; the deployed schema-placeholder guard and 128-token planner cap are active. The user confirmed multi-turn voice interaction works through systemd. Earlier native-abort, timeout, and placeholder failures are historical evidence, not current observed behavior. Reboot, reminder catch-up/retry/recurrence under systemd, sustained-session performance, and clean-image installation remain open.

### Install and operate the systemd service

The production unit runs one `orion --run-voice` process. That process owns voice interaction and the reminder scheduler together; do not start `--run-reminders` alongside it. The scheduler uses an exclusive per-database lock, so a second scheduler process exits instead of creating another reminder-delivery owner. `--run-reminders` remains available for maintenance/manual operation only while the systemd service is stopped.

Deployment record (2026-10-05): the dedicated `orion-service` account, `/opt/orion` runtime, locked Python 3.12 environment, `/var/cache/orion` TTS cache, migrated `/var/lib/orion/orion.sqlite3`, root-managed unit/environment file, and enabled `orion.service` are installed on the Pi. The service reached “Voice assistant started” after launch and after a manual systemd restart; the user confirmed multi-turn voice interaction through the running service. The deployed full suite passed 140 tests and 34 subtests. Reboot startup, reminder catch-up/retry/recurrence under systemd, and sustained-session behavior remain to be tested.

The unit uses a dedicated non-login account, `orion-service`, with no sudo membership and only `audio`/`gpio` supplementary groups. Application files, the venv, and VOSK model are staged under `/opt/orion`; they are group-readable and the unit's `ProtectSystem=strict` makes them read-only at runtime. SQLite state lives in `/var/lib/orion`; writable Hugging Face and uv caches live in `/var/cache/orion`. This avoids granting the service access to the interactive `/home/orion` checkout, uv install, SSH files, or the user's sudo-capable identity.

The current Pi deployment is already installed and enabled. For a fresh deployment, follow these steps once after the voice path passes its pre-deployment hardware check. They install the checkout without copying its `.git`, virtual environments, or existing model directory; copy the VOSK model separately so deployment ownership/modes are explicit:

```sh
sudo useradd --system --user-group --home-dir /var/lib/orion --no-create-home --shell /usr/sbin/nologin --groups audio,gpio orion-service
sudo install -d -o root -g orion-service -m 0750 /opt/orion
sudo install -d -o root -g orion-service -m 0750 /opt/orion/models
sudo install -d -o root -g orion-service -m 0750 /opt/orion/python
sudo install -d -o orion-service -g orion-service -m 0750 /var/lib/orion /var/cache/orion
sudo rsync -a --chown=root:orion-service --chmod=D750,F640 --exclude='.git/' --exclude='.venv*/' --exclude='models/' --exclude='.pytest_cache/' /home/orion/orion/ /opt/orion/
sudo rsync -a --chown=root:orion-service --chmod=D750,F640 /home/orion/orion/models/vosk-model-small-en-us-0.15/ /opt/orion/models/vosk-model-small-en-us-0.15/
sudo install -o root -g root -m 0755 /home/orion/.local/bin/uv /usr/local/bin/uv

sudo env UV_PYTHON_INSTALL_DIR=/opt/orion/python UV_CACHE_DIR=/var/cache/orion/uv-bootstrap /usr/local/bin/uv python install 3.12
sudo install -d -o orion-service -g orion-service -m 0750 /opt/orion/.venv-voice
sudo -u orion-service env HOME=/var/lib/orion UV_PYTHON_INSTALL_DIR=/opt/orion/python UV_CACHE_DIR=/var/cache/orion/uv-cache UV_PROJECT_ENVIRONMENT=/opt/orion/.venv-voice /usr/local/bin/uv sync --locked --no-install-project --extra voice --extra wakeword --extra pi --extra test --python 3.12 --project /opt/orion
sudo -u orion-service mkdir -p /var/cache/orion/build/src /var/cache/orion/wheels
sudo -u orion-service rsync -a --chown=orion-service:orion-service --chmod=D750,F640 /opt/orion/src/ /var/cache/orion/build/src/
sudo -u orion-service install -m 0640 /opt/orion/pyproject.toml /var/cache/orion/build/pyproject.toml
sudo -u orion-service install -m 0640 /opt/orion/README.md /var/cache/orion/build/README.md
sudo -u orion-service sh -c 'cd /var/cache/orion/build && HOME=/var/lib/orion UV_NO_CONFIG=1 UV_CACHE_DIR=/var/cache/orion/uv-cache /usr/local/bin/uv build --wheel --out-dir /var/cache/orion/wheels .'
sudo -u orion-service sh -c 'cd /var/cache/orion/wheels && HOME=/var/lib/orion UV_NO_CONFIG=1 UV_CACHE_DIR=/var/cache/orion/uv-cache /usr/local/bin/uv pip install --python /opt/orion/.venv-voice/bin/python --no-deps ./orion_assistant-0.1.0-py3-none-any.whl'

sudo install -d -o root -g orion-service -m 0750 /etc/orion
sudo install -o root -g orion-service -m 0640 systemd/orion.service /etc/systemd/system/orion.service
sudo install -o root -g orion-service -m 0640 systemd/orion.env.example /etc/orion/orion.env
sudo systemctl daemon-reload
```

Populate the dedicated account's TTS cache once while online, before setting offline mode for service operation:

```sh
sudo -u orion-service env HOME=/var/lib/orion HF_HOME=/var/cache/orion/huggingface HF_HUB_OFFLINE=0 /opt/orion/.venv-voice/bin/python -c 'from kittentts import KittenTTS; KittenTTS("KittenML/kitten-tts-nano-0.2").generate("Ready.", voice="expr-voice-3-f")'
```

The service creates `/var/lib/orion/orion.sqlite3` itself on a new install. On migration, stop every ORION process first and back up the old SQLite database using the online backup API as the old owner, then install the verified backup with the new ownership:

```sh
python3 -c 'import sqlite3; source=sqlite3.connect("/home/orion/.local/share/orion/orion.sqlite3"); target=sqlite3.connect("/tmp/orion.sqlite3"); source.backup(target); target.close(); source.close()'
sudo install -o orion-service -g orion-service -m 0600 /tmp/orion.sqlite3 /var/lib/orion/orion.sqlite3
rm /tmp/orion.sqlite3
```

Do not overwrite an existing `/var/lib/orion/orion.sqlite3`; take and verify a backup first. SQLite WAL state is included by the online backup API. Once the cache is populated, any database migration is complete, and Phase 9 voice acceptance is closed, enable the unit:

```sh
sudo systemctl enable --now orion.service
```

The environment file is root-owned and group-readable by `orion-service`; it contains no credentials. The source example selects legacy/VOSK, ALSA `default`, pinned llama.cpp loopback, the Qwen GGUF path, offline Hugging Face cache, uv cache/Python directories, and the systemd-owned database. Keep the deployed Pi on its previous revision until Phase 16 candidate acceptance. If the wired pins or device routing differ, edit the file before installation. Unit paths are intentionally explicit.

Inspect and control the service with:

```sh
sudo systemctl status orion.service
sudo journalctl -u orion.service -f
sudo systemctl restart orion.service
sudo systemctl stop orion.service
sudo systemctl disable orion.service
```

The unit restarts on unexpected process failure, not on a clean stop. It allows up to 150 seconds for orderly shutdown because an in-flight bounded model request may use the 120-second task limit. `SIGTERM` asks the service to stop and requests TTS interruption. If stop reaches the timeout, inspect the journal before using `systemctl kill`; do not launch a second voice process until the old process has exited.

For an upgrade, first stop the service, back up `/var/lib/orion/orion.sqlite3` using the online backup API above, then stage the new source into `/opt/orion` with the same rsync excludes/ownership and run `sudo -u orion-service env HOME=/var/lib/orion UV_PYTHON_INSTALL_DIR=/opt/orion/python UV_CACHE_DIR=/var/cache/orion/uv-cache UV_PROJECT_ENVIRONMENT=/opt/orion/.venv-voice /usr/local/bin/uv sync --locked --no-editable --extra voice --extra wakeword --extra pi --extra test --python 3.12 --project /opt/orion`. Then reload and restart the service. Verify `systemctl status`, the journal, local database access, and one voice/reminder check before considering the upgrade complete. For rollback, stop the service, restore the previous source revision and its matching locked environment, restore the pre-upgrade database backup if schema/data changed, then start and inspect the service. Keep backups outside `/var/lib/orion` and restrict them to the `orion-service` account.

Troubleshooting:

- `Permission denied` for GPIO/audio: check `id orion-service`, device ownership, and the `audio`/`gpio` groups, then restart the service after group changes. Do not solve this by running ORION as root.
- Audio device or sample-rate error: run `/opt/orion/.venv-voice/bin/python -m sounddevice` as `orion-service`, check `ORION_AUDIO_DEVICE` in `/etc/orion/orion.env`, then run input/output `check_*_settings` at ORION's configured rates. On the tested Pi, ALSA `default` accepts 16 kHz input and 24 kHz output; direct AIRHUG requires 48 kHz.
- `Another reminder scheduler already owns this database`: keep the systemd voice service as the sole owner; stop it before any manual `--run-reminders` maintenance process.
- TTS model unavailable with `HF_HUB_OFFLINE=1`: run the one-time online KittenTTS synthesis as `orion-service` using `HF_HOME=/var/cache/orion/huggingface`, then restore offline mode and restart the service.
- Local reasoning unavailable: check the configured `ORION_LLAMA_SERVER_PATH`, `ORION_MODEL_PATH`, model SHA-256, and `curl http://127.0.0.1:8080/health`. Keep llama.cpp bound to loopback.

The live Pi and source default remain on legacy/VOSK until the complete Phase 13 asset set is verified. The wakeword candidate uses GPIO23 for persistent microphone mute and GPIO16 to interrupt playback. `main.py`, `main_llama.py`, and `stt_lib.py` are historical upstream files, not supported launch commands.

### Phase 11 measurement capture

`scripts/phase11-capture.sh` writes a timestamped capture directory with target/model/package metadata, audio-device listings, systemd status, Ollama inventory, one-second resource samples, and the ORION journal window. Run it with sufficient privileges to read the service journal and Pi thermal/throttle data. Give each capture an explicit workload label so an idle baseline cannot be confused with a reasoning/STT/TTS run:

```sh
sudo /opt/orion/scripts/phase11-capture.sh 300 1 /var/lib/orion/phase11 "idle-listening"
```

The arguments are capture duration in seconds, sampling interval in seconds, output root, and workload label. During separate captures, exercise idle listening, repeated STT, several reasoning turns, TTS output, and reminder delivery. Run separate captures for cold model start and warm requests. A preliminary 30-second idle capture measured ORION RSS 488 MB, at least 7.69 GB available RAM, no swap, 46.1-48.3 C, and no throttling. A 120-second interactive capture across a time request and two reasoning turns measured peak ORION RSS 499 MB, Ollama RSS 2.39 GB, at least 5.29 GB available RAM, no swap, and 58.7 C maximum; planner calls took 10.9/8.1 seconds, STT 2.49/2.85 seconds, and TTS 6.11/14.74 seconds. The PTT wait was separately recorded and included user think time. These short captures are not cold-start peak or sustained results. For a sustained run, use at least 30 minutes and record power supply, cooling, and ambient temperature manually; software cannot reliably identify those physical conditions. Review `resources.csv`/`summary.txt` for RSS, CPU, memory headroom, swap, temperature, and throttling; CPU percentages are interval deltas summed across cores and can exceed 100%. Review `orion-journal.log` for stage timings and errors. The script records no microphone data or transcript content. Do not compare models unless prompt, context, quantization, runtime, and thermal conditions are held constant.

### Windows shortcuts

Run local tests and CLI smoke checks:

```powershell
.\scripts\test-local.ps1
```

The local shortcut invokes ORION with `python -m orion`; this avoids Windows Application Control blocking the generated `orion.exe` launcher.

Phase 14 is the production LLM runtime. First provision the pinned GGUF and `llama-server` build described in [Phase 14 model and runtime pins](docs/phase14-artifacts.md). The deploy script verifies both pins before stopping the service. It packages source while excluding virtual environments, models, secrets, and databases; then backs up source, systemd unit, environment, and SQLite before setting the managed model keys. On failure, it restores source, unit, environment, and the previous service; the SQLite backup remains for manual recovery. This promotes the LLM runtime only; wakeword, audio, sustained performance, and fresh-image acceptance remain separate gates.

```powershell
\.\scripts\deploy-pi.ps1
\.\scripts\deploy-pi.ps1 -Target orion.local
# Use an explicit identity only when SSH config alias is unavailable:
\.\scripts\deploy-pi.ps1 -Target orion@192.168.0.9 -SshIdentityFile "$HOME\.ssh\orion_pi_ed25519"
# Override if ORION_DATABASE_PATH differs from /var/lib/orion/orion.sqlite3:
\.\scripts\deploy-pi.ps1 -DatabasePath /custom/path/orion.sqlite3
```

Restart the Pi service without deploying:

```powershell
.\scripts\restart-pi.ps1
.\scripts\restart-pi.ps1 -Target orion@192.168.1.50
```

The Pi scripts require OpenSSH Client, working public-key authentication, and an SSH account authorized to run the required `sudo` commands. Use the host alias from SSH config or pass `-SshIdentityFile` when targeting an IP; the script checks key authentication first and never falls back to SSH password prompts. Deployment asks for the Pi sudo password once, keeps it in process memory, and reuses it through SSH standard input. It disposes the cached `SecureString` when the script exits; it does not write the password to disk or include it in remote command arguments. Deployment retains source, systemd, environment, and SQLite backups under `/var/backups/orion`. It does not provision models or migrate the database during file sync. The script changes only model runtime keys in `/etc/orion/orion.env`; it preserves audio and wake settings. The restarted application may perform its normal additive schema initialization. Rollback restores source, unit, and environment, not the database; keep the protected SQLite backup for manual recovery. Verify AIRHUG 01's PortAudio selector and test physical audio after deployment.