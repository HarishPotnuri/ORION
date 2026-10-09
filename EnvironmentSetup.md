# ORION Environment Setup

End-to-end setup for a new Windows development machine and a freshly imaged Raspberry Pi 5 8 GB. Commands are for the repository state dated 2026-10-07. Read this file from top to bottom for a first install.

Phase 14 is implemented, deployed, and validated on the Pi. The accepted operational GGUF on Pi and in `ORION-MODELS` has the fingerprint listed below. It differs from the upstream immutable-revision artifact fingerprint recorded in `docs/phase14-artifacts.md`; deployment pins the accepted Pi/depot copy. Never copy model files into the repository.

## 1. Target And Pins

| Item | Pin or value |
| --- | --- |
| Pi | Raspberry Pi 5, 8 GB, 64-bit Raspberry Pi OS Lite |
| Pi architecture | `aarch64` |
| Pi voice Python | 3.12, managed by uv |
| Python dependency lock | `uv.lock`, always use `uv sync --locked` |
| Current Pi voice STT profile | VOSK small English `0.15` |
| LLM model | Post-trained `Qwen/Qwen3.5-2B`, GGUF `Q4_K_M` |
| Accepted Pi/depot GGUF | `Qwen3.5-2B-Q4_K_M.gguf`, 1,312,165,056 bytes |
| Accepted Pi/depot SHA-256 | `a8b50de00b780c0f3c16f07cb041a1ecd678b6e22fd8cde91e57fe4b5031523b` |
| GGUF repository revision | `unsloth/Qwen3.5-2B-GGUF`, `f6d5376be1edb4d416d56da11e5397a961aca8ae` |
| llama.cpp source commit | `d81235049384534c167caea52b85a694f6103d14` (`v0.6.0`) |
| Context | 4096 total, 3072 input, up to 512 answer, 512 template/safety reserve |
| Planner output | 128 tokens |
| Pi model root | `/opt/orion/models` |
| Pi model path | `/opt/orion/models/LLM/Qwen3.5-2B-Q4_K_M.gguf` |
| Pi server path | `/usr/local/bin/llama-server` |
| Pi endpoint | `http://127.0.0.1:8080/v1`, loopback only |
| Pi state database | `/var/lib/orion/orion.sqlite3` |

`llama.cpp` GitHub release tag is unsigned. The immutable source commit is the pin. Model provenance, license, checksum, build command, and candidate status are also recorded in [docs/phase14-artifacts.md](docs/phase14-artifacts.md).

## 2. Windows Development Machine

### Install prerequisites

Install Git, Python 3.11, and OpenSSH Client. In an elevated PowerShell, install Git and Python:

```powershell
winget install --exact --id Git.Git
winget install --exact --id Python.Python.3.11
```

OpenSSH Client is included in current Windows versions. If missing, add the optional Windows capability from elevated PowerShell:

```powershell
Get-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
```

For local model builds, install CMake and Visual Studio 2022 Build Tools with **Desktop development with C++**. The optional C++ build is not needed for running tests or deploying to Pi. Restart PowerShell after installation so updated `PATH` values load.

Confirm Python 3.11 is available:

```powershell
py -3.11 --version
ssh.exe -V
git --version
```

If the repository is private, configure GitHub authentication using the approved account. Do not put tokens in this file or in command history.

### Clone and test ORION

```powershell
New-Item -ItemType Directory -Path (Join-Path $HOME 'source') -Force | Out-Null
cd $HOME\source
git clone https://github.com/HarishPotnuri/ORION.git
cd .\ORION
py -3.11 -m venv .venv-local
& .\.venv-local\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-local\Scripts\python.exe -m pip install -e '.[test]'
& .\.venv-local\Scripts\python.exe -m pytest -q
& .\.venv-local\Scripts\python.exe -m orion --check-config
& .\.venv-local\Scripts\python.exe -m orion --request 'What time is it?'
```

Run the repository's local test and CLI checks with:

```powershell
.\scripts\test-local.ps1
```

This creates a temporary database and removes it after completion.

### Use adjacent offline model depot

Keep `ORION-MODELS` adjacent to the `ORION` checkout, not inside the Git repository. The application maps its supplied uppercase directories directly. Set the root explicitly when it is not adjacent:

```powershell
$ModelRoot = Join-Path (Split-Path -Parent (Get-Location).Path) 'ORION-MODELS'
if (-not (Test-Path (Join-Path $ModelRoot 'LLM\Qwen3.5-2B-Q4_K_M.gguf'))) { throw "Q4 model missing under $ModelRoot" }
$env:ORION_MODEL_ROOT = $ModelRoot
$ModelFile = Join-Path $ModelRoot 'LLM\Qwen3.5-2B-Q4_K_M.gguf'
$ModelHash = (Get-FileHash -Algorithm SHA256 $ModelFile).Hash.ToLowerInvariant()
if ($ModelHash -ne 'a8b50de00b780c0f3c16f07cb041a1ecd678b6e22fd8cde91e57fe4b5031523b') { throw 'GGUF SHA-256 does not match accepted Pi/depot artifact.' }
```

Do not fetch models from application startup. Provision them into the depot through the separately approved offline process, and record each model's source, revision, checksum, and license before Pi use.

### Optional Windows llama.cpp build

Run from a Visual Studio Developer PowerShell with the x64 C++ toolchain active. This builds the same source commit for Windows; never copy the Windows executable to Pi.

```powershell
cd $HOME\source
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git llama.cpp-phase14
Set-Location .\llama.cpp-phase14
git checkout --detach d81235049384534c167caea52b85a694f6103d14
cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENCL=OFF
cmake --build build --config Release --target llama-server --parallel 4
$Server = Get-ChildItem .\build -Recurse -Filter llama-server.exe | Select-Object -First 1 -ExpandProperty FullName
if (-not $Server) { throw 'llama-server.exe was not produced.' }
& $Server --version
```

Start a loopback-only server in a separate terminal. Find the exact executable path from `$Server`; then configure ORION in the request terminal:

```powershell
& $Server -m $ModelFile --host 127.0.0.1 --port 8080 -c 4096 -t 4 --jinja --no-mmproj --reasoning-format deepseek
```

```powershell
$env:ORION_MODEL_BACKEND = 'llama.cpp'
$env:ORION_MODEL_BASE_URL = 'http://127.0.0.1:8080/v1'
$env:ORION_MODEL_ROOT = $ModelRoot
$env:ORION_MODEL_PATH = $ModelFile
$env:ORION_LLAMA_SERVER_PATH = $Server
$env:ORION_MODEL_CONTEXT_SIZE = '4096'
$env:ORION_MAX_CONTEXT_TOKENS = '3072'
$env:ORION_MODEL_ANSWER_TOKENS = '512'
$env:ORION_MODEL_PLANNER_TOKENS = '128'
& .\.venv-local\Scripts\python.exe -m orion --request 'Explain Kafka briefly.'
```

Close the server with `Ctrl+C`. Do not expose the server port beyond loopback.

## 3. Prepare Windows SSH Access To Pi

Install [Raspberry Pi Imager](https://www.raspberrypi.com/software/) on Windows and flash **Raspberry Pi OS Lite 64-bit** to the Pi SD card. Select current 64-bit Raspberry Pi OS Lite for Raspberry Pi 5. In Imager advanced settings:

1. Set hostname to `orion`.
2. Create login user `orion` and a strong password for local console/sudo recovery.
3. Configure Wi-Fi or Ethernet.
4. Enable SSH and install the Windows machine's Ed25519 public key. Keep the private key only on Windows.

If a key does not exist on Windows, create one:

```powershell
ssh-keygen -t ed25519 -f "$HOME\.ssh\orion_pi_ed25519"
Get-Content "$HOME\.ssh\orion_pi_ed25519.pub"
```

Enter public key in Imager. Do not paste the private key into Imager or chat.

If the private key has a passphrase, start Windows OpenSSH Agent and add the key in the current user session. Run service configuration once from elevated PowerShell if `ssh-agent` is disabled:

```powershell
Set-Service ssh-agent -StartupType Automatic
Start-Service ssh-agent
ssh-add "$HOME\.ssh\orion_pi_ed25519"
```

Type the key passphrase only into the local terminal prompt. `deploy-pi.ps1` uses `BatchMode=yes`; without an unlocked agent or unencrypted key, it stops instead of prompting for an SSH password.

Find the Pi IP in router DHCP leases or on the Pi console with `hostname -I`. Connect by IP for initial setup:

```powershell
ssh -i "$HOME\.ssh\orion_pi_ed25519" -o IdentitiesOnly=yes orion@<PI-IP>
```

After first successful connection, create `%USERPROFILE%\.ssh\config` with:

```sshconfig
Host orion.local
    HostName <PI-IP>
    User orion
    IdentityFile ~/.ssh/orion_pi_ed25519
    IdentitiesOnly yes
```

Replace `<PI-IP>` with the Pi's address. Verify the alias:

```powershell
ssh -o BatchMode=yes orion.local 'hostname; uname -m'
```

Expected architecture is `aarch64`. If `orion.local` does not resolve on Windows, use the IP plus `-i "$HOME\.ssh\orion_pi_ed25519" -o IdentitiesOnly=yes`. `scripts/deploy-pi.ps1` forces key-only SSH; it does not fall back to SSH password prompts.

## 4. Bootstrap A Fresh Pi OS

The following commands run in an SSH session as login user `orion`. Keep one session open. Type sudo password directly in that terminal when prompted. Never place it in a file or command argument.

### Update OS and install prerequisites

```sh
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y git cmake build-essential pkg-config python3-dev swig liblgpio-dev libportaudio2 espeak-ng unzip curl rsync sqlite3
sudo reboot
```

Reconnect after reboot, then verify:

```sh
ssh orion.local
hostname
uname -m
```

### Install service account and directories

```sh
getent passwd orion-service >/dev/null || sudo useradd --system --user-group --home-dir /var/lib/orion --no-create-home --shell /usr/sbin/nologin --groups audio,gpio orion-service
sudo install -d -o root -g orion-service -m 0750 /opt/orion /opt/orion/python
sudo install -d -o root -g orion-service -m 0755 /opt/orion/models
sudo install -d -o orion-service -g orion-service -m 0750 /var/lib/orion /var/cache/orion
sudo install -d -o orion-service -g orion-service -m 0750 /var/cache/orion/uv-cache /var/cache/orion/huggingface
```

### Copy checkout from Windows

Open PowerShell at the Windows ORION checkout. This transfers source and configuration, not virtual environments, model files, secrets, databases, or `.git` history:

```powershell
ssh orion.local 'mkdir -p /home/orion/orion'
scp -r .\src .\tests .\scripts .\systemd .\docs .\pyproject.toml .\uv.lock .\README.md .\Plan.md .\Issues.md orion.local:/home/orion/orion/
```

Return to the Pi SSH session. Verify `pyproject.toml` is at `/home/orion/orion/pyproject.toml`. Install source as root-owned read-only files:

```sh
sudo rsync -a --delete --chown=root:orion-service --chmod=D750,F640 /home/orion/orion/ /opt/orion/
sudo find /opt/orion/scripts -type f -name '*.sh' -exec chmod 0750 {} +
```

### Install uv and locked Python 3.12 environment

The Pi OS Python remains system-managed. ORION uses a separate uv-managed Python 3.12 because the voice dependency profile does not resolve on this Pi's Python 3.13.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo install -o root -g root -m 0755 /home/orion/.local/bin/uv /usr/local/bin/uv
sudo env UV_PYTHON_INSTALL_DIR=/opt/orion/python UV_CACHE_DIR=/var/cache/orion/uv-bootstrap /usr/local/bin/uv python install 3.12
sudo install -d -o orion-service -g orion-service -m 0750 /opt/orion/.venv-voice
sudo -u orion-service env HOME=/var/lib/orion UV_PYTHON_INSTALL_DIR=/opt/orion/python UV_CACHE_DIR=/var/cache/orion/uv-cache UV_PROJECT_ENVIRONMENT=/opt/orion/.venv-voice /usr/local/bin/uv sync --locked --no-install-project --no-install-package tflite-runtime --extra voice --extra wakeword --extra embedding --extra pi --extra test --python 3.12 --project /opt/orion
```

Build and install ORION wheel into that locked environment. This keeps the application tree root-owned and read-only:

```sh
sudo -u orion-service mkdir -p /var/cache/orion/build/src /var/cache/orion/wheels
sudo -u orion-service rsync -a --delete --chown=orion-service:orion-service --chmod=D750,F640 /opt/orion/src/ /var/cache/orion/build/src/
sudo -u orion-service install -m 0640 /opt/orion/pyproject.toml /var/cache/orion/build/pyproject.toml
sudo -u orion-service install -m 0640 /opt/orion/README.md /var/cache/orion/build/README.md
sudo -u orion-service sh -c 'cd /var/cache/orion/build && HOME=/var/lib/orion UV_NO_CONFIG=1 UV_CACHE_DIR=/var/cache/orion/uv-cache /usr/local/bin/uv build --wheel --out-dir /var/cache/orion/wheels .'
sudo -u orion-service sh -c 'cd /var/cache/orion/wheels && HOME=/var/lib/orion UV_NO_CONFIG=1 UV_CACHE_DIR=/var/cache/orion/uv-cache /usr/local/bin/uv pip install --python /opt/orion/.venv-voice/bin/python --no-deps ./orion_assistant-0.1.0-py3-none-any.whl'
sudo -u orion-service env HOME=/var/lib/orion UV_PYTHON_INSTALL_DIR=/opt/orion/python UV_CACHE_DIR=/var/cache/orion/uv-cache UV_PROJECT_ENVIRONMENT=/opt/orion/.venv-voice /usr/local/bin/uv pip check --python /opt/orion/.venv-voice/bin/python
```

### Copy offline depot and provision VOSK/KittenTTS fallback assets

From PowerShell at the ORION checkout, copy the adjacent depot once during provisioning. This is a manual asset transfer; model selection and runtime startup never download files.

```powershell
scp -r ..\ORION-MODELS orion.local:/home/orion/
```

On Pi, copy the uppercase depot into the configured runtime root. Keep the VOSK fallback directory alongside those depot categories; the depot itself does not include VOSK.

```sh
sudo install -d -o root -g orion-service -m 0755 /opt/orion/models
sudo rsync -a --chown=root:orion-service --chmod=D755,F644 /home/orion/ORION-MODELS/ /opt/orion/models/
```

Verify expected files before starting ORION:

```sh
find /opt/orion/models/LLM /opt/orion/models/STT /opt/orion/models/TTS /opt/orion/models/EMBEDDING -maxdepth 1 -type f -printf '%p\n'
```

Do not assume every listed model is Pi-compatible or licensed for redistribution. Run `orion-model list` and `orion-model validate <id>` after installation; benchmark candidate LLMs one at a time.

```sh
mkdir -p /home/orion/orion/models
curl -L --fail --retry 5 --retry-all-errors -o /tmp/vosk-model.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip -q /tmp/vosk-model.zip -d /home/orion/orion/models
rm /tmp/vosk-model.zip
sudo install -d -o root -g orion-service -m 0755 /opt/orion/models
sudo rsync -a --chown=root:orion-service --chmod=D755,F644 /home/orion/orion/models/vosk-model-small-en-us-0.15/ /opt/orion/models/vosk-model-small-en-us-0.15/
sudo -u orion-service env HOME=/var/lib/orion HF_HOME=/var/cache/orion/huggingface HF_HUB_OFFLINE=0 /opt/orion/.venv-voice/bin/python -c 'from kittentts import KittenTTS; KittenTTS("KittenML/kitten-tts-nano-0.2").generate("Ready.", voice="expr-voice-3-f")'
```

This online step provisions the KittenTTS cache only. Later runtime uses `HF_HUB_OFFLINE=1`. Do not proceed if licenses or model provenance are unacceptable. A local TTS directory can be configured with `ORION_TTS_MODEL_PATH` instead of the Hub cache.

## 5. Build And Install Pi LLM Runtime

Build on the Pi's ARM64 architecture. Do not build on Windows and copy that binary. The CMake configure step retrieves llama.cpp's bundled UI assets; the Pi needs internet during this provisioning step. Runtime inference itself stays on loopback and downloads no model.

```sh
cd /home/orion
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git llama.cpp-phase14
cd /home/orion/llama.cpp-phase14
git checkout --detach d81235049384534c167caea52b85a694f6103d14
test "$(git rev-parse HEAD)" = d81235049384534c167caea52b85a694f6103d14
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENCL=OFF
cmake --build build --config Release --target llama-server --parallel 4
./build/bin/llama-server --version
```

Expected version contains commit `d81235049`. Install the verified Q4 artifact from the copied offline depot. The application does not download model files.

```powershell
scp ..\ORION-MODELS\LLM\Qwen3.5-2B-Q4_K_M.gguf orion.local:/home/orion/ORION-MODELS/LLM/
```

On Pi, install binary and its colocated shared libraries. Both `orion` login and `orion-service` need read/execute access because deploy preflight runs as login user and systemd runs as service user. Model remains read-only:

```sh
printf '%s  %s\n' 'a8b50de00b780c0f3c16f07cb041a1ecd678b6e22fd8cde91e57fe4b5031523b' /home/orion/ORION-MODELS/LLM/Qwen3.5-2B-Q4_K_M.gguf | sha256sum --check
sudo install -d -o root -g orion-service -m 0755 /opt/orion/models/LLM /opt/orion/llama.cpp/bin
sudo install -o root -g orion-service -m 0644 /home/orion/ORION-MODELS/LLM/Qwen3.5-2B-Q4_K_M.gguf /opt/orion/models/LLM/Qwen3.5-2B-Q4_K_M.gguf
sudo cp -a /home/orion/llama.cpp-phase14/build/bin/. /opt/orion/llama.cpp/bin/
sudo chown -R root:orion-service /opt/orion/llama.cpp
sudo find /opt/orion/llama.cpp -type d -exec chmod 0755 {} +
sudo find /opt/orion/llama.cpp -type f -exec chmod 0644 {} +
sudo chmod 0755 /opt/orion/llama.cpp/bin/llama-server
sudo ln -sfn /opt/orion/llama.cpp/bin/llama-server /usr/local/bin/llama-server
```

Verify as login user:

```sh
test -x /usr/local/bin/llama-server
test -r /opt/orion/models/LLM/Qwen3.5-2B-Q4_K_M.gguf
printf '%s  %s\n' 'a8b50de00b780c0f3c16f07cb041a1ecd678b6e22fd8cde91e57fe4b5031523b' /opt/orion/models/LLM/Qwen3.5-2B-Q4_K_M.gguf | sha256sum --check
/usr/local/bin/llama-server --version
```

Start server only for a manual local smoke test, with no service active on port 8080:

```sh
/usr/local/bin/llama-server -m /opt/orion/models/LLM/Qwen3.5-2B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -c 4096 -t 4 --jinja --no-mmproj --reasoning-format deepseek
```

In another Pi shell, verify model health, the model chat template, tokenizer, and one streamed completion. Stop temporary server with `Ctrl+C` before starting systemd. Never bind it to `0.0.0.0`.

```sh
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/props
curl -fsS -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Hello"}]}' http://127.0.0.1:8080/apply-template
curl -fsS -H 'Content-Type: application/json' -d '{"content":"Hello","add_special":false,"parse_special":true}' http://127.0.0.1:8080/tokenize
curl -fsS -N -H 'Content-Type: application/json' -d '{"model":"local-model","messages":[{"role":"user","content":"Reply with exactly: smoke test passed"}],"max_tokens":32,"temperature":0,"stream":true,"stream_options":{"include_usage":true},"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false}}' http://127.0.0.1:8080/v1/chat/completions
```

Stream must contain text chunks, usage, a finish reason, and `data: [DONE]`. Confirm `/props` reports the exact GGUF path. Do not record a successful runtime test if model loading reports an error.

## 6. Configure And Start Systemd

Install the unit and environment template. Copying the template creates a fresh candidate config; edit only expected values and preserve settings you already verified for hardware. On this deployment, `ORION_AUDIO_DEVICE=sysdefault` is a measured capture setting. Some tests found `default` negotiates 16/24 kHz but can be near-silent; check the actual selected microphone and speaker after every fresh OS image.

```sh
sudo install -d -o root -g orion-service -m 0750 /etc/orion
sudo install -o root -g orion-service -m 0640 /opt/orion/systemd/orion.service /etc/systemd/system/orion.service
sudo install -o root -g orion-service -m 0640 /opt/orion/systemd/orion.env.example /etc/orion/orion.env
sudoedit /etc/orion/orion.env
```

Confirm these settings in `/etc/orion/orion.env`:

```ini
ORION_DATABASE_PATH=/var/lib/orion/orion.sqlite3
ORION_MODEL_ROOT=/opt/orion/models
ORION_VOSK_MODEL_PATH=/opt/orion/models/vosk-model-small-en-us-0.15
ORION_AUDIO_DEVICE=sysdefault
ORION_MODEL_BACKEND=llama.cpp
ORION_MODEL_BASE_URL=http://127.0.0.1:8080/v1
ORION_MODEL_PATH=/opt/orion/models/LLM/Qwen3.5-2B-Q4_K_M.gguf
ORION_LLAMA_SERVER_PATH=/usr/local/bin/llama-server
ORION_MODEL_CONTEXT_SIZE=4096
ORION_MAX_CONTEXT_TOKENS=3072
ORION_MODEL_ANSWER_TOKENS=512
ORION_MODEL_PLANNER_TOKENS=128
ORION_VOICE_PROFILE=legacy
ORION_STT_BACKEND=vosk
ORION_ONLINE_TOOLS_ENABLED=false
HF_HUB_OFFLINE=1
HF_HOME=/var/cache/orion/huggingface
UV_CACHE_DIR=/var/cache/orion/uv-cache
UV_PYTHON_INSTALL_DIR=/opt/orion/python
```

The selection file defaults to `/var/lib/orion/model-selection.json`, beside SQLite. Keep it writable by `orion-service`. Run model commands as that account so selection survives service restart:

```sh
sudo -u orion-service /opt/orion/.venv-voice/bin/orion-model list
sudo -u orion-service /opt/orion/.venv-voice/bin/orion-model current
sudo -u orion-service /opt/orion/.venv-voice/bin/orion-model validate qwen3.5-2b-q4
sudo -u orion-service /opt/orion/.venv-voice/bin/orion-model use qwen3.5-2b-q4
sudo systemctl restart orion.service
sudo systemctl stop orion.service
sudo -u orion-service /opt/orion/.venv-voice/bin/orion-model benchmark --all
sudo systemctl start orion.service
```

`orion-model use` updates persisted selection; it does not restart the service. Benchmarking loads each GGUF sequentially and can consume most available RAM. Run it only as a supervised hardware test. Results default to `/var/lib/orion/model-benchmarks.json`.

Do not add API keys or passwords. Keep wakeword disabled and VOSK selected until custom `Hey Orion` assets pass separate provenance, license, ONNX, and acoustic acceptance.

Run local checks as the service account. For manual CLI commands, source the environment file because systemd is the normal loader. Then start the unit:

```sh
sudo -u orion-service bash -lc 'set -a; . /etc/orion/orion.env; set +a; /opt/orion/.venv-voice/bin/orion --check-config'
sudo -u orion-service bash -lc 'set -a; . /etc/orion/orion.env; set +a; /opt/orion/.venv-voice/bin/orion --request "What time is it?"'
sudo systemctl daemon-reload
sudo systemctl enable --now orion.service
sudo systemctl is-active orion.service
sudo journalctl -u orion.service -n 50 --no-pager
```

The first command that reasons may start the managed llama-server. Deterministic time/date/tool requests do not require model initialization. Do not run a second `--run-voice` or `--run-reminders` process against the same database while systemd is active.

## 7. Windows Deployment To Pi

Use Windows OpenSSH Client and key authentication. Create key only if needed, and add its `.pub` content through Imager or an already authenticated Pi session:

```powershell
ssh-keygen -t ed25519 -f "$HOME\.ssh\orion_pi_ed25519"
Get-Content "$HOME\.ssh\orion_pi_ed25519.pub"
```

Recommended `%USERPROFILE%\.ssh\config` entry:

```sshconfig
Host orion.local
    HostName <PI-IP>
    User orion
    IdentityFile ~/.ssh/orion_pi_ed25519
    IdentitiesOnly yes
```

Test key auth before deploy:

```powershell
ssh -o BatchMode=yes orion.local 'hostname; uname -m'
```

Phase 14 remains the production LLM runtime. The deploy script checks the pinned Q4 hash and llama.cpp commit before stopping the service, backs up source/unit/environment/database, installs locked voice/wakeword/embedding dependencies, and rolls source/config back if deployment fails. It sets `ORION_MODEL_ROOT` and accepts Q4 under `LLM/` or at the historical flat path. It does not copy the depot or build llama.cpp; provision offline assets first. Use `-RemoteModelRoot` for a non-default Pi depot path. Wakeword, audio, sustained performance, and fresh-image acceptance remain separate gates.

After the depot is copied to `/opt/orion/models` (or the path passed through `-RemoteModelRoot`) and service prerequisites are present:

```powershell
.\scripts\deploy-pi.ps1 -Target orion.local
```

For a non-default depot location:

```powershell
.\scripts\deploy-pi.ps1 -Target orion.local -RemoteModelRoot /srv/orion-models
```

For an IP target without SSH alias configuration:

```powershell
.\scripts\deploy-pi.ps1 -Target orion@<PI-IP> -SshIdentityFile "$HOME\.ssh\orion_pi_ed25519"
```

Script asks once for sudo password; it keeps the value in process memory and does not write it to disk. SSH uses keys only and fails fast rather than asking for SSH password. Check every error before retrying. If DNS alias fails, fix alias or use the Pi IP; do not repeat an unreachable deployment.

## 8. Fresh-Image Acceptance And Recovery

Before relying on a fresh image, prove all of these on that image:

- `uname -m` is `aarch64`; Python voice environment uses 3.12 and `uv sync --locked` succeeds.
- Service account has only required `audio` and `gpio` groups. Application/model paths are read-only to service; database and caches are writable by `orion-service`.
- GGUF SHA-256 and llama.cpp commit match this guide. `/usr/local/bin/llama-server --version` includes `d81235049`.
- `orion --check-config`, deterministic typed request, voice check, local reasoning request, capture/playback, interruption, mute, and LED checks pass.
- Run audio/GPIO checks as `orion-service`: `sudo -u orion-service /opt/orion/.venv-voice/bin/python -m sounddevice` and `sudo -u orion-service /opt/orion/.venv-voice/bin/python -c 'import gpiozero, lgpio; print("GPIO imports OK")'`. Then run `orion --voice-check` with systemd stopped, test real microphone/speaker routing, buttons, and LEDs.
- Disable external network after model/TTS/artifact provisioning. Confirm inference works disconnected and weather/search remain disabled.
- Reboot. Verify systemd readiness, reminder recovery, mute persistence, no duplicate reminder, and clean shutdown.
- Measure cold start, repeated turns, 30-minute soak, available RAM, RSS, CPU, thermals, throttling, swap, speech latency, and wake metrics before marking Pi acceptance complete.

Never overwrite `/var/lib/orion/orion.sqlite3` on an existing Pi. Stop the service, make an online SQLite backup, verify backup integrity, and retain backup outside `/var/lib/orion` before any OS reflash or migration:

```sh
sudo systemctl stop orion.service
stamp=$(date +%Y%m%d-%H%M%S)
sudo install -d -o root -g root -m 0700 /var/backups/orion
temporary_db="/var/lib/orion/.orion-backup-$stamp.sqlite3"
sudo -u orion-service /opt/orion/.venv-voice/bin/python -c 'import sqlite3,sys; s=sqlite3.connect(sys.argv[1]); d=sqlite3.connect(sys.argv[2]); s.backup(d); d.close(); s.close()' /var/lib/orion/orion.sqlite3 "$temporary_db"
sudo install -o root -g root -m 0600 "$temporary_db" "/var/backups/orion/orion-$stamp.sqlite3"
sudo rm -f "$temporary_db"
sudo sqlite3 "/var/backups/orion/orion-$stamp.sqlite3" 'PRAGMA integrity_check;'
sudo install -o root -g root -m 0600 /etc/orion/orion.env "/var/backups/orion/orion.env-$stamp"
sudo install -o root -g root -m 0600 /etc/systemd/system/orion.service "/var/backups/orion/orion.service-$stamp"
sudo install -o orion -g orion -m 0600 "/var/backups/orion/orion-$stamp.sqlite3" "/home/orion/orion-$stamp.sqlite3"
sudo install -o orion -g orion -m 0600 "/var/backups/orion/orion.env-$stamp" "/home/orion/orion.env-$stamp"
sudo install -o orion -g orion -m 0600 "/var/backups/orion/orion.service-$stamp" "/home/orion/orion.service-$stamp"
```

Copy those three `/home/orion` backup files to a separate Windows disk before flashing:

```powershell
New-Item -ItemType Directory -Path 'D:\ORION-backup' -Force | Out-Null
scp orion.local:/home/orion/orion-<timestamp>.sqlite3 'D:\ORION-backup\'
scp orion.local:/home/orion/orion.env-<timestamp> 'D:\ORION-backup\'
scp orion.local:/home/orion/orion.service-<timestamp> 'D:\ORION-backup\'
```

Restore to a rebuilt Pi only after setting up the service account and directories. Copy the files back to `/home/orion`, inspect the saved environment before installing it, stop the service, verify SQLite integrity, then set ownership and mode:

```sh
sudo systemctl stop orion.service
sudo sqlite3 /home/orion/orion-<timestamp>.sqlite3 'PRAGMA integrity_check;'
sudo install -o orion-service -g orion-service -m 0600 /home/orion/orion-<timestamp>.sqlite3 /var/lib/orion/orion.sqlite3
sudo install -o root -g orion-service -m 0640 /home/orion/orion.env-<timestamp> /etc/orion/orion.env
```

Compare the old unit with the current repository unit; do not blindly restore an obsolete unit. Never upload a database into Git or include it in source archives.

A fresh flash replaces the old boot media. Keep a second backup copy off-Pi before flashing. Record image version, source revision, model SHA, llama.cpp commit, environment changes (excluding secrets), and acceptance results in the repo issue/handoff notes.

## 9. Current Pi State And Open Gates

As of 2026-10-09, Pi has the verified Qwen GGUF at `/opt/orion/models/LLM/Qwen3.5-2B-Q4_K_M.gguf` and llama.cpp at the pinned commit, installed under `/opt/orion/llama.cpp/bin` with `/usr/local/bin/llama-server`. SHA, file permissions, model load, chat template, tokenizer, `/props`, and streamed completion passed smoke tests. The owner confirms Phase 14 production deployment and validation on Pi. A trained Hey Orion candidate now exists in the adjacent local model depot and passes ONNX structural/inference checks; deployment is configured to upload it and select `hey-orion`. The Pi remains on legacy/VOSK until manifest validation and acoustic acceptance pass. Phase 15 model registry/Piper changes and remaining Phase 16 acceptance are not Pi-verified.

The training-produced `phase13-artifacts.json` manifest is not in this checkout or the local depot yet. Do not enable wakeword capture until that manifest validates on Pi. Pi performance, offline clean-boot, wake acoustics, mute/reboot persistence, reminders, and 30-minute soak still need measured acceptance.