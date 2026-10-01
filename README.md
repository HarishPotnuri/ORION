# ORION
Local voice assistant running on Raspberry Pi 5 powered by VOSK STT, local LLM (llama.cpp or Ollama), and KittenTTS

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

To test local reasoning with Ollama, set its local endpoint and model in the same PowerShell session:

```powershell
$env:ORION_MODEL_BACKEND = "ollama"
$env:ORION_MODEL_BASE_URL = "http://127.0.0.1:11434"
$env:ORION_MODEL_NAME = "qwen3:1.7b"
& .\.venv-local\Scripts\orion.exe --request "Explain Kafka briefly."
```

## Raspberry Pi Setup

ORION targets Raspberry Pi 5 with 8 GB RAM and 64-bit Raspberry Pi OS. The package requires Python `>=3.11,<3.12`; confirm that the chosen OS image provides Python 3.11 before installing. Exact OS image, ARM64 dependency compatibility, peripherals, and performance are not yet Pi-verified. Do not treat upstream script instructions or model recommendations as supported ORION setup.

From the repository root, create an isolated environment and install voice, GPIO, and test dependencies:

```sh
python3.11 --version
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[voice,pi,test]'
```

Install `wakeword` only when testing wake-word input:

```sh
python -m pip install -e '.[voice,pi,wakeword,test]'
```

Download and extract the VOSK model into the repository's `models/` directory:

```sh
mkdir -p models
curl -L --fail -o /tmp/vosk-model.zip https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip
unzip -q /tmp/vosk-model.zip -d models
rm /tmp/vosk-model.zip
```

Confirm `ORION_STT_MODEL_PATH` points to the extracted model directory. Connect the microphone, speaker, push-to-talk button, and status LED using the configured GPIO pins; verify wiring before starting the service.

Configure a local model service. For Ollama, use the installed local service and a model present on the Pi:

```sh
export ORION_MODEL_BACKEND=ollama
export ORION_MODEL_BASE_URL=http://127.0.0.1:11434
export ORION_MODEL_NAME=qwen3:1.7b
```

For an externally managed llama.cpp server, configure `ORION_MODEL_BACKEND=llama.cpp` and its loopback `ORION_MODEL_BASE_URL`. ORION can manage `llama-server` when `ORION_MODEL_PATH` names a local GGUF file. These Pi deployment modes remain unverified.

Run deterministic checks before opening hardware, then start the voice service as the normal user, not root:

```sh
orion --check-config
orion --request 'What time is it?'
orion --run-voice
```

Record the exact OS image/release, `uname -a`, Python version, `python -m pip freeze`, model runtime/model version, audio-device list, GPIO permissions, and test results. Run `python -m sounddevice` to enumerate audio devices. Test microphone capture, speaker playback, GPIO access, service-user database permissions, and local model responses separately. Do not report these checks as passed until they run on the target Pi.

`main.py`, `main_llama.py`, and `stt_lib.py` are historical upstream files, not supported ORION launch commands. No production systemd unit is provided yet; deployment as a managed service remains Phase 10 work.
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=duckida/RPi-local-voice-assistant&type=date&legend=top-left)](https://www.star-history.com/#duckida/RPi-local-voice-assistant&type=date&legend=top-left)
