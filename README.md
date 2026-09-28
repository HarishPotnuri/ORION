# RPi-local-voice-assistant
Local voice assistant running on Raspberry Pi 5 powered by VOSK STT, local LLM (llama.cpp or Ollama), and KittenTTS

## ORION local tools

### Local laptop testing

The default install targets typed local requests and tests. It does not install microphone, TTS, wake-word, or Raspberry Pi GPIO packages.

In PowerShell, create a separate Python 3.11 environment and install the test profile:

```powershell
py -3.11 -m venv .venv-local
& .\.venv-local\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-local\Scripts\python.exe -m pip install -e ".[test]"
```

Run tests and verify settings:

```powershell
& .\.venv-local\Scripts\python.exe -m pytest -q
& .\.venv-local\Scripts\orion.exe --check-config
```

Use one SQLite database for multi-command persistence checks:

```powershell
$env:ORION_DATABASE_PATH = Join-Path $env:TEMP "orion-local-test.sqlite3"
& .\.venv-local\Scripts\orion.exe --request "What time is it?"
& .\.venv-local\Scripts\orion.exe --request "Remind me tomorrow at 1 PM about the laptop test"
& .\.venv-local\Scripts\orion.exe --request "What reminders do I have today?"
```

Simple commands, memory, and reminders run without an LLM. Reasoning needs one local model service. The default backend is llama.cpp at `127.0.0.1:8080`; Ollama can be selected with `ORION_MODEL_BACKEND=ollama` and its local endpoint. Weather needs internet access. Web search stays disabled until you configure an HTTPS SearXNG instance and its exact allowed host.

This profile does not test live microphone capture, GPIO, wake-word inference, or Pi resource use. The continuous microphone-to-orchestrator loop is not connected yet. Raspberry Pi audio and GPIO dependencies are available as opt-in package extras.

Run the repeatable local test and typed-request smoke checks with:

```powershell
.\scripts\test-local.ps1
```

For reasoning requests, install and start Ollama using its Windows installer, then download one local model:

```powershell
ollama pull qwen3:1.7b
$env:ORION_MODEL_BACKEND = "ollama"
$env:ORION_MODEL_BASE_URL = "http://127.0.0.1:11434"
$env:ORION_MODEL_NAME = "qwen3:1.7b"
& .\.venv-local\Scripts\orion.exe --request "Why might a Kafka consumer slow down?"
```

Ollama and the model run locally. ORION does not use a cloud LLM fallback.

ORION typed requests run deterministic local tools without loading the local model:

```sh
orion --request "What time is it?"
orion --request "Remind me tomorrow at 1 PM about the project meeting"
```

Weather uses the Open-Meteo geocoding and forecast APIs. SearXNG search stays disabled until you configure a trusted HTTPS instance and its exact hostname:

```sh
export ORION_SEARXNG_BASE_URL="https://search.example.org"
export ORION_SEARXNG_ALLOWED_HOSTS="search.example.org"
orion --request "Search the web for the latest local result"
```

Online tools receive only the requested query or location. The local model receives compact factual results for interpretation. ORION does not send online requests to a cloud LLM. Do not configure public search hosts unless their query privacy policy meets your needs.

## Table of Contents
* [Features](#features)
* [Setup Instructions](#setup-instructions)
* [Notes](#notes)
* [Start on boot](#start-on-boot)
* [Audio details](#audio-details)
* [AIY Voice Kit setup](#aiy-voice-kit-setup)
* [Recommended models](#llm-tests)
* [Star History](#star-history)

## Features
- STT powered by VOSK 
- Ollama or llama.cpp local LLM processes queries
- TTS powered by [KittenTTS](https://github.com/KittenML/KittenTTS)
- Trigger with button on GPIO 23
- Status LED on GPIO 25

The project has been tested on the Raspberry Pi 5 (4GB RAM) and is designed for the AIY Voice Kit, though it works with any mic and speaker!

## Setup Instructions
Requirements: 
- Mic
- Speaker
- Ollama or llama.cpp installed with a model downloaded
- Raspberry Pi OS (tested on Trixie 64-bit)

## Notes
If using Ollama, by default the code uses the model [`exaone3.5:2.4b`](https://ollama.com/library/exaone3.5:2.4b). This can be changed in main.py.

If using llama.cpp, by default the code uses the model [`LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q8_0`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF) This can be changed in main_llama.py. 
We also recommend `mradermacher/Youtu-LLM-2B-i1-GGUF:Q4_K_M` for reasoning-heavy tasks and `unsloth/gemma-3n-E2B-it-GGUF:Q4_K_S` for models with more RAM

By default, the code uses the VOSK model [vosk-model-en-us-0.22-lgraph]([url](https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip)) which you will have to download. This can be changed in main.py.

## Instructions

1. Clone this repo: `git clone https://github.com/duckida/RPi-local-voice-assistant`
2. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
3. Download the VOSK model: `wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip && unzip vosk-model-en-us-0.22-lgraph.zip`
4. If using the Ollama version, run `uv run main.py`.
   If using the llama.cpp version, run `uv run main_llama.py`

## Start on boot
1. Create a service: `sudo systemctl edit --force --full local-ai-assistant.service`
2. Paste the following into the service:
```
[Unit]
Description=Local AI Assistant
After=multi-user.target

[Service]
Type=idle
ExecStart=/home/pi/.local/bin/uv run /home/pi/local-ai-assistant/main.py
Environment=XDG_RUNTIME_DIR=/run/user/1000
WorkingDirectory=/home/pi/local-ai-assistant
User=pi

[Install]
WantedBy=multi-user.target
```
3. Reload the systemd daemon: `sudo systemctl daemon-reload`
4. Enable the service: `sudo systemctl enable local-ai-assistant.service`
5. Reboot your Pi and it should start!

## Audio details
The code is designed for the latest Raspberry Pi OS Trixie, which uses PipeWire for audio. As such, the code uses `pw-play` to play audio and `python-sounddevice` for recording. It should use the default microphone and speaker connected, which can be changed thru the RPi Desktop UI.

## AIY Voice Kit setup
You will need to add `dtoverlay=googlevoicehat-soundcard` to the end of your `/boot/config/firmware.txt`

## LLM tests

We recommend using llama.cpp as it gives access to a nice web UI and more models compared to Ollama!

### Ollama models
- [Qwen3 1.7B](https://ollama.com/library/qwen3) - works as long as `/nothink` added to the prompt
- ✨ [EXAONE 3.5 2.4B](https://ollama.com/library/exaone3.5) - OK for world knowledge

### llama.cpp models
- ✨ [Gemma 4 e2B](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) - best all rounder - great world knowledge, slightly slower
- ✨ [Qwen 3.5 2B](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/tree/main) - good tool calling, same speed as Gemma 4, though less world knowledge
- ✨ [LFM2.5 1.5B](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF) - ideal for low-latency responses, can answer basic questions

- [Youtu LLM 2B](https://huggingface.co/mradermacher/Youtu-LLM-2B-GGUF) - good for reasoning but struggles in non-reasoning mode
- [Kanana Nano 1.2B](https://huggingface.co/DevQuasar/kakaocorp.kanana-1.5-2.1b-instruct-2505-GGUF/tree/main) - fast model, can answer basic questions
- [Qwen3 4B 2507 Instruct](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/tree/main?local-app=llama.cpp) - OK world knowledge, slower than Gemma

  
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=duckida/RPi-local-voice-assistant&type=date&legend=top-left)](https://www.star-history.com/#duckida/RPi-local-voice-assistant&type=date&legend=top-left)
