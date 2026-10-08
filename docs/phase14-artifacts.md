# Phase 14 Model and Runtime Pins

## Model

- Upstream weights: `Qwen/Qwen3.5-2B`, post-trained model, Apache-2.0.
- GGUF source: `unsloth/Qwen3.5-2B-GGUF`, pinned revision `f6d5376be1edb4d416d56da11e5397a961aca8ae`.
- File: `Qwen3.5-2B-Q4_K_M.gguf`.
- Expected size: 1,280,835,840 bytes.
- Verified SHA-256: `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`.
- The repository metadata identifies `Qwen/Qwen3.5-2B`, not `Qwen3.5-2B-Base`. Its model card states that the weights are post-trained.
- Do not provision `mmproj` files. ORION uses text-only inference.
- Keep model artifacts outside this source repository. Set `ORION_MODEL_PATH` to the verified local file.

Download from the pinned revision, then verify SHA-256 before use:

```powershell
$dir = Join-Path $env:LOCALAPPDATA 'ORION\models'
New-Item -ItemType Directory -Path $dir -Force | Out-Null
$file = Join-Path $dir 'Qwen3.5-2B-Q4_K_M.gguf'
Invoke-WebRequest -Uri 'https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-Q4_K_M.gguf?download=true' -OutFile $file
(Get-FileHash -Algorithm SHA256 $file).Hash.ToLower()
```

The local Windows download was verified at 1,280,835,840 bytes with the SHA-256 above. **Pi provisioning record (2026-10-07):** copied the verified GGUF to `/opt/orion/models/Qwen3.5-2B-Q4_K_M.gguf`; SHA-256 matches. Built llama.cpp at the pinned commit and installed the binary plus colocated libraries under `/opt/orion/llama.cpp/bin`, with `/usr/local/bin/llama-server` symlink. The installed server loaded the model, passed health, `/props`, `/apply-template`, `/tokenize`, and streamed completion checks. **Production validation (owner-reported, 2026-10-08):** owner confirms Phase 14 was deployed and validated on Pi. The repository deploy path uses Phase 14 by default. This session could not independently recheck the live host because SSH timed out. Broader Phase 16 acceptance remains separate.

For a new Pi, download the same immutable revision and verify before changing the service environment:

```sh
sudo install -d -o root -g orion-service -m 0755 /opt/orion/models
sudo curl -L --fail --output /opt/orion/models/Qwen3.5-2B-Q4_K_M.gguf 'https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-Q4_K_M.gguf?download=true'
printf '%s  %s\n' 'aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223' '/opt/orion/models/Qwen3.5-2B-Q4_K_M.gguf' | sha256sum --check
sudo chown root:orion-service /opt/orion/models/Qwen3.5-2B-Q4_K_M.gguf
sudo chmod 0644 /opt/orion/models/Qwen3.5-2B-Q4_K_M.gguf
```

## llama.cpp

- Upstream release: `ggml-org/llama.cpp` tag `v0.6.0`.
- Immutable source commit: `d81235049384534c167caea52b85a694f6103d14`.
- The GitHub release tag is unsigned. Pin and inspect this commit before building.
- Build on each target architecture; do not copy a Windows executable to Raspberry Pi.
- llama.cpp license: MIT.

Linux build and text-only smoke command:

```sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout d81235049384534c167caea52b85a694f6103d14
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-server --parallel 4
```

Install the binary and its colocated shared libraries. The deployment login and systemd service both need read/execute access. Keep model files read-only:

```sh
sudo install -d -o root -g orion-service -m 0755 /opt/orion/llama.cpp/bin
sudo cp -a build/bin/. /opt/orion/llama.cpp/bin/
sudo chown -R root:orion-service /opt/orion/llama.cpp
sudo find /opt/orion/llama.cpp -type d -exec chmod 0755 {} +
sudo find /opt/orion/llama.cpp -type f -exec chmod 0644 {} +
sudo chmod 0755 /opt/orion/llama.cpp/bin/llama-server
sudo ln -sfn /opt/orion/llama.cpp/bin/llama-server /usr/local/bin/llama-server
```

The server must expose `/apply-template`, `/tokenize`, and `/v1/chat/completions`. ORION checks the model-formatted prompt with the server tokenizer before every completion. It reserves 3072 input tokens, up to 512 answer tokens, and 512 template/safety tokens. Keep 8192-token context disabled until a separate Pi validation passes.

No fallback download or alternate model is permitted at service startup. A missing or mismatched file must fail before generation.
