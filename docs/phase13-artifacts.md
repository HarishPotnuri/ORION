# Phase 13 Local Artifact Requirements

Legacy/VOSK remains the default. Do not set `ORION_VOICE_PROFILE=wakeword` until all three ONNX files, their manifest, and runtime validation pass. No runtime component downloads a missing artifact or substitutes a different wake phrase or STT provider.

## Required Files

| Purpose | Expected Pi path | Requirement |
| --- | --- | --- |
| Custom wake classifier | `/opt/orion/models/wakeword/hey_orion.onnx` | Classifier trained for the exact phrase “Hey Orion”; input embedding contract below. |
| Wake feature model | `/opt/orion/models/wakeword/melspectrogram.onnx` | openWakeWord-compatible ONNX mel transform; must pair with embedding model below. |
| Wake embedding model | `/opt/orion/models/wakeword/embedding_model.onnx` | openWakeWord-compatible Google speech-embedding graph; must pair with mel and classifier. |
| Asset manifest | `/opt/orion/models/wakeword/phase13-artifacts.json` | Operator-supplied exact provenance, licenses, revisions, SHA-256 values, and compatibility declaration for all three files. |
| Whisper model | `/opt/orion/models/whisper/ggml-base.en.bin` | Provisioned English `base.en` GGML model. Record its source and model license. |
| Whisper executable | `/usr/local/bin/whisper-cli` | CPU-capable `whisper.cpp` v1.9.4, source commit `927cfce34f31707e17f2bff35c349632fb9e2c3a`. |
| Optional local TTS model | `/opt/orion/models/tts/kitten-tts-nano-0.2` | KittenTTS 0.1 `ONNX1` repository directory containing `config.json` and the referenced model/voice files; record source, hashes, and model license. |

## Verified Runtime Contract

ORION invokes `openwakeword==0.6.0` with `inference_framework="onnx"`, the custom classifier path, `melspec_model_path`, `embedding_model_path`, and configured CPU thread count. Audio arrives as 16 kHz mono signed int16 PCM. The detector passes `numpy.int16` frames to `Model.predict`; normal listener frames are 80 ms.

The versioned openWakeWord 0.6.0 implementation establishes these ONNX interfaces:

| Graph | Input contract | Output contract |
| --- | --- | --- |
| `melspectrogram.onnx` | One `tensor(float)` named `input`; rank 2 `[batch, samples]` | One float tensor; raw rank 4 `[time, 1, mel_frames, 32]`; openWakeWord squeezes batch item 0 to `[mel_frames, 32]` |
| `embedding_model.onnx` | One `tensor(float)` named `input_1`; rank 4 `[batch, 76, 32, 1]` | One float tensor; raw rank 4 `[batch, 1, 1, 96]`; openWakeWord squeezes to `[96]` |
| `hey_orion.onnx` | One float tensor; rank 3 `[batch, feature_frames, 96]` with positive fixed `feature_frames` | One float tensor; rank 3 `[batch, 1, 1]`, confidence in `[0, 1]` |

The validator opens each graph through ONNX Runtime CPU, checks tensor names/types/ranks/shapes, executes compatible zero-input smoke runs, checks finite outputs, and verifies that feature and classifier dimensions connect. It requires exactly `openwakeword==0.6.0`. The actual v0.6.0 ONNX feature graphs use rank-4 outputs; the openWakeWord code squeezes these internally. These structural checks cannot prove that a classifier was trained for “Hey Orion” or that its training used the declared feature-model revision. Manifest provenance and acoustic evaluation are also required.

Do not rename a built-in wake model to `hey_orion.onnx` or treat a different phrase as equivalent. Do not mix feature graphs from another openWakeWord release. The ONNX opset, pre/post-processing, and trained classifier must come from a traceable compatible set. The official openWakeWord v0.6.0 catalog has no “Hey Orion” classifier. The official feature graphs are staged locally as described in [the feature asset record](phase13-feature-assets.json); there is still no custom classifier or complete production manifest, and no Pi assets have been installed.

## Custom “Hey Orion” Classifier

The classifier does not exist yet and must be trained. The repository contains a [reproducible Colab training notebook](../notebooks/orion_hey_orion_training.ipynb) based on the official openWakeWord v0.6.0 [automatic training notebook](https://github.com/dscripka/openWakeWord/blob/v0.6.0/notebooks/automatic_model_training.ipynb). It pins the training source, uses exactly `Hey Orion`, targets 20,000 positive training examples plus a separate 4,000-example synthetic positive holdout, and records data/model revisions, licenses, and checksums. The official automatic notebook's 5,000-example setting is an instructional shortcut, not production acceptance. The Colab notebook excludes the unknown-license MIT RIR set and makes no RIR/far-field robustness claim.

The v0.6.0 training exporter can emit a binary classifier with a rank-2 `[batch, 1]` output, while ORION's validator and openWakeWord score extraction expect `[batch, 1, 1]`. The Colab notebook inspects the actual output and, only for rank 2, adds a value-preserving ONNX `Reshape` to `[batch, 1, 1]`; it records pre/post hashes and reruns ORT CPU plus ORION validator checks. Other output shapes stop the run. The classifier isn't considered Orion-compatible until that full check passes.

The openWakeWord trainer uses Piper sample generation. Its generator code is MIT-licensed, while the specific LibriTTS-R medium checkpoint's model card identifies LibriTTS-R under CC BY 4.0. The official openWakeWord README states its published pretrained models are CC BY-NC-SA 4.0; use that as the conservative interpretation for the staged feature-model release artifacts unless the maintainers provide a more specific grant. This is redistributable only subject to attribution, noncommercial use, and share-alike terms; commercial use is not covered. Confirm this is acceptable for the intended use before distributing the trained classifier or model bundle. Dataset licenses for any additional negative/background/test corpora must be checked individually. This is a license summary, not legal advice.

GHCP can automate repository scaffolding, pinned setup commands, config generation, metadata capture, and post-training validation. It cannot responsibly claim or produce an acoustically accepted model without actually running training, auditing dataset/checkpoint terms, and evaluating recordings representative of the Pi and room. Keep private recordings in the ignored training workspace and do not transfer them to the Pi. Start with `scripts/prepare-wakeword-training.ps1`; it creates local directories and a phrase seed only, and downloads nothing.

## Colab Training Instructions

Use [the reproducible `Hey Orion` Colab notebook](../notebooks/orion_hey_orion_training.ipynb). It pins openWakeWord v0.6.0, Piper sample-generator v2.0.0, and the official feature dataset revision. It uploads and hash-checks the two already staged ONNX feature models; it does not fetch substitute feature graphs. It creates at least 20,000 positive synthetic train clips and 4,000 separate synthetic positive validation clips, plus the upstream-generated adversarial negatives. Generic negative and false-positive validation features use the official openWakeWord dataset at its pinned revision, under the declared CC BY-NC-SA 4.0 terms. The notebook stops until the operator explicitly acknowledges those noncommercial/share-alike terms and the Piper/LibriTTS-R CC BY 4.0 attribution. Do not run for commercial use under those acknowledgements.

The upstream MIT impulse-response dataset card declares its license unknown, so the notebook excludes it and sets `rir_paths` empty. It does not download AudioSet, FMA, or raw third-party background audio. Augmentation still uses generated colored noise and the licensed generic-negative features; no RIR/far-field robustness claim is made. The notebook records this limitation.

1. In Colab, choose **File → Upload notebook** and select `notebooks/orion_hey_orion_training.ipynb` from this repository.
2. Choose **Runtime → Change runtime type → GPU**. Review the two acknowledgement booleans in the license-gate cell. Set them to `True` only if the stated CC BY-NC-SA 4.0 noncommercial/share-alike and CC BY 4.0 obligations fit the intended use. Otherwise stop; do not download the training assets.
3. Choose **Runtime → Run all**. When prompted, upload the two exact files from `artifacts/wakeword-staging/` and the current local `src/orion/audio/wakeword_assets.py` and `src/orion/config.py`. The notebook pins and records the upstream source commits and dataset revision. It blocks on missing pronunciation entries rather than downloading an unpinned phonemizer model.
4. Wait for the final `PASS` markers and `orion_hey_orion_training_bundle.zip` download. The bundle includes `hey_orion.onnx`, `hey_orion_metadata.json`, `phase13-artifacts.json`, `validation_report.json`, `evaluation.json`, `training_config.yaml`, `sha256sums.txt`, `provenance.json`, `training_completion.json`, `requirements_colab.lock`, and the manifests/licenses directories and archives. The notebook does not claim training completed until the official train stage returns and the ONNX artifact exists; it does not claim Pi acoustic acceptance.
5. Extract the bundle under `artifacts/wakeword-staging/hey_orion_v0_6_0/`. Keep the already staged `melspectrogram.onnx` and `embedding_model.onnx` in the parent `artifacts/wakeword-staging/` directory. Do not edit `systemd/orion.env.example`, `/etc/orion/orion.env`, or any Pi setting.

From the Orion repository root, install the local optional wakeword dependency and run this validator-only command. These variables last only for the current PowerShell process; remove them afterward. The command does not start audio capture or change the saved production profile:

```powershell
.\.venv-local\Scripts\python.exe -m pip install -e ".[wakeword]"
$env:ORION_VOICE_PROFILE = "wakeword"
$env:ORION_STT_BACKEND = "whisper.cpp"
$env:ORION_WAKE_WORD_MODEL_PATH = (Resolve-Path ".\artifacts\wakeword-staging\hey_orion_v0_6_0\hey_orion.onnx").Path
$env:ORION_WAKE_WORD_MELSPECTROGRAM_PATH = (Resolve-Path ".\artifacts\wakeword-staging\melspectrogram.onnx").Path
$env:ORION_WAKE_WORD_EMBEDDING_PATH = (Resolve-Path ".\artifacts\wakeword-staging\embedding_model.onnx").Path
$env:ORION_WAKE_WORD_MANIFEST_PATH = (Resolve-Path ".\artifacts\wakeword-staging\hey_orion_v0_6_0\phase13-artifacts.json").Path
.\.venv-local\Scripts\python.exe -m orion --validate-wakeword-assets
'ORION_VOICE_PROFILE','ORION_STT_BACKEND','ORION_WAKE_WORD_MODEL_PATH','ORION_WAKE_WORD_MELSPECTROGRAM_PATH','ORION_WAKE_WORD_EMBEDDING_PATH','ORION_WAKE_WORD_MANIFEST_PATH' | ForEach-Object { Remove-Item "Env:$_" -ErrorAction SilentlyContinue }
```

The existing validator checks the complete local ONNX set, manifest, hashes, tensor compatibility, runtime version, and zero-input smoke inference. A pass does not activate openWakeWord. Keep production and the Pi on VOSK until separate deployment approval and acoustic acceptance.

## Provenance And Checksums

The validator expects a JSON manifest with schema version 1. Populate every value from the actual source/training records. Do not copy example or placeholder values into the production manifest. A manifest is not proof by itself: its hashes must match the files, and its source, revision, license, and training records must be independently reviewable.

Required manifest fields:

- `schema_version`: integer `1`; `wake_phrase`: exact string `Hey Orion`; `sample_rate_hz`: integer `16000`.
- `runtime`: `name` `openwakeword`, exact `version` `0.6.0`, and `framework` `onnx`.
- `compatibility`: `sample_rate_hz` `16000`, `melspectrogram_bins` `32`, `embedding_window_frames` `76`, `embedding_dimension` `96`, `wake_output_classes` `1`, and `feature_model_sha256` containing the mel and embedding file hashes.
- `assets`: exactly `wakeword`, `melspectrogram`, and `embedding`. Each entry requires `file`, lowercase `sha256`, `source`, exact `revision`, and artifact-specific `license`. The wakeword entry also requires `training_recipe` and `training_data` provenance.

Generate and verify hashes after finalizing and copying files:

```sh
cd /opt/orion/models
sha256sum /opt/orion/models/wakeword/hey_orion.onnx /opt/orion/models/wakeword/melspectrogram.onnx /opt/orion/models/wakeword/embedding_model.onnx /opt/orion/models/whisper/ggml-base.en.bin
```

Compare each output against the trusted source manifest. Do not treat a hash generated after an untrusted download as trusted provenance. Do not commit model binaries or private training audio. Deployment packages exclude model directories, so artifact provisioning is a separate operator-controlled step. Set `ORION_TTS_MODEL_PATH` to load a provisioned local directory directly. When unset, the configured KittenTTS model identifier uses the pre-provisioned cache with Hugging Face offline mode enforced by ORION.

After installing the source and wakeword dependency extra on Pi, run validation as the service account. This command does not start audio capture or the assistant:

```sh
sudo -u orion-service bash -lc 'set -a; . /etc/orion/orion.env; set +a; /opt/orion/.venv-voice/bin/orion --validate-wakeword-assets'
```

The command exits non-zero for missing files, malformed ONNX, incompatible tensor/runtime interfaces, SHA mismatch, missing provenance/license metadata, or incompatible feature-model bindings. Keep `/etc/orion/orion.env` on `legacy` until it passes.

## Pi Provisioning

Build the pinned Whisper CLI on the Pi:

```sh
git clone --depth 1 --branch v1.9.4 https://github.com/ggml-org/whisper.cpp.git /tmp/whisper.cpp
cd /tmp/whisper.cpp
test "$(git rev-parse HEAD)" = 927cfce34f31707e17f2bff35c349632fb9e2c3a
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2 --target whisper-cli
```

Install the model assets so the service account can read, but not modify, them. Install the CLI's shared libraries too. The built executable has a runtime path into its build directory; it must also resolve libraries from `/usr/local/lib` after temporary build files are removed.

```sh
sudo install -d -o root -g orion-service -m 0750 /opt/orion/models/wakeword /opt/orion/models/whisper
sudo install -o root -g orion-service -m 0640 hey_orion.onnx /opt/orion/models/wakeword/hey_orion.onnx
sudo install -o root -g orion-service -m 0640 melspectrogram.onnx /opt/orion/models/wakeword/melspectrogram.onnx
sudo install -o root -g orion-service -m 0640 embedding_model.onnx /opt/orion/models/wakeword/embedding_model.onnx
sudo install -o root -g orion-service -m 0640 phase13-artifacts.json /opt/orion/models/wakeword/phase13-artifacts.json
sudo install -o root -g orion-service -m 0640 ggml-base.en.bin /opt/orion/models/whisper/ggml-base.en.bin
sudo install -o root -g root -m 0644 /tmp/whisper.cpp/build/bin/libwhisper.so.1.9.4 /usr/local/lib/libwhisper.so.1.9.4
sudo ln -sfn libwhisper.so.1.9.4 /usr/local/lib/libwhisper.so.1
sudo ln -sfn libwhisper.so.1 /usr/local/lib/libwhisper.so
sudo install -o root -g root -m 0644 /tmp/whisper.cpp/build/bin/libggml.so.0.23.0 /usr/local/lib/libggml.so.0.23.0
sudo ln -sfn libggml.so.0.23.0 /usr/local/lib/libggml.so.0
sudo ln -sfn libggml.so.0 /usr/local/lib/libggml.so
sudo install -o root -g root -m 0644 /tmp/whisper.cpp/build/bin/libggml-base.so.0.23.0 /usr/local/lib/libggml-base.so.0.23.0
sudo ln -sfn libggml-base.so.0.23.0 /usr/local/lib/libggml-base.so.0
sudo ln -sfn libggml-base.so.0 /usr/local/lib/libggml-base.so
sudo install -o root -g root -m 0644 /tmp/whisper.cpp/build/bin/libggml-cpu.so.0.23.0 /usr/local/lib/libggml-cpu.so.0.23.0
sudo ln -sfn libggml-cpu.so.0.23.0 /usr/local/lib/libggml-cpu.so.0
sudo ln -sfn libggml-cpu.so.0 /usr/local/lib/libggml-cpu.so
sudo ldconfig
sudo install -o root -g root -m 0755 /tmp/whisper.cpp/build/bin/whisper-cli /usr/local/bin/whisper-cli
```

Before enabling the profile, verify the SHA-256 manifest, `whisper-cli --version`, and that the CLI accepts WAV from stdin with `-f -` without writing an output file. The pinned source implements `-` as stdin and its default audio decoder accepts WAV data; still run a real transcription with the provisioned model:

```sh
cat /path/to/approved-16k-mono.wav | /usr/local/bin/whisper-cli -m /opt/orion/models/whisper/ggml-base.en.bin -f - -t 2 -l en -ng -nt -np -otxt
```

Test that malformed/missing paths fail startup, no network request occurs with Hugging Face offline, and no captured audio is written to disk. Then follow the Pi acceptance checklist in `Phase13-Handoff.md`.

## Controlled Profile Migration And Rollback

Keep the service on `legacy`/VOSK during provisioning. Do not edit `/etc/orion/orion.env` before the candidate assets pass validation.

1. Deploy the validator-capable source and pinned `openwakeword==0.6.0` dependency while the existing environment remains on VOSK. Confirm the systemd service returns active. Do not deploy while the active Pi has missing or changed environment values.
2. Transfer only operator-approved artifacts and their manifest. Install the three ONNX files, JSON manifest, Whisper model, CLI, and CLI libraries as described above. Keep private training audio off the Pi unless explicitly required by the recorded recipe.
3. Verify copied files against the independently trusted manifest. Then validate with candidate paths passed only to the one-shot validator; this does not change the service environment:

```sh
sudo -u orion-service env \
	ORION_VOICE_PROFILE=wakeword \
	ORION_STT_BACKEND=whisper.cpp \
	ORION_STT_MODEL_PATH=/opt/orion/models/whisper/ggml-base.en.bin \
	ORION_WHISPER_CLI_PATH=/usr/local/bin/whisper-cli \
	ORION_WAKE_WORD_MODEL_PATH=/opt/orion/models/wakeword/hey_orion.onnx \
	ORION_WAKE_WORD_MELSPECTROGRAM_PATH=/opt/orion/models/wakeword/melspectrogram.onnx \
	ORION_WAKE_WORD_EMBEDDING_PATH=/opt/orion/models/wakeword/embedding_model.onnx \
	ORION_WAKE_WORD_MANIFEST_PATH=/opt/orion/models/wakeword/phase13-artifacts.json \
	/opt/orion/.venv-voice/bin/orion --validate-wakeword-assets
```

Stop if validation returns non-zero. Do not weaken checks or regenerate manifest values from untrusted copies.

4. Back up the exact active environment, preserving owner and mode. Record backup path before editing:

```sh
backup="/var/backups/orion/orion.env.pre-wakeword.$(date +%Y%m%d-%H%M%S)"
sudo install -d -o root -g root -m 0700 /var/backups/orion
sudo cp -a /etc/orion/orion.env "$backup"
sudo chmod 0600 "$backup"
printf 'Environment backup: %s\n' "$backup"
```

5. Edit `/etc/orion/orion.env` with `sudoedit`. Set `ORION_VOICE_PROFILE=wakeword`, `ORION_STT_BACKEND=whisper.cpp`, the verified model/CLI paths, the three ONNX paths, and manifest path. Keep `ORION_AUDIO_DEVICE=default` and all unrelated values unchanged.
6. Run the asset validator using `/etc/orion/orion.env` as service account. If it fails, restore the backup and stop migration:

```sh
sudo -u orion-service bash -lc 'set -a; . /etc/orion/orion.env; set +a; /opt/orion/.venv-voice/bin/orion --validate-wakeword-assets'
```

7. Restart the service and check `systemctl is-active orion.service` plus recent journal output. Stop the service for the microphone smoke test so the service and one-shot command do not contend for capture. Run `orion --voice-check` as `orion-service`; say “Hey Orion”, then a short request. Confirm transcript and spoken readiness, no disk audio, GPIO23 mute/unmute, LED colors, and GPIO16 interruption. Restart the systemd service after the one-shot test and confirm it is active.
8. If validation, startup, or smoke test fails, restore the saved environment and return to legacy/VOSK:

```sh
sudo cp -a "$backup" /etc/orion/orion.env
sudo systemctl restart orion.service
sudo systemctl is-active orion.service
```

Keep newly installed model files in place during rollback; the legacy service does not use them. Do not remove backups until legacy recovery is confirmed. Record test conditions and wake false accepts/rejects before promoting the profile.

## Pi Provisioning Evidence (2026-10-06)

- Built `whisper.cpp` on ARM64 from commit `927cfce34f31707e17f2bff35c349632fb9e2c3a`; source license is MIT. The executable reports `1.9.4-dev`.
- Downloaded `ggml-base.en.bin` from `ggerganov/whisper.cpp` at revision `5359861c739e955e79d9a303bcbc70fb988958b1`; repository license is MIT. SHA-256: `a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002`.
- Verified service-account transcription from WAV stdin with `-f - -otxt`; text returned on stdout and no output file was created.
- This does not verify custom wake assets, acoustic behavior, or the full ORION voice service. Do not enable wakeword profile until remaining artifacts are provisioned.