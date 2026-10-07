# Local Wake-Word Model

`hey_jarvis.onnx` is the upstream openWakeWord pretrained Hey Jarvis classifier,
asset `hey_jarvis_v0.1.onnx` from the official [v0.5.1 release](https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx).
It is stored under ORION's configurable filename; the model itself is not
modified. Application startup does not download models.

Verified file size: 1,271,370 bytes  
SHA-256: `94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb`

## Verified Contract

The classifier accepts `float32` input `x.1` with shape `[1, 16, 96]` and
returns `float32` output `53` with shape `[1, 1]`. The staged feature models
remain unchanged:

- Melspectrogram: `input` `[batch_size, samples]` to 32-bin mel features.
- Embedding: `input_1` `[batch, 76, 32, 1]` to 96-dimensional embeddings.
- Audio contract: mono, signed 16-bit PCM at 16 kHz.

Loaded and run through ORION's `OpenWakeWordDetector` with OpenWakeWord 0.6.0
and ONNX Runtime 1.30.0 using the CPU provider. Three seconds of synthetic
audio completed inference and returned finite scores. A 1.61-second offline
Windows speech-synthesis probe of "Hey Jarvis" produced a maximum score of
`0.9984` and one activation at threshold `0.5`; threshold `1.0` produced none.

Back-to-back speech-synthesized phrases separated by 0.2 seconds caused two
activations with the former 2-second cooldown because model latency placed the
second score just over two seconds after the first. The configured default is
now 3 seconds: the same probe yields one activation at 0.2 seconds separation
and two at 2.5 seconds separation. Cooldown remains configurable.

On the Pi AIRHUG input path, a spoken live sample peaked at `0.094` with the
generic `0.5` threshold, while a separate quiet sample stayed below `0.004`.
The deployment script now uses a Pi-specific threshold of `0.08`; this is a
single-sample calibration and should be monitored for false activations.

The Pi's ALSA `default` PortAudio route intermittently captured a spoken probe
at RMS `94` with a peak score below `0.00002`. PortAudio `sysdefault` captured
the same phrase at RMS `1641`, scored `0.1874`, and triggered at `0.08`.
Configure `ORION_AUDIO_DEVICE=sysdefault` for this AIRHUG 01; leave the generic
application default unchanged for other hardware. The opt-in Pi deployment
script writes this device setting into `/etc/orion/orion.env` and uses it in
its wake-asset preflight checks to prevent a future redeploy from reverting to
the silent route.

## License Note

The upstream repository code is Apache-2.0. Its Hey Jarvis model card does not
state a separate license for the classifier weights. This copy is provided for
local development and testing; do not infer redistribution rights for the
weights from the repository code license.