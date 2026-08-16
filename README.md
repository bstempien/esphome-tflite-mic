# tflite_mic — ESPHome external component

Reads audio from an I2S INMP441 microphone, runs it through
**`trigger_model_int8.tflite`**, and exposes the result as a `binary_sensor`
that turns **on** when the trigger word is detected.

```
esphome-tflite-mic/
├── components/
│   └── tflite_mic/
│       ├── __init__.py              # hub component: I2S + feature config
│       ├── binary_sensor.py         # platform: entity per output value
│       ├── tflite_mic.h / .cpp      # I2S capture, spectrogram, inference
│       ├── binary_sensor_entity.h   # the BinarySensor C++ class
│       ├── model_data.h             # extern declarations
│       └── model_data.cc            # your model, embedded as a byte array
├── example.yaml
└── README.md
```

## What's in `trigger_model_int8.tflite`

I parsed your `.tflite` file directly (flatbuffer inspection — no other
tooling needed) and the component is built specifically around what it
found:

- **Input**: `[1, 124, 129, 1]`, int8, scale `0.10949513`, zero_point `-128`.
  A zero_point of exactly `-128` means the real (dequantized) values floor
  at `0.0` — a strong signature of a **magnitude spectrogram** (always
  ≥ 0), not a log/MFCC feature (which can go negative). Combined with the
  `124 × 129` shape, this is an exact match for TensorFlow's official
  ["Simple audio recognition"](https://www.tensorflow.org/tutorials/audio/simple_audio)
  tutorial pipeline:
  ```python
  spectrogram = tf.abs(tf.signal.stft(waveform, frame_length=255, frame_step=128))
  ```
  for a 1-second, 16kHz clip (`(16000-255)//128 + 1 = 124` frames,
  `256/2+1 = 129` frequency bins). **No mel filterbank, no log, no MFCC.**
  This component reproduces that pipeline on-device (Hann window → 256-pt
  FFT → magnitude), not a mel spectrogram — that's a change from an
  earlier draft of this component.
- **Output**: `[1, 1]`, int8, scale `0.00390625` (`= 1/256`), zero_point
  `-128`. The graph's last op is `LOGISTIC` (sigmoid), so this is a
  **single probability** that the trigger word is present — not a
  multi-class softmax. There's only one useful `class_index`: `0`.
- **Ops used** (all present in `esp-tflite-micro`'s
  `MicroMutableOpResolver`): `CONV_2D`, `MAX_POOL_2D`, `SHAPE`,
  `STRIDED_SLICE`, `PACK`, `RESHAPE`, `FULLY_CONNECTED`, `LOGISTIC`. The
  `SHAPE`/`STRIDED_SLICE`/`PACK` trio is what a Keras `Flatten` layer
  compiles to when the batch dimension is dynamic — it's just recomputing
  the flattened size at runtime, not doing anything with your audio. ReLU
  activations are fused into the `CONV_2D`/`FULLY_CONNECTED` ops directly
  (no separate op needed).
- **Architecture** (from tensor names/shapes): 3× `(Conv2D → MaxPool2D)`
  blocks (16→32→64 filters, all 3×3, 'same' padding, stride-2 pooling) →
  flatten → `Dense(64, relu)` → `Dense(1, sigmoid)`. The biggest single
  activation tensor is the first conv output at `124×129×16 ≈ 250KB` —
  that's the number that matters for arena sizing (see below), not the
  ~993KB model file, which is almost entirely the flatten→dense weight
  matrix (`64 × 15360 ≈ 960KB`) and stays resident in flash, not RAM.

If you retrain or swap in a different model, re-run the same kind of
inspection (or open it in [Netron](https://netron.app)) and update
`feature_type`/`frame_length`/`frame_step`/`fft_length` and the op resolver
in `tflite_mic.cpp` accordingly — the component logs a warning at boot if
the computed feature count doesn't match the model's input tensor size.

## 1. The model is already embedded

`model_data.cc` contains `trigger_model_int8.tflite` as a byte array
(`g_model` / `g_model_len`), generated directly from the file you uploaded.
Nothing to convert — just build.

## 2. Wire the INMP441

| INMP441 pin | ESP32          |
|-------------|----------------|
| VDD         | 3.3V           |
| GND         | GND            |
| SCK (BCLK)  | `i2s_bck_pin`  |
| WS (LRCL)   | `i2s_ws_pin`   |
| SD (DOUT)   | `i2s_data_pin` |
| L/R         | GND (selects left/mono channel — matches `I2S_CHANNEL_FMT_ONLY_LEFT` in the code) |

## 3. Configure YAML

See `example.yaml`. The defaults already match this model
(`feature_type: spectrogram`, `frame_length: 255`, `frame_step: 128`,
`fft_length: 256`, `clip_duration_ms: 1000`) — you shouldn't need to touch
those unless you retrain. What you will likely want to tune:

- **`tensor_arena_size`** (default `300kB`): the peak activation is
  ~250KB, plus TFLite Micro's own bookkeeping and any im2col scratch space
  the Conv2D kernels need. 300KB is a starting point — watch the boot log
  (`Arena used: X / Y bytes`) and raise `tensor_arena_size` if
  `AllocateTensors() failed` shows up instead.
- **PSRAM**: a 300KB+ arena won't fit in a plain ESP32's ~320KB internal
  DRAM alongside WiFi/BLE stacks and everything else ESPHome needs. Use a
  board with PSRAM (e.g. ESP32-WROVER) and add a `psram:` block to your
  YAML (see the commented-out example) — the component already prefers
  `MALLOC_CAP_SPIRAM` and only falls back to internal DRAM if no PSRAM is
  found.
- **Flash size**: the model itself is ~1MB. Make sure your board's flash
  size and ESPHome's chosen partition table leave room for that on top of
  the ESPHome firmware itself (a `min_spiffs` or custom partition CSV with
  a larger app partition may be needed on 4MB flash boards).
- **`mic_gain`**: the training pipeline's expected signal level is
  unknown from the `.tflite` file alone (that's determined by how the WAV
  files were normalized during training) — if detection seems
  insensitive or the input is clipping, adjust this and/or check the
  `input tensor` bytes logged at boot against expected quiet-room noise
  floor.
- **`inference_interval_ms`** (default `500`): how often a new 1-second
  window is classified. Lower = more responsive but more CPU; the model
  only sees whatever's in the trailing `clip_duration_ms` of audio each
  time, so this doesn't need to be as fast as the frame_step.

## 4. How detection works

1. `loop()` drains available I2S DMA samples into a rolling ring buffer
   every ESPHome loop iteration (non-blocking read).
2. Every `inference_interval_ms`, once at least one full `clip_duration_ms`
   of audio has been captured, it takes the most recent 1-second window,
   computes the STFT magnitude spectrogram (Hann-windowed 256-pt FFT every
   128 samples, 124 frames × 129 bins), quantizes it into the model's int8
   input tensor using the tensor's own scale/zero_point, and runs
   `Invoke()`.
3. The single sigmoid output is dequantized to a `0.0–1.0` probability.
   The registered `binary_sensor` (`class_index: 0`) compares it to
   `threshold` and publishes `on`/`off`, holding the `on` state for at
   least `min_trigger_interval` so Home Assistant sees a clean pulse
   rather than flapping.

## Known limitations / things to verify on real hardware

- The on-device FFT/windowing is a direct reimplementation of
  `tf.signal.stft`'s math (unnormalized DFT, periodic Hann window), not a
  call into the same library — double-check detection accuracy against a
  few known-good/known-bad clips once flashed, since a subtle windowing or
  normalization mismatch would degrade accuracy without necessarily
  crashing anything.
- The INMP441 sample-shift (`raw[i] >> 14`) assumes the mic's 24-bit
  output is left-justified in the 32-bit I2S word, standard for this part,
  but confirm signal levels with `mic_gain` if detection seems too
  quiet/clipped — the exact scaling the model was trained on isn't
  recoverable from the `.tflite` file itself.
- This was written and reasoned about from the model file alone — I don't
  have your training pipeline, a physical board, or the ability to run
  `idf.py build` here, so treat the arena size, op list, and feature
  pipeline as a well-grounded starting point to validate on hardware, not
  a guarantee.
