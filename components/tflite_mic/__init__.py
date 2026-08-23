"""
tflite_mic - ESPHome external component
Reads audio from an I2S microphone (e.g. INMP441), runs it through a
TensorFlow Lite Micro keyword-spotting model, and exposes the results
to one or more `binary_sensor.tflite_mic` entities.

Defaults here are tuned for trigger_model_int8.tflite specifically:
  - input:  [1, 124, 129, 1] int8   -- raw (linear) STFT magnitude spectrogram,
            i.e. abs(tf.signal.stft(waveform, frame_length=255, frame_step=128))
  - output: [1, 1] int8 (sigmoid)   -- a single "trigger present" probability

This file only defines the *hub* component (I2S + model + inference loop).
See binary_sensor.py for the platform that attaches sensors to detected
output values.
"""

import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import esp32
from esphome.const import CONF_ID

CODEOWNERS = ["@your-github-handle"]
DEPENDENCIES = ["esp32"]
AUTO_LOAD = ["binary_sensor"]

tflite_mic_ns = cg.esphome_ns.namespace("tflite_mic")
TFLiteMicComponent = tflite_mic_ns.class_("TFLiteMicComponent", cg.Component)

CONF_I2S_BCK_PIN = "i2s_bck_pin"
CONF_I2S_WS_PIN = "i2s_ws_pin"
CONF_I2S_DATA_PIN = "i2s_data_pin"
CONF_I2S_PORT = "i2s_port"
CONF_SAMPLE_RATE = "sample_rate"
CONF_TENSOR_ARENA_SIZE = "tensor_arena_size"
CONF_FEATURE_TYPE = "feature_type"
CONF_FRAME_LENGTH = "frame_length"
CONF_FRAME_STEP = "frame_step"
CONF_FFT_LENGTH = "fft_length"
CONF_CLIP_DURATION_MS = "clip_duration_ms"
CONF_INFERENCE_INTERVAL_MS = "inference_interval_ms"
CONF_MIC_GAIN = "mic_gain"

FEATURE_TYPES = {
    "spectrogram": 1,  # matches trigger_model_int8.tflite (default)
    "raw": 0,          # feed normalized raw PCM straight into the model
}

CONFIG_SCHEMA = cv.Schema(
    {
        cv.GenerateID(): cv.declare_id(TFLiteMicComponent),
        cv.Required(CONF_I2S_BCK_PIN): cv.int_range(min=0, max=44),
        cv.Required(CONF_I2S_WS_PIN): cv.int_range(min=0, max=44),
        cv.Required(CONF_I2S_DATA_PIN): cv.int_range(min=0, max=44),
        cv.Optional(CONF_I2S_PORT, default=0): cv.int_range(min=0, max=1),
        cv.Optional(CONF_SAMPLE_RATE, default=16000): cv.int_range(min=8000, max=48000),
        # This model's peak activation tensor alone is ~250KB; PSRAM strongly
        # recommended. See README for sizing guidance.
        cv.Optional(CONF_TENSOR_ARENA_SIZE, default="300kB"): cv.validate_bytes,
        cv.Optional(CONF_FEATURE_TYPE, default="spectrogram"): cv.enum(FEATURE_TYPES, lower=True),
        cv.Optional(CONF_FRAME_LENGTH, default=255): cv.int_range(min=32, max=4096),
        cv.Optional(CONF_FRAME_STEP, default=128): cv.int_range(min=16, max=4096),
        cv.Optional(CONF_FFT_LENGTH, default=256): cv.one_of(64, 128, 256, 512, 1024, 2048, int=True),
        cv.Optional(CONF_CLIP_DURATION_MS, default=1000): cv.int_range(min=200, max=5000),
        cv.Optional(CONF_INFERENCE_INTERVAL_MS, default=500): cv.int_range(min=50, max=5000),
        cv.Optional(CONF_MIC_GAIN, default=1.0): cv.float_range(min=0.1, max=32.0),
    }
).extend(cv.COMPONENT_SCHEMA)


async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    cg.add(var.set_i2s_pins(
        config[CONF_I2S_BCK_PIN],
        config[CONF_I2S_WS_PIN],
        config[CONF_I2S_DATA_PIN],
    ))
    cg.add(var.set_i2s_port(config[CONF_I2S_PORT]))
    cg.add(var.set_sample_rate(config[CONF_SAMPLE_RATE]))
    cg.add(var.set_tensor_arena_size(config[CONF_TENSOR_ARENA_SIZE]))
    cg.add(var.set_feature_type(config[CONF_FEATURE_TYPE]))
    cg.add(var.set_frame_length(config[CONF_FRAME_LENGTH]))
    cg.add(var.set_frame_step(config[CONF_FRAME_STEP]))
    cg.add(var.set_fft_length(config[CONF_FFT_LENGTH]))
    cg.add(var.set_clip_duration_ms(config[CONF_CLIP_DURATION_MS]))
    cg.add(var.set_inference_interval_ms(config[CONF_INFERENCE_INTERVAL_MS]))
    cg.add(var.set_mic_gain(config[CONF_MIC_GAIN]))

    esp32.add_idf_component(name="espressif/esp-tflite-micro", ref="1.3.1")
