#pragma once

#include <vector>
#include <cstdint>

#include "esphome/core/component.h"
#include "esphome/core/log.h"

#include "driver/i2s.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace esphome {
namespace tflite_mic {

// FEATURE_SPECTROGRAM matches trigger_model_int8.tflite: a raw (linear,
// non-mel) STFT magnitude spectrogram, i.e. exactly
//   tf.abs(tf.signal.stft(waveform, frame_length=255, frame_step=128))
// which is what TensorFlow's official "Simple audio recognition" tutorial
// produces. FEATURE_RAW is kept as an option for other models that take a
// raw waveform directly.
enum FeatureType : uint8_t {
  FEATURE_RAW = 0,
  FEATURE_SPECTROGRAM = 1,
};

class TFLiteMicBinarySensor;

class TFLiteMicComponent : public Component {
 public:
  // ---- config setters (called from generated code) ----
  void set_i2s_pins(uint8_t bck, uint8_t ws, uint8_t data) {
    this->bck_pin_ = bck;
    this->ws_pin_ = ws;
    this->data_pin_ = data;
  }
  void set_i2s_port(uint8_t port) { this->i2s_port_ = static_cast<i2s_port_t>(port); }
  void set_sample_rate(uint32_t rate) { this->sample_rate_ = rate; }
  void set_tensor_arena_size(uint32_t bytes) { this->tensor_arena_size_ = bytes; }
  void set_feature_type(uint8_t type) { this->feature_type_ = static_cast<FeatureType>(type); }
  void set_frame_length(uint32_t samples) { this->frame_length_ = samples; }
  void set_frame_step(uint32_t samples) { this->frame_step_ = samples; }
  void set_fft_length(uint32_t n) { this->fft_length_ = n; }
  void set_clip_duration_ms(uint32_t ms) { this->clip_duration_ms_ = ms; }
  void set_inference_interval_ms(uint32_t ms) { this->inference_interval_ms_ = ms; }
  void set_mic_gain(float gain) { this->mic_gain_ = gain; }

  void register_sensor(TFLiteMicBinarySensor *sensor) { this->sensors_.push_back(sensor); }

  // ---- Component overrides ----
  void setup() override;
  void loop() override;
  void dump_config() override;
  float get_setup_priority() const override { return setup_priority::LATE; }

 protected:
  bool init_i2s_();
  bool init_model_();
  bool init_feature_buffers_();

  size_t fill_ring_buffer_();
  bool build_input_tensor_();
  void extract_raw_(int8_t *dst_int8, float *dst_float);
  void extract_spectrogram_(int8_t *dst_int8, float *dst_float);
  void run_inference_and_publish_();

  // ---- I2S / audio config ----
  uint8_t bck_pin_{0};
  uint8_t ws_pin_{0};
  uint8_t data_pin_{0};
  i2s_port_t i2s_port_{I2S_NUM_0};
  uint32_t sample_rate_{16000};
  float mic_gain_{1.0f};

  // ---- feature extraction config ----
  FeatureType feature_type_{FEATURE_SPECTROGRAM};
  uint32_t frame_length_{255};   // samples per STFT frame (~16 ms @ 16kHz)
  uint32_t frame_step_{128};     // hop between frames (~8 ms @ 16kHz)
  uint32_t fft_length_{256};     // FFT size frame is zero-padded to
  uint32_t clip_duration_ms_{1000};
  uint32_t inference_interval_ms_{500};

  // ---- ring buffer of raw PCM samples ----
  std::vector<int16_t> ring_buffer_;
  size_t ring_write_pos_{0};
  size_t ring_capacity_{0};
  size_t ring_filled_samples_{0};
  uint32_t last_inference_time_{0};

  // ---- TFLite Micro ----
  uint32_t tensor_arena_size_{300 * 1024};
  uint8_t *tensor_arena_{nullptr};
  const tflite::Model *model_{nullptr};
  tflite::MicroInterpreter *interpreter_{nullptr};
  TfLiteTensor *input_{nullptr};
  TfLiteTensor *output_{nullptr};
  bool model_ready_{false};

  std::vector<TFLiteMicBinarySensor *> sensors_;
};

}  // namespace tflite_mic
}  // namespace esphome
