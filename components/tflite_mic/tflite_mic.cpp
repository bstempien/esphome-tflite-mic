#include "tflite_mic.h"
#include "binary_sensor_entity.h"
#include "model_data.h"
#include "model_data.cc"

#include <cmath>
#include <cstring>
#include <algorithm>

#include "esp_heap_caps.h"

namespace esphome {
namespace tflite_mic {

static const char *const TAG = "tflite_mic";

// Ops actually present in trigger_model_int8.tflite (verified by parsing
// the flatbuffer directly): CONV_2D, MAX_POOL_2D, SHAPE, STRIDED_SLICE,
// PACK, RESHAPE, FULLY_CONNECTED, LOGISTIC. The SHAPE/STRIDED_SLICE/PACK
// trio is what TF emits for a dynamic-batch Flatten layer -- it just
// re-derives the flattened size at runtime, it's not doing anything with
// your audio. ReLU activations are fused into the Conv2D/FullyConnected
// ops (no separate RELU op needed).
static constexpr int kNumOps = 8;

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

void TFLiteMicComponent::setup() {
  ESP_LOGCONFIG(TAG, "Setting up TFLite Mic...");

  if (!this->init_model_()) {
    ESP_LOGE(TAG, "Model init failed, component disabled");
    this->mark_failed();
    return;
  }
  if (!this->init_feature_buffers_()) {
    ESP_LOGE(TAG, "Feature buffer init failed, component disabled");
    this->mark_failed();
    return;
  }
  if (!this->init_i2s_()) {
    ESP_LOGE(TAG, "I2S init failed, component disabled");
    this->mark_failed();
    return;
  }

  this->model_ready_ = true;
  ESP_LOGCONFIG(TAG, "TFLite Mic setup complete");
}

bool TFLiteMicComponent::init_model_() {
  this->model_ = tflite::GetModel(g_model);
  if (this->model_->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Model schema version %d != supported %d", this->model_->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  // This model's peak activation tensor (the first conv2d output) is
  // ~250KB by itself, so the arena needs real headroom. Prefer PSRAM if
  // the board has it; fall back to internal DRAM (likely to fail
  // AllocateTensors() on a non-PSRAM board unless tensor_arena_size is
  // trimmed way down, which isn't really possible for this model without
  // shrinking it).
  this->tensor_arena_ =
      static_cast<uint8_t *>(heap_caps_malloc(this->tensor_arena_size_, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (this->tensor_arena_ == nullptr) {
    ESP_LOGW(TAG, "No PSRAM available (or allocation failed) -- trying internal DRAM");
    this->tensor_arena_ = static_cast<uint8_t *>(heap_caps_malloc(this->tensor_arena_size_, MALLOC_CAP_8BIT));
  }
  if (this->tensor_arena_ == nullptr) {
    ESP_LOGE(TAG, "Could not allocate %u byte tensor arena", this->tensor_arena_size_);
    return false;
  }

  static tflite::MicroMutableOpResolver<kNumOps> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddShape();
  resolver.AddStridedSlice();
  resolver.AddPack();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddLogistic();

  static tflite::MicroInterpreter static_interpreter(this->model_, resolver, this->tensor_arena_,
                                                       this->tensor_arena_size_);
  this->interpreter_ = &static_interpreter;

  if (this->interpreter_->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG, "AllocateTensors() failed -- tensor_arena_size (%u bytes) is too small; try increasing it",
             this->tensor_arena_size_);
    return false;
  }

  this->input_ = this->interpreter_->input(0);
  this->output_ = this->interpreter_->output(0);

  ESP_LOGCONFIG(TAG, "  Input tensor: %d dims [%d,%d,%d,%d], type=%d, bytes=%u", this->input_->dims->size,
                this->input_->dims->data[0],
                this->input_->dims->size > 1 ? this->input_->dims->data[1] : 0,
                this->input_->dims->size > 2 ? this->input_->dims->data[2] : 0,
                this->input_->dims->size > 3 ? this->input_->dims->data[3] : 0, this->input_->type,
                this->input_->bytes);
  ESP_LOGCONFIG(TAG, "  Output tensor: %d value(s), type=%d", this->output_->dims->data[this->output_->dims->size - 1],
                this->output_->type);
  ESP_LOGCONFIG(TAG, "  Arena used: %u / %u bytes", this->interpreter_->arena_used_bytes(),
                this->tensor_arena_size_);

  return true;
}

bool TFLiteMicComponent::init_feature_buffers_() {
  size_t clip_samples = (this->sample_rate_ * this->clip_duration_ms_) / 1000;
  this->ring_capacity_ = clip_samples + (this->sample_rate_ / 2);  // +0.5s headroom against tearing
  this->ring_buffer_.assign(this->ring_capacity_, 0);
  this->ring_write_pos_ = 0;
  this->ring_filled_samples_ = 0;

  // Sanity-check that frame_length/frame_step/clip_duration will actually
  // produce the number of frames the model expects (dim 1 of a 4D input).
  if (this->feature_type_ == FEATURE_SPECTROGRAM) {
    size_t expected_frames = 1 + (clip_samples - this->frame_length_) / this->frame_step_;
    ESP_LOGCONFIG(TAG, "  Spectrogram: %u frames x %u bins expected from frame_length=%u, frame_step=%u, clip=%ums",
                  expected_frames, this->fft_length_ / 2 + 1, this->frame_length_, this->frame_step_,
                  this->clip_duration_ms_);
  }

  return true;
}

bool TFLiteMicComponent::init_i2s_() {
  i2s_config_t i2s_config = {};
  i2s_config.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX);
  i2s_config.sample_rate = this->sample_rate_;
  i2s_config.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;  // INMP441 outputs 24-bit in a 32-bit frame
  i2s_config.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;   // tie L/R pin low on the mic -> left channel
  i2s_config.communication_format = static_cast<i2s_comm_format_t>(I2S_COMM_FORMAT_STAND_I2S);
  i2s_config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  i2s_config.dma_buf_count = 8;
  i2s_config.dma_buf_len = 256;
  i2s_config.use_apll = false;
  i2s_config.tx_desc_auto_clear = false;
  i2s_config.fixed_mclk = 0;

  esp_err_t err = i2s_driver_install(this->i2s_port_, &i2s_config, 0, nullptr);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "i2s_driver_install failed: %d", err);
    return false;
  }

  i2s_pin_config_t pin_config = {};
  pin_config.bck_io_num = this->bck_pin_;
  pin_config.ws_io_num = this->ws_pin_;
  pin_config.data_out_num = I2S_PIN_NO_CHANGE;
  pin_config.data_in_num = this->data_pin_;

  err = i2s_set_pin(this->i2s_port_, &pin_config);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "i2s_set_pin failed: %d", err);
    return false;
  }

  i2s_zero_dma_buffer(this->i2s_port_);
  return true;
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

void TFLiteMicComponent::loop() {
  if (!this->model_ready_) return;

  size_t new_samples = this->fill_ring_buffer_();
  this->ring_filled_samples_ = std::min(this->ring_filled_samples_ + new_samples, this->ring_capacity_);

  size_t clip_samples = (this->sample_rate_ * this->clip_duration_ms_) / 1000;
  uint32_t now = millis();

  bool have_full_clip = this->ring_filled_samples_ >= clip_samples;
  bool interval_elapsed = (now - this->last_inference_time_) >= this->inference_interval_ms_;

  if (have_full_clip && interval_elapsed) {
    this->last_inference_time_ = now;
    if (this->build_input_tensor_()) {
      this->run_inference_and_publish_();
    }
  }
}

size_t TFLiteMicComponent::fill_ring_buffer_() {
  // 32-bit stereo-slot samples from the I2S peripheral; INMP441 left-justifies
  // 24 significant bits into the upper part of the 32-bit word.
  static constexpr size_t kReadChunk = 256;
  int32_t raw[kReadChunk];
  size_t bytes_read = 0;

  esp_err_t err = i2s_read(this->i2s_port_, raw, sizeof(raw), &bytes_read, 0 /* don't block the ESPHome loop */);
  if (err != ESP_OK || bytes_read == 0) return 0;

  size_t samples_read = bytes_read / sizeof(int32_t);
  for (size_t i = 0; i < samples_read; i++) {
    int32_t sample32 = raw[i] >> 14;  // keep top ~18 bits, roughly 16-bit range
    float sample = static_cast<float>(sample32) * this->mic_gain_;
    sample = std::max(-32768.0f, std::min(32767.0f, sample));

    this->ring_buffer_[this->ring_write_pos_] = static_cast<int16_t>(sample);
    this->ring_write_pos_ = (this->ring_write_pos_ + 1) % this->ring_capacity_;
  }
  return samples_read;
}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

static void copy_last_clip(const std::vector<int16_t> &ring, size_t write_pos, size_t clip_samples,
                            std::vector<int16_t> &out) {
  out.resize(clip_samples);
  size_t capacity = ring.size();
  size_t start = (write_pos + capacity - clip_samples) % capacity;
  for (size_t i = 0; i < clip_samples; i++) {
    out[i] = ring[(start + i) % capacity];
  }
}

bool TFLiteMicComponent::build_input_tensor_() {
  int8_t *dst_int8 = nullptr;
  float *dst_float = nullptr;
  if (this->input_->type == kTfLiteInt8) {
    dst_int8 = this->input_->data.int8;
  } else if (this->input_->type == kTfLiteFloat32) {
    dst_float = this->input_->data.f;
  } else {
    ESP_LOGE(TAG, "Unsupported input tensor type %d (expected int8 or float32)", this->input_->type);
    return false;
  }

  if (this->feature_type_ == FEATURE_RAW) {
    this->extract_raw_(dst_int8, dst_float);
  } else {
    this->extract_spectrogram_(dst_int8, dst_float);
  }
  return true;
}

void TFLiteMicComponent::extract_raw_(int8_t *dst_int8, float *dst_float) {
  size_t clip_samples = (this->sample_rate_ * this->clip_duration_ms_) / 1000;
  std::vector<int16_t> clip;
  copy_last_clip(this->ring_buffer_, this->ring_write_pos_, clip_samples, clip);

  size_t elem_size = dst_int8 ? 1 : 4;
  size_t n = std::min<size_t>(clip_samples, this->input_->bytes / elem_size);

  float scale = 1.0f;
  int32_t zero_point = 0;
  if (dst_int8) {
    scale = this->input_->params.scale;
    zero_point = this->input_->params.zero_point;
  }

  for (size_t i = 0; i < n; i++) {
    float norm = clip[i] / 32768.0f;  // -1..1
    if (dst_int8) {
      int32_t q = static_cast<int32_t>(std::lround(norm / scale) + zero_point);
      q = std::max<int32_t>(-128, std::min<int32_t>(127, q));
      dst_int8[i] = static_cast<int8_t>(q);
    } else {
      dst_float[i] = norm;
    }
  }
}

// --- tiny iterative radix-2 FFT (in place), sizes must be a power of 2.
// Deliberately UNNORMALIZED (no 1/N scaling) to match tf.signal.stft /
// numpy.fft convention, since trigger_model_int8's input quantization
// (zero_point=-128, i.e. real value floors at exactly 0) was derived from
// tf.abs(tf.signal.stft(...)) with no extra scaling. ---
static void fft(std::vector<float> &re, std::vector<float> &im) {
  size_t n = re.size();
  for (size_t i = 1, j = 0; i < n; i++) {
    size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      std::swap(re[i], re[j]);
      std::swap(im[i], im[j]);
    }
  }
  for (size_t len = 2; len <= n; len <<= 1) {
    float ang = -2.0f * static_cast<float>(M_PI) / static_cast<float>(len);
    float wr = std::cos(ang), wi = std::sin(ang);
    for (size_t i = 0; i < n; i += len) {
      float cur_wr = 1.0f, cur_wi = 0.0f;
      for (size_t j = 0; j < len / 2; j++) {
        float ur = re[i + j], ui = im[i + j];
        float vr = re[i + j + len / 2] * cur_wr - im[i + j + len / 2] * cur_wi;
        float vi = re[i + j + len / 2] * cur_wi + im[i + j + len / 2] * cur_wr;
        re[i + j] = ur + vr;
        im[i + j] = ui + vi;
        re[i + j + len / 2] = ur - vr;
        im[i + j + len / 2] = ui - vi;
        float next_wr = cur_wr * wr - cur_wi * wi;
        float next_wi = cur_wr * wi + cur_wi * wr;
        cur_wr = next_wr;
        cur_wi = next_wi;
      }
    }
  }
}

// Computes a linear-frequency STFT magnitude spectrogram, matching:
//   spec = abs(tf.signal.stft(waveform, frame_length, frame_step))
// i.e. NO mel filterbank and NO log -- just periodic-Hann-windowed,
// zero-padded-to-fft_length, unnormalized DFT magnitude per bin.
void TFLiteMicComponent::extract_spectrogram_(int8_t *dst_int8, float *dst_float) {
  size_t clip_samples = (this->sample_rate_ * this->clip_duration_ms_) / 1000;
  size_t num_bins = this->fft_length_ / 2 + 1;
  size_t num_frames = 1 + (clip_samples - this->frame_length_) / this->frame_step_;

  std::vector<int16_t> clip;
  copy_last_clip(this->ring_buffer_, this->ring_write_pos_, clip_samples, clip);

  // Periodic Hann window (matches tf.signal.hann_window(..., periodic=True)):
  // w[n] = 0.5 - 0.5*cos(2*pi*n / frame_length)   -- note: divide by N, not N-1.
  std::vector<float> window(this->frame_length_);
  for (size_t i = 0; i < this->frame_length_; i++) {
    window[i] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * i / this->frame_length_);
  }

  float scale = 1.0f;
  int32_t zero_point = 0;
  if (dst_int8) {
    scale = this->input_->params.scale;
    zero_point = this->input_->params.zero_point;
  }
  size_t elem_size = dst_int8 ? 1 : 4;
  size_t max_out = this->input_->bytes / elem_size;

  std::vector<float> re(this->fft_length_), im(this->fft_length_);
  size_t out_idx = 0;

  for (size_t f = 0; f < num_frames; f++) {
    size_t start = f * this->frame_step_;
    std::fill(re.begin(), re.end(), 0.0f);
    std::fill(im.begin(), im.end(), 0.0f);

    // Waveform is assumed to be int16 PCM representing a [-1, 1] float
    // signal (i.e. int16/32768), matching how the training pipeline reads
    // WAV files via tf.audio.decode_wav.
    for (size_t i = 0; i < this->frame_length_ && (start + i) < clip.size(); i++) {
      re[i] = (clip[start + i] / 32768.0f) * window[i];
    }

    fft(re, im);

    for (size_t b = 0; b < num_bins; b++) {
      if (out_idx >= max_out) break;
      float magnitude = std::sqrt(re[b] * re[b] + im[b] * im[b]);
      if (dst_int8) {
        int32_t q = static_cast<int32_t>(std::lround(magnitude / scale) + zero_point);
        q = std::max<int32_t>(-128, std::min<int32_t>(127, q));
        dst_int8[out_idx] = static_cast<int8_t>(q);
      } else {
        dst_float[out_idx] = magnitude;
      }
      out_idx++;
    }
  }

  if (out_idx != max_out) {
    ESP_LOGW(TAG,
             "Spectrogram produced %u values but model input expects %u -- check frame_length/frame_step/"
             "clip_duration_ms against how the model was trained",
             out_idx, max_out);
  }
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

void TFLiteMicComponent::run_inference_and_publish_() {
  if (this->interpreter_->Invoke() != kTfLiteOk) {
    ESP_LOGW(TAG, "Invoke() failed");
    return;
  }

  // trigger_model_int8 has a single sigmoid output (shape [1,1]): one
  // probability that the trigger word is present. Multi-output models are
  // still supported generically here in case you swap models later.
  size_t num_outputs = this->output_->dims->data[this->output_->dims->size - 1];

  for (auto *sensor : this->sensors_) {
    uint8_t idx = sensor->get_class_index();
    if (idx >= num_outputs) {
      ESP_LOGW(TAG, "class_index %u out of range (model has %u output value(s))", idx, num_outputs);
      continue;
    }

    float score;
    if (this->output_->type == kTfLiteInt8) {
      int32_t raw = this->output_->data.int8[idx];
      score = (raw - this->output_->params.zero_point) * this->output_->params.scale;
    } else if (this->output_->type == kTfLiteUInt8) {
      int32_t raw = this->output_->data.uint8[idx];
      score = (raw - this->output_->params.zero_point) * this->output_->params.scale;
    } else {
      score = this->output_->data.f[idx];
    }
    ESP_LOGD(TAG, "class %u score = %.3f", idx, score);
    sensor->report_score(score);
  }
}

void TFLiteMicComponent::dump_config() {
  ESP_LOGCONFIG(TAG, "TFLite Mic:");
  ESP_LOGCONFIG(TAG, "  I2S port: %d, BCK: %u, WS: %u, DATA: %u", this->i2s_port_, this->bck_pin_, this->ws_pin_,
                this->data_pin_);
  ESP_LOGCONFIG(TAG, "  Sample rate: %u Hz", this->sample_rate_);
  ESP_LOGCONFIG(TAG, "  Feature type: %s", this->feature_type_ == FEATURE_SPECTROGRAM ? "spectrogram" : "raw");
  ESP_LOGCONFIG(TAG, "  Tensor arena: %u bytes", this->tensor_arena_size_);
  if (this->is_failed()) {
    ESP_LOGE(TAG, "  Setup FAILED -- check logs above");
  }
}

}  // namespace tflite_mic
}  // namespace esphome
