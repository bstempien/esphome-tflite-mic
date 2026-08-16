#pragma once

#include "esphome/core/component.h"
#include "esphome/core/hal.h"
#include "esphome/components/binary_sensor/binary_sensor.h"

namespace esphome {
namespace tflite_mic {

// One instance per YAML `binary_sensor: platform: tflite_mic` entry.
// The parent TFLiteMicComponent calls report_score() once per inference
// with the (0.0-1.0) probability for this sensor's class_index_.
class TFLiteMicBinarySensor : public binary_sensor::BinarySensor, public Component {
 public:
  void set_class_index(uint8_t index) { this->class_index_ = index; }
  void set_threshold(float threshold) { this->threshold_ = threshold; }
  void set_min_trigger_interval(uint32_t ms) { this->min_trigger_interval_ms_ = ms; }

  uint8_t get_class_index() const { return this->class_index_; }

  // Called by the parent component after every inference pass.
  void report_score(float score) {
    uint32_t now = millis();
    bool above = score >= this->threshold_;

    if (above && !this->state && (now - this->last_trigger_time_) >= this->min_trigger_interval_ms_) {
      this->publish_state(true);
      this->last_trigger_time_ = now;
      this->last_on_time_ = now;
    } else if (!above && this->state && (now - this->last_on_time_) >= this->min_trigger_interval_ms_) {
      // Auto-clear once the class score drops back below threshold and
      // we've held the ON state for at least one trigger interval. This
      // gives HA a clean pulse instead of a single-loop-iteration blip.
      this->publish_state(false);
    }
  }

  void dump_config() override {
    ESP_LOGCONFIG(TAG, "TFLite Mic Binary Sensor:");
    ESP_LOGCONFIG(TAG, "  Class index: %u", this->class_index_);
    ESP_LOGCONFIG(TAG, "  Threshold: %.2f", this->threshold_);
    ESP_LOGCONFIG(TAG, "  Min trigger interval: %ums", this->min_trigger_interval_ms_);
  }

 protected:
  static constexpr const char *TAG = "tflite_mic.binary_sensor";
  uint8_t class_index_{0};
  float threshold_{0.8f};
  uint32_t min_trigger_interval_ms_{1000};
  uint32_t last_trigger_time_{0};
  uint32_t last_on_time_{0};
};

}  // namespace tflite_mic
}  // namespace esphome
