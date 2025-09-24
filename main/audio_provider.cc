/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

NOTICE: This file has been modified from the original version:
- Adding jinja templated variables, so the project can be used in code generation.
- More recent i2s API from idf
- Use ES7210 ADC chip
- Remove unneeded code
==============================================================================*/

#include "audio_provider.h"

#include <cstdlib>
#include <cstring>

// FreeRTOS.h must be included before some of the following dependencies.
// Solves b/150260343.
// clang-format off
#include "freertos/FreeRTOS.h"
// clang-format on

#include <esp_check.h>
#include "es7210.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_spi_flash.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/task.h"
#include "ringbuf.h"
#include "micro_model_settings.h"

using namespace std;

// for c2 and c3, I2S support was added from IDF v4.4 onwards
#define NO_I2S_SUPPORT CONFIG_IDF_TARGET_ESP32C2 || \
                          (CONFIG_IDF_TARGET_ESP32C3 \
                          && (ESP_IDF_VERSION < ESP_IDF_VERSION_VAL(4, 4, 0)))

static const char* TAG = "TF_LITE_AUDIO_PROVIDER";
/* ringbuffer to hold the incoming audio data */
ringbuf_t* g_audio_capture_buffer;
volatile int32_t g_latest_audio_timestamp = 0;
/* model requires 20ms new data from g_audio_capture_buffer and 10ms old data
 * each time , storing old data in the histrory buffer , {
 * history_samples_to_keep = 10 * 16 } */
constexpr int32_t history_samples_to_keep =
    ((kFeatureDurationMs - kFeatureStrideMs) *
     (kAudioSampleFrequency / 1000));
/* new samples to get each time from ringbuffer, { new_samples_to_get =  20 * 16
 * } */
constexpr int32_t new_samples_to_get =
    (kFeatureStrideMs * (kAudioSampleFrequency / 1000));

const int32_t kAudioCaptureBufferSize = 40000;
const int32_t i2s_bytes_to_read = 6400;

namespace {
int16_t g_audio_output_buffer[kMaxAudioSampleSize * 32];
bool g_is_audio_initialized = false;
int16_t g_history_buffer[history_samples_to_keep];

#if !NO_I2S_SUPPORT
uint8_t g_i2s_read_buffer[i2s_bytes_to_read] = {};
#if CONFIG_IDF_TARGET_ESP32
i2s_port_t i2s_port = I2S_NUM_1; // for esp32-eye
#else
i2s_port_t i2s_port = I2S_NUM_0; // for esp32-s3-eye
#endif
#endif
}  // namespace

#if NO_I2S_SUPPORT
  // nothing to be done here
#else

static int es7210_codec_init() {
    ESP_LOGI(TAG, "Init I2C used to configure ES7210");
    i2c_config_t i2c_conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = GPIO_NUM_17,
        .scl_io_num = GPIO_NUM_18,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master = {
            .clk_speed = 100000,
        }
    };
    ESP_RETURN_ON_ERROR(i2c_param_config(I2C_NUM_0, &i2c_conf), TAG, "Failed to configure I2C parameters");
    ESP_RETURN_ON_ERROR(i2c_driver_install(I2C_NUM_0, i2c_conf.mode, 0, 0, 0), TAG, "Failed to install I2C driver");

    /* Create ES7210 device handle */
    es7210_dev_handle_t es7210_handle = NULL;
    es7210_i2c_config_t es7210_i2c_conf = {
        .i2c_port = I2C_NUM_0,
        .i2c_addr = 0x40
    };
    ESP_RETURN_ON_ERROR(es7210_new_codec(&es7210_i2c_conf, &es7210_handle), TAG, "Failed to instantiate codec.");

    ESP_LOGI(TAG, "Configure ES7210 codec parameters");
    es7210_codec_config_t codec_conf = {
        .sample_rate_hz = 16000,
        .mclk_ratio = I2S_MCLK_MULTIPLE_256,
        .i2s_format = ES7210_I2S_FMT_I2S,
        .bit_width = (es7210_i2s_bits_t)(I2S_DATA_BIT_WIDTH_32BIT),
        .mic_bias = ES7210_MIC_BIAS_2V87,
        .mic_gain = ES7210_MIC_GAIN_30DB,
        .flags = {
            .tdm_enable = true
        }
    };
    ESP_RETURN_ON_ERROR(es7210_config_codec(es7210_handle, &codec_conf), TAG, "Failed to config codec");
    ESP_RETURN_ON_ERROR(es7210_config_volume(es7210_handle, 0), TAG, "Failed to config volume");
    return ESP_OK;
}

static int i2s_init(i2s_chan_handle_t &rx_handle) {
  // Start listening for audio: MONO @ 16KHz
  i2s_std_config_t std_config = {
    .clk_cfg = {
      .sample_rate_hz = 16000,
        .clk_src = I2S_CLK_SRC_DEFAULT,
      .mclk_multiple = I2S_MCLK_MULTIPLE_256,
    },
    .slot_cfg = {
      .data_bit_width = I2S_DATA_BIT_WIDTH_32BIT,
      .slot_bit_width = I2S_SLOT_BIT_WIDTH_AUTO,
      .slot_mode = I2S_SLOT_MODE_MONO,
      .slot_mask = I2S_STD_SLOT_LEFT,
      .ws_width = I2S_DATA_BIT_WIDTH_32BIT,  // TODO: THIS MUST BE THE SAME AS DATA BIT WIDTH
      .ws_pol = false,
      .bit_shift = true,
      .left_align = true,
      .big_endian = false,
      .bit_order_lsb = false
    },
    .gpio_cfg = {
        .mclk = GPIO_NUM_16,
        .bclk = GPIO_NUM_9,
        .ws = GPIO_NUM_45,
        .dout = I2S_GPIO_UNUSED,
        .din = GPIO_NUM_10,
        .invert_flags = {
          .mclk_inv = false,
          .bclk_inv = false,
          .ws_inv = false,
        }
    }
  };
  i2s_chan_config_t chan_config = {
      .id = i2s_port,
      .role = I2S_ROLE_MASTER,
      .dma_desc_num = 512,
      .dma_frame_num = 8,
      .auto_clear = false
  };
  ESP_RETURN_ON_ERROR(i2s_new_channel(&chan_config, nullptr, &rx_handle), TAG, "Couldn't create new channel");
  ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(rx_handle, &std_config), TAG, "Couldn't init i2s mode");
  ESP_RETURN_ON_ERROR(i2s_channel_enable(rx_handle), TAG, "Couldn't enable channel");
  ESP_LOGI(TAG, "I2S initialized");
  return ESP_OK;
}
#endif

static void CaptureSamples(void* arg) {
  if (es7210_codec_init() != ESP_OK) {
    ESP_LOGE(TAG, "Can't configure ADC");
    return;
  }

  size_t bytes_read = i2s_bytes_to_read;
  i2s_chan_handle_t rx_handle;
  if (i2s_init(rx_handle) != ESP_OK) {
    ESP_LOGE(TAG, "No i2s RX handle");
    return;
  }
  while (true) {
    /* read 100ms data at once from i2s */
    i2s_channel_read(rx_handle, (void*)g_i2s_read_buffer, i2s_bytes_to_read,
             &bytes_read, 100);

    if (bytes_read <= 0) {
      ESP_LOGE(TAG, "Error in I2S read : %d", bytes_read);
    } else {
      if (bytes_read < i2s_bytes_to_read) {
        ESP_LOGW(TAG, "Partial I2S read");
      }
#if CONFIG_IDF_TARGET_ESP32S3
      // rescale the data
      for (int i = 0; i < bytes_read / 4; ++i) {
        ((int16_t *) g_i2s_read_buffer)[i] = ((int32_t *) g_i2s_read_buffer)[i] >> 16;
      }
      bytes_read = bytes_read / 2;
#endif
      /* write bytes read by i2s into ring buffer */
      int bytes_written = rb_write(g_audio_capture_buffer,
                                   (uint8_t*)g_i2s_read_buffer, bytes_read, pdMS_TO_TICKS(100));
      if (bytes_written != bytes_read) {
        ESP_LOGI(TAG, "Could only write %d bytes out of %d", bytes_written, bytes_read);
      }
      /* update the timestamp (in ms) to let the model know that new data has
       * arrived */
      g_latest_audio_timestamp = g_latest_audio_timestamp +
          ((1000 * (bytes_written / 2)) / kAudioSampleFrequency);
      if (bytes_written <= 0) {
        ESP_LOGE(TAG, "Could Not Write in Ring Buffer: %d ", bytes_written);
      } else if (bytes_written < bytes_read) {
        ESP_LOGW(TAG, "Partial Write");
      }
    }
  }
  vTaskDelete(NULL);
}

TfLiteStatus InitAudioRecording() {
  g_audio_capture_buffer = rb_init("tf_ringbuffer", kAudioCaptureBufferSize);
  if (!g_audio_capture_buffer) {
    ESP_LOGE(TAG, "Error creating ring buffer");
    return kTfLiteError;
  }
  /* create CaptureSamples Task which will get the i2s_data from mic and fill it
   * in the ring buffer */
  xTaskCreate(CaptureSamples, "CaptureSamples", 1024 * 4, NULL, 10, NULL);
  while (!g_latest_audio_timestamp) {
    vTaskDelay(1); // one tick delay to avoid watchdog
  }
  ESP_LOGI(TAG, "Audio Recording started");
  return kTfLiteOk;
}


TfLiteStatus GetAudioSamples(int* audio_samples_size, int16_t** audio_samples) {
  if (!g_is_audio_initialized) {
    TfLiteStatus init_status = InitAudioRecording();
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    g_is_audio_initialized = true;
  }

  // We're doing sliding windows, so read one stride worth of samples and get
  // the rest of the window from history.
  // History first
  memcpy((void*)(g_audio_output_buffer), (void*)(g_history_buffer),
         history_samples_to_keep * sizeof(int16_t));
  // Then new samples
  int bytes_read =
      rb_read(g_audio_capture_buffer,
              ((uint8_t*)(g_audio_output_buffer + history_samples_to_keep)),
              new_samples_to_get * sizeof(int16_t), pdMS_TO_TICKS(200));
  if (bytes_read < 0) {
    ESP_LOGE(TAG, " Model Could not read data from Ring Buffer");
  } else if (bytes_read < new_samples_to_get * sizeof(int16_t)) {
    ESP_LOGD(TAG, "RB FILLED RIGHT NOW IS %d",
             rb_filled(g_audio_capture_buffer));
    ESP_LOGD(TAG, " Partial Read of Data by Model ");
    ESP_LOGV(TAG, " Could only read %d bytes when required %d bytes ",
             bytes_read, (int) (new_samples_to_get * sizeof(int16_t)));
  }

  // update history with the new samples we read
  memcpy((void*)(g_history_buffer),
         (void*)(g_audio_output_buffer + new_samples_to_get),
         history_samples_to_keep * sizeof(int16_t));

  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_audio_output_buffer;
  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() { return g_latest_audio_timestamp; }
