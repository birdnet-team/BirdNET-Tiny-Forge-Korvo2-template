/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

 NOTICE: this file was modified such that the feature provider's feature extraction
 could run in its own task, in parallel with inference rather than serially.
==============================================================================*/

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <esp_log.h>

#include <cstring>
#include <esp_timer.h>
#include "feature_provider.h"

#include "audio_provider.h"
#include "micro_features_generator.h"
#include "micro_model_settings.h"
#include "tensorflow/lite/micro/micro_log.h"


Features g_features;
const char *TAG = "feature_provider";

FeatureProvider::FeatureProvider(int feature_size, int8_t* feature_data)
    : feature_size_(feature_size),
      feature_data_(feature_data),
      is_first_run_(true),
      task_params{},
      n_new_slices(0){
  // Initialize the feature data to default values.
  for (int n = 0; n < feature_size_; ++n) {
    feature_data_[n] = 0;
  }
}

FeatureProvider::~FeatureProvider() {}


static void ComputeFeatures(void *pvParameters) {
  TickType_t xLastWakeTime;
  const TickType_t xFrequency = pdMS_TO_TICKS(kFeatureStrideMs) > 0 ? pdMS_TO_TICKS(kFeatureStrideMs) : 1;
  ESP_LOGI(TAG, "ticks: %lu", xFrequency);
  xLastWakeTime = xTaskGetTickCount();
  ESP_LOGI(TAG, "Feature provider task starting");
  auto *params = (fp_task_params_t *)pvParameters;
  int how_many_new_slices = 0;
  int32_t previous_time = 0;
  while(true) {
    ESP_LOGD(TAG, "Feature provider running at tick: %lu", xTaskGetTickCount());
    const int32_t current_time = LatestAudioTimestamp();
    ESP_LOGD(TAG, "Last time: %ld, cur time: %ld", previous_time, current_time);
    *(params->n_new_slices) = 0;
    TfLiteStatus feature_status = params->populate_func(previous_time, current_time, params->n_new_slices);
    if (feature_status != kTfLiteOk) {
      MicroPrintf("Feature generation failed");
      vTaskDelete(nullptr);
      return;
    }
    previous_time = current_time;
    xTaskDelayUntil(&xLastWakeTime, xFrequency);
  }
}


TfLiteStatus FeatureProvider::InitFeatureExtraction() {
  task_params.populate_func = [this](auto && PH1, auto && PH2, auto && PH3) {
    return this->PopulateFeatureData(
      std::forward<decltype(PH1)>(PH1),
      std::forward<decltype(PH2)>(PH2),
      std::forward<decltype(PH3)>(PH3));
  };
  task_params.n_new_slices = &n_new_slices;

  xTaskCreatePinnedToCore(
    ComputeFeatures,
    "ComputeFeatures",
    20000,                  // Stack size in bytes
    &task_params,           // Task parameters
    22,                      // Task priority (0-25, higher = more priority)
    nullptr,                // Task handle (not needed here)
    0                       // Core id: we pin the preprocessing to core 0, and the classifier to core 1
  );
  ESP_LOGI(TAG, "Periodic task created successfully");
  return kTfLiteOk;
}


TfLiteStatus FeatureProvider::PopulateFeatureData(
    int32_t last_time_in_ms, int32_t time_in_ms, std::atomic<int>* how_many_new_slices) {
  if (feature_size_ != kFeatureElementCount) {
    MicroPrintf("Requested feature_data_ size %d doesn't match %d",
                feature_size_, kFeatureElementCount);
    return kTfLiteError;
  }

  // Quantize the time into steps as long as each window stride, so we can
  // figure out which audio data we need to fetch.
  const int last_step = (last_time_in_ms / kFeatureStrideMs);
  const int current_step = (time_in_ms / kFeatureStrideMs);

  int slices_needed = current_step - last_step;
  ESP_LOGD(TAG, "Slices needed: %d", slices_needed);
  // If this is the first call, make sure we don't use any cached information.
  if (is_first_run_) {
    TfLiteStatus init_status = InitializeMicroFeatures();
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    ESP_LOGI(TAG, "InitializeMicroFeatures successful");
    is_first_run_ = false;
    slices_needed = kFeatureCount;
  }

  if (slices_needed > kFeatureCount) {
    slices_needed = kFeatureCount;
  }

  const int slices_to_keep = kFeatureCount - slices_needed;
  const int slices_to_drop = kFeatureCount - slices_to_keep;
  // If we can avoid recalculating some slices, just move the existing data
  // up in the spectrogram, to perform something like this:
  // last time = 80ms          current time = 120ms
  // +-----------+             +-----------+
  // | data@20ms |         --> | data@60ms |
  // +-----------+       --    +-----------+
  // | data@40ms |     --  --> | data@80ms |
  // +-----------+   --  --    +-----------+
  // | data@60ms | --  --      |  <empty>  |
  // +-----------+   --        +-----------+
  // | data@80ms | --          |  <empty>  |
  // +-----------+             +-----------+
  if (slices_to_keep > 0) {
    for (int dest_slice = 0; dest_slice < slices_to_keep; ++dest_slice) {
      int8_t* dest_slice_data =
          feature_data_ + (dest_slice * kFeatureSize);
      const int src_slice = dest_slice + slices_to_drop;
      const int8_t* src_slice_data =
          feature_data_ + (src_slice * kFeatureSize);
      for (int i = 0; i < kFeatureSize; ++i) {
        dest_slice_data[i] = src_slice_data[i];
      }
    }
  }
  // Any slices that need to be filled in with feature data have their
  // appropriate audio data pulled, and features calculated for that slice.
  if (slices_needed > 0) {
    for (int new_slice = slices_to_keep; new_slice < kFeatureCount;
         ++new_slice) {
      const int new_step = (current_step - kFeatureCount + 1) + new_slice;
      const int32_t slice_start_ms = (new_step * kFeatureStrideMs);
      int16_t* audio_samples = nullptr;
      int audio_samples_size = 0;
      // TODO(petewarden): Fix bug that leads to non-zero slice_start_ms
      GetAudioSamples(&audio_samples_size, &audio_samples);
      if (audio_samples_size < kMaxAudioSampleSize) {
        ESP_LOGI(TAG, "Audio data size %d too small, want %d",
                    audio_samples_size, kMaxAudioSampleSize);
        return kTfLiteError;
      }
      int8_t* new_slice_data = feature_data_ + (new_slice * kFeatureSize);

      TfLiteStatus generate_status = GenerateFeatures(
            audio_samples, audio_samples_size, &g_features);
      if (generate_status != kTfLiteOk) {
        return generate_status;
      }

      // copy features
      for (int j = 0; j < kFeatureSize; ++j) {
        new_slice_data[j] = g_features[0][j];
      }
    }
  }
  *how_many_new_slices = slices_needed;
  return kTfLiteOk;
}

int FeatureProvider::GetNewSlicesN() {
  return n_new_slices;
}
