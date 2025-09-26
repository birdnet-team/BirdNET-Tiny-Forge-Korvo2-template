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
==============================================================================*/
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "main_functions.h"

[[noreturn]] void tf_main() {
  setup();
  while (true) {
    loop();
  }
}

extern "C" void app_main() {
  xTaskCreatePinnedToCore((TaskFunction_t)&tf_main, "tensorflow", 8 * 1024, nullptr, 8, nullptr, 1);
  vTaskDelete(nullptr);
}
