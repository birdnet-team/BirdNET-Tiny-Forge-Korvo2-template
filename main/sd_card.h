# pragma once
#include "esp_err.h"


namespace sdcard {
esp_err_t mount();
void unmount();
void logPredictions(float *predictions);
bool writeBytes(char* filename, const void* data, size_t size);
}  // namespace sdcard