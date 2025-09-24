# pragma once
#include "esp_err.h"


namespace sdcard {
esp_err_t mount();
void unmount();
void logPredictions(float *predictions);
}  // namespace sdcard