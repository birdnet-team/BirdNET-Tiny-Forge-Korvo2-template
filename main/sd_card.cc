#include <cerrno>
#include "esp_vfs.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdmmc_host.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "micro_model_settings.h"

#define MOUNT_POINT "/sdcard"

static const char *TAG = "sd";

namespace sdcard {
esp_err_t mount() {
  esp_err_t ret;

  esp_vfs_fat_sdmmc_mount_config_t mount_config = {
    .format_if_mount_failed = false,
    .max_files = 5,
    .allocation_unit_size = 16 * 1024
  };

  sdmmc_card_t *card;
  sdmmc_host_t host = SDMMC_HOST_DEFAULT();

  sdmmc_slot_config_t slot_config = {
    .clk = GPIO_NUM_15,
    .cmd = GPIO_NUM_7,
    .d0 = GPIO_NUM_4,
    .d1 = GPIO_NUM_NC,
    .d2 = GPIO_NUM_NC,
    .d3 = GPIO_NUM_NC,
    .cd = SDMMC_SLOT_NO_CD,
    .wp = SDMMC_SLOT_NO_WP,
    .width   = 1,
    .flags = SDMMC_SLOT_FLAG_INTERNAL_PULLUP,
  };

  ret = esp_vfs_fat_sdmmc_mount(MOUNT_POINT, &host, &slot_config, &mount_config, &card);

  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to mount SD card (%s)", esp_err_to_name(ret));
    return ret;
  }

  ESP_LOGI(TAG, "SD card mounted at %s", MOUNT_POINT);
  sdmmc_card_print_info(stdout, card);
  return ESP_OK;
}

void unmount() {
  esp_vfs_fat_sdcard_unmount(MOUNT_POINT, nullptr);
  ESP_LOGI(TAG, "SD card unmounted");
}

void logPredictions(float* predictions) {
  static FILE* prediction_file = nullptr;
  static char current_filename[64] = {0};
  static unsigned int file_index = 0;
  static bool index_initialized = false;
  const size_t MAX_FILE_SIZE = 512 * 1024; // 512KB

  if (predictions == nullptr) {
    ESP_LOGE(TAG, "Predictions array is null");
    return;
  }

  // Initialize file index on first run by finding the highest existing index
  if (!index_initialized) {
    DIR* dir = opendir("/sdcard");
    if (dir == nullptr) {
      ESP_LOGE(TAG, "Failed to open /sdcard directory: %s", strerror(errno));
      return;
    }

    struct dirent* entry;
    uint32_t max_index = 0;
    while ((entry = readdir(dir)) != nullptr) {
      unsigned int index;
      if (sscanf(entry->d_name, "%u.csv", &index) == 1) {
        if (index > max_index) {
          max_index = index;
        }
      }
    }
    closedir(dir);
    file_index = max_index;
    ESP_LOGI(TAG, "Curr file index: %d", file_index);

    // Check if current max file exists and is under size limit
    snprintf(current_filename, sizeof(current_filename), "/sdcard/%u.csv", file_index);
    struct stat file_stat = {};
    if (stat(current_filename, &file_stat) == 0 && file_stat.st_size < MAX_FILE_SIZE) {
      // Current file exists and has space, continue using it
      ESP_LOGI(TAG, "Continuing with existing file: %s (size: %ld bytes)",
               current_filename, file_stat.st_size);
    } else {
      // Need new file
      file_index++;
      snprintf(current_filename, sizeof(current_filename), "/sdcard/%u.csv", file_index);
    }
    index_initialized = true;
  }

  // Check if current file needs rotation (size limit reached)
  bool need_new_file = false;
  if (prediction_file != nullptr) {
    if (fseek(prediction_file, 0, SEEK_END) != 0) {
      ESP_LOGE(TAG, "Failed to seek file, closing: %s", strerror(errno));
      fclose(prediction_file);
      prediction_file = nullptr;
      return;
    }

    long current_size = ftell(prediction_file);
    if (current_size >= MAX_FILE_SIZE) {
      need_new_file = true;
      fclose(prediction_file);
      prediction_file = nullptr;
      ESP_LOGI(TAG, "File %s reached size limit (%ld bytes), rotating",
               current_filename, current_size);
    }
  }

  // Open file if needed (first run or after rotation)
  if (prediction_file == nullptr) {
    if (need_new_file) {
      file_index++;
      snprintf(current_filename, sizeof(current_filename), "/sdcard/%u.csv", file_index);
    }

    prediction_file = fopen(current_filename, "a");
    if (prediction_file == nullptr) {
      ESP_LOGE(TAG, "Failed to open predictions file %s: %s", current_filename, strerror(errno));
      return;
    }

    ESP_LOGI(TAG, "Opened prediction file: %s", current_filename);

    // Write header if file is new (check if empty)
    fseek(prediction_file, 0, SEEK_END);
    if (ftell(prediction_file) == 0) {
      fprintf(prediction_file, "timestamp");
      for (auto kCategoryLabel : kCategoryLabels) {
        fprintf(prediction_file, ",%s", kCategoryLabel);
      }
      fprintf(prediction_file, "\n");
      ESP_LOGD(TAG, "Written CSV header to new file");
    }
  }

  // Write prediction data
  int64_t timestamp_ms = esp_timer_get_time() / 1000;
  if (fprintf(prediction_file, "%lld", timestamp_ms) < 0) {
    ESP_LOGE(TAG, "Failed to write to file: %s", strerror(errno));
    return;
  }

  for (int i = 0; i < kCategoryCount; i++) {
    if (fprintf(prediction_file, ",%.4f", predictions[i]) < 0) {
      ESP_LOGE(TAG, "Failed to write prediction data: %s", strerror(errno));
      return;
    }
  }
  fprintf(prediction_file, "\n");

  // Ensure data is written immediately
  if (fflush(prediction_file) != 0) {
    ESP_LOGE(TAG, "Failed to flush file: %s", strerror(errno));
  }
  if (fsync(fileno(prediction_file)) != 0) {
    ESP_LOGE(TAG, "Failed to sync file: %s", strerror(errno));
  }

  ESP_LOGD(TAG, "Logged predictions to %s", current_filename);
  ESP_LOGD(TAG, "File size %lu/%zu", ftell(prediction_file), MAX_FILE_SIZE);
}

bool writeBytes(char* filename, const void* data, size_t size) {
  if (filename == nullptr || data == nullptr || size == 0) {
    ESP_LOGE(TAG, "Invalid parameters: filename=%p, data=%p, size=%zu",
             filename, data, size);
    return false;
  }

  FILE* file = fopen(filename, "ab");
  if (file == nullptr) {
    ESP_LOGE(TAG, "Failed to open file %s: %s", filename, strerror(errno));
    return false;
  }

  size_t written = fwrite(data, 1, size, file);
  if (written != size) {
    ESP_LOGE(TAG, "Write incomplete: %zu/%zu bytes to %s", written, size, filename);
    fclose(file);
    return false;
  }

  if (fflush(file) != 0) {
    ESP_LOGE(TAG, "Failed to flush %s: %s", filename, strerror(errno));
    fclose(file);
    return false;
  }

  if (fsync(fileno(file)) != 0) {
    ESP_LOGE(TAG, "Failed to sync %s: %s", filename, strerror(errno));
    fclose(file);
    return false;
  }

  fclose(file);
  ESP_LOGD(TAG, "Successfully appended %zu bytes to %s", size, filename);
  return true;
}


}  // namespace sdcard
