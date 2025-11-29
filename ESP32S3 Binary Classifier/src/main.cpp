#include <WiFi.h>
#include "esp_camera.h"
#include <ESPAsyncWebServer.h>
#include "img_converters.h"
#include "esp_task_wdt.h"

#include "model_data.h"
#include "labels.h"

// TFLite Micro
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

// Mutex for camera access
SemaphoreHandle_t camera_mutex = nullptr;

// Task handles
TaskHandle_t inferenceTaskHandle = nullptr;

// Cached JPEG buffer for stream
static uint8_t *cached_jpg_buf = nullptr;
static size_t cached_jpg_len = 0;
static SemaphoreHandle_t jpg_cache_mutex = nullptr;

// wifi
const char* WIFI_SSID = "WIFI NAME";
const char* WIFI_PASS = "WIFI PASSWORD";

// camera config
camera_config_t camera_config = {
    .pin_pwdn = -1,
    .pin_reset = -1,
    .pin_xclk = 15,
    .pin_sscb_sda = 4,
    .pin_sscb_scl = 5,

    .pin_d7 = 16,
    .pin_d6 = 17,
    .pin_d5 = 18,
    .pin_d4 = 12,
    .pin_d3 = 10,
    .pin_d2 = 8,
    .pin_d1 = 9,
    .pin_d0 = 11,

    .pin_vsync = 6,
    .pin_href = 7,
    .pin_pclk = 13,

    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_GRAYSCALE,  // Grayscale for inference
    .frame_size = FRAMESIZE_96X96,
    .jpeg_quality = 12,
    .fb_count = 2,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

// TFLite globals
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input_tensor = nullptr;
  TfLiteTensor* output_tensor = nullptr;

  constexpr int kTensorArenaSize = 250 * 1024;
  // Allocate tensor arena in PSRAM to avoid DRAM overflow
  static uint8_t *tensor_arena = nullptr;
}  // namespace

static float last_probs[kNumLabels] = {0};


AsyncWebServer server(80);

// Simple index page
const char INDEX_HTML[] PROGMEM = R"HTML(
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>ESP32-S3 TinyML Cam</title>
<style>
body { font-family: sans-serif; background:#111; color:#eee; text-align:center; }
#bar p { margin:0.2em; }
#stream { max-width: 640px; border: 2px solid #555; }
</style>
</head>
<body>
<h1>Watch vs ESP32 classifier</h1>
<img id="stream" src="/stream" />
<div id="bar">
  <p><b>ESP32:</b> <span id="esp32">0</span>%</p>
  <p><b>Watch:</b> <span id="watch">0</span>%</p>
</div>
<script>
async function poll(){
  try{
    const r = await fetch('/predict');
    const j = await r.json();
    document.getElementById('esp32').textContent = (j.ESP32*100).toFixed(1);
    document.getElementById('watch').textContent = (j.Watch*100).toFixed(1);
  }catch(e){}
  setTimeout(poll, 500);
}
function refreshImage(){
  document.getElementById('stream').src = '/stream?t=' + new Date().getTime();
  setTimeout(refreshImage, 1000);
}
poll();
refreshImage();
</script>
</body>
</html>
)HTML";

// preprocess 96x96 grayscale frame into INT8 input tensor
void preprocess_frame_to_input(camera_fb_t *fb)
{
  // model input shape: [1,96,96,1]
  float scale = input_tensor->params.scale;
  int zero_point = input_tensor->params.zero_point;

  const uint8_t* src = fb->buf;
  int8_t* dst = input_tensor->data.int8;

  static bool first_run = true;
  if (first_run) {
    Serial.printf("Input quantization - scale: %f, zero: %d\n", scale, zero_point);
    Serial.printf("Frame size: %dx%d, format: %d, len: %d\n", fb->width, fb->height, fb->format, fb->len);
    Serial.printf("First 10 pixels: ");
    for (int i = 0; i < 10; i++) {
      Serial.printf("%d ", src[i]);
    }
    Serial.println();
    first_run = false;
  }

  for (int i = 0; i < fb->width * fb->height; ++i) {
    float x = static_cast<float>(src[i]) / 255.0f; // 0..1
    int32_t q = static_cast<int32_t>(roundf(x / scale)) + zero_point;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    dst[i] = static_cast<int8_t>(q);
  }
}


void handle_root(AsyncWebServerRequest *request)
{
  request->send_P(200, "text/html", INDEX_HTML);
}

void handle_predict(AsyncWebServerRequest *request)
{
  // assumes labels[0] = ESP32, labels[1] = Watch
  // Swap these or programmatically read from kLabels in labels.h
  char buf[128];
  snprintf(buf, sizeof(buf),
           "{\"ESP32\":%.3f,\"Watch\":%.3f}",
           last_probs[0], last_probs[1]);
  request->send(200, "application/json", buf);
}

// JPEG snapshot handler - serve cached JPEG from background task
void handle_stream(AsyncWebServerRequest *request)
{
  if (xSemaphoreTake(jpg_cache_mutex, pdMS_TO_TICKS(100)) != pdTRUE) {
    request->send(503, "text/plain", "Image not ready");
    return;
  }

  if (cached_jpg_buf && cached_jpg_len > 0) {
    AsyncWebServerResponse *response = request->beginResponse_P(200, "image/jpeg", cached_jpg_buf, cached_jpg_len);
    response->addHeader("Content-Disposition", "inline; filename=capture.jpg");
    response->addHeader("Cache-Control", "no-cache, no-store, must-revalidate");
    request->send(response);
  } else {
    request->send(500, "text/plain", "No image available");
  }

  xSemaphoreGive(jpg_cache_mutex);
}

void setup_tflite()
{
  // Allocate tensor arena in PSRAM
  // Without this, it allocates to internal DRAM and will fail to build
  tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
  if (!tensor_arena) {
    Serial.println("Failed to allocate tensor arena in PSRAM");
    return;
  }
  Serial.printf("Allocated %d KB tensor arena in PSRAM\n", kTensorArenaSize / 1024);

  // Set up TF logging
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    Serial.println("Model schema version mismatch");
    return;
  }

  static tflite::AllOpsResolver resolver;

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize,
      error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    Serial.println("AllocateTensors failed");
    return;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.printf("Input dims: %d %d %d %d\n",
    input_tensor->dims->data[0],
    input_tensor->dims->data[1],
    input_tensor->dims->data[2],
    input_tensor->dims->data[3]);
}

void setup_camera()
{
  if (esp_camera_init(&camera_config) != ESP_OK) {
    Serial.println("Camera init failed");
    while (true) delay(1000);
  }
  sensor_t *s = esp_camera_sensor_get();
  s->set_framesize(s, FRAMESIZE_96X96);
  // Keep JPEG format for web streaming, is converted for inference
  Serial.println("Camera initialized");
}

void setup_wifi()
{
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("Connecting to %s", WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
}

// Background task running on Core 0, handles camera and inference
// Might get better performance with camera and web server on Core 1 instead
void inferenceTask(void *parameter)
{
  Serial.println("Inference task started on core " + String(xPortGetCoreID()));

  while (true) {
    // Run inference
    if (xSemaphoreTake(camera_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
      camera_fb_t *fb = esp_camera_fb_get();
      if (fb) {
        // Do inference
        unsigned long start = millis();
        preprocess_frame_to_input(fb);

        if (interpreter->Invoke() == kTfLiteOk) {
          unsigned long elapsed = millis() - start;
          float out_scale = output_tensor->params.scale;
          int out_zero = output_tensor->params.zero_point;

          // Debug quantization parameters
          Serial.printf("Inference took %lu ms\n", elapsed);
          Serial.printf("Output scale: %f, zero: %d\n", out_scale, out_zero);

          for (int i = 0; i < kNumLabels; ++i) {
            int8_t v = output_tensor->data.int8[i];
            float p = (v - out_zero) * out_scale;
            last_probs[i] = p;
            Serial.printf("  Label %d: raw=%d, prob=%.4f\n", i, v, p);
          }

          Serial.printf("ESP32: %.2f  Watch: %.2f\n", last_probs[0], last_probs[1]);
        } else {
          Serial.println("Invoke failed!");
        }

        // Update cached JPEG for streaming
        if (xSemaphoreTake(jpg_cache_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
          // Free old buffer
          if (cached_jpg_buf) {
            free(cached_jpg_buf);
            cached_jpg_buf = nullptr;
          }

          // Convert to JPEG
          if (fmt2jpg(fb->buf, fb->len, fb->width, fb->height, PIXFORMAT_GRAYSCALE, 80, &cached_jpg_buf, &cached_jpg_len)) {
            // Created new cache
          }

          xSemaphoreGive(jpg_cache_mutex);
        }

        esp_camera_fb_return(fb);
      }
      xSemaphoreGive(camera_mutex);
    }

    // yield task
    // Unsure if we actually hit this
    taskYIELD();
  }
}

void setup_server()
{
  server.on("/", HTTP_GET, handle_root);
  server.on("/predict", HTTP_GET, handle_predict);
  server.on("/stream", HTTP_GET, handle_stream);
  server.begin();
}

void setup()
{
  Serial.begin(115200);
  delay(1000);

  Serial.println("Main setup running on core " + String(xPortGetCoreID()));


  // Disable watchdog for both cores, TensorFlow inference takes too long
  // This isn't a good practice in general, but it works for demonstration purposes
  disableCore0WDT();
  disableCore1WDT();
  Serial.println("Watchdog disabled on both cores");


  // Create mutexes
  camera_mutex = xSemaphoreCreateMutex();
  jpg_cache_mutex = xSemaphoreCreateMutex();
  if (!camera_mutex || !jpg_cache_mutex) {
    Serial.println("Failed to create mutexes");
    while(1) delay(1000);
  }

  setup_wifi();
  setup_camera();
  setup_tflite();
  setup_server();

  // Create inference task pinned to Core 0
  // Core 1 will handle WiFi/AsyncTCP/Web server
  xTaskCreatePinnedToCore(
    inferenceTask,
    "InferenceTask", 
    8192, 
    NULL,
    1, 
    &inferenceTaskHandle,
    0 
  );

  Serial.println("Inference task created on Core 0");
  Serial.println("Web server running on Core 1");
}

void loop()
{
  delay(10);
}