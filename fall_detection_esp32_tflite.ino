#include <Arduino.h>
#include <Wire.h>
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h" // 변환된 모델 헤더 파일 (별도 생성 필요)
#include <SPIFFS.h>
#include <SD.h>
#include <WiFi.h>
#include <time.h>

// MPU6050 I2C 설정
#define MPU6050_ADDR 0x68
#define PWR_MGMT_1 0x6B
#define SMPLRT_DIV 0x19
#define CONFIG 0x1A
#define GYRO_CONFIG 0x1B
#define ACCEL_CONFIG 0x1C
#define ACCEL_XOUT_H 0x3B
#define GYRO_XOUT_H 0x43

// 핀 설정
#define LED_PIN 2      // 낙상 경고용 LED
#define BUZZER_PIN 4   // 낙상 경고용 부저
#define BUTTON_PIN 15  // 알람 중지 버튼
#define SD_CS_PIN 5    // SD 카드 CS 핀

// 시스템 설정
#define SEQ_LENGTH 100  // 시퀀스 길이
#define NUM_FEATURES 9  // 특징 수 (AccX, AccY, AccZ, GyrX, GyrY, GyrZ, EulerX, EulerY, EulerZ)
#define SAMPLING_RATE 100  // Hz (데이터 샘플링 속도)
#define LOG_DATA true     // 데이터 로깅 활성화
#define WIFI_ENABLED false // WiFi 연결 활성화

// WiFi 설정 (필요한 경우)
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// 스케일러 파라미터 (모델 학습 후 생성된 값으로 업데이트 필요)
float scaler_mean[NUM_FEATURES] = {
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  // 실제 값으로 업데이트 필요
};
float scaler_scale[NUM_FEATURES] = {
  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  // 실제 값으로 업데이트 필요
};

// 데이터 버퍼
float data_buffer[SEQ_LENGTH][NUM_FEATURES] = {0};
int buffer_index = 0;
bool buffer_filled = false;

// 알람 상태
bool alarm_active = false;
unsigned long alarm_start_time = 0;
const unsigned long ALARM_DURATION = 10000; // 10초

// 데이터 로깅
File logFile;
unsigned long log_start_time = 0;
const unsigned long LOG_INTERVAL = 600000; // 10분마다 새 로그 파일 (밀리초)
unsigned long last_log_time = 0;
int frame_counter = 0;

// TensorFlow Lite 객체
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

// 모델 입출력 텐서를 위한 메모리 할당
constexpr int kTensorArenaSize = 32 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// 함수 선언
void setupMPU6050();
void readMPU6050Data(float* acc_x, float* acc_y, float* acc_z, 
                     float* gyr_x, float* gyr_y, float* gyr_z);
void calculateEulerAngles(float acc_x, float acc_y, float acc_z, 
                          float* euler_x, float* euler_y, float* euler_z);
void addDataToBuffer(float acc_x, float acc_y, float acc_z, 
                     float gyr_x, float gyr_y, float gyr_z,
                     float euler_x, float euler_y, float euler_z);
void normalizeData(float normalized_buffer[][NUM_FEATURES]);
bool predictFall();
void triggerAlarm();
void stopAlarm();
void buttonInterrupt();
void initLogging();
void logSensorData(float acc_x, float acc_y, float acc_z, 
                  float gyr_x, float gyr_y, float gyr_z,
                  float euler_x, float euler_y, float euler_z);
String getTimestamp();
void initWiFi();
void syncTime();

void setup() {
  // 시리얼 통신 초기화
  Serial.begin(115200);
  Serial.println("낙상 감지 시스템 초기화 중...");
  
  // I2C 통신 초기화
  Wire.begin();
  
  // MPU6050 설정
  setupMPU6050();
  
  // GPIO 설정
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  // LED 테스트
  digitalWrite(LED_PIN, HIGH);
  delay(500);
  digitalWrite(LED_PIN, LOW);
  
  // 버튼 인터럽트 설정
  attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), buttonInterrupt, FALLING);
  
  // WiFi 및 시간 동기화 (필요한 경우)
  if (WIFI_ENABLED) {
    initWiFi();
    syncTime();
  }
  
  // 로깅 초기화 (필요한 경우)
  if (LOG_DATA) {
    initLogging();
  }
  
  // TFLite 모델 로드
  model = tflite::GetModel(g_model_data); // model_data.h 에서 가져옴
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("모델 버전이 맞지 않습니다!");
    while (1);
  }
  
  // 인터프리터 설정
  interpreter = new tflite::MicroInterpreter(
    model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
  
  // 텐서 할당
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("텐서 할당 실패!");
    while (1);
  }
  
  // 입력 텐서 정보 출력
  TfLiteTensor* input = interpreter->input(0);
  Serial.print("입력 텐서 크기: ");
  Serial.print(input->dims->data[0]); Serial.print(", ");
  Serial.print(input->dims->data[1]); Serial.print(", ");
  Serial.println(input->dims->data[2]);
  
  // 시작 시간 저장
  log_start_time = millis();
  
  Serial.println("낙상 감지 시스템 준비 완료");
}

void loop() {
  // IMU 센서 데이터 읽기
  float acc_x, acc_y, acc_z;
  float gyr_x, gyr_y, gyr_z;
  float euler_x, euler_y, euler_z;
  
  readMPU6050Data(&acc_x, &acc_y, &acc_z, &gyr_x, &gyr_y, &gyr_z);
  calculateEulerAngles(acc_x, acc_y, acc_z, &euler_x, &euler_y, &euler_z);
  
  // 프레임 카운터 증가
  frame_counter++;
  
  // 데이터 로깅 (필요한 경우)
  if (LOG_DATA) {
    logSensorData(acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, euler_x, euler_y, euler_z);
  }
  
  // 데이터 버퍼에 추가
  addDataToBuffer(acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, euler_x, euler_y, euler_z);
  
  // 디버그 출력 (100개의 샘플마다)
  static int debug_counter = 0;
  if (++debug_counter >= 100) {
    Serial.print("가속도: X="); Serial.print(acc_x);
    Serial.print(", Y="); Serial.print(acc_y);
    Serial.print(", Z="); Serial.println(acc_z);
    Serial.print("자이로: X="); Serial.print(gyr_x);
    Serial.print(", Y="); Serial.print(gyr_y);
    Serial.print(", Z="); Serial.println(gyr_z);
    debug_counter = 0;
  }
  
  // 버퍼가 다 찼을 때만 예측 수행
  if (buffer_filled) {
    // 낙상 예측
    bool is_fall = predictFall();
    
    // 낙상으로 예측된 경우
    if (is_fall) {
      Serial.println("낙상 감지! 알람 발생");
      triggerAlarm();
    }
  }
  
  // 알람 시간 체크 (자동 종료)
  if (alarm_active && (millis() - alarm_start_time > ALARM_DURATION)) {
    stopAlarm();
  }
  
  // 샘플링 속도 조절
  static unsigned long last_sample_time = 0;
  unsigned long sample_delay = (1000 / SAMPLING_RATE);
  while (millis() - last_sample_time < sample_delay) {
    // 샘플링 속도 유지를 위해 대기
    delayMicroseconds(100);
  }
  last_sample_time = millis();
}

// MPU6050 초기 설정
void setupMPU6050() {
  // 전원 관리 설정 - 슬립 모드 해제
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(PWR_MGMT_1);
  Wire.write(0);
  Wire.endTransmission(true);
  delay(100);  // 안정화 시간
  
  // 샘플링 속도 설정 - 100Hz (1000 / (1 + 9))
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(SMPLRT_DIV);
  Wire.write(9);
  Wire.endTransmission(true);
  
  // 디지털 저역통과 필터 설정
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(CONFIG);
  Wire.write(0);
  Wire.endTransmission(true);
  
  // 자이로스코프 설정 - ±250°/s 범위
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(GYRO_CONFIG);
  Wire.write(0);
  Wire.endTransmission(true);
  
  // 가속도계 설정 - ±2g 범위
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(ACCEL_CONFIG);
  Wire.write(0);
  Wire.endTransmission(true);
  
  Serial.println("MPU6050 초기화 완료");
}

// 16비트 워드 읽기 (2바이트)
int16_t readWord(int reg) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, 2, true);
  
  int16_t value = Wire.read() << 8 | Wire.read();
  return value;
}

// MPU6050 데이터 읽기
void readMPU6050Data(float* acc_x, float* acc_y, float* acc_z, 
                     float* gyr_x, float* gyr_y, float* gyr_z) {
  // 가속도계 데이터 (g 단위로 변환: ±2g 범위에서 16384 LSB/g)
  *acc_x = readWord(ACCEL_XOUT_H) / 16384.0;
  *acc_y = readWord(ACCEL_XOUT_H + 2) / 16384.0;
  *acc_z = readWord(ACCEL_XOUT_H + 4) / 16384.0;
  
  // 자이로스코프 데이터 (°/s 단위로 변환: ±250°/s 범위에서 131 LSB/°/s)
  *gyr_x = readWord(GYRO_XOUT_H) / 131.0;
  *gyr_y = readWord(GYRO_XOUT_H + 2) / 131.0;
  *gyr_z = readWord(GYRO_XOUT_H + 4) / 131.0;
}

// 오일러 각도 계산
void calculateEulerAngles(float acc_x, float acc_y, float acc_z, 
                          float* euler_x, float* euler_y, float* euler_z) {
  // 단순화된 오일러 각도 계산
  *euler_x = atan2(acc_y, sqrt(acc_x * acc_x + acc_z * acc_z)) * 180.0 / PI;
  *euler_y = atan2(-acc_x, sqrt(acc_y * acc_y + acc_z * acc_z)) * 180.0 / PI;
  *euler_z = atan2(acc_z, sqrt(acc_x * acc_x + acc_y * acc_y)) * 180.0 / PI;
}

// 데이터 버퍼에 추가
void addDataToBuffer(float acc_x, float acc_y, float acc_z, 
                     float gyr_x, float gyr_y, float gyr_z,
                     float euler_x, float euler_y, float euler_z) {
  // 현재 인덱스에 데이터 저장
  data_buffer[buffer_index][0] = acc_x;
  data_buffer[buffer_index][1] = acc_y;
  data_buffer[buffer_index][2] = acc_z;
  data_buffer[buffer_index][3] = gyr_x;
  data_buffer[buffer_index][4] = gyr_y;
  data_buffer[buffer_index][5] = gyr_z;
  data_buffer[buffer_index][6] = euler_x;
  data_buffer[buffer_index][7] = euler_y;
  data_buffer[buffer_index][8] = euler_z;
  
  // 인덱스 증가
  buffer_index = (buffer_index + 1) % SEQ_LENGTH;
  
  // 버퍼가 다 찼는지 확인
  if (buffer_index == 0) {
    buffer_filled = true;
  }
}

// 데이터 정규화
void normalizeData(float normalized_buffer[][NUM_FEATURES]) {
  // 버퍼에서 데이터 가져와 정규화
  for (int i = 0; i < SEQ_LENGTH; i++) {
    int idx = (buffer_index + i) % SEQ_LENGTH;  // 순환 버퍼에서 올바른 순서로 가져오기
    
    for (int j = 0; j < NUM_FEATURES; j++) {
      normalized_buffer[i][j] = (data_buffer[idx][j] - scaler_mean[j]) / scaler_scale[j];
    }
  }
}

// 낙상 예측
bool predictFall() {
  // 데이터가 충분하지 않으면 예측 안함
  if (!buffer_filled) {
    return false;
  }
  
  // 정규화된 데이터를 위한 버퍼
  float normalized_buffer[SEQ_LENGTH][NUM_FEATURES];
  normalizeData(normalized_buffer);
  
  // 입력 텐서 가져오기
  TfLiteTensor* input = interpreter->input(0);
  
  // 입력 텐서에 데이터 복사
  for (int i = 0; i < SEQ_LENGTH; i++) {
    for (int j = 0; j < NUM_FEATURES; j++) {
      input->data.f[(i * NUM_FEATURES) + j] = normalized_buffer[i][j];
    }
  }
  
  // 모델 실행
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("모델 실행 실패!");
    return false;
  }
  
  // 출력 텐서 가져오기
  TfLiteTensor* output = interpreter->output(0);
  
  // 클래스 1(낙상)의 확률
  float fall_prob = output->data.f[1];
  
  // 임계값을 넘으면 낙상으로 판단 (0.7은 조정 가능)
  return (fall_prob > 0.7);
}

// 알람 발생
void triggerAlarm() {
  if (!alarm_active) {
    alarm_active = true;
    alarm_start_time = millis();
    
    digitalWrite(LED_PIN, HIGH);
    
    // 부저 소리 (간단한 패턴)
    for (int i = 0; i < 3; i++) {
      digitalWrite(BUZZER_PIN, HIGH);
      delay(200);
      digitalWrite(BUZZER_PIN, LOW);
      delay(200);
    }
    
    digitalWrite(BUZZER_PIN, HIGH);  // 지속적인 경고음
    
    // 낙상 감지 로그
    Serial.println("낙상 감지! - " + getTimestamp());
  }
}

// 알람 중지
void stopAlarm() {
  alarm_active = false;
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.println("알람 중지");
}

// 버튼 인터럽트 처리
void IRAM_ATTR buttonInterrupt() {
  // 디바운싱
  static unsigned long last_interrupt_time = 0;
  unsigned long interrupt_time = millis();
  
  if (interrupt_time - last_interrupt_time > 300) {
    stopAlarm();
  }
  
  last_interrupt_time = interrupt_time;
}

// 로깅 초기화
void initLogging() {
  // SD 카드 초기화
  if (!SD.begin(SD_CS_PIN)) {
    Serial.println("SD 카드 초기화 실패!");
    return;
  }
  
  Serial.println("SD 카드 초기화 성공");
  
  // 로그 파일 생성
  String filename = "/fall_data_" + getTimestamp() + ".csv";
  logFile = SD.open(filename, FILE_WRITE);
  
  if (!logFile) {
    Serial.println("로그 파일 생성 실패!");
    return;
  }
  
  // CSV 헤더 작성
  logFile.println("TimeStamp(s),FrameCounter,AccX,AccY,AccZ,GyrX,GyrY,GyrZ,EulerX,EulerY,EulerZ");
  logFile.flush();
  
  Serial.println("로깅 시스템 준비 완료: " + filename);
}

// 센서 데이터 로깅
void logSensorData(float acc_x, float acc_y, float acc_z, 
                  float gyr_x, float gyr_y, float gyr_z,
                  float euler_x, float euler_y, float euler_z) {
  // 로그 파일이 열려있지 않으면 반환
  if (!logFile) {
    return;
  }
  
  // 10분마다 새로운 로그 파일 생성
  unsigned long current_time = millis();
  if (current_time - log_start_time > LOG_INTERVAL) {
    logFile.close();
    
    String filename = "/fall_data_" + getTimestamp() + ".csv";
    logFile = SD.open(filename, FILE_WRITE);
    
    if (logFile) {
      logFile.println("TimeStamp(s),FrameCounter,AccX,AccY,AccZ,GyrX,GyrY,GyrZ,EulerX,EulerY,EulerZ");
      log_start_time = current_time;
      Serial.println("새 로그 파일 생성: " + filename);
    }
  }
  
  // 로그 데이터 작성
  if (logFile) {
    // CSV 형식으로 데이터 기록
    logFile.print(millis() / 1000.0, 3); // 초 단위 타임스탬프
    logFile.print(",");
    logFile.print(frame_counter);
    logFile.print(",");
    logFile.print(acc_x, 6);
    logFile.print(",");
    logFile.print(acc_y, 6);
    logFile.print(",");
    logFile.print(acc_z, 6);
    logFile.print(",");
    logFile.print(gyr_x, 6);
    logFile.print(",");
    logFile.print(gyr_y, 6);
    logFile.print(",");
    logFile.print(gyr_z, 6);
    logFile.print(",");
    logFile.print(euler_x, 6);
    logFile.print(",");
    logFile.print(euler_y, 6);
    logFile.print(",");
    logFile.println(euler_z, 6);
    
    // 500개 샘플마다 파일 플러시
    if (frame_counter % 500 == 0) {
      logFile.flush();
    }
  }
}

// 시간 문자열 생성
String getTimestamp() {
  if (WIFI_ENABLED) {
    struct tm timeinfo;
    if (!getLocalTime(&timeinfo)) {
      return String(millis());
    }
    
    char buffer[24];
    strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &timeinfo);
    return String(buffer);
  } else {
    // WiFi가 없으면 밀리초로 타임스탬프 생성
    return String(millis());
  }
}

// WiFi 초기화
void initWiFi() {
  Serial.print("WiFi 연결 중...");
  WiFi.begin(ssid, password);
  
  int attempt = 0;
  while (WiFi.status() != WL_CONNECTED && attempt < 20) {
    delay(500);
    Serial.print(".");
    attempt++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi 연결 성공!");
    Serial.print("IP 주소: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi 연결 실패!");
  }
}

// 시간 동기화
void syncTime() {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }
  
  Serial.println("NTP로 시간 동기화 중...");
  configTime(9 * 3600, 0, "pool.ntp.org"); // KST (UTC+9)
  
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    Serial.println("시간 동기화 실패!");
    return;
  }
  
  Serial.print("현재 시간: ");
  Serial.print(timeinfo.tm_year + 1900);
  Serial.print('-');
  Serial.print(timeinfo.tm_mon + 1);
  Serial.print('-');
  Serial.print(timeinfo.tm_mday);
  Serial.print(' ');
  Serial.print(timeinfo.tm_hour);
  Serial.print(':');
  Serial.print(timeinfo.tm_min);
  Serial.print(':');
  Serial.println(timeinfo.tm_sec);
}

/*
 * 모델 배포 지침:
 * 
 * 1. TensorFlow Lite 모델(.tflite)을 xxd 유틸리티로 바이너리 배열로 변환:
 *    xxd -i cnn_fall_detection.tflite > model_data.h
 * 
 * 2. model_data.h 파일을 다음과 같이 수정:
 *    const unsigned char g_model_data[] = { ... };
 *    const unsigned int g_model_data_len = ...;
 * 
 * 3. 스케일러 파라미터 업데이트:
 *    - cnn_fall_detection_scaler_mean.npy에서 배열 복사하여 scaler_mean 배열 값 설정
 *    - cnn_fall_detection_scaler_scale.npy에서 배열 복사하여 scaler_scale 배열 값 설정
 * 
 * 4. ESP32 환경 설정:
 *    - PlatformIO 사용 시 platformio.ini 설정:
 *      [env:esp32dev]
 *      platform = espressif32
 *      board = esp32dev
 *      framework = arduino
 *      monitor_speed = 115200
 *      lib_deps = 
 *        tanakamasayuki/TensorFlowLite_ESP32@^0.9.0
 *        # 기타 필요한 라이브러리
 * 
 *    - Arduino IDE 사용 시:
 *      "TensorFlowLite_ESP32" 라이브러리 설치
 */
