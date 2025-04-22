import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
import RPi.GPIO as GPIO
import smbus
import threading
import signal
import sys

# GPIO 핀 설정
LED_PIN = 17      # 낙상 경고용 LED
BUZZER_PIN = 18   # 낙상 경고용 부저
BUTTON_PIN = 27   # 알람 중지 버튼

# MPU6050 I2C 설정
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

# 모델 및 데이터 설정
MODEL_PATH = 'cnn_fall_detection.tflite'
SCALER_MEAN_PATH = 'cnn_fall_detection_scaler_mean.npy'
SCALER_SCALE_PATH = 'cnn_fall_detection_scaler_scale.npy'
SEQ_LENGTH = 100  # 시퀀스 길이
N_FEATURES = 9    # 특징 수 (AccX, AccY, AccZ, GyrX, GyrY, GyrZ, EulerX, EulerY, EulerZ)
SAMPLING_RATE = 100  # Hz (데이터셋의 샘플링 레이트에 맞춤)

# CSV 로깅 설정
LOG_DATA = True
LOG_FILE = 'imu_data.csv'
LOG_INTERVAL = 600  # 10분마다 새 로그 파일

class IMUSensor:
    def __init__(self):
        """IMU 센서 (MPU6050) 초기화 및 I2C 설정"""
        self.bus = smbus.SMBus(1)  # I2C 버스 1 사용
        self.setup_mpu6050()
        self.frame_counter = 0
        
    def setup_mpu6050(self):
        """MPU6050 센서 초기 설정"""
        # 전원 관리 설정 - 슬립 모드 해제
        self.bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)  # 안정화 시간
        
        # 샘플링 속도 설정 - 100Hz (1000 / (1 + 9))
        self.bus.write_byte_data(MPU6050_ADDR, SMPLRT_DIV, 9)
        
        # 디지털 저역통과 필터 설정
        self.bus.write_byte_data(MPU6050_ADDR, CONFIG, 0)
        
        # 자이로스코프 설정 - ±250°/s 범위
        self.bus.write_byte_data(MPU6050_ADDR, GYRO_CONFIG, 0)
        
        # 가속도계 설정 - ±2g 범위
        self.bus.write_byte_data(MPU6050_ADDR, ACCEL_CONFIG, 0)
    
    def read_word(self, reg):
        """16비트 워드 읽기 (2바이트)"""
        high = self.bus.read_byte_data(MPU6050_ADDR, reg)
        low = self.bus.read_byte_data(MPU6050_ADDR, reg + 1)
        value = (high << 8) + low
        return value
    
    def read_word_2c(self, reg):
        """2의 보수 값으로 변환"""
        val = self.read_word(reg)
        if val >= 0x8000:
            return -((65535 - val) + 1)
        else:
            return val
    
    def get_data(self):
        """IMU 센서 데이터 읽기"""
        # 가속도계 데이터 (g 단위로 변환: ±2g 범위에서 16384 LSB/g)
        accel_x = self.read_word_2c(ACCEL_XOUT_H) / 16384.0
        accel_y = self.read_word_2c(ACCEL_XOUT_H + 2) / 16384.0
        accel_z = self.read_word_2c(ACCEL_XOUT_H + 4) / 16384.0
        
        # 자이로스코프 데이터 (°/s 단위로 변환: ±250°/s 범위에서 131 LSB/°/s)
        gyro_x = self.read_word_2c(GYRO_XOUT_H) / 131.0
        gyro_y = self.read_word_2c(GYRO_XOUT_H + 2) / 131.0
        gyro_z = self.read_word_2c(GYRO_XOUT_H + 4) / 131.0
        
        # 오일러 각도 계산 (단순화된 방법)
        accel_xangle = np.arctan2(accel_y, np.sqrt(accel_x**2 + accel_z**2)) * 180 / np.pi
        accel_yangle = np.arctan2(-accel_x, np.sqrt(accel_y**2 + accel_z**2)) * 180 / np.pi
        accel_zangle = np.arctan2(accel_z, np.sqrt(accel_x**2 + accel_y**2)) * 180 / np.pi
        
        # 프레임 카운터 증가
        self.frame_counter += 1
        
        # 원본 데이터셋 구조와 일치하는 데이터 반환
        return {
            'TimeStamp(s)': time.time(),
            'FrameCounter': self.frame_counter,
            'AccX': accel_x,
            'AccY': accel_y,
            'AccZ': accel_z,
            'GyrX': gyro_x,
            'GyrY': gyro_y,
            'GyrZ': gyro_z,
            'EulerX': accel_xangle,
            'EulerY': accel_yangle,
            'EulerZ': accel_zangle
        }

class DataLogger:
    def __init__(self, log_file=LOG_FILE, interval=LOG_INTERVAL):
        """데이터 로깅 클래스"""
        self.log_file = log_file
        self.interval = interval  # 로그 파일 변경 간격(초)
        self.start_time = time.time()
        self.log_count = 0
        self.header_written = False
        self.current_file = self._get_log_filename()
        
    def _get_log_filename(self):
        """타임스탬프가 포함된 로그 파일 이름 생성"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(self.log_file)
        return f"{base}_{timestamp}{ext}"
    
    def log_data(self, data):
        """센서 데이터를 CSV 파일에 기록"""
        # 일정 시간마다 새 로그 파일 생성
        current_time = time.time()
        if current_time - self.start_time > self.interval:
            self.start_time = current_time
            self.current_file = self._get_log_filename()
            self.header_written = False
        
        # 데이터를 DataFrame으로 변환하여 저장
        df = pd.DataFrame([data])
        
        # 파일이 존재하지 않거나 헤더가 아직 쓰여지지 않은 경우
        if not os.path.exists(self.current_file) or not self.header_written:
            df.to_csv(self.current_file, mode='w', index=False)
            self.header_written = True
        else:
            df.to_csv(self.current_file, mode='a', header=False, index=False)
        
        self.log_count += 1
        if self.log_count % 1000 == 0:
            print(f"로그 데이터 {self.log_count}개 저장됨")


class FallDetector:
    def __init__(self, model_path, scaler_mean_path, scaler_scale_path, seq_length=100, n_features=9):
        """낙상 감지 모델 초기화"""
        self.seq_length = seq_length
        self.n_features = n_features
        self.data_buffer = deque(maxlen=seq_length)
        
        # GPIO 설정
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(LED_PIN, GPIO.OUT)
        GPIO.setup(BUZZER_PIN, GPIO.OUT)
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # 부저 PWM 설정
        self.buzzer = GPIO.PWM(BUZZER_PIN, 440)  # 440Hz (A4 음)
        
        # 알람 상태
        self.alarm_active = False
        
        # TFLite 모델 로드
        self.interpreter = self.load_model(model_path)
        
        # 스케일러 파라미터 로드
        self.scaler_mean = np.load(scaler_mean_path)
        self.scaler_scale = np.load(scaler_scale_path)
        
        # 입력/출력 텐서 설정
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("모델 로드 완료")
        print(f"입력 형태: {self.input_details[0]['shape']}")
        print(f"출력 형태: {self.output_details[0]['shape']}")
        
        # 버튼 인터럽트 설정
        GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, 
                             callback=self.stop_alarm, bouncetime=300)
        
        # 데이터 로거 설정
        self.logger = DataLogger() if LOG_DATA else None
    
    def load_model(self, model_path):
        """TFLite 모델 로드"""
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    
    def add_data_point(self, data_point):
        """데이터 버퍼에 새 데이터 포인트 추가"""
        # 센서 데이터를 배열로 변환
        data_array = [
            data_point['AccX'], data_point['AccY'], data_point['AccZ'],
            data_point['GyrX'], data_point['GyrY'], data_point['GyrZ'],
            data_point['EulerX'], data_point['EulerY'], data_point['EulerZ']
        ]
        self.data_buffer.append(data_array)
        
        # 데이터 로깅 (필요한 경우)
        if LOG_DATA and self.logger:
            self.logger.log_data(data_point)
    
    def normalize_data(self, data):
        """데이터 정규화"""
        # (seq_length, n_features) -> (seq_length * n_features)
        data_flat = data.reshape(-1, self.n_features)
        
        # 정규화 적용
        data_norm = (data_flat - self.scaler_mean) / self.scaler_scale
        
        # 원래 형태로 복원
        return data_norm.reshape(1, self.seq_length, self.n_features)
    
    def predict(self):
        """낙상 예측 수행"""
        if len(self.data_buffer) < self.seq_length:
            return None  # 충분한 데이터가 없음
        
        # 버퍼에서 데이터 추출 및 배열로 변환
        data = np.array(list(self.data_buffer))
        
        # 데이터 정규화
        data_norm = self.normalize_data(data)
        
        # 모델 입력 설정
        input_data = data_norm.astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # 추론 실행
        self.interpreter.invoke()
        
        # 결과 가져오기
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # 낙상 확률 (클래스 1의 확률)
        fall_prob = output_data[0][1]
        
        # 예측 결과 (0: 정상, 1: 낙상)
        prediction = np.argmax(output_data[0])
        
        return {
            'prediction': int(prediction),
            'fall_probability': float(fall_prob)
        }
    
    def trigger_alarm(self):
        """낙상 감지 시 알람 발생"""
        if not self.alarm_active:
            self.alarm_active = True
            GPIO.output(LED_PIN, GPIO.HIGH)
            self.buzzer.start(50)  # 50% 듀티 사이클
            print("낙상 감지! 알람 발생")
    
    def stop_alarm(self, channel=None):
        """알람 중지"""
        if self.alarm_active:
            self.alarm_active = False
            GPIO.output(LED_PIN, GPIO.LOW)
            self.buzzer.stop()
            print("알람 중지")
    
    def cleanup(self):
        """리소스 정리"""
        self.stop_alarm()
        GPIO.cleanup()


def main():
    """메인 함수"""
    print("낙상 감지 시스템 시작")
    
    try:
        # IMU 센서 초기화
        imu_sensor = IMUSensor()
        
        # 낙상 감지기 초기화
        detector = FallDetector(
            model_path=MODEL_PATH, 
            scaler_mean_path=SCALER_MEAN_PATH,
            scaler_scale_path=SCALER_SCALE_PATH,
            seq_length=SEQ_LENGTH,
            n_features=N_FEATURES
        )
        
        # Ctrl+C 시그널 핸들러
        def signal_handler(sig, frame):
            print("\n프로그램 종료")
            detector.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # 낙상 감지 루프
        print("센서 데이터 수집 중...")
        
        # 초기 데이터 버퍼 채우기
        print(f"초기 데이터 버퍼 채우는 중 ({SEQ_LENGTH} 샘플)...")
        for _ in range(SEQ_LENGTH):
            sensor_data = imu_sensor.get_data()
            detector.add_data_point(sensor_data)
            time.sleep(1.0 / SAMPLING_RATE)  # 100Hz 샘플링
        
        print("실시간 낙상 감지 시작")
        
        # 메인 감지 루프
        last_time = time.time()
        while True:
            # 센서 데이터 읽기
            sensor_data = imu_sensor.get_data()
            
            # 디버그 출력 (1초마다)
            current_time = time.time()
            if current_time - last_time >= 1.0:
                print(f"현재 가속도: X={sensor_data['AccX']:.2f}, Y={sensor_data['AccY']:.2f}, Z={sensor_data['AccZ']:.2f}")
                last_time = current_time
            
            # 데이터 버퍼에 추가
            detector.add_data_point(sensor_data)
            
            # 낙상 예측
            result = detector.predict()
            
            # 예측 결과가 있고 낙상으로 예측된 경우
            if result and result['prediction'] == 1:
                print(f"낙상 감지! 확률: {result['fall_probability']:.2%}")
                detector.trigger_alarm()
            
            # 샘플링 속도 유지
            time.sleep(1.0 / SAMPLING_RATE)
            
    except KeyboardInterrupt:
        print("\n프로그램 종료")
    except Exception as e:
        print(f"오류 발생: {str(e)}")
    finally:
        detector.cleanup()


if __name__ == "__main__":
    main()
