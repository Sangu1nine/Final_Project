import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
import threading
import signal
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 하드웨어 에뮬레이션 모드 설정
HARDWARE_MODE = "SIMULATION"  # "SIMULATION" 또는 "RASPBERRY_PI"

# 실제 하드웨어가 사용 가능한 경우만 임포트
if HARDWARE_MODE == "RASPBERRY_PI":
    try:
        import RPi.GPIO as GPIO
        import smbus
        HARDWARE_AVAILABLE = True
    except ImportError:
        print("라즈베리 파이 하드웨어 라이브러리를 임포트할 수 없습니다. 시뮬레이션 모드로 실행합니다.")
        HARDWARE_MODE = "SIMULATION"
        HARDWARE_AVAILABLE = False
else:
    HARDWARE_AVAILABLE = False

# GPIO 핀 설정 (라즈베리 파이 모드에서만 사용)
LED_PIN = 17      # 낙상 경고용 LED
BUZZER_PIN = 18   # 낙상 경고용 부저
BUTTON_PIN = 27   # 알람 중지 버튼

# MPU6050 I2C 설정 (라즈베리 파이 모드에서만 사용)
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

# 모델 및 데이터 설정
MODEL_PATH = 'C:/gitproject/Final_Project/data/models/tflite_model/fall_detection_compatible.tflite'
SCALER_MEAN_PATH = 'C:/gitproject/Final_Project/data/models/tflite_model/extracted_scaler_mean.npy'
SCALER_SCALE_PATH = 'C:/gitproject/Final_Project/data/models/tflite_model/extracted_scaler_scale.npy'
TEST_DATA_PATH = 'C:/gitproject/Final_Project/data/test'  # 테스트 데이터 경로
SEQ_LENGTH = 50   # 시퀀스 길이
N_FEATURES = 9    # 특징 수 (AccX, AccY, AccZ, GyrX, GyrY, GyrZ, EulerX, EulerY, EulerZ)
SAMPLING_RATE = 100  # Hz (데이터셋의 샘플링 레이트에 맞춤)

# CSV 로깅 설정
LOG_DATA = True
LOG_DIR = 'logs'
LOG_FILE = 'imu_data.csv'
LOG_INTERVAL = 600  # 10분마다 새 로그 파일

# 시뮬레이션 데이터 경로
SIM_DATA_PATH = TEST_DATA_PATH  # 테스트 데이터를 시뮬레이션에 활용

# 시뮬레이션된 MPU6050 센서 클래스
class SimulatedIMUSensor:
    def __init__(self, data_path=SIM_DATA_PATH):
        """시뮬레이션된 IMU 센서 초기화"""
        print("시뮬레이션된 IMU 센서 초기화 중...")
        self.frame_counter = 0
        self.sim_data = self.load_simulation_data(data_path)
        self.data_index = 0
        self.sim_data_length = len(self.sim_data)
        print(f"시뮬레이션 데이터 로드 완료: {self.sim_data_length}개 샘플")
    
    def load_simulation_data(self, data_path):
        """시뮬레이션 데이터 로드"""
        # 테스트 데이터 파일 경로
        X_test_path = os.path.join(data_path, 'X_test.npy')
        y_test_path = os.path.join(data_path, 'y_test.npy')
        
        if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
            print(f"시뮬레이션 데이터를 찾을 수 없습니다. 경로: {data_path}")
            # 더미 데이터 생성
            return self.generate_dummy_data()
        
        try:
            # 테스트 데이터 로드
            X_test = np.load(X_test_path)
            y_test = np.load(y_test_path)
            
            print(f"시뮬레이션 데이터 로드 완료: X_test shape={X_test.shape}, y_test shape={y_test.shape}")
            
            # 시뮬레이션에 사용할 데이터 준비
            sim_data = []
            for i in range(len(X_test)):
                for j in range(X_test.shape[1]):
                    # X_test[i, j]를 딕셔너리로 변환
                    sample = {
                        'TimeStamp(s)': time.time() + (j * 0.01),  # 100Hz 가정
                        'FrameCounter': self.frame_counter,
                        'AccX': X_test[i, j, 0],
                        'AccY': X_test[i, j, 1],
                        'AccZ': X_test[i, j, 2],
                        'GyrX': X_test[i, j, 3],
                        'GyrY': X_test[i, j, 4],
                        'GyrZ': X_test[i, j, 5],
                        'EulerX': X_test[i, j, 6],
                        'EulerY': X_test[i, j, 7],
                        'EulerZ': X_test[i, j, 8],
                        'Label': y_test[i]  # 레이블 추가 (낙상 여부)
                    }
                    sim_data.append(sample)
                    self.frame_counter += 1
            
            return sim_data
        except Exception as e:
            print(f"시뮬레이션 데이터 로드 오류: {str(e)}")
            return self.generate_dummy_data()
    
    def generate_dummy_data(self, n_samples=10000):
        """테스트 데이터가 없을 경우 더미 데이터 생성"""
        print("더미 IMU 데이터 생성 중...")
        dummy_data = []
        
        for i in range(n_samples):
            # 가끔 낙상 이벤트 생성
            is_fall = np.random.random() < 0.05  # 5%의 낙상 확률
            
            if is_fall:
                # 낙상 이벤트 시뮬레이션 (갑작스러운 가속도 변화)
                acc_magnitude = np.random.uniform(1.5, 3.0)  # g 단위
                gyro_magnitude = np.random.uniform(100, 200)  # °/s 단위
            else:
                # 일반 움직임 시뮬레이션
                acc_magnitude = np.random.uniform(0.8, 1.2)  # g 단위
                gyro_magnitude = np.random.uniform(5, 30)  # °/s 단위
            
            # 랜덤 방향
            acc_direction = np.random.normal(0, 1, 3)
            acc_direction = acc_direction / np.linalg.norm(acc_direction)
            
            gyro_direction = np.random.normal(0, 1, 3)
            gyro_direction = gyro_direction / np.linalg.norm(gyro_direction)
            
            # 가속도 및 자이로스코프 값
            accel_x = acc_direction[0] * acc_magnitude
            accel_y = acc_direction[1] * acc_magnitude
            accel_z = acc_direction[2] * acc_magnitude
            
            gyro_x = gyro_direction[0] * gyro_magnitude
            gyro_y = gyro_direction[1] * gyro_magnitude
            gyro_z = gyro_direction[2] * gyro_magnitude
            
            # 오일러 각도 (단순화된 계산)
            euler_x = np.arctan2(accel_y, np.sqrt(accel_x**2 + accel_z**2)) * 180 / np.pi
            euler_y = np.arctan2(-accel_x, np.sqrt(accel_y**2 + accel_z**2)) * 180 / np.pi
            euler_z = np.arctan2(accel_z, np.sqrt(accel_x**2 + accel_y**2)) * 180 / np.pi
            
            # 샘플 데이터
            sample = {
                'TimeStamp(s)': time.time() + (i * 0.01),  # 100Hz 가정
                'FrameCounter': self.frame_counter,
                'AccX': accel_x,
                'AccY': accel_y,
                'AccZ': accel_z,
                'GyrX': gyro_x,
                'GyrY': gyro_y,
                'GyrZ': gyro_z,
                'EulerX': euler_x,
                'EulerY': euler_y,
                'EulerZ': euler_z,
                'Label': 1 if is_fall else 0  # 낙상 여부
            }
            
            dummy_data.append(sample)
            self.frame_counter += 1
        
        print(f"더미 데이터 생성 완료: {len(dummy_data)}개 샘플")
        return dummy_data
    
    def get_data(self):
        """다음 IMU 센서 데이터 가져오기"""
        # 시뮬레이션 데이터 순환
        if self.data_index >= self.sim_data_length:
            self.data_index = 0
            print("시뮬레이션 데이터를 처음부터 다시 사용합니다.")
        
        data = self.sim_data[self.data_index]
        self.data_index += 1
        
        # 현재 타임스탬프로 업데이트
        data['TimeStamp(s)'] = time.time()
        
        return data

# 실제 MPU6050 센서 클래스 (라즈베리 파이에서만 사용)
class MPU6050Sensor:
    def __init__(self):
        """실제 IMU 센서 (MPU6050) 초기화 및 I2C 설정"""
        if not HARDWARE_AVAILABLE:
            raise ImportError("하드웨어 라이브러리가 설치되어 있지 않습니다.")
        
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

# 데이터 로깅 클래스
class DataLogger:
    def __init__(self, log_dir=LOG_DIR, log_file=LOG_FILE, interval=LOG_INTERVAL):
        """데이터 로깅 클래스"""
        self.log_dir = log_dir
        self.log_file = log_file
        self.interval = interval  # 로그 파일 변경 간격(초)
        self.start_time = time.time()
        self.log_count = 0
        self.header_written = False
        
        # 로그 디렉터리 생성
        os.makedirs(log_dir, exist_ok=True)
        
        self.current_file = self._get_log_filename()
        
    def _get_log_filename(self):
        """타임스탬프가 포함된 로그 파일 이름 생성"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(self.log_file)
        return os.path.join(self.log_dir, f"{base}_{timestamp}{ext}")
    
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

# 출력 장치 에뮬레이션 클래스
class OutputDeviceEmulator:
    def __init__(self):
        """출력 장치 에뮬레이션 (LED, 부저 등)"""
        self.led_state = False
        self.buzzer_state = False
        self.alarm_active = False
        
        print("출력 장치 에뮬레이션 모드 활성화")
    
    def set_led(self, state):
        """LED 상태 설정"""
        self.led_state = state
        print(f"LED {'켜짐' if state else '꺼짐'}")
    
    def set_buzzer(self, state, frequency=440):
        """부저 상태 설정"""
        self.buzzer_state = state
        if state:
            print(f"부저 켜짐 (주파수: {frequency}Hz)")
        else:
            print("부저 꺼짐")
    
    def cleanup(self):
        """리소스 정리"""
        self.set_led(False)
        self.set_buzzer(False)
        print("출력 장치 정리 완료")

# 실제 출력 장치 제어 클래스 (라즈베리 파이에서만 사용)
class HardwareOutputDevices:
    def __init__(self, led_pin=LED_PIN, buzzer_pin=BUZZER_PIN, button_pin=BUTTON_PIN):
        """실제 하드웨어 출력 장치 초기화"""
        if not HARDWARE_AVAILABLE:
            raise ImportError("하드웨어 라이브러리가 설치되어 있지 않습니다.")
        
        # GPIO 설정
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(led_pin, GPIO.OUT)
        GPIO.setup(buzzer_pin, GPIO.OUT)
        GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        self.led_pin = led_pin
        self.buzzer_pin = buzzer_pin
        self.button_pin = button_pin
        
        # 부저 PWM 설정
        self.buzzer = GPIO.PWM(buzzer_pin, 440)  # 440Hz (A4 음)
        self.alarm_active = False
        
        print("하드웨어 출력 장치 초기화 완료")
    
    def set_led(self, state):
        """LED 상태 설정"""
        GPIO.output(self.led_pin, GPIO.HIGH if state else GPIO.LOW)
    
    def set_buzzer(self, state, frequency=440):
        """부저 상태 설정"""
        if state and not self.alarm_active:
            self.buzzer.ChangeFrequency(frequency)
            self.buzzer.start(50)  # 50% 듀티 사이클
            self.alarm_active = True
        elif not state and self.alarm_active:
            self.buzzer.stop()
            self.alarm_active = False
    
    def cleanup(self):
        """GPIO 리소스 정리"""
        self.set_led(False)
        self.set_buzzer(False)
        GPIO.cleanup()

# 낙상 감지기 클래스
class FallDetector:
    def __init__(self, model_path, scaler_mean_path, scaler_scale_path, seq_length=50, n_features=9):
        """낙상 감지 모델 초기화"""
        self.seq_length = seq_length
        self.n_features = n_features
        self.data_buffer = deque(maxlen=seq_length)
        
        # 하드웨어 또는 에뮬레이션 모드에 따라 출력 장치 초기화
        if HARDWARE_MODE == "RASPBERRY_PI" and HARDWARE_AVAILABLE:
            try:
                self.output_devices = HardwareOutputDevices()
                print("실제 하드웨어 출력 장치 사용")
            except Exception as e:
                print(f"하드웨어 출력 장치 초기화 실패: {e}")
                self.output_devices = OutputDeviceEmulator()
        else:
            self.output_devices = OutputDeviceEmulator()
        
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
        
        # 데이터 로거 설정
        self.logger = DataLogger() if LOG_DATA else None
        
        # 통계 정보
        self.detection_stats = {
            'true_detections': 0,
            'false_alarms': 0,
            'total_samples': 0
        }
    
    def load_model(self, model_path):
        """TFLite 모델 로드"""
        try:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {str(e)}")
            raise
    
    def add_data_point(self, data_point):
        """데이터 버퍼에 새 데이터 포인트 추가"""
        # 센서 데이터를 배열로 변환
        data_array = np.array([
            data_point['AccX'], data_point['AccY'], data_point['AccZ'],
            data_point['GyrX'], data_point['GyrY'], data_point['GyrZ'],
            data_point['EulerX'], data_point['EulerY'], data_point['EulerZ']
        ])
        
        self.data_buffer.append(data_array)
        
        # 통계 카운터 증가
        self.detection_stats['total_samples'] += 1
        
        # 데이터 로깅 (필요한 경우)
        if LOG_DATA and self.logger:
            self.logger.log_data(data_point)
    
    def normalize_data(self, data):
        """데이터 정규화"""
        try:
            # (seq_length, n_features) -> (seq_length * n_features)
            data_flat = data.reshape(-1, self.n_features)
            
            # 정규화 적용
            data_norm = (data_flat - self.scaler_mean) / self.scaler_scale
            
            # 원래 형태로 복원
            return data_norm.reshape(1, self.seq_length, self.n_features)
        except Exception as e:
            print(f"정규화 중 오류: {str(e)}")
            print(f"데이터 형태: {data.shape}")
            print(f"스케일러 평균 형태: {self.scaler_mean.shape}")
            print(f"스케일러 스케일 형태: {self.scaler_scale.shape}")
            raise
    
    def predict(self):
        """낙상 예측 수행"""
        try:
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
            
            # 출력 형태에 따라 다르게 처리
            if output_data.shape[1] == 1:  # 시그모이드 출력 (0~1 사이 하나의 값)
                fall_prob = float(output_data[0][0])
                prediction = 1 if fall_prob >= 0.5 else 0
            else:  # 소프트맥스 출력 (클래스별 확률)
                fall_prob = float(output_data[0][1]) if output_data.shape[1] > 1 else float(output_data[0][0])
                prediction = np.argmax(output_data[0])
            
            return {
                'prediction': int(prediction),
                'fall_probability': float(fall_prob)
            }
        except Exception as e:
            print(f"예측 중 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def trigger_alarm(self, true_label=None):
        """낙상 감지 시 알람 발생"""
        if not self.alarm_active:
            self.alarm_active = True
            self.output_devices.set_led(True)
            self.output_devices.set_buzzer(True)
            print("낙상 감지! 알람 발생")
            
            # 낙상 감지 정확도 통계 업데이트
            if true_label is not None:
                if true_label == 1:
                    self.detection_stats['true_detections'] += 1
                else:
                    self.detection_stats['false_alarms'] += 1
    
    def stop_alarm(self):
        """알람 중지"""
        if self.alarm_active:
            self.alarm_active = False
            self.output_devices.set_led(False)
            self.output_devices.set_buzzer(False)
            print("알람 중지")
    
    def get_stats(self):
        """감지 통계 정보 반환"""
        stats = self.detection_stats.copy()
        
        if stats['total_samples'] > 0:
            stats['detection_rate'] = stats['true_detections'] / stats['total_samples']
            stats['false_alarm_rate'] = stats['false_alarms'] / stats['total_samples']
        else:
            stats['detection_rate'] = 0
            stats['false_alarm_rate'] = 0
            
        return stats
    
    def cleanup(self):
        """리소스 정리"""
        self.stop_alarm()
        self.output_devices.cleanup()
        print("낙상 감지기 정리 완료")

def main():
    """메인 함수"""
    print("낙상 감지 시스템 시작")
    
    try:
        # 하드웨어 모드에 따라 센서 초기화
        if HARDWARE_MODE == "RASPBERRY_PI" and HARDWARE_AVAILABLE:
            try:
                sensor = MPU6050Sensor()
                print("실제 MPU6050 센서 사용")
            except Exception as e:
                print(f"MPU6050 센서 초기화 실패: {e}")
                print("시뮬레이션 모드로 전환합니다.")
                sensor = SimulatedIMUSensor()
        else:
            # 시뮬레이션 모드
            sensor = SimulatedIMUSensor()
        
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
            
            # 통계 출력
            stats = detector.get_stats()
            print("\n=== 감지 통계 ===")
            print(f"총 샘플 수: {stats['total_samples']}")
            print(f"정확한 감지: {stats['true_detections']}")
            print(f"오탐지: {stats['false_alarms']}")
            if stats['total_samples'] > 0:
                print(f"감지율: {stats['detection_rate']:.2%}")
                print(f"오탐지율: {stats['false_alarm_rate']:.2%}")
            
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # 낙상 감지 루프
        print("센서 데이터 수집 중...")
        
        # 초기 데이터 버퍼 채우기
        print(f"초기 데이터 버퍼 채우는 중 ({SEQ_LENGTH} 샘플)...")
        for _ in range(SEQ_LENGTH):
            sensor_data = sensor.get_data()
            detector.add_data_point(sensor_data)
            time.sleep(1.0 / SAMPLING_RATE)  # 100Hz 샘플링
        
        print("실시간 낙상 감지 시작")
        
        # 메인 감지 루프
        last_time = time.time()
        alarm_duration = 0  # 알람 지속 시간 추적
        
        while True:
            # 센서 데이터 읽기
            sensor_data = sensor.get_data()
            
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
                
                # 실제 레이블이 있는 경우 (시뮬레이션 모드)
                true_label = sensor_data.get('Label', None)
                detector.trigger_alarm(true_label)
                
                # 알람 시작 시간 기록
                if not detector.alarm_active:
                    alarm_start_time = time.time()
            
            # 알람이 활성화된 경우, 3초 후 자동으로 끄기
            if detector.alarm_active:
                if current_time - alarm_start_time >= 3.0:
                    detector.stop_alarm()
            
            # 샘플링 속도 유지
            time.sleep(1.0 / SAMPLING_RATE)
            
    except KeyboardInterrupt:
        print("\n프로그램 종료")
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'detector' in locals():
            detector.cleanup()


if __name__ == "__main__":
    main()
