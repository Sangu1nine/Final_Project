import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from google.colab import drive

# Google Drive 마운트
drive.mount('/content/drive')

# 데이터셋 경로 설정
dataset_path = '/content/drive/MyDrive/KFall_dataset'
sensor_data_path = os.path.join(dataset_path, 'sensor_data_new')
label_data_path = os.path.join(dataset_path, 'label_data_new')

class FallDetectionDataset:
    def __init__(self, sensor_path, label_path, sampling_rate=100, window_size=50, stride=10):
        self.sensor_path = sensor_path
        self.label_path = label_path
        self.sampling_rate = sampling_rate
        self.window_size = window_size  # 0.5초 (50프레임)
        self.stride = stride  # 슬라이딩 윈도우 스트라이드
        self.data = []
        self.labels = []
        
        # 제외할 subject ID (SA01~SA05, SA34)
        self.excluded_subjects = ['SA01', 'SA02', 'SA03', 'SA04', 'SA05', 'SA34']
        
        # 로우패스 필터 설정 (Butterworth 5Hz)
        self.cutoff_freq = 5.0  # Hz
        self.nyquist_freq = sampling_rate / 2
        self.filter_order = 4
        self.b, self.a = butter(self.filter_order, self.cutoff_freq / self.nyquist_freq, btype='low')
    
    def apply_lowpass_filter(self, data):
        """5Hz Butterworth 로우패스 필터 적용"""
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = filtfilt(self.b, self.a, data[:, i])
        return filtered_data
    
    def load_data(self):
        """센서 데이터와 라벨 데이터를 로드하고 전처리합니다."""
        print("데이터 로딩 시작...")
        
        # 모든 라벨 파일들 불러오기
        label_files = glob(os.path.join(self.label_path, '*.xlsx'))
        
        for label_file in label_files:
            subject_id = os.path.basename(label_file).split('_')[0]  # SA##
            
            # 제외할 subject 체크
            if subject_id in self.excluded_subjects:
                print(f"Skipping {subject_id} (excluded)")
                continue
                
            print(f"\nProcessing {subject_id}...")
            
            # 라벨 데이터 로드
            labels_df = pd.read_excel(label_file)
            
            for idx, row in labels_df.iterrows():
                task_code = row['Task Code (Task ID)']
                trial_id = int(row['Trial ID'])
                
                # NaN 체크
                if pd.isna(task_code):
                    continue
                
                # task_code를 문자열로 변환하고 정리
                task_code_str = str(task_code).strip()
                
                # 'F01 (20)' 형식의 경우 'F01'만 추출
                if ' ' in task_code_str:
                    task_code_str = task_code_str.split(' ')[0]
                
                # 낙상 데이터인지 확인 (F로 시작하는 task)
                is_fall = task_code_str.startswith('F')
                
                # Task code를 파일명 형식에 맞게 변환
                if task_code_str.isdigit():
                    # 이미 숫자형인 경우 (01, 02, ...)
                    task_num = f"T{int(task_code_str):02d}"
                elif task_code_str.startswith('F'):
                    # F01 -> T20, F02 -> T21, ..., F15 -> T34
                    fall_num = int(task_code_str[1:])
                    task_num = f"T{fall_num + 19:02d}"
                elif task_code_str.startswith('D'):
                    if task_code_str in ['D20', 'D21']:
                        # D20 -> T35, D21 -> T36
                        d_num = int(task_code_str[1:])
                        task_num = f"T{d_num + 15:02d}"
                    else:
                        # D01 -> T01, D02 -> T02, ..., D19 -> T19
                        task_num = f"T{int(task_code_str[1:]):02d}"
                else:
                    continue
                
                # 센서 데이터 파일 경로 생성
                subject_num = subject_id[2:]  # SA06 -> 06
                
                # trial_id가 5 이하인 경우만 처리하고, 파일이 존재하는지 확인
                found_data = False
                for test_trial_id in range(1, 6):  # 1부터 5까지 시도
                    sensor_filename = f"S{subject_num}{task_num}R{test_trial_id:02d}.csv"
                    sensor_filepath = os.path.join(self.sensor_path, subject_id, sensor_filename)
                    
                    # 파일이 존재하면 데이터 처리
                    if os.path.exists(sensor_filepath):
                        print(f"파일 발견: {sensor_filename}")
                        found_data = True
                        sensor_data = pd.read_csv(sensor_filepath)
                        
                        # 로우패스 필터 적용
                        features = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'EulerX', 'EulerY', 'EulerZ']
                        filtered_data = sensor_data.copy()
                        filtered_data[features] = self.apply_lowpass_filter(sensor_data[features].values)
                        
                        if is_fall:
                            # 낙상 데이터 처리
                            fall_onset_frame = int(row['Fall_onset_frame'])
                            fall_impact_frame = int(row['Fall_impact_frame'])
                            
                            if pd.notna(fall_onset_frame) and pd.notna(fall_impact_frame):
                                self._process_fall_data(filtered_data, fall_onset_frame, fall_impact_frame)
                        else:
                            # ADL(일상 활동) 데이터 처리
                            self._process_adl_data(filtered_data)
                        
                        # 파일 하나만 처리하고 종료
                        break
                
                if not found_data:
                    print(f"파일을 찾을 수 없음: {subject_id}/{task_num}R##.csv")
        
        # 리스트를 numpy 배열로 변환
        if len(self.data) > 0:
            self.data = np.array(self.data)
            self.labels = np.array(self.labels)
            
            print(f"총 {len(self.data)}개의 샘플 로딩 완료")
            print(f"낙상 샘플: {np.sum(self.labels)}, ADL 샘플: {len(self.labels) - np.sum(self.labels)}")
        else:
            print("로드된 데이터가 없습니다!")
            self.data = np.array([])
            self.labels = np.array([])
        
        return self.data, self.labels
    
    def _process_fall_data(self, sensor_data, fall_onset_frame, fall_impact_frame):
        """낙상 데이터 처리 - 사전 감지를 위해 낙상 시작 전과 낙상 구간을 윈도우로 분할"""
        # 낙상 전 데이터 (Fall_onset_frame 이전)
        pre_fall_data = sensor_data[sensor_data['FrameCounter'] < fall_onset_frame]
        self._extract_windows(pre_fall_data, label=0)  # Non-fall로 레이블링
        
        # 낙상 데이터 (Fall_onset_frame부터 Fall_impact_frame까지)
        fall_data = sensor_data[(sensor_data['FrameCounter'] >= fall_onset_frame) & 
                               (sensor_data['FrameCounter'] <= fall_impact_frame)]
        self._extract_windows(fall_data, label=1)  # Fall로 레이블링
    
    def _process_adl_data(self, sensor_data):
        """ADL 데이터 처리 - 전체를 윈도우로 분할"""
        self._extract_windows(sensor_data, label=0)  # Non-fall로 레이블링
    
    def _extract_windows(self, data, label):
        """데이터를 윈도우로 분할하여 추출"""
        features = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'EulerX', 'EulerY', 'EulerZ']
        data_values = data[features].values
        
        for i in range(0, len(data_values) - self.window_size + 1, self.stride):
            window = data_values[i:i + self.window_size]
            if len(window) == self.window_size:
                self.data.append(window)
                self.labels.append(label)

def preprocess_data():
    """데이터를 전처리하고 저장하는 메인 함수"""
    # 데이터셋 준비
    print("데이터셋 준비 중...")
    dataset = FallDetectionDataset(sensor_data_path, label_data_path)
    X, y = dataset.load_data()

    # 데이터 전처리 및 저장
    print("\n데이터 전처리 중...")
    if len(X) > 0:
        # Train/Validation/Test 분할
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)
        
        # 스케일링
        scaler = StandardScaler()
        n_samples, n_timesteps, n_features = X_train.shape
        
        X_train_scaled = np.zeros_like(X_train)
        X_val_scaled = np.zeros_like(X_val)
        X_test_scaled = np.zeros_like(X_test)
        
        for i in range(n_features):
            X_train_scaled[:,:,i] = scaler.fit_transform(X_train[:,:,i])
            X_val_scaled[:,:,i] = scaler.transform(X_val[:,:,i])
            X_test_scaled[:,:,i] = scaler.transform(X_test[:,:,i])
        
        # 클래스 가중치 계산
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))
        
        print(f"Train shape: {X_train_scaled.shape}, Train labels: {y_train.shape}")
        print(f"Validation shape: {X_val_scaled.shape}, Validation labels: {y_val.shape}")
        print(f"Test shape: {X_test_scaled.shape}, Test labels: {y_test.shape}")
        print(f"Class weights: {class_weight_dict}")
        
        # 전처리된 데이터 저장
        np.save(os.path.join(dataset_path, 'X_train.npy'), X_train_scaled)
        np.save(os.path.join(dataset_path, 'y_train.npy'), y_train)
        np.save(os.path.join(dataset_path, 'X_test.npy'), X_test_scaled)
        np.save(os.path.join(dataset_path, 'y_test.npy'), y_test)
        np.save(os.path.join(dataset_path, 'X_val.npy'), X_val_scaled)
        np.save(os.path.join(dataset_path, 'y_val.npy'), y_val)
        np.save(os.path.join(dataset_path, 'class_weights.npy'), class_weight_dict)
        
        print("\n전처리된 데이터가 저장되었습니다.")
    else:
        print("데이터가 없어 전처리를 진행할 수 없습니다.")

if __name__ == "__main__":
    preprocess_data()