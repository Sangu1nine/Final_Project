import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os
from glob import glob
import seaborn as sns
from google.colab import drive
from scipy.signal import butter, filtfilt

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
                    # D01 -> T01, D02 -> T02, ..., D21 -> T21
                    task_num = f"T{int(task_code_str[1:]):02d}"
                else:
                    continue
                
                # 센서 데이터 파일 경로 생성
                subject_num = subject_id[2:]  # SA06 -> 06
                sensor_filename = f"S{subject_num}{task_num}R{trial_id:02d}.csv"
                sensor_filepath = os.path.join(self.sensor_path, subject_id, sensor_filename)
                
                # 파일이 존재하면 데이터 처리
                if os.path.exists(sensor_filepath):
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

# 데이터셋 준비
print("데이터셋 준비 중...")
dataset = FallDetectionDataset(sensor_data_path, label_data_path)
X, y = dataset.load_data()

# 데이터 전처리
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
    
    # LSTM 모델 구축 (TFLite 호환성을 위해 명시적으로 설정)
    def create_lstm_model(input_shape):
        # GPU 환경에서도 CuDNN이 아닌 기본 LSTM 구현 사용
        import os
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        
        model = Sequential([
            # implementation=1은 non-CuDNN 버전 강제
            LSTM(64, return_sequences=True, 
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 implementation=1,
                 use_bias=True,
                 unroll=False,
                 input_shape=input_shape),
            Dropout(0.3),
            LSTM(32,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 implementation=1,
                 use_bias=True,
                 unroll=False),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    # 모델 생성
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    model = create_lstm_model(input_shape)
    model.summary()
    
    # 콜백 설정
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('/content/drive/MyDrive/KFall_dataset/best_model.h5', 
                                     monitor='val_loss', 
                                     save_best_only=True)
    
    # 모델 학습
    print("\n모델 학습 시작...")
    history = model.fit(X_train_scaled, y_train,
                       validation_data=(X_val_scaled, y_val),
                       epochs=100,
                       batch_size=32,
                       class_weight=class_weight_dict,
                       callbacks=[early_stopping, model_checkpoint],
                       verbose=1)
    
    # 학습 곡선 시각화
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/KFall_dataset/training_history.png')
    plt.show()
    
    # 모델 평가
    print("\n모델 평가 중...")
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # 예측 및 성능 메트릭 계산
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Non-Fall', 'Fall'],
               yticklabels=['Non-Fall', 'Fall'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('/content/drive/MyDrive/KFall_dataset/confusion_matrix.png')
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Fall', 'Fall']))
    
    # 리드 타임 계산
    fall_indices = np.where(y_test == 1)[0]
    lead_times = []
    
    for idx in fall_indices:
        prob_sequence = model.predict(X_test_scaled[idx:idx+1])[0][0]
        if prob_sequence > 0.5:
            lead_time = dataset.window_size * (1/dataset.sampling_rate)
            lead_times.append(lead_time)
    
    if lead_times:
        avg_lead_time = np.mean(lead_times) * 1000  # ms로 변환
        print(f"\n평균 리드 타임: {avg_lead_time:.2f} ms")
    
    # TensorFlow Lite 변환 (CuDNN 문제 해결)
    print("\nTensorFlow Lite 모델로 변환 중...")
    
    # 어제 성공한 방식대로 직접 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 동적 범위 양자화 (모델 크기 감소)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # TFLite 모델로 변환
    tflite_converted = converter.convert()
    
    # TFLite 모델 저장
    tflite_path = '/content/drive/MyDrive/KFall_dataset/fall_detection_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_converted)
    
    print(f"TFLite 모델 저장 완료: {tflite_path}")
    
    # 모델과 학습 데이터 저장
    model.save('/content/drive/MyDrive/KFall_dataset/lstm_model.h5')
    np.save('/content/drive/MyDrive/KFall_dataset/X_train.npy', X_train)
    np.save('/content/drive/MyDrive/KFall_dataset/y_train.npy', y_train)
    np.save('/content/drive/MyDrive/KFall_dataset/X_test.npy', X_test)
    np.save('/content/drive/MyDrive/KFall_dataset/y_test.npy', y_test)
    np.save('/content/drive/MyDrive/KFall_dataset/X_val.npy', X_val)
    np.save('/content/drive/MyDrive/KFall_dataset/y_val.npy', y_val)
    
    print("\n모든 파일이 저장되었습니다.")
else:
    print("데이터가 없어 학습을 진행할 수 없습니다.")
