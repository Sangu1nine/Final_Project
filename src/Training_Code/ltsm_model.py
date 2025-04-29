import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os
import seaborn as sns
from google.colab import drive

# 전처리된 데이터 로딩 함수
def load_preprocessed_data(data_path='/content/drive/MyDrive/KFall_dataset'):
    """전처리된 데이터를 로드합니다."""
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    X_val = np.load(os.path.join(data_path, 'X_val.npy'))
    y_val = np.load(os.path.join(data_path, 'y_val.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))
    
    # 클래스 가중치 딕셔너리도 로드
    class_weight_dict = np.load(os.path.join(data_path, 'class_weights.npy'), allow_pickle=True).item()
    
    return X_train, y_train, X_val, y_val, X_test, y_test, class_weight_dict

# LSTM 모델 구축
def create_lstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train_model(X_train, y_train, X_val, y_val, class_weight_dict, output_path='/content/drive/MyDrive/KFall_dataset'):
    """모델을 학습하고 시각화합니다."""
    # 모델 생성
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape)
    model.summary()
    
    # 콜백 설정
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(os.path.join(output_path, 'best_model.keras'), 
                                     monitor='val_loss', 
                                     save_best_only=True)
    
    # 모델 학습
    print("\n모델 학습 시작...")
    history = model.fit(X_train, y_train,
                       validation_data=(X_val, y_val),
                       epochs=50,
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
    plt.savefig(os.path.join(output_path, 'training_history.png'))
    plt.show()
    
    return model, history

def evaluate_model(model, X_test, y_test, dataset_window_size=50, sampling_rate=100, output_path='/content/drive/MyDrive/KFall_dataset'):
    """모델을 평가하고 성능 메트릭을 출력합니다."""
    # 모델 평가
    print("\n모델 평가 중...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # 예측 및 성능 메트릭 계산
    y_pred_prob = model.predict(X_test)
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
    plt.savefig(os.path.join(output_path, 'confusion_matrix.png'))
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Fall', 'Fall']))
    
    # 리드 타임 계산
    fall_indices = np.where(y_test == 1)[0]
    lead_times = []
    
    for idx in fall_indices:
        prob_sequence = model.predict(X_test[idx:idx+1])[0][0]
        if prob_sequence > 0.5:
            lead_time = dataset_window_size * (1/sampling_rate)
            lead_times.append(lead_time)
    
    if lead_times:
        avg_lead_time = np.mean(lead_times) * 1000  # ms로 변환
        print(f"\n평균 리드 타임: {avg_lead_time:.2f} ms")
    
    return y_pred, y_pred_prob

def save_model(model, output_path='/content/drive/MyDrive/KFall_dataset'):
    """모델을 저장합니다."""
    model.save(os.path.join(output_path, 'lstm_model.keras'))
    print(f"\n모델이 {os.path.join(output_path, 'lstm_model.keras')}에 저장되었습니다.")

# 메인 실행 부분
if __name__ == "__main__":
    # Google Drive 마운트
    drive.mount('/content/drive')
    
    # 데이터셋 경로 설정
    dataset_path = '/content/drive/MyDrive/KFall_dataset'
    
    # 전처리된 데이터 로드
    print("전처리된 데이터 로드 중...")
    X_train, y_train, X_val, y_val, X_test, y_test, class_weight_dict = load_preprocessed_data(dataset_path)
    
    print(f"Train shape: {X_train.shape}, Train labels: {y_train.shape}")
    print(f"Validation shape: {X_val.shape}, Validation labels: {y_val.shape}")
    print(f"Test shape: {X_test.shape}, Test labels: {y_test.shape}")
    print(f"Class weights: {class_weight_dict}")
    
    # 모델 학습
    model, history = train_model(X_train, y_train, X_val, y_val, class_weight_dict, dataset_path)
    
    # 모델 평가
    y_pred, y_pred_prob = evaluate_model(model, X_test, y_test, 50, 100, dataset_path)
    
    # 모델 저장
    save_model(model, dataset_path)
