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

# 모델 및 데이터 설정
MODEL_PATH = 'Final_Project/data/models/cnn_fall_detection.tflite'
SCALER_MEAN_PATH = 'Final_Project/data/models/cnn_fall_detection_scaler_mean.npy'
SCALER_SCALE_PATH = 'Final_Project/data/models/cnn_fall_detection_scaler_scale.npy'
TEST_DATA_PATH = '/content/drive/MyDrive/KFall_dataset/test_data'  # 테스트 데이터 경로
SEQ_LENGTH = 100  # 시퀀스 길이
N_FEATURES = 9    # 특징 수 (AccX, AccY, AccZ, GyrX, GyrY, GyrZ, EulerX, EulerY, EulerZ)

class FallDetectorTest:
    def __init__(self, model_path, scaler_mean_path, scaler_scale_path, seq_length=100, n_features=9):
        """낙상 감지 모델 초기화"""
        self.seq_length = seq_length
        self.n_features = n_features
        self.data_buffer = deque(maxlen=seq_length)
        
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
    
    def load_model(self, model_path):
        """TFLite 모델 로드"""
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    
    def normalize_data(self, data):
        """데이터 정규화"""
        # (seq_length, n_features) -> (seq_length * n_features)
        data_flat = data.reshape(-1, self.n_features)
        
        # 정규화 적용
        data_norm = (data_flat - self.scaler_mean) / self.scaler_scale
        
        # 원래 형태로 복원
        return data_norm.reshape(1, self.seq_length, self.n_features)
    
    def predict(self, data_sequence):
        """낙상 예측 수행"""
        if len(data_sequence) < self.seq_length:
            return None  # 충분한 데이터가 없음
        
        # 데이터 정규화
        data_norm = self.normalize_data(data_sequence)
        
        # 모델 입력 설정
        input_data = data_norm.astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # 추론 실행
        self.interpreter.invoke()
        
        # 결과 가져오기
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # 낙상 확률
        fall_prob = output_data[0][0]  # sigmoid 출력이므로 [0][0]
        
        # 예측 결과 (0: 정상, 1: 낙상)
        prediction = 1 if fall_prob >= 0.5 else 0
        
        return {
            'prediction': int(prediction),
            'fall_probability': float(fall_prob)
        }
    
    def evaluate_on_test_data(self, X_test, y_test):
        """테스트 데이터셋에 대한 평가 수행"""
        predictions = []
        probabilities = []
        
        print(f"테스트 데이터 평가 시작 ({X_test.shape[0]}개 샘플)")
        
        for i in range(X_test.shape[0]):
            if i % 100 == 0:
                print(f"진행률: {i}/{X_test.shape[0]} ({i/X_test.shape[0]*100:.1f}%)")
            
            result = self.predict(X_test[i])
            if result:
                predictions.append(result['prediction'])
                probabilities.append(result['fall_probability'])
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # 성능 평가
        print("\n=== 모델 성능 평가 결과 ===")
        print(classification_report(y_test, predictions))
        
        # 혼동 행렬 시각화
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.xticks([0.5, 1.5], ['Non-Fall', 'Fall'])
        plt.yticks([0.5, 1.5], ['Non-Fall', 'Fall'])
        plt.show()
        
        # ROC 곡선
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()
        
        return predictions, probabilities
    
    def simulate_realtime_detection(self, test_sequence, true_label, show_plot=True):
        """실시간 감지 시뮬레이션"""
        self.data_buffer.clear()
        predictions = []
        
        print(f"\n실시간 감지 시뮬레이션 시작 (실제 레이블: {'낙상' if true_label == 1 else '정상'})")
        
        for i in range(test_sequence.shape[0]):
            self.data_buffer.append(test_sequence[i])
            
            if len(self.data_buffer) >= self.seq_length:
                data_array = np.array(list(self.data_buffer))
                result = self.predict(data_array)
                
                if result:
                    predictions.append(result['fall_probability'])
                    
                    if result['prediction'] == 1:
                        print(f"[프레임 {i}] 낙상 감지! 확률: {result['fall_probability']:.2%}")
                else:
                    predictions.append(0)
        
        if show_plot and predictions:
            plt.figure(figsize=(10, 6))
            plt.plot(predictions, label='Fall Probability')
            plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
            plt.title(f'Real-time Fall Detection Simulation (True Label: {"Fall" if true_label == 1 else "Non-Fall"})')
            plt.xlabel('Frame')
            plt.ylabel('Fall Probability')
            plt.legend()
            plt.show()
        
        return predictions


def load_test_data():
    """테스트 데이터 로드"""
    test_data_path = TEST_DATA_PATH
    
    # 테스트 데이터 파일 경로
    X_test_path = os.path.join(test_data_path, 'X_test.npy')
    y_test_path = os.path.join(test_data_path, 'y_test.npy')
    
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        print(f"테스트 데이터를 찾을 수 없습니다. 경로: {test_data_path}")
        return None, None
    
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    print(f"테스트 데이터 로드 완료: X_test shape={X_test.shape}, y_test shape={y_test.shape}")
    return X_test, y_test


def main():
    """메인 함수"""
    print("낙상 감지 시스템 테스트 시작")
    
    try:
        # 낙상 감지기 초기화
        detector = FallDetectorTest(
            model_path=MODEL_PATH, 
            scaler_mean_path=SCALER_MEAN_PATH,
            scaler_scale_path=SCALER_SCALE_PATH,
            seq_length=SEQ_LENGTH,
            n_features=N_FEATURES
        )
        
        # 테스트 데이터 로드
        X_test, y_test = load_test_data()
        
        if X_test is None or y_test is None:
            print("테스트 데이터 로드 실패")
            return
        
        # 전체 테스트 데이터셋에 대한 평가
        predictions, probabilities = detector.evaluate_on_test_data(X_test, y_test)
        
        # 낙상과 비낙상 샘플 각각 하나씩 시뮬레이션
        fall_indices = np.where(y_test == 1)[0]
        non_fall_indices = np.where(y_test == 0)[0]
        
        if len(fall_indices) > 0:
            fall_idx = fall_indices[0]
            print("\n=== 낙상 샘플 시뮬레이션 ===")
            detector.simulate_realtime_detection(X_test[fall_idx], y_test[fall_idx])
        
        if len(non_fall_indices) > 0:
            non_fall_idx = non_fall_indices[0]
            print("\n=== 비낙상 샘플 시뮬레이션 ===")
            detector.simulate_realtime_detection(X_test[non_fall_idx], y_test[non_fall_idx])
        
        # 성능 요약
        print("\n=== 최종 성능 요약 ===")
        accuracy = np.mean(predictions == y_test)
        print(f"전체 정확도: {accuracy:.2%}")
        
        # 클래스별 정확도
        fall_accuracy = np.mean(predictions[y_test == 1] == y_test[y_test == 1])
        non_fall_accuracy = np.mean(predictions[y_test == 0] == y_test[y_test == 0])
        print(f"낙상 감지 정확도: {fall_accuracy:.2%}")
        print(f"비낙상 감지 정확도: {non_fall_accuracy:.2%}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()