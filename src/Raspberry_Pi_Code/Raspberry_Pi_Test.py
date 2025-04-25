import os
import time
import numpy as np
import pandas as pd
# TensorFlow 대신 TFLite 런타임 사용
import tflite_runtime.interpreter as tflite
from collections import deque
import threading
import signal
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 라즈베리 파이 경로로 변경
MODEL_PATH = '/home/pi/Final_Project/data/models/tflite_model/fall_detection_compatible.tflite'
SCALER_MEAN_PATH = '/home/pi/Final_Project/data/models/tflite_model/extracted_scaler_mean.npy'
SCALER_SCALE_PATH = '/home/pi/Final_Project/data/models/tflite_model/extracted_scaler_scale.npy'
TEST_DATA_PATH = '/home/pi/Final_Project/data/test'  # 테스트 데이터 경로
SEQ_LENGTH = 50  # 시퀀스 길이 
N_FEATURES = 9    # 특징 수 (AccX, AccY, AccZ, GyrX, GyrY, GyrZ, EulerX, EulerY, EulerZ)

class FallDetectorTest:
    def __init__(self, model_path, scaler_mean_path, scaler_scale_path, seq_length=50, n_features=9):
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
        try:
            # TensorFlow 대신 TFLite 런타임 사용
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {str(e)}")
            raise
    
    def normalize_data(self, data):
        """데이터 정규화"""
        try:
            # 입력 데이터 형태 확인
            if data.shape[0] != self.seq_length or data.shape[1] != self.n_features:
                print(f"경고: 데이터 형태가 예상과 다릅니다. 예상: ({self.seq_length}, {self.n_features}), 실제: {data.shape}")
                # 필요시 데이터 형태 조정
                if data.shape[0] > self.seq_length:
                    data = data[-self.seq_length:]  # 마지막 seq_length 개만 사용
            
            # (seq_length, n_features) -> (seq_length * n_features)
            data_flat = data.reshape(-1, self.n_features)
            
            # 스케일러 형태 확인
            if self.scaler_mean.shape[0] != self.n_features or self.scaler_scale.shape[0] != self.n_features:
                print(f"경고: 스케일러 형태가 맞지 않습니다. 특성 수: {self.n_features}, "
                      f"평균 형태: {self.scaler_mean.shape}, 스케일 형태: {self.scaler_scale.shape}")
            
            # 정규화 적용
            data_norm = (data_flat - self.scaler_mean) / self.scaler_scale
            
            # 원래 형태로 복원 (배치 차원 추가)
            return data_norm.reshape(1, self.seq_length, self.n_features)
        except Exception as e:
            print(f"정규화 중 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            # 오류 시 정규화 생략
            return np.expand_dims(data, axis=0)
    
    def predict(self, data_sequence):
        """낙상 예측 수행"""
        try:
            if len(data_sequence) < self.seq_length:
                print(f"데이터 길이 부족: {len(data_sequence)} < {self.seq_length}")
                return None  # 충분한 데이터가 없음
            
            # 입력 차원 확인 및 조정
            if len(data_sequence.shape) != 2:
                print(f"경고: 입력 데이터 차원이 예상과 다릅니다: {data_sequence.shape}")
                if len(data_sequence.shape) == 3 and data_sequence.shape[0] == 1:
                    data_sequence = data_sequence[0]  # 첫 번째 배치 항목만 사용
            
            # 데이터 정규화
            data_norm = self.normalize_data(data_sequence)
            
            # 모델 입력 형태 확인
            expected_shape = tuple(self.input_details[0]['shape'])
            if data_norm.shape != expected_shape:
                print(f"경고: 입력 형태가 맞지 않습니다. 예상: {expected_shape}, 실제: {data_norm.shape}")
                # 필요시 차원 조정 시도
                data_norm = data_norm.reshape(expected_shape)
            
            # 모델 입력 설정
            input_data = data_norm.astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # 추론 실행
            self.interpreter.invoke()
            
            # 결과 가져오기
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # 출력 형태에 따라 다르게 처리
            if output_data.size == 1:
                # 단일 값 출력인 경우
                fall_prob = float(output_data.flatten()[0])
            else:
                # 다차원 출력인 경우
                fall_prob = float(output_data[0][0])
            
            # 예측 결과 (0: 정상, 1: 낙상)
            prediction = 1 if fall_prob >= 0.5 else 0
            
            return {
                'prediction': int(prediction),
                'fall_probability': float(fall_prob)
            }
        except Exception as e:
            print(f"예측 중 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_on_test_data(self, X_test, y_test):
        """테스트 데이터셋에 대한 평가 수행"""
        predictions = []
        probabilities = []
        
        print(f"테스트 데이터 평가 시작 ({X_test.shape[0]}개 샘플)")
        
        for i in range(X_test.shape[0]):
            if i % 100 == 0:
                print(f"진행률: {i}/{X_test.shape[0]} ({i/X_test.shape[0]*100:.1f}%)")
            
            # 길이 확인 및 조정
            if X_test[i].shape[0] != self.seq_length:
                print(f"경고: 샘플 {i}의 길이가 맞지 않습니다: {X_test[i].shape[0]} != {self.seq_length}")
                # 길이 조정 필요시 처리
                if X_test[i].shape[0] > self.seq_length:
                    sample_data = X_test[i][-self.seq_length:]  # 마지막 seq_length 개 사용
                else:
                    # 부족한 경우 패딩 (첫 번째 값으로)
                    pad_length = self.seq_length - X_test[i].shape[0]
                    padding = np.tile(X_test[i][0:1], (pad_length, 1))
                    sample_data = np.vstack([padding, X_test[i]])
            else:
                sample_data = X_test[i]
            
            result = self.predict(sample_data)
            
            # 디버깅용 - 처음 5개 결과와 매 500번째 결과 출력
            if i < 5 or i % 500 == 0:
                print(f"샘플 {i} 결과: {result}")
            
            if result is not None:
                predictions.append(result['prediction'])
                probabilities.append(result['fall_probability'])
            else:
                # None 결과인 경우 기본값 사용
                print(f"샘플 {i}에 대한 예측 실패, 기본값 사용")
                predictions.append(0)  # 기본값 'Non-Fall'
                probabilities.append(0.0)
        
        # 결과가 비어있는지 확인
        if len(predictions) == 0:
            print("경고: 예측 결과가 없습니다!")
            return np.array([]), np.array([])
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # 결과 개수 확인
        print(f"총 예측 결과 수: {len(predictions)}")
        
        # 성능 평가
        print("\n=== 모델 성능 평가 결과 ===")
        
        # 길이 확인
        if len(predictions) != len(y_test):
            print(f"경고: 예측 결과 수({len(predictions)})와 실제 레이블 수({len(y_test)})가 일치하지 않습니다.")
            # 길이가 다르면 맞추기
            min_len = min(len(predictions), len(y_test))
            predictions = predictions[:min_len]
            y_test_truncated = y_test[:min_len]
            print(f"평가를 위해 처음 {min_len}개 샘플만 사용합니다.")
        else:
            y_test_truncated = y_test
        
        # 레이블 요약
        print(f"레이블 분포: 낙상={sum(y_test_truncated == 1)}, 비낙상={sum(y_test_truncated == 0)}")
        print(f"예측 분포: 낙상={sum(predictions == 1)}, 비낙상={sum(predictions == 0)}")
        
        print(classification_report(y_test_truncated, predictions))
        
        # 라즈베리 파이에서는 그래프 표시를 선택적으로 활성화
        try:
            # 혼동 행렬 시각화
            cm = confusion_matrix(y_test_truncated, predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.xticks([0.5, 1.5], ['Non-Fall', 'Fall'])
            plt.yticks([0.5, 1.5], ['Non-Fall', 'Fall'])
            
            # 그래프를 이미지로 저장 (GUI가 없는 경우를 위해)
            plt.savefig('confusion_matrix.png')
            print("혼동 행렬 그래프가 'confusion_matrix.png'로 저장되었습니다.")
            
            try:
                plt.show()  # 라즈베리 파이에 GUI가 있는 경우
            except:
                pass
            
            # ROC 곡선
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_test_truncated, probabilities[:min_len])
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
            
            # 그래프를 이미지로 저장
            plt.savefig('roc_curve.png')
            print("ROC 곡선 그래프가 'roc_curve.png'로 저장되었습니다.")
            
            try:
                plt.show()  # 라즈베리 파이에 GUI가 있는 경우
            except:
                pass
        except Exception as e:
            print(f"그래프 생성 중 오류 발생: {e}")
            print("그래프 생성을 건너뜁니다.")
        
        return predictions, probabilities
    
    def simulate_realtime_detection(self, test_sequence, true_label, show_plot=True):
        """실시간 감지 시뮬레이션"""
        try:
            self.data_buffer.clear()
            predictions = []
            
            print(f"\n실시간 감지 시뮬레이션 시작 (실제 레이블: {'낙상' if true_label == 1 else '정상'})")
            
            # 시퀀스 길이 체크 및 조정
            if test_sequence.shape[0] < self.seq_length:
                print(f"경고: 테스트 시퀀스 길이({test_sequence.shape[0]})가 필요한 길이({self.seq_length})보다 짧습니다.")
                # 필요시 패딩 처리할 수 있음
            
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
                        # 예측 실패 시 기본값 추가
                        predictions.append(0)
                        print(f"[프레임 {i}] 예측 실패")
            
            # 라즈베리 파이에서는 그래프 표시를 선택적으로 활성화
            if show_plot and predictions:
                try:
                    plt.figure(figsize=(10, 6))
                    plt.plot(predictions, label='Fall Probability')
                    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
                    plt.title(f'Real-time Fall Detection Simulation (True Label: {"Fall" if true_label == 1 else "Non-Fall"})')
                    plt.xlabel('Frame')
                    plt.ylabel('Fall Probability')
                    plt.legend()
                    
                    # 그래프를 이미지로 저장
                    plt.savefig(f'simulation_{"fall" if true_label == 1 else "nonfall"}.png')
                    print(f"시뮬레이션 그래프가 'simulation_{'fall' if true_label == 1 else 'nonfall'}.png'로 저장되었습니다.")
                    
                    try:
                        plt.show()  # 라즈베리 파이에 GUI가 있는 경우
                    except:
                        pass
                except Exception as e:
                    print(f"그래프 생성 중 오류 발생: {e}")
                    print("그래프 생성을 건너뜁니다.")
            
            return predictions
        except Exception as e:
            print(f"시뮬레이션 중 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            return []


def load_test_data():
    """테스트 데이터 로드"""
    test_data_path = TEST_DATA_PATH
    
    # 테스트 데이터 파일 경로
    X_test_path = os.path.join(test_data_path, 'X_test.npy')
    y_test_path = os.path.join(test_data_path, 'y_test.npy')
    
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        print(f"테스트 데이터를 찾을 수 없습니다. 경로: {test_data_path}")
        return None, None
    
    try:
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        
        # 데이터 형태 및 타입 확인
        print(f"테스트 데이터 로드 완료: X_test shape={X_test.shape}, dtype={X_test.dtype}")
        print(f"레이블 데이터 로드 완료: y_test shape={y_test.shape}, dtype={y_test.dtype}")
        
        # 데이터 유효성 검사
        if len(X_test) != len(y_test):
            print(f"경고: X_test({len(X_test)})와 y_test({len(y_test)})의 샘플 수가 다릅니다.")
        
        return X_test, y_test
    except Exception as e:
        print(f"데이터 로드 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """메인 함수"""
    print("낙상 감지 시스템 테스트 시작")
    
    try:
        # 파일 경로 확인
        for path, description in [
            (MODEL_PATH, "모델 파일"),
            (SCALER_MEAN_PATH, "스케일러 평균 파일"),
            (SCALER_SCALE_PATH, "스케일러 스케일 파일"),
            (TEST_DATA_PATH, "테스트 데이터 경로")
        ]:
            if os.path.exists(path):
                print(f"{description} 확인: {path} (존재함)")
            else:
                print(f"경고: {description}를 찾을 수 없습니다: {path}")

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
        
        # 시퀀스 길이 체크
        if X_test.shape[1] != SEQ_LENGTH:
            print(f"경고: 데이터의 시퀀스 길이({X_test.shape[1]})가 설정된 시퀀스 길이({SEQ_LENGTH})와 다릅니다.")
            if X_test.shape[1] > SEQ_LENGTH:
                print(f"데이터를 {X_test.shape[1]}에서 {SEQ_LENGTH}로 잘라냅니다.")
                X_test_resized = X_test[:, :SEQ_LENGTH, :]
            else:
                print(f"데이터 길이가 짧아 처리할 수 없습니다. 설정을 확인하세요.")
                return
        else:
            X_test_resized = X_test
        
        # 전체 테스트 데이터셋에 대한 평가
        predictions, probabilities = detector.evaluate_on_test_data(X_test_resized, y_test)
        
        # 결과가 비어있는지 확인
        if len(predictions) == 0:
            print("경고: 예측 결과가 없어 성능 평가를 건너뜁니다.")
            return
        
        # 낙상과 비낙상 샘플 각각 하나씩 시뮬레이션
        fall_indices = np.where(y_test == 1)[0]
        non_fall_indices = np.where(y_test == 0)[0]
        
        if len(fall_indices) > 0:
            fall_idx = fall_indices[0]
            print("\n=== 낙상 샘플 시뮬레이션 ===")
            detector.simulate_realtime_detection(X_test_resized[fall_idx], y_test[fall_idx])
        
        if len(non_fall_indices) > 0:
            non_fall_idx = non_fall_indices[0]
            print("\n=== 비낙상 샘플 시뮬레이션 ===")
            detector.simulate_realtime_detection(X_test_resized[non_fall_idx], y_test[non_fall_idx])
        
        # 성능 요약
        print("\n=== 최종 성능 요약 ===")
        
        # 길이 확인
        if len(predictions) != len(y_test):
            min_len = min(len(predictions), len(y_test))
            predictions_summary = predictions[:min_len]
            y_test_summary = y_test[:min_len]
            print(f"길이가 다르므로 처음 {min_len}개 샘플만 사용하여 요약합니다.")
        else:
            predictions_summary = predictions
            y_test_summary = y_test
        
        accuracy = np.mean(predictions_summary == y_test_summary)
        print(f"전체 정확도: {accuracy:.2%}")
        
        # 클래스별 정확도
        fall_mask = y_test_summary == 1
        non_fall_mask = y_test_summary == 0
        
        if sum(fall_mask) > 0:
            fall_accuracy = np.mean(predictions_summary[fall_mask] == y_test_summary[fall_mask])
            print(f"낙상 감지 정확도: {fall_accuracy:.2%} ({sum(fall_mask)}개 샘플)")
        else:
            print("낙상 샘플이 없습니다.")
        
        if sum(non_fall_mask) > 0:
            non_fall_accuracy = np.mean(predictions_summary[non_fall_mask] == y_test_summary[non_fall_mask])
            print(f"비낙상 감지 정확도: {non_fall_accuracy:.2%} ({sum(non_fall_mask)}개 샘플)")
        else:
            print("비낙상 샘플이 없습니다.")
        
        print("\n테스트 완료!")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()