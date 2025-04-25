import tensorflow as tf
import os

# 모델 경로 설정
original_model_path = '/content/drive/MyDrive/KFall_dataset/models/tflite_model/best_tflite_friendly_model.keras'
output_dir = '/content/drive/MyDrive/KFall_dataset/models/tflite_model/'

# 출력 디렉토리 확인
os.makedirs(output_dir, exist_ok=True)

# 원본 Keras 모델 로드
print(f"모델 로드 중: {original_model_path}")
model = tf.keras.models.load_model(original_model_path)
print("모델 로드 완료")

# 모델 요약
model.summary()

# 모델의 입력과 출력 형태를 확인
input_shape = model.input_shape
print(f"입력 형태: {input_shape}")

# 여러 변환 방법을 순차적으로 시도하는 함수
def convert_to_tflite(model, output_path, model_name):
    methods_tried = 0
    
    # 방법 1: 고정된 입력 크기로 모델 다시 생성
    try:
        print("\n방법 1: 고정 입력 크기 모델 변환 시도")
        tf_input = tf.keras.layers.Input(shape=input_shape[1:], batch_size=1)
        outputs = model(tf_input)
        fixed_model = tf.keras.Model(inputs=tf_input, outputs=outputs)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.optimization_default = False
        converter.allow_custom_ops = True
        
        tflite_model = converter.convert()
        model_path = os.path.join(output_path, f'{model_name}_method1.tflite')
        with open(model_path, 'wb') as f:
            f.write(tflite_model)
        print(f"성공: TFLite 모델 저장 완료: {model_path}")
        methods_tried += 1
        return True
    except Exception as e:
        print(f"방법 1 실패: {str(e)}")
    
    # 방법 2: 모델 저장 후 다시 로드
    try:
        print("\n방법 2: H5 저장 후 변환 시도")
        temp_model_path = os.path.join(output_path, 'temp_model.h5')
        model.save(temp_model_path, save_format='h5')
        h5_model = tf.keras.models.load_model(temp_model_path)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(h5_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        tflite_model = converter.convert()
        model_path = os.path.join(output_path, f'{model_name}_method2.tflite')
        with open(model_path, 'wb') as f:
            f.write(tflite_model)
        print(f"성공: TFLite 모델 저장 완료: {model_path}")
        
        # 임시 파일 삭제
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
            print(f"임시 파일 {temp_model_path} 삭제 완료")
            
        methods_tried += 1
        return True
    except Exception as e:
        print(f"방법 2 실패: {str(e)}")
        # 임시 파일 삭제 시도
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
    
    # 방법 3: Concrete Function 사용
    try:
        print("\n방법 3: Concrete Function 사용 변환 시도")
        run_model = tf.function(lambda x: model(x))
        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec([1] + list(model.input_shape[1:]), model.input.dtype))
        
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        tflite_model = converter.convert()
        model_path = os.path.join(output_path, f'{model_name}_method3.tflite')
        with open(model_path, 'wb') as f:
            f.write(tflite_model)
        print(f"성공: TFLite 모델 저장 완료: {model_path}")
        methods_tried += 1
        return True
    except Exception as e:
        print(f"방법 3 실패: {str(e)}")
    
    if methods_tried == 0:
        print("\n모든 변환 방법이 실패했습니다.")
        return False
    
    return True

# 변환 실행
model_name = 'fall_detection'
result = convert_to_tflite(model, output_dir, model_name)

if result:
    print("\n최소 하나의 방법으로 변환에 성공했습니다.")
else:
    print("\n모든 변환 방법이 실패했습니다. 모델 구조 또는 TensorFlow 버전을 확인해보세요.")

print("\n변환 과정 완료.")