import os
import pandas as pd
import numpy as np
import glob
import re
import tqdm
from pathlib import Path
import shutil

# 기본 디렉토리 경로
LABEL_BASE_DIR = '/content/drive/MyDrive/KFall_dataset/data/label_data_new'
SENSOR_BASE_DIR = '/content/drive/MyDrive/KFall_dataset/data/sensor_data_new'
OUTPUT_BASE_DIR = '/content/drive/MyDrive/KFall_dataset/data/extracted_data'

# 파라미터 설정
FRAMES_BEFORE_ONSET = 150
FRAMES_AFTER_IMPACT = 150
TARGET_TASKS = list(range(20, 35))  # T20부터 T34까지

# 이전 출력 데이터 삭제 함수
def clean_previous_output(output_dir=OUTPUT_BASE_DIR, ask_confirmation=True):
    """이전에 생성된 출력 데이터 삭제"""
    if os.path.exists(output_dir):
        if ask_confirmation:
            confirm = input(f"기존 출력 폴더 '{output_dir}'가 존재합니다. 삭제하시겠습니까? (y/n): ")
            if confirm.lower() != 'y':
                print("기존 데이터 유지. 프로그램을 종료합니다.")
                return False

        print(f"기존 출력 폴더 '{output_dir}' 삭제 중...")
        try:
            shutil.rmtree(output_dir)
            print("기존 출력 폴더 삭제 완료")
        except Exception as e:
            print(f"폴더 삭제 중 오류 발생: {e}")
            return False

    # 출력 디렉토리 새로 생성
    os.makedirs(output_dir, exist_ok=True)
    return True

# 모든 레이블 파일과 센서 데이터 폴더를 처리하는 함수
def process_all_data():
    # 결과 저장용 리스트
    extraction_summary = []
    failed_files = []

    # 레이블 파일 처리 (SA06~SA38, SA34 제외)
    for subject_num in range(6, 39):
        if subject_num == 34:  # SA34 건너뛰기
            continue

        subject_id = f'SA{subject_num:02d}'
        label_file_path = os.path.join(LABEL_BASE_DIR, f"{subject_id}_label.xlsx")

        if not os.path.exists(label_file_path):
            print(f"Label file not found: {label_file_path}")
            continue

        # 센서 데이터 폴더 확인
        sensor_dir = os.path.join(SENSOR_BASE_DIR, subject_id)
        if not os.path.exists(sensor_dir):
            print(f"Sensor data folder not found: {sensor_dir}")
            continue

        # 출력 디렉토리 생성
        output_subject_dir = os.path.join(OUTPUT_BASE_DIR, subject_id)
        os.makedirs(output_subject_dir, exist_ok=True)

        # 레이블 파일 읽기
        try:
            label_df = pd.read_excel(label_file_path)
            print(f"\nProcessing: {subject_id} - Loaded label data ({len(label_df)} items)")

            # Task Code에서 숫자 추출 (예: "F01 (20)" -> 20)
            def extract_task_id(task_code):
                if pd.isna(task_code) or task_code == '':
                    return None
                match = re.search(r'\((\d+)\)', task_code)
                if match:
                    return int(match.group(1))
                return None

            # 레이블 데이터에 Task ID 열 추가
            label_df['Numeric_Task_ID'] = label_df['Task Code (Task ID)'].apply(extract_task_id)

            # Task ID가 있는 첫 번째 행 기준으로 그룹화
            task_groups = []
            current_group = None

            for idx, row in label_df.iterrows():
                if not pd.isna(row['Task Code (Task ID)']) and row['Task Code (Task ID)'] != '':
                    # 새 그룹 시작
                    current_group = row['Numeric_Task_ID']

                # 현재 행에 그룹 ID 할당
                label_df.at[idx, 'Group_Task_ID'] = current_group

            # 그룹 Task ID를 숫자로 변환
            label_df['Group_Task_ID'] = pd.to_numeric(label_df['Group_Task_ID'])

            # T20~T34 범위의 레이블만 필터링
            label_df = label_df[label_df['Group_Task_ID'].isin(TARGET_TASKS)]

            if label_df.empty:
                print(f"No labels in T20-T34 range for {subject_id}")
                continue

        except Exception as e:
            print(f"Error reading label file: {label_file_path}, Error: {e}")
            continue

        # 센서 데이터 파일 목록 가져오기
        sensor_files = glob.glob(os.path.join(sensor_dir, "*.csv"))

        # T20~T34 파일만 필터링
        target_files = []
        for file_path in sensor_files:
            file_name = os.path.basename(file_path)
            t_match = re.search(r'T(\d+)', file_name)
            if t_match and int(t_match.group(1)) in TARGET_TASKS:
                target_files.append(file_path)

        print(f"{subject_id} - Found {len(target_files)} sensor data files in T20-T34 range")

        # 각 센서 파일 처리
        for sensor_file in tqdm.tqdm(target_files, desc=f"Processing {subject_id}"):
            try:
                file_name = os.path.basename(sensor_file)

                # 파일명에서 정보 추출 (예: S06T20R01.csv)
                t_match = re.search(r'T(\d+)', file_name)
                r_match = re.search(r'R(\d+)', file_name)

                if not t_match or not r_match:
                    print(f"Cannot parse filename: {file_name}")
                    failed_files.append({'subject_id': subject_id, 'file': file_name, 'reason': 'filename_parse_error'})
                    continue

                task_id = int(t_match.group(1))
                trial_id = int(r_match.group(1))

                # 해당 작업과 시험 번호에 대한 레이블 찾기
                relevant_labels = label_df[(label_df['Group_Task_ID'] == task_id) &
                                          (label_df['Trial ID'] == trial_id)]

                if relevant_labels.empty:
                    print(f"No label found for: {subject_id}, Task {task_id}, Trial {trial_id}")
                    failed_files.append({'subject_id': subject_id, 'file': file_name, 'reason': 'label_not_found'})
                    continue

                # 센서 데이터 읽기
                sensor_data = pd.read_csv(sensor_file)

                # Onset과 Impact 프레임 가져오기
                onset_frame = relevant_labels['Fall_onset_frame'].iloc[0]
                impact_frame = relevant_labels['Fall_impact_frame'].iloc[0]

                # 프레임 범위 계산
                start_frame = max(1, onset_frame - FRAMES_BEFORE_ONSET)
                end_frame = impact_frame + FRAMES_AFTER_IMPACT

                # 필요한 데이터 추출 (FrameCounter 열을 기준으로)
                extracted_data = sensor_data[(sensor_data['FrameCounter'] >= start_frame) &
                                             (sensor_data['FrameCounter'] <= end_frame)].copy()

                if len(extracted_data) == 0:
                    print(f"No data extracted: {file_name}, Frame range: {start_frame}-{end_frame}")
                    failed_files.append({'subject_id': subject_id, 'file': file_name,
                                        'reason': 'no_frames_in_range',
                                        'start_frame': start_frame,
                                        'end_frame': end_frame})
                    continue

                # 메타데이터 추가
                extracted_data['original_frame'] = extracted_data['FrameCounter'].copy()
                extracted_data['rel_to_onset'] = extracted_data['FrameCounter'] - onset_frame
                extracted_data['rel_to_impact'] = extracted_data['FrameCounter'] - impact_frame
                extracted_data['task_id'] = task_id
                extracted_data['trial_id'] = trial_id

                # 파일 저장
                output_file = os.path.join(output_subject_dir, f"extracted_{file_name}")
                extracted_data.to_csv(output_file, index=False)

                # 요약 정보 저장
                extraction_summary.append({
                    'subject_id': subject_id,
                    'task_id': task_id,
                    'trial_id': trial_id,
                    'original_file': file_name,
                    'original_frames': len(sensor_data),
                    'extracted_frames': len(extracted_data),
                    'onset_frame': onset_frame,
                    'impact_frame': impact_frame,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'frame_range': end_frame - start_frame + 1,
                    'output_file': os.path.basename(output_file)
                })

            except Exception as e:
                print(f"Error processing file: {sensor_file}, Error: {e}")
                failed_files.append({'subject_id': subject_id, 'file': os.path.basename(sensor_file),
                                    'reason': f'processing_error: {str(e)}'})
                continue

    # 결과 요약 저장
    summary_df = pd.DataFrame(extraction_summary)
    failed_df = pd.DataFrame(failed_files)

    if not summary_df.empty:
        summary_df.to_csv(os.path.join(OUTPUT_BASE_DIR, 'extraction_summary.csv'), index=False)

        # 추가 분석 수행
        task_stats = summary_df.groupby('task_id').agg({
            'original_frames': 'sum',
            'extracted_frames': 'sum',
            'subject_id': 'nunique',
            'original_file': 'count'
        }).reset_index()

        task_stats['extraction_ratio'] = task_stats['extracted_frames'] / task_stats['original_frames'] * 100
        task_stats.to_csv(os.path.join(OUTPUT_BASE_DIR, 'task_statistics.csv'), index=False)

        subject_stats = summary_df.groupby('subject_id').agg({
            'original_frames': 'sum',
            'extracted_frames': 'sum',
            'task_id': 'nunique',
            'original_file': 'count'
        }).reset_index()

        subject_stats['extraction_ratio'] = subject_stats['extracted_frames'] / subject_stats['original_frames'] * 100
        subject_stats.to_csv(os.path.join(OUTPUT_BASE_DIR, 'subject_statistics.csv'), index=False)

        # 통계 출력
        total_original_frames = summary_df['original_frames'].sum()
        total_extracted_frames = summary_df['extracted_frames'].sum()
        total_files = len(summary_df)

        print(f"\nExtraction Summary:")
        print(f"Total files processed successfully: {total_files}")
        print(f"Total original frames: {total_original_frames}")
        print(f"Total extracted frames: {total_extracted_frames}")
        print(f"Overall extraction ratio: {(total_extracted_frames / total_original_frames * 100):.2f}%")

    if not failed_df.empty:
        failed_df.to_csv(os.path.join(OUTPUT_BASE_DIR, 'failed_files.csv'), index=False)
        print(f"Failed files: {len(failed_df)}")

        # 실패 원인 분석
        reason_counts = failed_df['reason'].value_counts()
        print("Failure reasons:")
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count}")

    return summary_df, failed_df

# 간단한 시각화 기능
def create_basic_visualizations(summary_df, output_dir=OUTPUT_BASE_DIR):
    """데이터 추출 결과에 대한 기본 시각화 생성"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if summary_df is None or summary_df.empty:
        print("No data to visualize")
        return

    # 시각화 저장 폴더 생성
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # 1. 작업별(Task) 추출된 프레임 수
    plt.figure(figsize=(12, 6))
    task_frames = summary_df.groupby('task_id')['extracted_frames'].sum().sort_index()
    sns.barplot(x=task_frames.index, y=task_frames.values)
    plt.title('Total Extracted Frames by Task')
    plt.xlabel('Task ID')
    plt.ylabel('Number of Frames')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'frames_by_task.png'))
    plt.close()

    # 2. 피험자별 추출된 파일 수
    plt.figure(figsize=(14, 6))
    subject_counts = summary_df['subject_id'].value_counts().sort_index()
    sns.barplot(x=subject_counts.index, y=subject_counts.values)
    plt.title('Number of Processed Files by Subject')
    plt.xlabel('Subject ID')
    plt.ylabel('Number of Files')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'files_by_subject.png'))
    plt.close()

    print(f"Basic visualizations saved to {viz_dir}")

# 메인 실행 함수
def main():
    print(f"Data Extraction Settings:")
    print(f"- Frames before onset: {FRAMES_BEFORE_ONSET}")
    print(f"- Frames after impact: {FRAMES_AFTER_IMPACT}")
    print(f"- Target task range: T{min(TARGET_TASKS)}-T{max(TARGET_TASKS)}")
    print(f"- Output directory: {OUTPUT_BASE_DIR}")
    print("-" * 50)

    # 이전 출력 삭제
    if not clean_previous_output(OUTPUT_BASE_DIR):
        return

    print("Starting data extraction process...")
    summary_df, failed_df = process_all_data()

    if summary_df is not None and not summary_df.empty:
        create_basic_visualizations(summary_df)
        print("Data extraction complete!")

        # 처리 완료된 파일 수 출력
        if not failed_df.empty:
            success_rate = len(summary_df) / (len(summary_df) + len(failed_df)) * 100
            print(f"Successfully processed {len(summary_df)} files ({success_rate:.1f}%)")
            print(f"Failed to process {len(failed_df)} files")
        else:
            print(f"Successfully processed all {len(summary_df)} files (100%)")
    else:
        print("No data was extracted. Check for errors.")

if __name__ == "__main__":
    main()