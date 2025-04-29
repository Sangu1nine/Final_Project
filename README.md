# 📂 KFall Dataset 소개

KFall Dataset은 낙상(Fall) 및 일반 동작 데이터를 수집한 센서 기반 데이터셋입니다.  
본 데이터셋은 인공지능 기반 낙상 감지 연구 및 다양한 시계열 분석 실험에 활용할 수 있습니다.

---

## 📁 데이터 구성

- **[`label_data_new.zip`](https://drive.google.com/file/d/1iTApqf7RRix-OTL6bJE0LrIKOjrm5os8/view?usp=drive_link)**  
  낙상에 대한 레이블(활동 구분 정보) 파일
- **[`sensor_data_new.zip`](https://drive.google.com/file/d/1uZ3tDh_qiFN49vuf409WfV0wIiiwhThJ/view?usp=drive_link)**  
  IMU 센서를 통해 수집된 가속도, 자이로, Euler 각도 데이터 파일
- **[`sliced_sensor_data.zip`](https://drive.google.com/file/d/1m7_0oCtt0m5oADnPAZSrTqiw7MgF3_kb/view?usp=sharing)**  
  fall_data_slicing.py를 통해 sensor data의 낙상 중 onset, impact 전후 150 프레임만 slicing한 파일
- **[`selected_tasks_data.zip`](https://drive.google.com/file/d/10IypyM-quIUcgKZmGf3uCOuNCOw4GMJe/view?usp=sharing)**  
  sensor_task_select.py를 통해 sensor data의 일상 행동 중 낙상과 헷갈릴 수 있는 동작들만 선별.



---

## 📝 참고사항
- 데이터셋은 낙상 인식 알고리즘 개발, 착용형 디바이스 연구 등에 적합합니다.
- 사용 전 반드시 데이터 구조와 포맷을 확인하세요.
- 파일 내에는 각 동작에 대한 자세한 타임스탬프 및 레이블링 정보가 포함되어 있습니다.

## 데이터 추출 요약
		    num_windows_mean	num_windows_sum
D03	03	32.1				      5034				Pick up an object from the floor
D04	04	20.6				      3305				Gently jump (try to reach an object)
D09	09	61.2				      9241				Jog quickly with turn (4m)
D10	10	57.1				      9077				Stumble while walking
D14	14	35.5				      5509				Sit down to a chair quickly, and get up from a chair quickly
D15	15	35.6				      5592				Sit a moment, trying to get up, and collapse into a chair
D19	19	47.7				      7109				Sit a moment, lie down to the bed quickly, and get up quickly
D21	36	68				        10129			  Walk upstairs and downstairs quickly (5 steps)
                          54,996

            num_windows_count  num_windows_mean  num_windows_sum
subject_id                                                      
SA06                       71         22.676056             1610
SA07                       65         23.953846             1557
SA08                       74         22.216216             1644
SA09                       75         22.933333             1720
SA10                       75         23.093333             1732
SA11                       77         22.376623             1723
SA12                       77         22.766234             1753
SA13                       75         21.560000             1617
SA14                       72         22.305556             1606
SA15                       75         22.746667             1706
SA16                       76         22.131579             1682
SA17                       74         22.756757             1684
SA18                       75         23.173333             1738
SA19                       73         22.753425             1661
SA20                       72         22.847222             1645
SA21                       73         22.917808             1673
SA22                       74         22.918919             1696
SA23                       76         22.789474             1732
SA24                       68         22.485294             1529
SA25                       75         23.186667             1739
SA26                       74         23.405405             1732
SA27                       72         23.694444             1706
SA28                       70         23.100000             1617
SA29                       74         23.121622             1711
SA30                       74         23.324324             1726
SA31                       70         22.942857             1606
SA32                       75         22.906667             1718
SA33                       74         22.364865             1655
SA35                       74         23.189189             1716
SA36                       75         23.106667             1733
SA37                       68         22.441176             1526
SA38                       74         22.864865             1692
                                                            53,585