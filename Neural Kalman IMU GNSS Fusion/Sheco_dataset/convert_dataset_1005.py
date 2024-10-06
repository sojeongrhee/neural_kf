import pandas as pd

# Magnetometer 값을 생성하는 함수 (Gyro 및 Acc 데이터를 기반으로)
def generate_magnetometer(ax, ay, az, gx, gy, gz):
    """
    가속도계 및 자이로스코프 데이터를 기반으로 임의의 Magnetometer 데이터를 생성합니다.
    실제 물리적인 자기장 값을 반영하지 않으므로 학습 목적의 임의 값임을 주의하세요.
    """
    # 대략적인 지구 자기장의 범위 (단위: uT, 마이크로테슬라)
    earth_magnetic_field = 50  # 임의로 설정된 자기장 강도 (평균적으로 25~65 uT 사이)

    # 자이로스코프 데이터를 기반으로 방향성 추가 (여기서는 임의의 값을 사용)
    mx = earth_magnetic_field + 0.1 * gx
    my = earth_magnetic_field + 0.1 * gy
    mz = earth_magnetic_field + 0.1 * gz
    
    # 가속도계의 축을 기준으로 약간의 변동을 추가
    mx += 0.01 * ax
    my += 0.01 * ay
    mz += 0.01 * az
    
    return mx, my, mz

# Dataset 변환 함수 (Magnetometer 생성 포함)
def convert_dataset(input_csv, output_csv):
    # Input dataset 읽기
    df = pd.read_csv(input_csv)
    
    # 새 DataFrame 생성
    df_new = pd.DataFrame()
    df_new['Frame'] = range(1, 1 + len(df))  # Frame 번호 생성

    # Time (Seconds): 타임스탬프 기준으로 첫 번째 값에서 뺀 값 (초로 변환)
    first_time = df['%time'].iloc[0]
    df_new['Time (Seconds)'] = (df['%time'] - first_time) / 1e9  # 나노초를 초로 변환

    # X, Z: lon을 X로, lat을 Z로 변경
    df_new['X'] = df['lon']
    df_new['Z'] = df['lat']

    # IMU (s): Time (Seconds) 값 그대로 사용
    df_new['IMU (s)'] = df_new['Time (Seconds)']

    # Ax, Ay, Az: 가속도 데이터 (이미 ax, ay, az로 존재)
    df_new['Ax'] = df['ax']
    df_new['Ay'] = df['ay']
    df_new['Az'] = df['az']

    # Gx, Gy, Gz: 자이로스코프 데이터 (이미 wx, wy, wz로 존재)
    df_new['Gx'] = df['wx']
    df_new['Gy'] = df['wy']
    df_new['Gz'] = df['wz']

    # # Magnetometer 값 생성 (Gyro와 Acc 기반)
    # magnetometer_data = [generate_magnetometer(ax, ay, az, gx, gy, gz)
    #                      for ax, ay, az, gx, gy, gz in zip(df['ax'], df['ay'], df['az'], df['wx'], df['wy'], df['wz'])]
    
    # # Magnetometer 데이터 열로 추가
    # df_new['Mx'], df_new['My'], df_new['Mz'] = zip(*magnetometer_data)

    df_new['Mx'] = df['roll']
    df_new['My'] = df['pitch']
    df_new['Mz'] = df['yaw']

    # 새로운 데이터셋 CSV로 저장
    df_new.to_csv(output_csv, index=False)

# Example usage
input_csv = 'merged_output_py_200_final3.csv'
output_csv = 'sheco3_200.csv'
convert_dataset(input_csv, output_csv)
