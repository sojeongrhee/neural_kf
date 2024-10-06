from math import sqrt
import numpy as np
from traj_utils import *
from tqdm import tqdm
import tensorflow as tf

# Magnetometer 노이즈 분산 제거
ACCELEROMETER_NOISE_VARIANCE = 0.00615490459  # (m/s^2)^2
GYROSCOPE_NOISE_VARIANCE = 0.0000030462       # (rad/s)^2
GYROSCOPE_ARW = [3.09, 2.7, 5.4]              # deg/sqrt(hr)
GYROSCOPE_BI = [88.91, 78.07, 211.4]          # deg/hr

# GPS_VELOCITY_NOISE_VARIANCE = 0.0025
GPS_VELOCITY_NOISE_VARIANCE = 0.1
GPS_POSITION_NOISE_VARIANCE = 1.5**2

def kalman_predict(X, P, Q, A, B, G, T):
    X = A @ X + B @ T
    P = A @ P @ np.transpose(A) + G @ Q @ np.transpose(G)
    return X, P

def kalman_update(X, P, z, R, H):
    K = (P @ np.transpose(H)) @ np.linalg.inv(H @ P @ np.transpose(H) + R)
    X = X + K @ (z - H @ X)
    P = P - K @ H @ P
    return X, P

def neural_ekf_gnss_imu_2(net_inp_mat, GT_vel_x, GT_vel_y, size_of_each,
                        x0_list, y0_list, file_idx, window_size, stride,
                        gps_decimation_factor, my_model):

    fused_pos_x = []
    fused_pos_y = []
    dt = stride / (window_size - stride)

    # 가상 GPS 생성
    Gvx_gps, Gvy_gps, Gvx_vel, Gvy_vel, _ = gen_GPS_values_all_traj(
        GT_vel_x, GT_vel_y, size_of_each, x0_list, y0_list,
        window_size, stride, gps_decimation_factor,
        GPS_POSITION_NOISE_VARIANCE, GPS_VELOCITY_NOISE_VARIANCE
    )

    GPS_x = Gvx_gps[file_idx]
    GPS_y = Gvy_gps[file_idx]
    GPS_vel_x = Gvx_vel[file_idx]
    GPS_vel_y = Gvy_vel[file_idx]

    # 현재 트랙의 입력 데이터 선택
    if file_idx == 0:
        cur_inp = net_inp_mat[0:size_of_each[0], :, :]
    else:
        start_idx = np.sum(size_of_each[0:file_idx])
        end_idx = np.sum(size_of_each[0:file_idx + 1])
        cur_inp = net_inp_mat[start_idx:end_idx, :, :]

    X = np.array([x0_list[file_idx], y0_list[file_idx], 0.0, 0.0]).reshape(4, 1)
    A = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    H = np.identity(4)
    B = np.array([
        [dt, 0.0],
        [0.0, dt],
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    R = np.diag([
        GPS_POSITION_NOISE_VARIANCE,
        GPS_POSITION_NOISE_VARIANCE,
        GPS_VELOCITY_NOISE_VARIANCE,
        GPS_VELOCITY_NOISE_VARIANCE
    ])

    # P = 1e-5 * np.zeros((4, 4))
    P = 1e-5 * np.eye(4)
    gps_counter = 1

    # 입력 데이터의 채널 수에 맞게 변수 설정
    num_channels = cur_inp.shape[2]  # 채널 수 조정 (예: 7 또는 6)

    for i in tqdm(range(cur_inp.shape[0])):
        image = tf.cast(cur_inp[i, :, :].reshape(1, cur_inp.shape[1], num_channels), tf.float32)
        with tf.GradientTape(persistent=True) as t:
            t.watch(image)
            pred = my_model(image)
            vx_jacob = pred[0][0]
            vy_jacob = pred[1][0]

        my_grad1 = np.array(t.gradient(vx_jacob, image)).reshape(cur_inp.shape[1], num_channels)
        my_grad2 = np.array(t.gradient(vy_jacob, image)).reshape(cur_inp.shape[1], num_channels)

        T = np.array(pred).flatten().reshape(2, 1)

        # G 행렬 크기 및 내용 수정
        G = np.zeros((4, num_channels))
        for j in range(num_channels):
            grad1_sum = np.sum(np.abs(my_grad1[:, j]))
            grad2_sum = np.sum(np.abs(my_grad2[:, j]))
            total_grad1 = np.sum(np.abs(my_grad1))
            total_grad2 = np.sum(np.abs(my_grad2))
            G[0, j] = dt * grad1_sum / total_grad1
            G[1, j] = dt * grad2_sum / total_grad2
            G[2, j] = grad1_sum / total_grad1
            G[3, j] = grad2_sum / total_grad2

        # Q 행렬 수정
        Q = np.diag([
            ACCELEROMETER_NOISE_VARIANCE,
            ACCELEROMETER_NOISE_VARIANCE,
            ACCELEROMETER_NOISE_VARIANCE,
            ((np.deg2rad(GYROSCOPE_ARW[0]) / 60.0) * sqrt(dt)) ** 2 +
            (np.deg2rad(GYROSCOPE_BI[0]) / 3600.0) ** 2 + GYROSCOPE_NOISE_VARIANCE,
            ((np.deg2rad(GYROSCOPE_ARW[1]) / 60.0) * sqrt(dt)) ** 2 +
            (np.deg2rad(GYROSCOPE_BI[1]) / 3600.0) ** 2 + GYROSCOPE_NOISE_VARIANCE,
            ((np.deg2rad(GYROSCOPE_ARW[2]) / 60.0) * sqrt(dt)) ** 2 +
            (np.deg2rad(GYROSCOPE_BI[2]) / 3600.0) ** 2 + GYROSCOPE_NOISE_VARIANCE,
            3 * ACCELEROMETER_NOISE_VARIANCE  # Physics Vector의 노이즈 분산
        ])

        X, P = kalman_predict(X, P, Q, A, B, G, T)

        if (i % gps_decimation_factor == 0 and gps_counter < len(GPS_x)):
            z = np.array([
                GPS_x[gps_counter],
                GPS_y[gps_counter],
                GPS_vel_x[gps_counter],
                GPS_vel_y[gps_counter]
            ]).reshape(4, 1)
            X, P = kalman_update(X, P, z, R, H)
            gps_counter += 1

        fused_pos_x.append(X[0, 0])
        fused_pos_y.append(X[1, 0])

    return fused_pos_x, fused_pos_y, GPS_x, GPS_y
