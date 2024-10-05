import os
import warnings
warnings.filterwarnings('ignore')

# TensorFlow의 로깅 레벨을 설정하여 불필요한 경고를 숨깁니다.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from gtda.time_series import SlidingWindow
import matplotlib.pyplot as plt
from math import atan2, pi, sqrt, cos, sin, floor

import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPooling1D, Flatten, Input
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model
from tcn import TCN
from sklearn.model_selection._split import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import csv
import random
import itertools
import math
import time

from Sheco_dataset.data_utils_0 import *
from traj_utils import *

# TensorFlow 2.x에서는 세션 설정이 필요하지 않습니다.
# GPU 메모리 증가 허용이나 디바이스 설정은 자동으로 처리됩니다.

""" 데이터셋 로드 """
window_size = 100
stride = 20

f = 'Sheco_dataset/'
X_train, Y_Pos_train, Physics_Vec_train, x_vel_train, y_vel_train, \
x0_list_train, y0_list_train, size_of_each_train = import_sheco_dataset_p1(
    dataset_folder=f, type_flag=1, window_size=window_size, stride=stride)

P = np.repeat(Physics_Vec_train, window_size).reshape(
    (Physics_Vec_train.shape[0], window_size, 1))
X_train = np.concatenate((X_train, P), axis=2)

X_test, Y_Pos_test, Physics_Vec_test, x_vel_test, y_vel_test, \
x0_list_test, y0_list_test, size_of_each_test = import_sheco_dataset_p1(
    type_flag=2, dataset_folder=f, window_size=window_size, stride=stride)

P_test = np.repeat(Physics_Vec_test, window_size).reshape(
    (Physics_Vec_test.shape[0], window_size, 1))
X_test = np.concatenate((X_test, P_test), axis=2)

""" 모델 학습 """
nb_filters = 32
kernel_size = 5
dilations = [1, 2, 4, 8, 16, 32, 64, 128]
dropout_rate = 0.0
use_skip_connections = True

batch_size, timesteps, input_dim = 256, window_size, X_train.shape[2]
inputs = Input(shape=(timesteps, input_dim))

x = TCN(nb_filters=nb_filters, kernel_size=kernel_size, dilations=dilations,
        dropout_rate=dropout_rate, use_skip_connections=use_skip_connections)(inputs)

x = tf.keras.layers.Reshape((nb_filters, 1))(x)
x = MaxPooling1D(pool_size=2)(x)
x = Flatten()(x)
x = Dense(32, activation='linear', name='pre')(x)
output1 = Dense(1, activation='linear', name='velx')(x)
output2 = Dense(1, activation='linear', name='vely')(x)

model = Model(inputs=inputs, outputs=[output1, output2])
model.compile(loss={'velx': 'mse', 'vely': 'mse'}, optimizer='adam')
model.summary()

model_name = 'Sheco_First_TCN.hdf5'

# 모델 체크포인트 콜백
checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True)

# TensorBoard 콜백 설정
log_dir = "logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# 모델 학습
history = model.fit(x=X_train, y=[x_vel_train, y_vel_train], epochs=4000, shuffle=True,
                    callbacks=[checkpoint, tensorboard_callback], batch_size=batch_size)

# 학습 과정에서의 loss 그리기
plt.plot(history.history['loss'], label='Total Loss')
plt.plot(history.history['velx_loss'], label='velx_loss')
plt.plot(history.history['vely_loss'], label='vely_loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('logs/training_loss_with_tensorboard.png')  # 학습 중 loss 그래프 저장
# plt.show()

""" 모델 평가 """
model = load_model(model_name, custom_objects={'TCN': TCN})

## TensorBoard 실행 방법 안내
print("To view the logs in TensorBoard, run the following command in your terminal:")
print(f"tensorboard --logdir={log_dir}")

## 새로운 궤적에 대한 평가
ATE = []
RTE = []
ATE_dist = []
RTE_dist = []
for i in range(len(size_of_each_test)):
    Pvx, Pvy = model_pos_generator(X_test, size_of_each_test,
                                   x0_list_test, y0_list_test, window_size, stride, i, model)
    Gvx, Gvy = GT_pos_generator(x_vel_test, y_vel_test, size_of_each_test,
                                x0_list_test, y0_list_test, window_size, stride, i)

    at, rt, at_all, rt_all = Cal_TE(Gvx, Gvy, Pvx, Pvy,
                                    sampling_rate=100, window_size=window_size, stride=stride)
    ATE.append(at)
    RTE.append(rt)
    ATE_dist.append(Cal_len_meters(Gvx, Gvy))
    RTE_dist.append(Cal_len_meters(Gvx, Gvy, 600))
    print('ATE, RTE, Trajectory Length, Trajectory Length (60 seconds):',
          ATE[i], RTE[i], ATE_dist[i], RTE_dist[i])

print('Median ATE and RTE:', np.median(ATE), np.median(RTE))
print('Mean ATE and RTE:', np.mean(ATE), np.mean(RTE))
print('STD ATE and RTE:', np.std(ATE), np.std(RTE))

## 학습된 궤적에 대한 평가
ATE = []
RTE = []
ATE_dist = []
RTE_dist = []
for i in range(len(size_of_each_train)):
    Pvx, Pvy = model_pos_generator(X_train, size_of_each_train,
                                   x0_list_train, y0_list_train, window_size, stride, i, model)
    Gvx, Gvy = GT_pos_generator(x_vel_train, y_vel_train, size_of_each_train,
                                x0_list_train, y0_list_train, window_size, stride, i)

    at, rt, at_all, rt_all = Cal_TE(Gvx, Gvy, Pvx, Pvy,
                                    sampling_rate=100, window_size=window_size, stride=stride)
    ATE.append(at)
    RTE.append(rt)
    ATE_dist.append(Cal_len_meters(Gvx, Gvy))
    RTE_dist.append(Cal_len_meters(Gvx, Gvy, 600))
    print('ATE, RTE, Trajectory Length, Trajectory Length (60 seconds):',
          ATE[i], RTE[i], ATE_dist[i], RTE_dist[i])

print('Median ATE and RTE:', np.median(ATE), np.median(RTE))
print('Mean ATE and RTE:', np.mean(ATE), np.mean(RTE))
print('STD ATE and RTE:', np.std(ATE), np.std(RTE))

## 결과 플롯

Pvx, Pvy = model_pos_generator(X_train, size_of_each_train,
                               x0_list_train, y0_list_train, window_size, stride, 0, model)
Gvx, Gvy = GT_pos_generator(x_vel_train, y_vel_train, size_of_each_train,
                            x0_list_train, y0_list_train, window_size, stride, 0)

plt.figure()
plt.plot(Gvx[0:2000], Gvy[0:2000], label='Ground Truth')
plt.plot(Pvx[0:2000], Pvy[0:2000], label='Predicted')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectory')
plt.savefig('Sheco_dataset/results/trajectory.png')
# plt.show()

print('Trajectory Length:', Cal_len_meters(Gvx[0:2250], Gvy[0:2250]))
