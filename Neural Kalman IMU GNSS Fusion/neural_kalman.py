import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pandas as pd
import numpy as np
from gtda.time_series import SlidingWindow
import matplotlib.pyplot as plt
from math import atan2, pi, sqrt, cos, sin, floor
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True  
config.log_device_placement = True  
sess2 = tf.compat.v1.Session(config=config)
set_session(sess2)  
from tensorflow.keras.layers import Dense, MaxPooling1D, Flatten
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.compat.v1.keras.backend as K
from tensorflow.keras.models import load_model
from tcn import TCN, tcn_full_summary
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform
from keras_flops import get_flops
import pickle
import csv
import random
import itertools
import math
import time
import sys
# sys.path.append('/home/sjrhee/sjrhee/neural_kf_0926/sheco/sheco\ Dataset/')

from Sheco_dataset.data_utils_0 import *
from traj_utils import *
# from neural_ekf_2 import *
from neural_ekf import *

# Create the 'results' directory if it doesn't exist
if not os.path.exists('Sheco_dataset/results'):
    os.makedirs('Sheco_dataset/results')

window_size = 100
stride = 20

f = 'Sheco_dataset/'
X_train,Y_Pos_train, Physics_Vec_train, x_vel_train, y_vel_train, x0_list_train, y0_list_train, size_of_each_train = import_sheco_dataset_p1(dataset_folder=f, type_flag=1, window_size=window_size, stride=stride)
P = np.repeat(Physics_Vec_train,window_size).reshape((Physics_Vec_train.shape[0],window_size,1))
X_train = np.concatenate((X_train,P),axis=2)

X_test,Y_Pos_test, Physics_Vec_test, x_vel_test, y_vel_test, x0_list_test, y0_list_test, size_of_each_test= import_sheco_dataset_p1(type_flag = 2, dataset_folder=f,window_size=window_size, stride=stride)
P_test = np.repeat(Physics_Vec_test,window_size).reshape((Physics_Vec_test.shape[0],window_size,1))
X_test = np.concatenate((X_test,P_test),axis=2)
# print("X_test : ", X_test, len(X_test))
model = load_model('Sheco_First_TCN.hdf5', custom_objects={'TCN':TCN})

ATE = []
RTE = []
for i in range(len(size_of_each_train)):

    # fused_pos_x, fused_pos_y, GPS_x, GPS_y =  neural_ekf_gnss_imu_2(X_train, x_vel_train,y_vel_train, 
    #             size_of_each_train,
    #             x0_list_train, y0_list_train,i,window_size,stride,60*5,
    #              model)
    fused_pos_x, fused_pos_y, GPS_x, GPS_y =  neural_ekf_gnss_imu(X_train, x_vel_train,y_vel_train, 
                size_of_each_train,
                x0_list_train, y0_list_train,i,window_size,stride,60*5,
                 model)
    act_x,act_y =  GT_pos_generator(x_vel_train, y_vel_train, size_of_each_train, 
                   x0_list_train, y0_list_train, window_size, stride,i)
    
    
        
    at, rt, at_all, rt_all = Cal_TE(act_x, act_y, fused_pos_x, fused_pos_y,
                                    sampling_rate=100,window_size=window_size,stride=stride)

    ATE.append(at)
    RTE.append(rt)
    print('ATE, RTE:',ATE[i],RTE[i])

    a = 0
    b = 3000
    plt.figure()  # Create a new figure
    plt.plot(act_x[a:b],act_y[a:b],label='Ground Truth',linestyle='-')
    plt.plot(fused_pos_x[a:b],fused_pos_y[a:b],label='Neurl-KF',linestyle='-')
    plt.scatter(GPS_x[math.ceil(a/5):math.ceil(b/5)],GPS_y[math.ceil(a/5):math.ceil(b/5)],
                marker='.',label='GPS only')
    plt.xlim([-50,15])
    plt.ylim([-50,20])
    plt.grid('minor')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('Phase 1, with GPS')
    plt.legend(loc='best')

    # Save the plot in 'results' folder
    plt.savefig(f'Sheco_dataset/results/phase1_with_GPS_{i}.png', dpi=300)
    print(f"Plot saved in 'results/phase1_with_GPS_{i}.png'.")

    # Optionally display the plot (useful if not in a headless environment)
    # plt.show()

    
print('Median ATE and RTE', np.median(ATE),np.median(RTE))
print('Mean ATE and RTE', np.mean(ATE),np.mean(RTE))
print('STD ATE and RTE', np.std(ATE),np.std(RTE))

"""test dataset plot"""

ATE = []
RTE = []
for i in range(len(size_of_each_test)):

    # fused_pos_x, fused_pos_y, GPS_x, GPS_y =  neural_ekf_gnss_imu_2(X_test, x_vel_test,y_vel_test, 
    #             size_of_each_test,
    #             x0_list_test, y0_list_test,i,window_size,stride,5,
    #              model)
    fused_pos_x, fused_pos_y, GPS_x, GPS_y =  neural_ekf_gnss_imu(X_test, x_vel_test,y_vel_test, 
                size_of_each_test,
                x0_list_test, y0_list_test,i,window_size,stride,5,
                 model)
    act_x,act_y =  GT_pos_generator(x_vel_test, y_vel_test, size_of_each_test, 
                   x0_list_test, y0_list_test, window_size, stride,i)
    
    
        
    at, rt, at_all, rt_all = Cal_TE(act_x, act_y, fused_pos_x, fused_pos_y,
                                    sampling_rate=100,window_size=window_size,stride=stride)

    ATE.append(at)
    RTE.append(rt)
    print('test ATE, test RTE:',ATE[i],RTE[i])

    a = 0
    b = 3000
    plt.figure()  # Create a new figure
    plt.plot(act_x[a:b],act_y[a:b],label='Ground Truth',linestyle='-')
    plt.plot(fused_pos_x[a:b],fused_pos_y[a:b],label='Neurl-KF',linestyle='-')
    plt.scatter(GPS_x[math.ceil(a/5):math.ceil(b/5)],GPS_y[math.ceil(a/5):math.ceil(b/5)],
                marker='.',label='GPS only')
    plt.xlim([-50,15])
    plt.ylim([-50,20])
    plt.grid('minor')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('Phase 1, with GPS')
    plt.legend(loc='best')

    # Save the plot in 'results' folder
    plt.savefig(f'Sheco_dataset/results/phase1_with_GPS_test_{i}.png', dpi=300)
    print(f"plot saved in 'results/phase1_with_GPS_test_{i}.png'.")

    # Optionally display the plot (useful if not in a headless environment)
    # plt.show()

    
print('Median ATE and RTE', np.median(ATE),np.median(RTE))
print('Mean ATE and RTE', np.mean(ATE),np.mean(RTE))
print('STD ATE and RTE', np.std(ATE),np.std(RTE))



