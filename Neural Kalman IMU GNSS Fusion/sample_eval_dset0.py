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
import matplotlib
from math import atan2, pi, sqrt, cos, sin, floor
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

# If running in a headless environment, use Agg backend
matplotlib.use('Agg')

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
from Sheco_dataset.data_utils_0 import *
from traj_utils import *

""" Import Dataset """
window_size = 100
stride = 20

f = 'Sheco_dataset/'
X_train,Y_Pos_train, Physics_Vec_train, x_vel_train, y_vel_train, x0_list_train, y0_list_train, size_of_each_train = import_sheco_dataset_p1(dataset_folder=f, type_flag=1, window_size=window_size, stride=stride)
P = np.repeat(Physics_Vec_train,window_size).reshape((Physics_Vec_train.shape[0],window_size,1))
X_train = np.concatenate((X_train,P),axis=2)

X_test,Y_Pos_test, Physics_Vec_test, x_vel_test, y_vel_test, x0_list_test, y0_list_test, size_of_each_test= import_sheco_dataset_p1(type_flag = 2, dataset_folder=f,window_size=window_size, stride=stride)
P_test = np.repeat(Physics_Vec_test,window_size).reshape((Physics_Vec_test.shape[0],window_size,1))
X_test = np.concatenate((X_test,P_test),axis=2)

""" Evaluation """
model_name = 'Sheco_First_TCN.hdf5'
model = load_model(model_name,custom_objects={'TCN':TCN})

## Unseen Trajectories
ATE = []
RTE = []
ATE_dist = []
RTE_dist = []
for i in range(len(size_of_each_test)):
    Pvx, Pvy = model_pos_generator(X_test, size_of_each_test, 
                   x0_list_test, y0_list_test, window_size, stride,i,model)   
    Gvx, Gvy = GT_pos_generator(x_vel_test,y_vel_test,size_of_each_test,
                                x0_list_test, y0_list_test, window_size, stride,i)
    
    at, rt, at_all, rt_all = Cal_TE(Gvx, Gvy, Pvx, Pvy,
                                    sampling_rate=100,window_size=window_size,stride=stride)
    ATE.append(at)
    RTE.append(rt)
    ATE_dist.append(Cal_len_meters(Gvx, Gvy))
    RTE_dist.append(Cal_len_meters(Gvx, Gvy, 600))
    print('ATE, RTE, Trajectory Length, Trajectory Length (60 seconds)',ATE[i],RTE[i],ATE_dist[i],RTE_dist[i])


    min_len = min(len(Gvx), len(Gvy), len(Pvx), len(Pvy))

    plt.figure()  # Create a new figure
    plt.plot(Gvx[0:min_len], Gvy[0:min_len], label='Ground Truth', linestyle='-')
    plt.plot(Pvx[0:min_len], Pvy[0:min_len], label='Predicted Trajectory', linestyle='-')

    # Add labels and title
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('Trajectory Comparison')
    plt.legend(loc='best')

    # Save the figure as PNG
    plt.savefig('Sheco_dataset/results/trajectory_comparison_test.png', dpi=300)
    print("Figure saved as 'trajectory_comparison_test.png'.")

    # Optional: if you are in an environment where plt.show() is supported
    # plt.show()  # Uncomment this if plt.show() is supported and needed.

    
print('Median ATE and RTE', np.median(ATE),np.median(RTE))
print('Mean ATE and RTE', np.mean(ATE),np.mean(RTE))
print('STD ATE and RTE', np.std(ATE),np.std(RTE))

## Seen Trajectories

ATE = []
RTE = []
ATE_dist = []
RTE_dist = []
for i in range(len(size_of_each_train)):
    Pvx, Pvy = model_pos_generator(X_train, size_of_each_train, 
                   x0_list_train, y0_list_train, window_size, stride,i,model)   
    Gvx, Gvy = GT_pos_generator(x_vel_train,y_vel_train,size_of_each_train,
                                x0_list_train, y0_list_train, window_size, stride,i)
    
    at, rt, at_all, rt_all = Cal_TE(Gvx, Gvy, Pvx, Pvy,
                                    sampling_rate=100,window_size=window_size,stride=stride)

    
print('Median ATE and RTE', np.median(ATE),np.median(RTE))
print('Mean ATE and RTE', np.mean(ATE),np.mean(RTE))
print('STD ATE and RTE', np.std(ATE),np.std(RTE))

## Plot

Pvx, Pvy = model_pos_generator(X_train, size_of_each_train, 
               x0_list_train, y0_list_train, window_size, stride,0,model)   
Gvx, Gvy = GT_pos_generator(x_vel_train,y_vel_train,size_of_each_train,
                            x0_list_train, y0_list_train, window_size, stride,0)

min_len = min(len(Gvx), len(Gvy), len(Pvx), len(Pvy))

plt.figure()  # Create a new figure
plt.plot(Gvx[0:min_len], Gvy[0:min_len], label='Ground Truth', linestyle='-')
plt.plot(Pvx[0:min_len], Pvy[0:min_len], label='Predicted Trajectory', linestyle='-')

# Add labels and title
plt.xlabel('East (m)')
plt.ylabel('North (m)')
plt.title('Trajectory Comparison')
plt.legend(loc='best')

# Save the figure as PNG
plt.savefig('Sheco_dataset/results/trajectory_comparison.png', dpi=300)
print("Figure saved as 'trajectory_comparison.png'.")

# Optional: if you are in an environment where plt.show() is supported
# plt.show()  # Uncomment this if plt.show() is supported and needed.
