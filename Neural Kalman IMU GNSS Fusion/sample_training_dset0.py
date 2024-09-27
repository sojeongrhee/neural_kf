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
from Sheco_dataset.data_utils_0 import *
from traj_utils import *

""" Import Dataset """
window_size = 100
stride = 20

f = 'Sheco_dataset/'
X_train,Y_Pos_train, Physics_Vec_train, x_vel_train, y_vel_train, x0_list_train, y0_list_train, size_of_each_train = import_agrobot_dataset_p1(dataset_folder=f, type_flag=1, window_size=window_size, stride=stride)
P = np.repeat(Physics_Vec_train,window_size).reshape((Physics_Vec_train.shape[0],window_size,1))
X_train = np.concatenate((X_train,P),axis=2)

X_test,Y_Pos_test, Physics_Vec_test, x_vel_test, y_vel_test, x0_list_test, y0_list_test, size_of_each_test= import_agrobot_dataset_p1(type_flag = 2, dataset_folder=f,window_size=window_size, stride=stride)
P_test = np.repeat(Physics_Vec_test,window_size).reshape((Physics_Vec_test.shape[0],window_size,1))
X_test = np.concatenate((X_test,P_test),axis=2)

""" Model Training """
nb_filters = 32
kernel_size = 5
dilations = [1,2,4,8,16,32,64,128]
dropout_rate = 0.0
use_skip_connections = True

batch_size, timesteps, input_dim = 256, window_size, X_train.shape[2]
i = Input(shape=(timesteps, input_dim))

m = TCN(nb_filters=nb_filters,kernel_size=kernel_size,dilations=dilations,dropout_rate=dropout_rate,
            use_skip_connections=use_skip_connections)(i)  

m = tf.reshape(m, [-1, nb_filters, 1])
m = MaxPooling1D(pool_size=(2))(m)
m = Flatten()(m)
m = Dense(32, activation='linear', name='pre')(m)
output1 = Dense(1, activation='linear', name='velx')(m)
output2 = Dense(1, activation='linear', name='vely')(m)
model = Model(inputs=[i], outputs=[output1, output2])
opt = tf.keras.optimizers.Adam()
model.compile(loss={'velx': 'mse','vely':'mse'},optimizer=opt)  
model.summary()

model_name = 'Sheco_First_TCN.hdf5'
checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True)
model.fit(x=X_train, y=[x_vel_train, y_vel_train],epochs=3000, shuffle=True,callbacks=[checkpoint],batch_size=batch_size)     

""" Evaluation """
#model_name = 'Sheco_First_TCN.hdf5'
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
    ATE.append(at)
    RTE.append(rt)
    ATE_dist.append(Cal_len_meters(Gvx, Gvy))
    RTE_dist.append(Cal_len_meters(Gvx, Gvy, 600))
    print('ATE, RTE, Trajectory Length, Trajectory Length (60 seconds)',ATE[i],RTE[i],ATE_dist[i],RTE_dist[i])
    
print('Median ATE and RTE', np.median(ATE),np.median(RTE))
print('Mean ATE and RTE', np.mean(ATE),np.mean(RTE))
print('STD ATE and RTE', np.std(ATE),np.std(RTE))

## Plot

Pvx, Pvy = model_pos_generator(X_train, size_of_each_train, 
               x0_list_train, y0_list_train, window_size, stride,0,model)   
Gvx, Gvy = GT_pos_generator(x_vel_train,y_vel_train,size_of_each_train,
                            x0_list_train, y0_list_train, window_size, stride,0)

plt.plot(Gvx[0:2000],Gvy[0:2000])
plt.plot(Pvx[0:2000],Pvy[0:2000])

print(Cal_len_meters(Gvx[0:2250],Gvy[0:2250]))