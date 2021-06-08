import numpy as np
import tensorflow as tf
import random as rn
import os
import json
import mat73

from keras import metrics, regularizers, optimizers, backend
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, DepthwiseConv2D, SeparableConv2D, Flatten, pooling, AveragePooling2D
from keras.utils import np_utils, vis_utils
from tensorflow.keras.models import Sequential

import scipy.io as sio

# ===================================== Fix random seed ======================================================
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(2021)
rn.seed(2021)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=4)
tf.random.set_seed(2021)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
backend.set_session(sess)

L = 500
nClass = 13
train_model = 31

# ===================================== Load Train Data ======================================================
x_data_mat = mat73.loadmat('train_data.mat')
x_data_complex = x_data_mat['train_data']
x_data_real = x_data_complex.real
x_data_imag = x_data_complex.imag
x_data_real = x_data_real.reshape((x_data_real.shape[0], L, 1))
x_data_imag = x_data_imag.reshape((x_data_imag.shape[0], L, 1))
x_train = np.stack((x_data_real, x_data_imag), axis=1)
y_data_mat = mat73.loadmat('train_label.mat')
y_data = y_data_mat['train_label']
y_train = np_utils.to_categorical(y_data, nClass)

# ===================================== Train Data Shuffle ======================================================
index = np.arange(y_train.shape[0])
np.random.shuffle(index)
x_train = x_train[index,:]
y_train = y_train[index]

print(x_train.shape[1], x_train.shape[2])
_in_ = Input(shape = (x_train.shape[1], x_train.shape[2], 1))

if train_model == 0:
    ot = Conv2D(filters=64, kernel_size=(2,4), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = Conv2D(filters=16, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    #ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    ot = Dense(16, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 1:
    ot = Conv2D(filters=256, kernel_size=(1,3), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    #ot = Conv2D(filters=256, kernel_size=(2,3), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=80, kernel_size=(1,3), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=80, kernel_size=(1,3), strides=1, padding='valid', use_bias=True, activation='relu')(ot)    
    ot = Flatten()(ot)

    ot = Dense(128, use_bias=True, activation='relu')(ot)
    _out_ = Dense(13, use_bias=True, activation='softmax')(ot)

elif train_model == 2:
    ot = Conv2D(filters=64, kernel_size=(2,8), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = Conv2D(filters=32, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    #ot = AveragePooling2D()(ot)
    ot = Conv2D(filters=16, kernel_size=(1,3), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    #ot = AveragePooling2D()(ot)
    ot = Flatten()(ot)

    #ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(84, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 3:
    ot = Conv2D(filters=64, kernel_size=(2,8), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = Conv2D(filters=32, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    #ot = Conv2D(filters=16, kernel_size=(1,3), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(84, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 4:
    ot = Conv2D(filters=64, kernel_size=(2,8), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = DepthwiseConv2D(kernel_size=(1,4), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    #ot = Conv2D(filters=16, kernel_size=(1,3), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(84, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 5:
    ot = Conv2D(filters=64, kernel_size=(2,8), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=32, kernel_size=(1,4), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    #ot = Conv2D(filters=16, kernel_size=(1,3), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(84, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 6:
    ot = SeparableConv2D(filters=64, kernel_size=(2,8), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=32, kernel_size=(1,4), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    #ot = Conv2D(filters=16, kernel_size=(1,3), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(84, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 7:
    ot = DepthwiseConv2D(kernel_size=(2,8), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=32, kernel_size=(1,4), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    #ot = Conv2D(filters=16, kernel_size=(1,3), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(84, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 8:
    ot = Conv2D(filters=64, kernel_size=(2,8), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=32, kernel_size=(1,4), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    #ot = Conv2D(filters=16, kernel_size=(1,3), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    ot = Dense(16, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 9:
    ot = Conv2D(filters=128, kernel_size=(2,8), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,4), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    #ot = Conv2D(filters=16, kernel_size=(1,3), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(160, use_bias=True, activation='relu')(ot)
    ot = Dense(100, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 10:
    ot = Conv2D(filters=128, kernel_size=(2,16), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,16), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=8, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 11:
    ot = Conv2D(filters=128, kernel_size=(2,16), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,8), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=8, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 12:
    ot = Conv2D(filters=128, kernel_size=(2,8), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,8), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=8, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 13:
    ot = Conv2D(filters=128, kernel_size=(2,4), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,4), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=8, kernel_size=(1,2), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 14:
    ot = Conv2D(filters=128, kernel_size=(2,4), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,4), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=16, kernel_size=(1,2), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 15:
    ot = Conv2D(filters=128, kernel_size=(2,4), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,4), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=16, kernel_size=(1,2), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = SeparableConv2D(filters=8, kernel_size=(1,2), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 16:
    ot = Conv2D(filters=128, kernel_size=(2,4), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,4), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=16, kernel_size=(1,2), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = SeparableConv2D(filters=8, kernel_size=(1,2), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(84, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 17:
    ot = Conv2D(filters=128, kernel_size=(2,4), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,4), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=16, kernel_size=(1,2), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=8, kernel_size=(1,2), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(84, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 18:
    ot = Conv2D(filters=128, kernel_size=(2,16), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,16), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=8, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    ot = Dense(32, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 19:
    ot = Conv2D(filters=128, kernel_size=(2,16), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,16), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=8, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 20:
    ot = Conv2D(filters=128, kernel_size=(2,16), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,16), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=16, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 21:
    ot = Conv2D(filters=128, kernel_size=(2,16), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,8), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=8, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 22:
    ot = Conv2D(filters=128, kernel_size=(2,8), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,8), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=8, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 23:
    ot = Conv2D(filters=128, kernel_size=(2,4), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,8), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=8, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 24:
    ot = Conv2D(filters=128, kernel_size=(2,4), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,8), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=8, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 25:
    ot = Conv2D(filters=128, kernel_size=(2,4), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,8), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=8, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 26:
    ot = Conv2D(filters=128, kernel_size=(2,4), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,8), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Conv2D(filters=8, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 27:
    ot = Conv2D(filters=128, kernel_size=(2,4), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = SeparableConv2D(filters=64, kernel_size=(1,8), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Conv2D(filters=8, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 28:
    ot = Conv2D(filters=128, kernel_size=(2,4), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,8), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=16, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 29:
    ot = Conv2D(filters=128, kernel_size=(2,4), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,8), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=32, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 30:
    ot = Conv2D(filters=128, kernel_size=(2,4), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,8), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=32, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=16, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)

elif train_model == 31:
    ot = Conv2D(filters=128, kernel_size=(2,4), strides=1, padding='valid', use_bias=True, activation='relu')(_in_)
    ot = SeparableConv2D(filters=64, kernel_size=(1,8), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=32, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = SeparableConv2D(filters=32, kernel_size=(1,4), strides=1, padding='valid', depth_multiplier=1, use_bias=True, activation='relu')(ot)
    ot = Conv2D(filters=16, kernel_size=(1,4), strides=1, padding='valid', use_bias=True, activation='relu')(ot)
    ot = Flatten()(ot)

    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(120, use_bias=True, activation='relu')(ot)
    ot = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(ot)
    ot = Dense(64, use_bias=True, activation='relu')(ot)
    _out_ = Dense(nClass, activation='softmax')(ot)



model = Model(_in_, _out_)

tensor_board = TensorBoard(log_dir='./tensorboard_log', histogram_freq=0, write_graph=True, write_images=False,
                            embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
               optimizer=adam, 
               metrics=['categorical_accuracy'])
model.fit(x_train,
          y_train, 
          epochs=500, 
          batch_size=100,
          validation_split=0.1,
          shuffle=True,
          callbacks=[tensor_board, early_stopping])
scores = model.evaluate(x_train, y_train)
print(" %s %f" % (model.metrics_names[1], scores[1]))
model.summary()

with open('model_struct_model' + str(train_model) + '.json', 'w') as f:
    json.dump(model.to_json(), f)    
model.save_weights('model_weights_model' + str(train_model) + '.h5')