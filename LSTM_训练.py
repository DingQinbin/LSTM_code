#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Author: DingQinbin
# @Time  : 2020/10/26 9:39

import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import Callback
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import pandas as pd
import os
import keras.callbacks
import matplotlib.pyplot as plt

# 设定为自增长
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)


def create_dataset(data, n_predictions, n_next):
    '''
    对数据进行处理
    '''
    dim = data.shape[1]
    train_X, train_Y = [], [] 
    for i in range(data.shape[0] - n_predictions - n_next - 1):
        a = data[i:(i + n_predictions), :]
        train_X.append(a)
        tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
        b = []
        for j in range(len(tempb)):
            for k in range(dim):
                b.append(tempb[j, k])
        train_Y.append(b)
    train_X = np.array(train_X, dtype='float64')
    train_Y = np.array(train_Y, dtype='float64')

    test_X, test_Y = [], []
    i = data.shape[0] - n_predictions - n_next - 1
    a = data[i:(i + n_predictions), :]
    test_X.append(a)
    tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
    b = []
    for j in range(len(tempb)):
        for k in range(dim):
            b.append(tempb[j, k])
    test_Y.append(b)
    test_X = np.array(test_X, dtype='float64')
    test_Y = np.array(test_Y, dtype='float64')

    return train_X, train_Y, test_X, test_Y


def NormalizeMult(data, set_range):
    '''
    返回归一化后的数据和最大最小值
    '''
    normalize = np.arange(2 * data.shape[1], dtype='float64')
    normalize = normalize.reshape(data.shape[1], 2)

    for i in range(0, data.shape[1]):
        if set_range == True:
            list = data[:, i]
            listlow, listhigh = np.percentile(list, [0, 100])
        else:
            if i == 0:
                listlow = -90
                listhigh = 90
            else:
                listlow = -180
                listhigh = 180

        normalize[i, 0] = listlow
        normalize[i, 1] = listhigh

        delta = listhigh - listlow
        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - listlow) / delta

    return data, normalize


def trainModel(train_X, train_Y):
    '''
    trainX，trainY: 训练LSTM模型所需要的数据
    '''
    model = Sequential()
    model.add(LSTM(
        120,
        input_shape=(train_X.shape[1], train_X.shape[2]),
        return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(
        120,
        return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(
        train_Y.shape[1]))
    model.add(Activation("relu"))

    model.compile(loss='mse', optimizer='adam', metrics=['acc'])
    model.fit(train_X, train_Y, epochs=100, batch_size=64, verbose=1)
    model.summary()

    return model


if __name__ == "__main__":
    train_num = 6
    per_num = 1
    # set_range = False
    set_range = True

    # 读入时间序列的文件数据
    data = pd.read_csv('20080403010747.txt', sep=',').iloc[:, 0:2].values
    print("样本数：{0}，维度：{1}".format(data.shape[0], data.shape[1]))
    # print(data)

    # 画样本数据库
    plt.scatter(data[:, 1], data[:, 0], c='b', marker='o', label='traj_A')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

    # 归一化
    data, normalize = NormalizeMult(data, set_range)
    # print(normalize)

    # 生成训练数据
    train_X, train_Y, test_X, test_Y = create_dataset(data, train_num, per_num)
    print("x\n", train_X.shape)
    print("y\n", train_Y.shape)

    # 训练模型
    model = trainModel(train_X, train_Y)
    loss, acc = model.evaluate(train_X, train_Y, verbose=2)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))

    # 保存模型
    np.save("./traj_model_trueNorm.npy", normalize)
    model.save("./traj_model_120.h5")
