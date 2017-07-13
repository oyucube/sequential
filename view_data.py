# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 04:46:33 2016

@author: oyu
"""
import os
# os.environ["CHAINER_TYPE_CHECK"] = "0" #ここでオフに  オンにしたかったら1にするかコメントアウト
import numpy as np
# 乱数のシード固定
#
# i = np.random()
# np.random.seed()

import argparse
import chainer
from sklearn.datasets import fetch_mldata
from chainer import cuda, serializers
import sys
from tqdm import tqdm
import datetime
from multi_object_model import SAF
import pickle
from chainer import function_hooks
from data.libtraindata import make_train_data, draw_data
import time
import matplotlib.pyplot as plt

model_id = "s09"
# 教師データの読み込み
mnist = fetch_mldata('MNIST original', data_home=".")
with open('data/' + model_id + '.pickle', 'rb') as f:
    dic = pickle.load(f)
data = dic["data"]
num_class = dic["num_class"]

# train_data, train_target = make_train_data(data, mnist.data, num_class, size=112)

data_max = data.shape[0]
img_size = 112
comment = ""
n_target = num_class

# data to img


draw_data(data[115], mnist.data, size=img_size)
draw_data(data[1215], mnist.data, size=img_size)
draw_data(data[5315], mnist.data, size=img_size)
draw_data(data[9414], mnist.data, size=img_size)
draw_data(data[9404], mnist.data, size=img_size)
err = np.zeros(100)
# acc = serializers.load_npz("buf/acc.npy")
# err = np.load("buf/acc.npy")
# for i in range(100):
#     if err[i] == 0:
#         err[i] = None
# plt.plot(err)
# plt.ylim(0, 1)
# plt.savefig("buf/graph.png")
# plt.show()
#
