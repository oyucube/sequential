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
from data.libtraindata import make_sequential_train_data
import time
import importlib
from data.libdraw import draw_attention, draw_digit
import make_sampled_image
#  input
#
xp = np
file_id = "v2"
model_id = "m69"
model_file_name = "sample"
num_step = 4

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


train_var = 0.02
gpu_id = -1
sss = importlib.import_module(model_file_name)
# モデルの作成
model = sss.SAF(n_out=n_target, img_size=img_size, var=train_var, gpu_id=gpu_id)
# model load
serializers.load_npz('model/' + file_id + model_id + '.model', model)

test_b = int(data_max * 0.2)

test_b = 300
perm = np.random.permutation(data_max)

# data to img
xx, tt, t2 = make_sequential_train_data(data[perm[0:test_b]], mnist.data, num_class, size=112)
xx = xx.reshape(test_b, 1, img_size, img_size).astype(xp.float32)
tt = tt.astype(xp.float32)
x = chainer.Variable(xp.asarray(xx), volatile="off")
t = chainer.Variable(xp.asarray(tt), volatile="off")
t2 = chainer.Variable(xp.asarray(t2), volatile="off")
# 順伝播
acc, y, lac, l, s = model(x, t, t2, test_b, train=2, n_step=num_step)
# 描画を書く
print(acc)
print(np.sum(lac))
s0 = np.log10(s[0]) + 1
g = make_sampled_image.generate_xm(l[0], s0, xx, test_b, 20, img_size=112)
for i in range(20):
    # print(y[i])
    print(lac[i] * test_b)
    # print(l[:, i] * 112)
    # print(s[:, i] * 112)
    # print(s[:, i])
    # print(data[perm[i]][3])
    # draw_digit(xx[i], img_size)
    # save_name = ("buf/{}".format(i))
    for j in range(num_step):
        save_name = ("buf/{}{}".format(i, j))
        draw_attention(xx[i], 112, l[[j], i], s[[j], i], view_double=False, save=save_name)
    # draw_attention(xx[i], 112, l[:, i], s[:, i], view_double=True, save=save_name)
    # draw_digit(g[i], 20)
