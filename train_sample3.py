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
import pickle
from data.libtraindata import make_sequential_train_data
import importlib

#  引数分解
#    
parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, default=1,
                    help="batch size")
parser.add_argument("-e", "--epoch", type=int, default=50,
                    help="iterate training given epoch times")
parser.add_argument("-d", "--data", type=int, default=-1,
                    help="data name")
parser.add_argument("-m", "--num_l", type=int, default=40,
                    help="a number of sample ")
parser.add_argument("-s", "--step", type=int, default=2,
                    help="look step")
parser.add_argument("-v", "--var", type=float, default=0.02,
                    help="sample variation")
parser.add_argument("-g", "--gpu", type=int, default=-1,
                    help="use gpu")
# train id
parser.add_argument("-i", "--id", type=str, default="sample",
                    help="data id")
parser.add_argument("-a", "--am", type=str, default="sample",
                    help="attention model")
# load model id
parser.add_argument("-l", "--l", type=str, default="",
                    help="load model name")

# model save id
parser.add_argument("-o", "--filename", type=str, default="v2",
                    help="prefix of output file names")
args = parser.parse_args()

file_id = args.filename
model_id = args.id
n_data = args.data
num_lm = args.num_l
n_epoch = args.epoch
train_id = args.id
num_step = args.step
train_b = args.batch_size
train_var = args.var
gpu_id = args.gpu

xp = cuda.cupy if gpu_id >= 0 else np

# 教師データの読み込み
mnist = fetch_mldata('MNIST original', data_home=".")
with open('data/' + model_id + '.pickle', 'rb') as f:
    dic = pickle.load(f)
data = dic["data"]
num_class = dic["num_class"]
num_class = 10
# target_c = dic["target_combinations"]
target_c = ""
target_c = ""
train_data, train_target, train_target2 = make_sequential_train_data(data, mnist.data, num_class, size=112)

data_max = data.shape[0]
img_size = 112
comment = ""
n_target = num_class

# 仮設定　train = test
test_data = train_data
test_target = train_target
num_testdata = data_max
target_number = dic["target_number"]

train_data = train_data.reshape(data_max, 1, img_size, img_size).astype(xp.float32)
test_data = test_data.reshape(num_testdata, 1, img_size, img_size).astype(xp.float32)
train_target = train_target.astype(xp.float32)
test_target = test_target.astype(xp.float32)
test_target2 = test_target.astype(xp.float32)
test_b = 1000

model_file_name = args.am
sss = importlib.import_module(model_file_name)
# モデルの作成
model = sss.SAF(n_out=n_target, img_size=img_size, var=train_var, gpu_id=gpu_id)
# model load
if len(args.l) != 0:
    print("load model model/my{}{}.model".format(args.l, model_id))
    serializers.load_npz('model/my' + args.l + model_id + '.model', model)

# オプティマイザの設定
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

# gpuの設定
if gpu_id >= 0:
    cuda.get_device(gpu_id).use()
    model.to_gpu()

# ログの設定　精度、エラー率
if n_data == -1:
    n_data = int(data_max)
acc1_array = np.zeros(n_epoch)
acc2_array = np.zeros(n_epoch)
max_acc = 0
date_id = datetime.datetime.now().strftime("%m%d%H%M")
log_dir = "log/" + train_id + file_id + date_id
os.mkdir(log_dir)
out_file_name = log_dir + "/log"

log_filename = out_file_name + '.txt'
f = open(log_filename, 'w')
f.write("{} class recognition\nclass:{} use {} data set".format(num_class, target_c, model_id))
f.write("model:{}".format(model_file_name))
f.write("parameter")
f.write("step:{}\nnum_sample:{} \nbatch_size{}\nvar:{}".format(num_step, num_lm, train_b, train_var))
f.write("log dir:{}".format(out_file_name))
f.write("going to train {} epoch".format(n_epoch))
f.close()  # ファイルを閉じる

print("{} class recognition\nclass:{} use {} data set".format(num_class, target_c, model_id))
print("model:{}".format(model_file_name))
print("parameter")
print("step:{} num_sample:{} batch_size:{} var:{}".format(num_step, num_lm, train_b, train_var))
print("log dir:{}".format(out_file_name))
print("going to train {} epoch".format(n_epoch))

#
# 訓練開始
#


for epoch in range(n_epoch):
    sys.stdout.write("(epoch: {})\n".format(epoch + 1))
    sys.stdout.flush()
    #   学習    
    perm = np.random.permutation(data_max)
    for i in tqdm(range(0, n_data - test_b, train_b), ncols=60):
        x = chainer.Variable(xp.asarray(xp.tile(train_data[perm[i:i+train_b]], (num_lm, 1, 1, 1))), volatile="off")
        t = chainer.Variable(xp.asarray(xp.tile(train_target[perm[i:i+train_b]], (num_lm, 1))), volatile="off")
        t2 = chainer.Variable(xp.asarray(xp.tile(train_target2[perm[i:i+train_b]], (num_lm, 1))), volatile="off")
        # 順伝播
        model.cleargrads()
        loss_func = 0
        # loss_func += model(x, t, t, num_lm * train_b, batch_size=train_b, train=1, n_step=1)
        loss_func += model(x, t2, t2, num_lm * train_b, batch_size=train_b, train=1, n_step=2)

        loss_func.backward()
        loss_func.unchain_backward()  # truncate
        optimizer.update()

    # evaluate
    x = chainer.Variable(xp.asarray(test_data[perm[n_data - test_b:n_data]]), volatile="off")
    t = chainer.Variable(xp.asarray(test_target[perm[n_data - test_b:n_data]]), volatile="off")
    t2 = chainer.Variable(xp.asarray(test_target2[perm[n_data - test_b:n_data]]), volatile="off")
    # 順伝播
    model.reset()
    acc1, acc2 = model(x, t, t2, test_b, train=0, n_step=num_step)

    # 分類精度の記録
    acc1_array[epoch] = acc1
    acc2_array[epoch] = acc2
    print("acc1:{:1.4f} acc2:{:1.4f}".format(acc1_array[epoch], acc2_array[epoch]))

    best = ""
    if acc1 < max_acc:
        max_acc = acc1
        best = "best"
    # 分類精度の保存
    with open(log_filename, mode='a') as fh:
        fh.write("acc1:{:1.4f} acc2:{:1.4f}\n".format(acc1_array[epoch], acc2_array[epoch]))

    np.save(log_dir + "/acc1.npy", acc1_array)
    np.save(log_dir + "/acc2.npy", acc2_array)
    # モデルの保存
    if gpu_id >= 0:
        model.to_cpu()
        serializers.save_npz(log_dir + "/" + best + file_id + train_id + '.model', model)
        model.to_gpu()
    else:
        serializers.save_npz(log_dir + "/" + best + file_id + train_id + '.model', model)

with open(log_filename, mode='a') as fh:
    fh.write("last acc:{}  max_acc:{}\n".format(acc1_array[n_epoch - 1], max_acc))
