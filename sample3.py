# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 04:46:24 2016

@author: oyu
"""

from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable,cuda
import math
import numpy as np
import time
import make_sampled_image
from env import xp


class SAF(chainer.Chain):
    def __init__(self, n_units=128, n_out=0, img_size=112, var=0.18, gpu_id=-1):
        super(SAF, self).__init__(
            # the size of the inputs to each layer will be inferred
            # glimpse network
            # 切り取られた画像を処理する部分　位置情報 (glimpse loc)と画像特徴量の積を出力
            glimpse_cnn_1=L.Convolution2D(1, 20, 5),  # in 28 out 24
            glimpse_cnn_2=L.Convolution2D(20, 40, 5),  # in 24 out 20
            glimpse_cnn_3=L.Convolution2D(40, 80, 5),  # in 20 out 16
            glimpse_full=L.Linear(8 * 8 * 80, n_units),
            glimpse_loc=L.Linear(2, n_units),

            # baseline network 強化学習の期待値を学習し、バイアスbとする
            baseline=L.Linear(n_units, 1),

            l_norm_c1=L.BatchNormalization(20),
            l_norm_c2=L.BatchNormalization(40),
            l_norm_c3=L.BatchNormalization(80),

            # 記憶を用いるLSTM部分
            rnn_1=L.LSTM(n_units, n_units),
            rnn_2=L.LSTM(n_units, n_units),

            # 注意領域を選択するネットワーク
            attention_loc=L.Linear(n_units, 2),
            attention_scale=L.Linear(n_units, 1),

            # 入力画像を処理するネットワーク
            context_cnn_1=L.Convolution2D(1, 2, 5),  # 56 to 52 pooling: 26
            context_cnn_2=L.Convolution2D(2, 2, 5),  # 26 to 22 pooling
            context_cnn_3=L.Convolution2D(2, 2, 4),  # 22 to 16

            l_norm_cc1=L.BatchNormalization(2),
            l_norm_cc2=L.BatchNormalization(2),
            l_norm_cc3=L.BatchNormalization(2),

            class_full=L.Linear(n_units, n_out)
        )

        #
        # img parameter
        #
        if gpu_id == 0:
            self.use_gpu = True
        else:
            self.use_gpu = False
        self.img_size = img_size
        self.gsize = 20

        self.var = 0.015
        self.vars = 0.015
        self.n_unit = n_units
        self.num_class = n_out
        # r determine the rate of position
        self.r = 0.5

    def reset(self):
        self.rnn_1.reset_state()
        self.rnn_2.reset_state()

    def my_name(self):
        return "grid serch lv 0.015 sv 0.02"

    def __call__(self, x, target, target2, num_lm, batch_size=1, train=1, debug=0, n_step=1):

        if train == 1:
            self.reset()
            r_buf = xp.zeros((num_lm, 1))
            l, s, b = self.first_forward(x, num_lm)
            loss_buf = 0
            for i in range(n_step):
                if i + 1 == n_step:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=1)
                    l1, s1, y, b1 = self.recurrent_forward(xm, lm, sm)
                    loss, r = self.cul_loss(y, target2, l, s, lm, sm, r_buf, b, num_lm)
                    loss_buf += loss
                    return loss_buf / num_lm
                else:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=0)
                    l1, s1, y, b1 = self.recurrent_forward(xm, lm, sm)
                    # loss, r = self.cul_loss(y, target, l, s, lm, sm, r_buf, b, num_lm)
                    # r_buf += r test delete
                    # loss_buf += loss
                l = l1
                s = s1
                b = b1

        elif train == 0:
            self.reset()
            l, s, b1 = self.first_forward(x, num_lm)
            for i in range(n_step):
                if i + 1 == n_step:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=0)
                    l1, s1, y, b = self.recurrent_forward(xm, lm, sm)

                    acc2 = y.data * target2.data
                    return xp.sum(acc1) / num_lm, xp.sum(acc2) / num_lm
                else:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=0)
                    l1, s1, y, b = self.recurrent_forward(xm, lm, sm)
                    acc1 = target.data * y.data
                l = l1
                s = s1
        elif train == 2:
            sum_accuracy = 0
            ydata = xp.zeros((num_lm, self.num_class))
            self.reset()
            l_list = xp.zeros((n_step, num_lm, 2))
            s_list = xp.zeros((n_step, num_lm, 1))
            l, s, b1 = self.first_forward(x, num_lm)
            l_list[0] = l.data
            s_list[0] = s.data
            for i in range(n_step):
                if i + 1 == n_step:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=0)
                    l1, s1, y, b = self.recurrent_forward(xm, lm, sm)

                    accuracy = y.data * target.data
                    sum_accuracy += xp.sum(accuracy)
                    ydata += y.data
                    z = np.power(10, s_list - 1)
                    return sum_accuracy / (num_lm * n_step), ydata / n_step, xp.sum(accuracy, axis=1) / num_lm, l_list, z
                else:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=0)
                    l1, s1, y, b = self.recurrent_forward(xm, lm, sm)
                    accuracy = y.data * target.data
                    sum_accuracy += xp.sum(accuracy)
                    ydata += y.data
                l = l1
                s = s1
                l_list[i + 1] = l.data
                s_list[i + 1] = s.data
            return False

    def first_forward(self, x, num_lm, test=False):
        self.rnn_1(Variable(xp.zeros((num_lm, self.n_unit)).astype(xp.float32)))
        h2 = F.relu(self.l_norm_cc1(self.context_cnn_1(F.max_pooling_2d(x, 2, stride=2))))
        h3 = F.relu(self.l_norm_cc2(self.context_cnn_2(F.max_pooling_2d(h2, 2, stride=2))))
        h4 = F.relu(self.l_norm_cc3(self.context_cnn_3(F.max_pooling_2d(h3, 2, stride=2))))
        h5 = F.relu(self.rnn_2(h4))

        l = F.sigmoid(self.attention_loc(h5))
        s = F.sigmoid(self.attention_scale(h5))
        b = F.relu(self.baseline(Variable(h5.data)))
        return l, s, b

    def recurrent_forward(self, xm, lm, sm, test=False):
        hgl = F.relu(self.glimpse_loc(lm))
        hg1 = F.relu(self.l_norm_c1(self.glimpse_cnn_1(Variable(xm))))
        hg2 = F.relu(self.l_norm_c2(self.glimpse_cnn_2(hg1)))
        hg3 = F.relu(self.l_norm_c3(self.glimpse_cnn_3(hg2)))
        hgf = F.relu(self.glimpse_full(hg3))

        hr1 = F.relu(self.rnn_1(hgl * hgf))
        # ベクトルの積
        hr2 = F.relu(self.rnn_2(hr1))
        l = F.sigmoid(self.attention_loc(hr2))
        s = F.sigmoid(self.attention_scale(hr2))
        y = F.softmax(self.class_full(hr1))
        b = F.relu(self.baseline(Variable(hr2.data)))
        return l, s, y, b

    # loss 関数を計算

    def cul_loss(self, y, target, l, s, lm, sm, r_buf, b, num_lm):

        zm = xp.power(10, sm.data - 1)

        l1, l2 = F.split_axis(l, indices_or_sections=2, axis=1)
        m1, m2 = F.split_axis(lm, indices_or_sections=2, axis=1)
        ln_p = ((l1 - m1) * (l1 - m1) + (l2 - m2) * (l2 - m2)) / self.var / zm / zm / 2
        # size
        size_p = (sm - s) * (sm - s) / self.vars + ln_p

        accuracy = y * target

        loss = -F.sum(accuracy)

        r = xp.where(
            xp.argmax(y.data, axis=1) == xp.argmax(target.data, axis=1), 1, 0).reshape((num_lm, 1)).astype(xp.float32)
        r += r_buf
        loss += F.sum((r - b) * (r - b))
        loss_m = self.r * (r - b.data)
        loss += F.sum(Variable(loss_m) * size_p)
        return loss, r

    def make_img(self, x, l, s, num_lm, random=0):
        if random == 0:
            lm = Variable(xp.clip(l.data, 0, 1))
            sm = Variable(xp.clip(s.data, 0, 1))
        else:
            eps = xp.random.normal(0, 1, size=l.data.shape).astype(xp.float32)
            epss = xp.random.normal(0, 1, size=s.data.shape).astype(xp.float32)
            sm = xp.clip((s.data + xp.sqrt(self.var) * epss), 0, 1).astype(xp.float32)
            lm = xp.clip(l.data + xp.power(10, sm - 1) * eps * xp.sqrt(self.vars), 0, 1)
            sm = Variable(sm)
            lm = Variable(lm.astype(xp.float32))
        if self.use_gpu:
            xm = make_sampled_image.generate_xm_gpu(lm.data, sm.data, x.data, num_lm, g_size=self.gsize)
        else:
            xm = make_sampled_image.generate_xm(lm.data, sm.data, x.data, num_lm, g_size=self.gsize)
        return xm, lm, sm
