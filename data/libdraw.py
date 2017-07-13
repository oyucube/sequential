# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 20:05:52 2016

@author: oyu
"""

from sklearn.datasets import fetch_mldata
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
from itertools import combinations


def afin(a, b, x):
    k = b/a
    r = np.empty((a, a))

    for i in range(0, a):
        for j in range(0, a):
            ax = int(k*i)
            ay = int(k*j)
            r[i][j] = x[ax][ay]
    return r


# s = 0~1 not log
def draw_attention(data, size, l, s, view_double=False, save=""):
    plt.figure(figsize=(2.5, 3))

    x, y = np.meshgrid(range(size), range(size))
    z = data.reshape(size, size)   # convert from vector to 28x28 matrix
    z = z[::-1, :]             # flip vertical
    plt.xlim(0, size - 1)
    plt.ylim(0, size - 1)
    plt.pcolor(x, y, z)
    plt.gray()
    draw_window(plt, size, l[0], s[0], 'r')
    if view_double:
        draw_window(plt, size, l[1], s[1], 'y')
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")
    if len(save) > 0:
        plt.savefig(save + ".png")
    plt.show()


def draw_window(plt, size, l, s, color):
    p1 = (size * (l - s / 2)).astype(np.int32)
    p2 = (size * (l + s / 2)).astype(np.int32)
    p1[0] = size - p1[0]
    p2[0] = size - p2[0]
    plt.plot([p1[1], p1[1]], [p1[0], p2[0]], color=color, lw=2)
    plt.plot([p2[1], p2[1]], [p1[0], p2[0]], color=color, lw=2)
    plt.plot([p1[1], p2[1]], [p1[0], p1[0]], color=color, lw=2)
    plt.plot([p1[1], p2[1]], [p2[0], p2[0]], color=color, lw=2)


def draw_digit(data, size):
    plt.figure(figsize=(2.5, 3))

    x, y = np.meshgrid(range(size), range(size))
    z = data.reshape(size, size)   # convert from vector to 28x28 matrix
    z = z[::-1, :]             # flip vertical
    plt.xlim(0, size - 1)
    plt.ylim(0, size - 1)
    plt.pcolor(x, y, z)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")
    plt.show()


def data_to_img(data, mnist, size=112):
    black = np.zeros((size, size))
    img1 = mnist[int(data[0])].reshape(28, 28)
    x = int(data[1] * size)
    y = int(data[2] * size)
    size_n1 = int(data[3])
    black[x:x + size_n1, y:y + size_n1] = afin(size_n1, 28, img1)

    if data[4] > -1:
        img2 = mnist[int(data[4])].reshape(28, 28)
        x = int(data[5] * size)
        y = int(data[6] * size)
        size_n1 = int(data[7])
        black[x:x + size_n1, y:y + size_n1] = afin(size_n1, 28, img2)
    return black


def draw_data(data, mnist, size):
    draw_digit(data_to_img(data, mnist, size), size)


def make_train_data(data, mnist, num_class, size=112):
    data_max = data.shape[0]
    train_data = np.empty((data_max, size, size))
    train_target = np.empty((data_max, num_class))
    for i in range(data_max):
        train_data[i] = data_to_img(data[i], mnist, size)
        train_target[i] = one_of_k(num_class, data[i][8])
    return train_data, train_target


def one_of_k(num_class, c):
    r = np.zeros(num_class)
    r[int(c)] = 1
    return r
