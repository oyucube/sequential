# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 20:05:52 2016

@author: oyu
"""

from sklearn.datasets import fetch_mldata
import numpy as np
import pickle
import libtraindata
from libtraindata import draw_digit
import math
from itertools import combinations



def main():
    mnist = fetch_mldata('MNIST original', data_home=".")
    numin = 70000
    out_item = 9

    mnist_label = np.load("./mnist/label.npy")
    mnist_count = np.load("./mnist/count.npy")

# s single n no scale
# ここで出力を指定
    trainsize = 112    
    file_id = "s09"
    target_number = "0123456789"
    num_output_data = 2000
    testout = 50
    comment = ""
##
##
    min_size = 20
    max_size = 80
    # sizer サイズ

    count_size = np.zeros(100)

    target_ar = np.array((list(target_number))).astype(np.int32)
    num_class = target_ar.shape[0]
    print("target {}\n num class:{}".format(target_ar, num_class))
    print("count {}".format(mnist_count))
    output_data = np.zeros((num_output_data * num_class, out_item))
    data_max = num_output_data * num_class

    # output_data:
    #    num1 num2
    # [i][0]  [4] index of MNIST
    #    [1]  [5] x position
    #    [2]  [6] y position
    #    [3]  [7] size
    #
    #         [8] class label

    for i in range(num_class):
        print(i)
        n1 = mnist_count[target_ar[i]]

        rn1 = np.random.randint(0, n1, num_output_data)

        for j in range(num_output_data):
            id = num_output_data * i + j

            size = np.random.randint(min_size, max_size, (2, 1))
            position = (1 - size / 112) * np.random.rand(2, 2)

            output_data[id][0] = mnist_label[target_ar[i]][rn1[j]]
            output_data[id][1] = position[0][0]
            output_data[id][2] = position[0][1]
            output_data[id][3] = size[0][0]
            output_data[id][4] = -1
            output_data[id][8] = i

    for i in range(data_max):
        count_size[int(output_data[i][3])] += 1
        count_size[int(output_data[i][7])] += 1
    print(count_size)

    dic = {
        "data": output_data,
        "num_class": num_class,
        "target_number": target_number
    }

    with open(file_id + '.pickle', 'wb') as f:
        pickle.dump(dic, f)

    libtraindata.draw_data(output_data[0], mnist.data, 112)
    p = np.random.permutation(data_max)
    for i in range(10):
        libtraindata.draw_data(output_data[p[i]], mnist.data, 112)

if __name__ == '__main__':
    main()
    