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
    out_item = 10

    mnist_label = np.load("./mnist/label.npy")
    mnist_count = np.load("./mnist/count.npy")

# ここで出力を指定
    trainsize = 112    
    file_id = "test"
    target_number = "0123456789"
    num_output_data = 3
    testout = 50
    comment = ""
##
##
    # target_number


    min_size = 20
    max_size = 80
    # sizer サイズ

    count_size = np.zeros(100)

    target_combinations = np.array((list(combinations(target_number, 2)))).astype(np.int32)
    num_class = target_combinations.shape[0]
    print("target {}\n num class:{}".format(target_combinations, num_class))
    # print("count {}".format(mnist_count))
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
    #         [9] class label

    for i in range(num_class):
        n1 = mnist_count[target_combinations[i][0]]
        n2 = mnist_count[target_combinations[i][1]]

        rn1 = np.random.randint(0, n1, num_output_data)
        rn2 = np.random.randint(0, n2, num_output_data)

        for j in range(num_output_data):
            id = num_output_data * i + j
            print(id)
            while(True):
                size = np.random.randint(min_size, max_size, (2, 1))
                print(size)
                position = (1 - size / 112) * np.random.rand(2, 2)
                center = position + size / 112 / 2

                test = np.abs(center[0] - center[1]) - np.sum(size / 112) / 2
                t = np.sum(np.sign(test))
                if t > -2:
                    output_data[id][0] = mnist_label[target_combinations[i][0]][rn1[j]]
                    output_data[id][1] = position[0][0]
                    output_data[id][2] = position[0][1]
                    output_data[id][3] = size[0][0]
                    output_data[id][4] = mnist_label[target_combinations[i][1]][rn2[j]]
                    output_data[id][5] = position[1][0]
                    output_data[id][6] = position[1][1]
                    output_data[id][7] = size[1][0]
                    if size[0][0] > size[1][0]:
                        output_data[id][8] = target_combinations[i][0]
                        output_data[id][9] = target_combinations[i][1]
                    else:
                        output_data[id][8] = target_combinations[i][1]
                        output_data[id][9] = target_combinations[i][0]
                    break

    for i in range(data_max):
        count_size[int(output_data[i][3])] += 1
        count_size[int(output_data[i][7])] += 1
    # print(count_size)

    dic = {
        "data": output_data,
        "num_class": num_class,
        "target_number": target_number,
        "target_combinations": target_combinations
    }

    with open(file_id + '.pickle', 'wb') as f:
        pickle.dump(dic, f)
    print(output_data[1])
    print(output_data[2])
    libtraindata.draw_data(output_data[0], mnist.data, 112)
    libtraindata.draw_data(output_data[10], mnist.data, 112)
if __name__ == '__main__':
    main()
