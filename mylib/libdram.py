# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:41:35 2016

@author: waka-lab
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
fp = FontProperties(fname='C:/Users/waka-lab/.matplotlib/font/ipaexg.ttf')

def draw_heatmap(data, row_labels, column_labels):
    # 描画する

    plt.set_context("poster",1.0)
    plt.heatmap(data,annot=True,annot_kws={"size": 10},fmt='.2g', cmap='Blues')
    plt.show

    return

def draw_g(array):
     plt.plot(array)
     plt.show()
     return
    
    


def draw_train_digit(data):
    size = 56
    plt.figure(figsize=(2.5, 3))

    X, Y = np.meshgrid(range(size),range(size))
    Z = data.reshape(size,size)   
    Z = Z[::-1,:]             # flip vertical
    plt.xlim(0,55)
    plt.ylim(0,55)
    plt.pcolor(X, Y, Z)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

    plt.show()

def draw_digit(data,size):
    plt.figure(figsize=(2.5, 3))

    X, Y = np.meshgrid(range(size),range(size))
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]             # flip vertical
    plt.xlim(0,size -1)
    plt.ylim(0,size -1)
    plt.pcolor(X, Y, Z)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

    plt.show()
    
def load_data(filename):
    td = np.load('traindata/' + filename+'train_data.npy').astype(np.float32)
    tt = np.load('traindata/' + filename+'train_target.npy').astype(np.float32)
    testd = np.load('traindata/' + filename+'test_data.npy').astype(np.float32)
    testt = np.load('traindata/' + filename+'test_target.npy').astype(np.float32)
    return td,tt,testd,testt
    
def dic_to_data(dic):
    td = dic['train_data'].astype(np.float32)
    tt = dic['train_target'].astype(np.float32)
    testd = dic['test_data'].astype(np.float32)
    testt = dic['test_target'].astype(np.float32)
    return td,tt,testd,testt