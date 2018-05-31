#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 20:04:48 2018

@author: icedeath
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


i =322

data = sio.loadmat('mnist_shifted.mat', appendmat=False)
for j in data:
    locals()[j] = data[j]
del data
del j
    
    
xtest=np.squeeze(x_test[i])
xtest0=np.squeeze(x_test0[i])
xtest1=np.squeeze(x_test1[i])
ytest = y_test1[i]

xtrain=np.squeeze(x_train[i])
xtrain0=np.squeeze(x_train0[i])
xtrain1=np.squeeze(x_train1[i])
ytrain = y_train1[i]

print('train_label:', ytrain)
img = np.concatenate([xtrain,xtrain0, xtrain1],axis = 1)*255
plt.figure(1)
plt.imshow(img)

print('test_label:', ytest)
plt.figure(2)
img = np.concatenate([xtest,xtest0, xtest1],axis = 1)*255
plt.imshow(img)