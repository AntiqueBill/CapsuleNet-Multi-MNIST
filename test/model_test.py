#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 09:56:06 2018

@author: icedeath
"""
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.layers import Lambda
import matplotlib.pyplot as plt
import tensorflow as tf
from capsulelayers2 import CapsuleLayer, PrimaryCap, Length, Mask
from keras import callbacks
import argparse
import scipy.io as sio
'''
n_class = 10
input_shape=x_train.shape[1:]

x = layers.Input(shape=input_shape)
conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', 
                      activation='relu', name='conv1')(x)
primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9,
                         strides=2, padding='valid')
digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=3,
                         name='digitcaps')(primarycaps)
out_caps = Length(name='capsnet')(digitcaps)

y = layers.Input(shape=(2,))

masked_by_y = Mask()([digitcaps, y])
masked_by_y0 =  Lambda(lambda x: x[:,0,:])(masked_by_y)
masked_by_y1 =  Lambda(lambda x: x[:,1,:])(masked_by_y)

masked = Mask()(digitcaps)
masked0 =  Lambda(lambda x: x[:,0,:])(masked)
masked1 =  Lambda(lambda x: x[:,1,:])(masked)

decoder = models.Sequential(name='decoder')
decoder.add(layers.Dense(512, activation='sigmoid', input_dim=16))
decoder.add(layers.Dense(1024, activation='sigmoid'))
decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

train_model = models.Model([x, y], [out_caps, decoder(masked_by_y0), decoder(masked_by_y1)])
eval_model = models.Model(x, [out_caps, decoder(masked0), decoder(masked1)])

train_model.load_weights('caps2.h5')
'''
i=322
y_pred, x_recon0, x_recon1 = eval_model.predict(np.expand_dims(x_test[i], 0))
_, y_pred1 = tf.nn.top_k(y_pred[0], 2)
y_pred1 = K.eval(y_pred1)
plt.figure()
img = np.concatenate([np.squeeze(x_test[i]),np.squeeze(x_recon0), np.squeeze(x_recon1)],axis = 1)*255
plt.imshow(img)




