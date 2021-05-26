# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:51:46 2021

@author: biela
"""
#import pydot_ng as pydot

from keras.utils.vis_utils import plot_model
from vgg import *
from easy_cnn import *
#model =  VGG('VGG19')
#model =  Face_CNN()
#model = Sequential()
#plot_model(model, to_file='model.png')

import os
import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
 
print('tf version: {}'.format(tf.__version__))
 
 
# Returns a short sequential model
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
 
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
 
    return model
 
 
# Create a basic model instance
model = create_model()
plot_model(model, './model.bmp', show_shapes=True)
