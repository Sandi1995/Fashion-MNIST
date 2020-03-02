 # -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:48:24 2019

@author: Sandipan Paul
"""

#import tensorflow as tf
from tensorflow.keras import models,layers
import numpy as np
import gzip
import os

dirname = 'C:\\Users\\691823\\Desktop\\ML\\fashion_MNIST'
#base = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']
#fashion_mnist = keras.datasets.fashion_mnist
paths=[]

for fname in files:
    paths.append(os.path.join(dirname,fname))

with gzip.open(paths[0], 'rb') as lbpath:
    train_label = np.frombuffer(lbpath.read(), np.uint8, offset=8)

with gzip.open(paths[1], 'rb') as imgpath:
    train_image = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(train_label), 28, 28,1)

with gzip.open(paths[2], 'rb') as lbpath:
    test_label = np.frombuffer(lbpath.read(), np.uint8, offset=8)

with gzip.open(paths[3], 'rb') as imgpath:
    test_image = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(test_label), 28, 28,1)

#Linear Regression
# =============================================================================
# train_image,test_image=train_image/255.0,test_image/255.0
# model=keras.Sequential([
#         keras.layers.Flatten(input_shape=(28,28)),
#         keras.layers.Dense(128,activation='relu'),
#         keras.layers.Dense(10,activation='sigmoid')
#         ])
#     
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# 
# model.fit(train_image,train_label,epochs=20)
# 
# test_loss, test_acc = model.evaluate(test_image,  test_label, verbose=2)
# 
# print('\nTest accuracy:', test_acc)
# =============================================================================
    
#CNN model
    

train_image,test_image=train_image/255.0,test_image/255.0

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.Dropout(0.40))
model.add(layers.Flatten())
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

history = model.fit(train_image, train_label, epochs=10, 
                    validation_data=(test_image, test_label))

test_loss, test_acc = model.evaluate(test_image,  test_label, verbose=2)
print('\nTest accuracy:', test_acc)
