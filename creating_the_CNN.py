import pickle 
import time 
import os 
import cv2
import random 
import matplotlib.pyplot as plt 
import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout
from tensorflow.keras.callbacks import TensorBoard

NAME = f'plant leaf good or bad predictor-{int(time.time())}'
tb = TensorBoard(log_dir=f'./logs')

pickle_in = open('X.pkl', 'rb')
X = pickle.load(pickle_in)

pickle_in = open('Y.pkl', 'rb')
Y = pickle.load(pickle_in)


model = Sequential()

model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))
model.add(Dense(15,activation = 'softmax'))
# model.add(Dropout(rate = 0.1,seed=100)) gotta learn how to use this

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(X,Y, epochs =6, validation_split = 0.1) #lower epochs had higher val_acc i think but ill have to see

model.save('64x3-CNN.model')
