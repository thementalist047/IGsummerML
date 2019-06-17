# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import np_utils
np.random.seed(101)
#import matplotlib.pyplot as plt
(train_img , train_labels) , (test_img , test_labels) = mnist.load_data()
Num_pixels = 784
x_train = train_img.reshape(60000 , Num_pixels) 
x_test = test_img.reshape(10000 , Num_pixels)
y_train = np_utils.to_categorical(train_labels)
y_test = np_utils.to_categorical(test_labels)
x_train = x_train/225
x_test = x_test/225 #Normalization
model = Sequential()
model.add(Dense(200 , input_dim = Num_pixels , activation = 'sigmoid'))
model.add(Dense(150 , activation = 'sigmoid'))
model.add(Dense(10 , activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train , y_train , validation_data = (x_test , y_test) , epochs = 20 , batch_size = 100 , verbose = 2)
scores = model.evaluate(x_test , y_test , verbose = 2)
prediction = model.predict(x_test)
print("Test set error : " , 100-scores[1]*100)