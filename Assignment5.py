# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.datasets import mnist
import keras.layers as kl
from keras.utils import np_utils
np.random.seed(101)
import matplotlib.pyplot as plt
(train_img , train_labels) , (test_img , test_labels) = mnist.load_data()
Num_pixels = 784
y_train = np_utils.to_categorical(train_labels)
y_test = np_utils.to_categorical(test_labels)
x_train = train_img.astype(np.float32)/225
x_test = test_img.astype(np.float32)/225  #Normalization
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1) #Adding channel width(1 in this case(greyscale))
model = Sequential()
model.add(kl.Conv2D(filters = 32 , kernel_size = 3 , padding = 'same', input_shape = (28,28,1)))
model.add(kl.Activation('relu'))
model.add(kl.MaxPooling2D(pool_size = 2))
model.add(kl.Conv2D(filters = 64 , kernel_size = 3 , padding = 'same' , activation = 'relu' ))
model.add(kl.MaxPooling2D(pool_size = 2))
model.add(kl.Flatten())
model.add(kl.Dense(500 , activation = 'relu'))
model.add(kl.Dense(10 , activation = 'softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train , y_train , validation_data = (x_test , y_test) , epochs = 4 , batch_size = 60)
scores = model.evaluate(x_test , y_test , verbose = 2)
print("Test set error percentage : " , 100-scores[1]*100)