# -*- coding: utf-8 -*-
import numpy as np
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense , Dropout , Flatten , Conv2D , MaxPooling2D , Activation
from keras.utils import to_categorical

Batch_Size = 100
Epochs = 50
Num_labels = 100
Dropout_rate = 0.4

(x_train , y_train) , (x_test , y_test) = cifar100.load_data(label_mode = 'fine')
y_train = to_categorical(y_train , Num_labels)
y_test = to_categorical(y_test , Num_labels)

x_train = x_train/255 #Normalization
x_test = x_test/255

model = Sequential()
model.add(Conv2D(filters = 512 , kernel_size = 3 , padding = 'same' , input_shape = x_train.shape[1:]))
model.add(Conv2D(filters = 256 , kernel_size = 3 , padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(Dropout_rate))

model.add(Conv2D(filters = 512 , kernel_size = 3 , padding = 'same'))
model.add(Conv2D(filters = 256 , kernel_size = 3 , padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(Dropout_rate))
 
model.add(Flatten())

model.add(Dense(2000 , activation = 'relu'))
model.add(Dense(800 , activation = 'relu'))
model.add(Dense(100 , activation = 'softmax'))

model.summary()
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.fit(x_train , y_train , validation_data = [x_test , y_test] , batch_size= Batch_Size , epochs = Epochs)
scores = model.evaluate(x_test, y_test)
print("Test data error percentage : " , 100-scores[1]*100)
