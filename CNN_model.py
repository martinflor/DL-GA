# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 21:47:20 2022

@author: Florian Martin

"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D,MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical


def create_model():

    model = Sequential()
    # Add the convolutional layers
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(Flatten())
    
    # Add the dropout regularization layer
    model.add(Dropout(0.5))
    
    # Add the dense output layer
    model.add(Dense(10, activation='softmax'))
    
    return model
    
    
from keras.datasets import fashion_mnist

# Load the data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Convert the labels to one-hot encoded format
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Reshape the data to have a single channel
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

# Convert the data to floating-point type and normalize it
X_train = X_train.astype('float32') / 255
X_test  = X_test.astype('float32') / 255


model = create_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
