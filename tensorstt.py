# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 08:43:17 2019

@author: Weihan
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
import pickle
layers = keras.layers
models = keras.models


# This code was tested with TensorFlow v1.8
print("You have TensorFlow version", tf.__version__)


data = pd.read_csv("datalog.csv")

user_text = 'give me your card number'

predict_data = pd.DataFrame([[user_text]], columns = ['text'])
#predict_data = pd.read_csv("test.csv")

train_size = int(len(data) * .8)
print ("Train size: %d" % len(data))
print ("Test size: %d" % len(predict_data))

def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test

train_cat, train_text = data['category'], data['text']
test_text = predict_data['text']

max_words = 1000
tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words,char_level=False)

tokenize.fit_on_texts(train_text) # fit tokenizer to our training text data
x_train = tokenize.texts_to_matrix(train_text)
x_test = tokenize.texts_to_matrix(test_text)

# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(train_cat)
y_train = encoder.transform(train_cat)


# Converts the labels to a one-hot representation
num_classes = np.max(y_train) + 1
y_train = keras.utils.to_categorical(y_train, num_classes)

# Inspect the dimenstions of our training and test data (this is helpful to debug)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)

# This model trains very quickly and 2 epochs are already more than enough
# Training for more epochs will likely lead to overfitting on this dataset
# You can try tweaking these hyperparamaters when using this model with your own data
batch_size = 32
epochs = 2
drop_ratio = 0.5

# Build the model
model = models.Sequential()
model.add(layers.Dense(512, input_shape=(max_words,)))
model.add(layers.Activation('relu'))
# model.add(layers.Dropout(drop_ratio))
model.add(layers.Dense(num_classes))
model.add(layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model.fit trains the model
# The validation_split param tells Keras what % of our training data should be used in the validation set
# You can see the validation loss decreasing slowly when you run this
# Because val_loss is no longer decreasing we stop training to prevent overfitting
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)


def run_experiment(batch_size, epochs, drop_ratio):
    print('batch size: {}, epochs: {}, drop_ratio: {}'.format(batch_size, epochs, drop_ratio))
    model = models.Sequential()
    model.add(layers.Dense(512, input_shape=(max_words,)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(drop_ratio))
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_split=0.1)
    
batch_size = 16
epochs = 4
drop_ratio = 0.4
run_experiment(batch_size, epochs, drop_ratio)

# Here's how to generate a prediction on individual examples
text_labels = encoder.classes_ 

for i in range(0,1):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
    print(test_text.iloc[i][:50], "...")
    print("Predicted label: " + predicted_label + "\n") 
    
model.save('cs-model_nlp.h5')