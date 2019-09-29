# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:39:24 2019

@author: Weihan
"""

import speech_recognition as sr

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
layers = keras.layers
models = keras.models

data = pd.read_csv("datalog.csv")
train_cat, train_text = data['category'], data['text']
max_words = 1000
tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words,char_level=False)
tokenize.fit_on_texts(train_text) # fit tokenizer to our training text data
x_train = tokenize.texts_to_matrix(train_text)
# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(train_cat)
y_train = encoder.transform(train_cat)
# Converts the labels to a one-hot representation
num_classes = np.max(y_train) + 1
y_train = keras.utils.to_categorical(y_train, num_classes)
model = keras.models.load_model('cs-model_nlp.h5')
# Here's how to generate a prediction on individual examples
text_labels = encoder.classes_ 

print('start...')




r = sr.Recognizer()
run = True
while(run):
    with sr.Microphone() as source:
        print('speak')
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)    
    
    try:
        user_text = r.recognize_google(audio)
        print('processing...')
        predict_data = pd.DataFrame([[user_text]], columns = ['text'])
        test_text = predict_data['text']
        x_test = tokenize.texts_to_matrix(test_text)
        prediction = model.predict(np.array([x_test[0]]))
        predicted_label = text_labels[np.argmax(prediction)]
        print(test_text.iloc[0][:50])
        print("Predicted: " + predicted_label + "\n") 
        
        if 'exit' in user_text:
            run = False
    except: 
        print('no aud')

    