import numpy as np
from numpy import argmax
import pandas as pd
import csv
import pickle
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras import losses
from keras.layers import Dropout
import tensorflow as tf


train = pd.read_csv("CICIDS2017_Wed_train.zip",header=None)
test = pd.read_csv("CICIDS2017_Wed_test.zip",header=None)
train_x = train.drop([78],axis=1)
train_y = train[78]
test_x = test.drop([78],axis=1)
test_y = test[78]


model = Sequential()
model.add(Dense(78, input_dim=78, activation='relu'))
model.add(Dense(78, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(6, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
model.fit(train_x, train_y, epochs=100, batch_size=256)


pred = model.predict(test_x)
pred_test_y = pred.argmax(axis=1)


model.save("pred_model.h5")
pickle.dump(pred_test_y, open( "pred_test_y.p", "wb" ) )
