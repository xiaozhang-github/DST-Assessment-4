import numpy as np
import pandas as pd
import csv
import pickle
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras import losses
from keras.layers import Dropout


train = pd.read_csv("CICIDS2017_Wed_train.zip",header=None)
test = pd.read_csv("CICIDS2017_Wed_test.zip",header=None)
train_x = train.drop([78],axis=1)
train_y = train[78]
test_x = test.drop([78],axis=1)
test_y = test[78]


dropout_rate = [0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
train_acc=list()
test_acc=list()

for i in range(0,11) :
        model = Sequential()
        model.add(Dense(78, input_dim=78, activation='relu'))
        model.add(Dense(78, activation='relu'))
        model.add(Dropout(dropout_rate[i]))
        model.add(Dense(6, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("dropout_rate : " + str(dropout_rate[i]))
        model.fit(train_x, train_y, epochs=25, batch_size=256)
        scores = model.evaluate(train_x, train_y)
        test_scores = model.evaluate(test_x, test_y)
        train_acc.append(scores[1])
        test_acc.append(test_scores[1])


pickle.dump(train_acc, open( "dropout_train_acc.p", "wb" ) )
pickle.dump(test_acc, open( "dropout_test_acc.p", "wb" ) )
