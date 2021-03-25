import numpy as np
import pandas as pd
import csv
import pickle
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras import losses


train = pd.read_csv("CICIDS2017_Wed_train.zip",header=None)
test = pd.read_csv("CICIDS2017_Wed_test.zip",header=None)
train_x = train.drop([78],axis=1)
train_y = train[78]
test_x = test.drop([78],axis=1)
test_y = test[78]


neuron=list()
train_acc=list()
test_acc=list()
deltaepochs=1
n=6

for i in range(0,13) :
        model = Sequential()
        model.add(Dense(n, input_dim=78, activation='relu'))
        model.add(Dense(n, activation='relu'))
        model.add(Dense(6, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("neuron : " + str(n))
        model.fit(train_x, train_y, epochs=10, batch_size=256)
        scores = model.evaluate(train_x, train_y)
        test_scores = model.evaluate(test_x, test_y)
        neuron.append(n)
        train_acc.append(scores[1])
        test_acc.append(test_scores[1])
        n+=6


pickle.dump(neuron, open( "neuron.p", "wb" ) )
pickle.dump(train_acc, open( "neuron_train_acc.p", "wb" ) )
pickle.dump(test_acc, open( "neuron_test_acc.p", "wb" ) )
