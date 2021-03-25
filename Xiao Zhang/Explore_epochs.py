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

model = Sequential()
model.add(Dense(42, input_dim=78, activation='relu'))
model.add(Dense(42, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs=list()
train_acc=list()
test_acc=list()
deltaepochs=1
eon=0
for i in range(0,200) :
        print("epoch : " + str(eon))
        model.fit(train_x, train_y, epochs=deltaepochs, batch_size=256)
        scores = model.evaluate(train_x, train_y)
        test_scores = model.evaluate(test_x, test_y)
        eon+=deltaepochs
        epochs.append(eon)
        train_acc.append(scores[1])
        test_acc.append(test_scores[1])

model.save('epochs_model.h5')

pickle.dump(epochs, open( "epochs.p", "wb" ) )
pickle.dump(train_acc, open( "epochs_train_acc.p", "wb" ) )
pickle.dump(test_acc, open( "epochs_test_acc.p", "wb" ) )
