from sklearn.datasets import make_classification
import pandas as pd
#import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
import csv
#import urllib2
import pandas as pd
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from sklearn.cross_validation import train_test_split
import numpy
from tensorflow.keras.models import Sequential
#from tensorflow.scikeras.wrappers import KerasClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

seed=7
numpy.random.seed(seed)

train=pd.read_csv("CICIDS2017_Wed_train.zip",header=None)
test=pd.read_csv("CICIDS2017_Wed_test.zip",header=None)

train75,train25=train_test_split(train,test_size=0.25,random_state=42)

train75x=train75.drop([78],axis=1)

train25x=train25.drop([78],axis=1)

train25y=train25[78]

testx=test.drop([78],axis=1)

n_inputs=train75x.shape[1]

ini=Input(shape=(n_inputs,))

e=Dense(n_inputs*2)(ini)
e=BatchNormalization()(e)
e=LeakyReLU()(e)


e=Dense(n_inputs)(e)
e=BatchNormalization()(e)
e=LeakyReLU()(e)

n_bottleneck=n_inputs
bottleneck=Dense(n_bottleneck)(e)

d=Dense(n_inputs)(bottleneck)
d=BatchNormalization()(d)
d=LeakyReLU()(d)

d=Dense(n_inputs*2)(d)
d=BatchNormalization()(d)
d=LeakyReLU()(d)

output=Dense(n_inputs,activation='linear')(d)

automodel=Model(inputs=ini,outputs=output)

automodel.compile(optimizer='adam',loss='mse')

history=automodel.fit(train75x,train75x,epochs=1,batch_size=1,verbose=2,validation_data=(testx,testx))

encoder=Model(inputs=ini,outputs=bottleneck)

encoder.save('encoder.h5')

#encoder=load_model('encoder.h5')

train25xen=encoder.predict(train25x)

# Function to create model, required for KerasClassifier
def create_model(optimizer='rmsprop'):
    #def create_model(optimizer='rmsprop', init='glorot_uniform'):
    # create model
    model = Sequential()
    model.add(Dense(78, input_dim=78, activation='relu'))
    model.add(Dense(32,  activation='relu'))
    model.add(Dense(6,  activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

train25y = train25[78]
# create model
model = KerasClassifier(build_fn=create_model)
# grid search epochs, batch size and optimizer
optimizers = ['rmsprop', 'adam']
#init = ['glorot_uniform', 'normal', 'uniform']
epochs = numpy.array([10, 20, 30])
batches = numpy.array([50, 100, 150])
#param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(train25xen, train25y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))