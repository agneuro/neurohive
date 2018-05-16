import numpy as np
import matplotlib.pyplot as plt
rd
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.constraints import maxnorm

path0 = "/home/m/salad/data"

# the data, shuffled and split between tran and test sets
X_train, Y_train = np.load(path0+'trainsetx.npy'), np.load(path0+'trainsety.npy')
X_test, Y_test = np.load(path0+'testsetx.npy'), np.load(path0+'testsety.npy')
print("X_train original shape", X_train.shape)
print("y_train original shape", Y_train.shape)

#X_train = X_train.reshape(len(X_train), 400, 400, 3)
#X_test = X_test.reshape(len(X_test), 400, 400, 3)
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(400, 400, 3), padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(48, (3, 3), activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(24, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

adam = keras.optimizers.Adam(lr=0.0001)
model.compile(loss='mean_squared_error', optimizer=adam)

model.load_weights('weights')
model.fit(X_train, Y_train,
          batch_size=8, epochs=400,
          verbose=1,
          validation_data=(X_test, Y_test))


score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score)

# save everything
open('model','w').write(model.to_yaml())
model.save_weights('weights')
