import numpy as np
import os,sys
import math

from datafunction import addrandomnoise,delay_series,butter_lowpass_filter
from retrieveData import get_data

import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras import metrics

#init data
data_len = 60*60
train_trials = 14
number_of_trials = 19
fs = 1
cutoff = .5
order = 15
#0 is interior temperature
#1 is exterior temperature 
#5 is timesteps
raw_data = get_data([0,1,5])

for i in range(raw_data.shape[0]): #all trials
    for j in range(0,2): #only temperature readings
        raw_data[i,:,j] = addrandomnoise(raw_data[i,:,j]) #add noise to data for fun
        #raw_data[i,:,j] = butter_lowpass_filter(raw_data[i,:,j],cutoff,fs,order)
        # resize data here 

#new_data = reduce_data_size3d(raw_data, 19*3600)
#new_data = np.random.shuffle(raw_data) 
        
data_len = 3600
scalers = {}
scaled = raw_data
for i in range(scaled.shape[0]):
    scalers[i] = MinMaxScaler(feature_range=(0, 1))
    scaled[i,:, :] = scalers[i].fit_transform(raw_data[i,:,:]) 
#scaled = scaler.fit_transform(raw_data)

for t in range(0,scaled.shape[0]):
    scaled[t,:,:] =  delay_series(scaled[t,:,1:],scaled[t,:,0],5)
local_batch_size = 180
model = Sequential()
model.add(LSTM(10, batch_input_shape=(local_batch_size,  1, scaled[0,:,1:].shape[1]),activation='softsign', stateful=True, return_sequences=False))
model.add(Dropout(0.01))
model.add(Dense(1))
model.add(Activation('linear'))
ad = optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='MSE', optimizer=ad)

for epo in range(0, 50):
    for i in range(0, train_trials):
        
        train = scaled[i,:,:]  #select first trial
        test = scaled[(i+train_trials) % number_of_trials-1,:,:] 
        # split into input and outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        # for p in range(0, train_shape):
            # pyplot.subplot(train_shape, 1, p+1)
            # pyplot.plot(train_X[:,0,p])
        # pyplot.show()

        history = model.fit(train_X, train_y, epochs=1, batch_size=local_batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    model.reset_states()
        # plot history
    #pyplot.plot(history.history['loss'], label='train')
    #pyplot.plot(history.history['val_loss'], label='test')
    #pyplot.legend()
    #pyplot.self.model.reset_states()show()
 
# make a prediction
for i in range(0,number_of_trials+1):
    test = scaled[i,:, :] 
    test_X, test_y = test[:, :-1], test[:, -1]
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    yhat = model.predict(test_X, batch_size = local_batch_size)
    test_shape = test_X.shape[2]

    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    scaler = MinMaxScaler(feature_range=(0,1)).fit(inv_yhat)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    #yhat = model.predict(test_X[data_len*i:data_len*(i+1)-1,:,:])
    pyplot.plot(scaled[i,:, -1]) #select first trial
    pyplot.plot(yhat)
    pyplot.show()