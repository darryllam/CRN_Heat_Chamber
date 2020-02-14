import math
import numpy as np
#import scipy.io
import os, sys
from mat4py import loadmat
import random

from scipy.signal import butter, lfilter,freqz

import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Dropout


def sigmoid(x,rise):
    #print(x)
    return 1 / (1 + rise/10 * math.exp(-1/(rise/20)*(x)))

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

def convertfile(infile):
    mat = loadmat(infile)
    if(infile.find('outer') > 0):
    #    print(infile.find('outer'))
        mat = np.array(mat['outT'])
    else:
        mat = np.array(mat['centerT'])
    return mat

def addrandomnoise(array):
    return array + np.random.randn(len(array))

def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = False)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def delay_series(data, data_to_predict, delay_time):
    #pass data as data np array 
    len_row = data.shape[0]
    len_col = data.shape[1]
    

    for td in range(0,delay_time):
        for i in range(0, len_col):
                #extend all Cols
                data = np.insert(data, 0, data[0,:],axis = 0) 
                data = np.delete(data, -1, axis = 0)
    data = np.column_stack((data, data_to_predict))

    return data


#init data
data_len = 60*60+1
number_of_trials = 0
train_trials = 3
fs = 1
cutoff = .05
order = 15
res_init_temp = []
res_external = []
res_internal = []
res_rise_temp = []
res_final_temp = []
time_arr = []

data_folder = os.getcwd() + "/data/"
for filename in os.listdir(data_folder):
    if(filename == "outer_folder"):
        continue 
    #print(filename)   
    if(int(filename[12:14]) != 20):
        continue
    if(int(filename[16:-4]) != 100):
        continue
    number_of_trials = number_of_trials + 1
    folder_filename_center = data_folder + filename
    folder_filename_outer = data_folder + "outer_folder/" +"outer" + filename[6:]
    result_data_center = convertfile(folder_filename_center)
    result_data_outer = convertfile(folder_filename_outer)
    result_data_center_noise = addrandomnoise(result_data_center)
    result_data_outer_noise = addrandomnoise(result_data_outer)
    res_init_temp.extend([filename[12:14] for x in range(60*60+1)])
    res_rise_temp.extend([filename[14:16] for x in range(60*60+1)])
    res_final_temp.extend([filename[16:-4] for x in range(60*60+1)])
    res_internal.extend(result_data_center_noise)
    res_external.extend(result_data_outer_noise)
    time_arr.extend([x/60 for x in range(60*60+1)])

raw_data = np.array(res_internal, ndmin = 1)
raw_data = np.column_stack((raw_data, np.array(res_external, ndmin = 1)))
#raw_data = np.column_stack((raw_data, np.array(res_init_temp, ndmin = 1)))
#raw_data = np.column_stack((raw_data, np.array(res_rise_temp, ndmin = 1)))
#raw_data = np.column_stack((raw_data, np.array(res_final_temp, ndmin = 1)))
raw_data = np.column_stack((raw_data, np.array(time_arr, ndmin = 1)))
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(raw_data)

# pyplot.plot(train_X[:,:,5])
# pyplot.subplot(6,1,6)
for t in range(0,number_of_trials):
    scaled[(data_len*t):(data_len*(t+1)),0] = butter_lowpass_filter(scaled[(data_len*t):(data_len*(t+1)),0] ,cutoff,fs,order)
    scaled[(data_len*t):(data_len*(t+1)),1] = butter_lowpass_filter(scaled[(data_len*t):(data_len*(t+1)),1] ,cutoff,fs,order)
for t in range(0,number_of_trials):
    scaled[(data_len*t):(data_len*(t+1)),:] =  delay_series(scaled[(data_len*t):(data_len*(t+1)),1:],scaled[(data_len*t):(data_len*(t+1)),0],35)

local_batch_size = 277
model = Sequential()
#model.add(LSTM(25, activation='relu', batch_input_shape(local_batch_size, scaled[:,1:].shape[1], 1), input_shape=(1, scaled[:,1:].shape[1]), stateful=True, return_sequences=False))
model.add(LSTM(10, batch_input_shape=(local_batch_size, 1, scaled[:,1:].shape[1]),activation='softsign', stateful=True, return_sequences=False))
#model.add(SimpleRNN(25, return_sequences=False, stateful=True))
model.add(Dropout(0.2))
model.add(Dense(1,activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
n_train_hours = data_len * (number_of_trials - train_trials) #keep 3 to test rest are used to train
n_tests = number_of_trials - train_trials

# train = values[:n_train_hours, :]
# test = values[n_train_hours:, :]
# print(values)

# data_len = 60*60+1

# # split into input and outputs
# train_X, train_y = train[:, :-1], train[:, -1]
# test_X, test_y = test[:, :-1], test[:, -1]
for epo in range(0, 100):
    for i in range(0, train_trials):

        train = scaled[(data_len*(i)):(data_len*(i+1)), :]  #select first trial
        test = scaled[(data_len*(i+train_trials)):(data_len*(i+1+train_trials)), :] 
        # split into input and outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        print(train_X.shape[1])
        print(train_X.shape[2])
        train_shape = train_X.shape[2]
        # for p in range(0, train_shape):
        #     pyplot.subplot(train_shape+2, 1, p+1)
        #     pyplot.plot(train_X[:,0,p])
        # pyplot.subplot(train_shape+2, 1, train_shape+2)
        # pyplot.plot(test[:,-1])
        # pyplot.show()

        history = model.fit(train_X, train_y, epochs=3, batch_size=local_batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)
        # plot history
    #pyplot.plot(history.history['loss'], label='train')
    #pyplot.plot(history.history['val_loss'], label='test')
    #pyplot.legend()
    #pyplot.show()
 
# make a prediction
for i in range(0,number_of_trials+1):
    test = scaled[(data_len*(i)):(data_len*(i+1)), :] 
    test_X, test_y = test[:, :-1], test[:, -1]
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    yhat = model.predict(test_X, batch_size = local_batch_size)
    test_shape = test_X.shape[2]
    for p in range(0, test_shape):
        pyplot.subplot(train_shape+1, 1, p+1)
        pyplot.plot(test_X[:,0,p])

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
    pyplot.plot(yhat)
    pyplot.plot(scaled[(data_len*i):(data_len*(i+1)), -1]) #select first trial
    pyplot.show()
