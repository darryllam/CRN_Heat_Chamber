import numpy as np
import argparse, os, sys
import math

from datafunction import addrandomnoise,delay_series,butter_lowpass_filter
from retrieveData import get_data

import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.models import load_model
from keras import optimizers
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras import metrics

###
#Takes an input directory and tries to append current dir to it
#Checks if the input directory is a directory
###
def dir_path(string):
    string2 = os.getcwd() + string
    if os.path.isdir(string2):
        return string2
    elif os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
###
#Implements command line arguemnts
#-p is file path point to data
#-o is a output file which model will be saved too
#-i is a input file which model is loaded from 
#   If -i is input then the model will not be trained and will just use input weights
#Output Example: python3 lstmMethod.py -p /data2/ -o weights.h5
# Input Example: python3 lstmMethod.py -p /data2/ -i weights.h5
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='file path', type=dir_path)
    parser.add_argument('-i', '--in_file', help='Input file for Network weights')
    parser.add_argument('-o', '--out_file', help='Output file for Network weights')
    return parser.parse_args()

#init data
train_trials = 184 #Number of trials to train on 
number_of_trials = 184 #Number of trials to test on
data_len = 3601
only_predict_flag = 0 #Flag to determine if train or ONLY predict
local_batch_size = 277 #data_len/20, must be multiple of data_len
epochs_end = 200 #Number of epochs to train on
scalers = {}
#Filter parameters 
fs = 1
cutoff = .5
order = 15


parsed_args = parse_arguments()
in_file_name = parsed_args.in_file
out_file_name = parsed_args.out_file
#0 is interior temperature
#1 is exterior temperature 
#5 is timesteps
#[tlist', outT', centerT', radCSV, specificHeatCSV, thermalConductivityCSV, massDensityCSV];
raw_data = get_data([0,1,3],2, parsed_args.path)
if((in_file_name) != None):
    if os.path.isfile(in_file_name):
        only_predict_flag = 1
    else:
        raise FileNotFoundError(in_file_name)

# for i in range(raw_data.shape[0]): #all trials
#     for j in range(0,2): #only temperature readings
#         raw_data[i,:,j] = addrandomnoise(raw_data[i,:,j]) #add noise to data for fun
#         #raw_data[i,:,j] = butter_lowpass_filter(raw_data[i,:,j],cutoff,fs,order)
#         # resize data here 

#new_data = reduce_data_size3d(raw_data, 19*3600)
#new_data = np.random.shuffle(raw_data) 
        
scaled = raw_data
for i in range(scaled.shape[0]):
    scalers[i] = MinMaxScaler(feature_range=(0, 1))
    scaled[i,:, :] = scalers[i].fit_transform(raw_data[i,:,:]) 
#scaled = scaler.fit_transform(raw_data)

for t in range(0,scaled.shape[0]):
    scaled[t,:,:] =  delay_series(scaled[t,:,1:],scaled[t,:,0],5)

#Build keras model
model = Sequential()
model.add(LSTM(25, batch_input_shape=(local_batch_size,  1, scaled[0,:,1:].shape[1]),activation='relu', stateful=True, return_sequences=False))
#model.add(Dropout(0.01))
model.add(Dense(1))
model.add(Activation('linear'))
ad = optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='MSE', optimizer=ad)

history_log = {'loss' : [0]*epochs_end*train_trials, \
               'val': [0]*epochs_end*train_trials}
if(only_predict_flag == 0):
    for epo in range(0, epochs_end):
        for i in range(0, train_trials):
            num = epo*train_trials + i
            train = scaled[i,:,:]  #select first trial
            test = scaled[(i+train_trials) % number_of_trials-1,:,:] 
            # split into input and outputs
            train_X, train_y = train[:, :-1], train[:, -1]
            test_X, test_y = test[:, :-1], test[:, -1]
            # reshape input to be 3D [samples, timesteps, features]
            train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
            test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
            #Fit the model for a single epoch on each trial but do this many times
            history = model.fit(train_X, train_y, epochs=1, batch_size=local_batch_size, \
                validation_data=(test_X, test_y), verbose=2, shuffle=True)
            #Store loss and val loss in these dictionaries 
            history_log['loss'][num] = history.history['loss'][0]+history_log['loss'][num]  
            history_log['val'][num] = history.history['val_loss'][0]+history_log['val'][num]
        model.reset_states()
    pyplot.plot(history_log['loss'], label='train')
    pyplot.plot(history_log['val'], label='test')
    pyplot.legend()
    pyplot.show()
else:
    model.load_weights(in_file_name)

model = load_model(file_name)

# make a prediction
for i in range(0,number_of_trials):
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
    pyplot.plot(scaled[i,:,0])
    pyplot.plot(scaled[i,:,1])
    pyplot.show()

if((out_file_name) == None):
    #Just set some default name in case you forget to set filename
    out_file_name = "default_name_weights.h5"
model.save_weights(out_file_name) #save weights
model.save("model_" + out_file_name) #save entire model