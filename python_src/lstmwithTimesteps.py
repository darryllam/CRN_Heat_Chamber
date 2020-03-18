import numpy as np #numpy
import argparse, os, sys 
import math #math 
import time #time for sleeps
#Functions for importing and adjusting data
from datafunction import addrandomnoise,delay_series,butter_lowpass_filter, reshape_with_timestep
from retrieveData import get_data
#Plotting functions 
import matplotlib.pyplot as plt
from matplotlib import pyplot

import pickle #Use this to save a scaler
from sklearn.preprocessing import MinMaxScaler #Scaled inputs
from sklearn.metrics import mean_squared_error #Find errors
#Use these to build a LSTM model 
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
    parser.add_argument('-tp', '--test_path', help='file path', type=dir_path)
    parser.add_argument('-vp', '--val_path', help='file path', type=dir_path)
    parser.add_argument('-i', '--in_file', help='Input file for Network weights')
    parser.add_argument('-o', '--out_file', help='Output file for Network weights')
    return parser.parse_args()

#init data
train_trials = 0 #Number of trials to train on 
number_of_trials = 27 #Number of trials to test on
data_len = 3600
only_predict_flag = 0 #Flag to determine if train or ONLY predict
local_batch_size = 180 #data_len/20, must be multiple of data_len
epochs_end = 10 #Number of epochs to train on
scalers = {}
val_scalers = {}
timesteps = 180
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
raw_data = get_data([0,1,3],2, parsed_args.test_path, 3600)
val_data = get_data([0,1,3],2, parsed_args.val_path, 3600)
if((in_file_name) != None):
    if os.path.isfile(in_file_name):
        only_predict_flag = 1
    else:
        raise FileNotFoundError(in_file_name)

# for i in range(raw_data.shape[0]): #all trials
#     for j in range(0,2): #only temperature readings
        #raw_data[i,:,j] = addrandomnoise(raw_data[i,:,j]) #add noise to data for fun
        #raw_data[i,:,j] = butter_lowpass_filter(raw_data[i,:,j],cutoff,fs,order)
        # resize data here 
       
val_scaled = val_data
scaled = raw_data
raw_reshape = reshape_with_timestep(scaled, 360,10) #360 * 10 is data length 3600
raw_val_reshape = reshape_with_timestep(val_scaled, 360,10) #360 * 10 is data length 3600

#Scale data that change during run this way
for i in range(scaled.shape[0]):
    scalers[i] = MinMaxScaler(feature_range=(0, 1))
    scaled[i,:, :3] = scalers[i].fit_transform(raw_data[i,:,:3])
for i in range(val_scaled.shape[0]):    
    val_scalers[i] = MinMaxScaler(feature_range=(0, 1)) 
    val_scaled[i,:,:3] = val_scalers[i].fit_transform(val_data[i,:,:3])

#Scale data that changes per run this way
col_scaler = MinMaxScaler(feature_range=(0, 1))
col_scaler.fit(np.vstack((raw_data[:,:,3], val_data[:,:,3])))
scaled[:,:, 3] = col_scaler.transform(raw_data[:,:,3])
val_scaled[:,:, 3] = col_scaler.transform(val_data[:,:,3])


for t in range(0,val_scaled.shape[0]):
    val_scaled[t,:,:] = delay_series(val_scaled[t,:,1:],val_scaled[t,:,0],5)
for t in range(0,scaled.shape[0]):
    scaled[t,:,:] =  delay_series(scaled[t,:,1:],scaled[t,:,0],5)
scaled_reshape = reshape_with_timestep(scaled, 360,10) #360 * 10 is data length 3600
val_scaled_reshape = reshape_with_timestep(val_scaled, 360,10) #360 * 10 is data length 3600
quit()
#pyplot.plot(scaled[0,:,:])
#pyplot.plot(val_scaled[0,:,:])
#pyplot.show()


local_batch_size = 36
#Build keras model
model = Sequential()
model.add(LSTM(50, batch_input_shape=(local_batch_size,scaled_reshape.shape[2], scaled_reshape.shape[3]-1),activation='softsign', stateful=True, return_sequences=False))
model.add(Dropout(0.0005))
model.add(Dense(1))
model.add(Activation('linear'))
ad = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='MSE', optimizer=ad)
train_trials = scaled_reshape.shape[0]
history_log = {'loss' : [0]*epochs_end*train_trials, \
               'val': [0]*epochs_end*train_trials}
if(only_predict_flag == 0): 
    for epo in range(0, epochs_end):
        for i in range(0, train_trials):
            num = epo*train_trials + i
            train = scaled_reshape[i,:,:,:]  #select first trial
            test = val_scaled_reshape[i % val_scaled_reshape.shape[0],:,:,:] 
            # split into input and outputs
            train_X, train_y = train[:,:, :-1], train[:,0, -1]
            test_X, test_y = test[:,:, :-1], test[:,0, -1]
            #Fit the model for a single epoch on each trial but do this many times
            history = model.fit(train_X, train_y, epochs=1, batch_size=local_batch_size, \
                validation_data=(test_X, test_y), verbose=2, shuffle=False)
            #Store loss and val loss in these dictionaries 
            history_log['loss'][num] = history.history['loss'][0]+history_log['loss'][num]  
            history_log['val'][num] = history.history['val_loss'][0]+history_log['val'][num]
        yhat = model.predict(test_X, batch_size = local_batch_size)
        val_scaler = MinMaxScaler(feature_range=(0,1)).fit(raw_val_reshape[i% val_scaled_reshape.shape[0],:,0,2:3])
        inv_yhat_out = val_scaler.inverse_transform(yhat)
        inv_reshape = val_scaler.inverse_transform(val_scaled_reshape[i % val_scaled_reshape.shape[0],:,0,-1:])
        pyplot.plot(inv_reshape) #select first trial
        pyplot.plot(inv_yhat_out)
        pyplot.show()
        model.reset_states()
    pyplot.plot(history_log['loss'], label='train')
    pyplot.plot(history_log['val'], label='test')
    pyplot.legend()
    pyplot.show()
    if((out_file_name) == None):
       #Just set some default name in case you forget to set filename
        out_file_name = "default_name_weights.h5"
    model.save_weights(out_file_name) #save weights
    model.save("model_" + out_file_name) #save entire model
    # make a prediction
else:
    print(in_file_name)
    model.load_weights(in_file_name)

model = load_model("model_" + in_file_name)
i = 0
test = val_scaled_reshape[i % val_scaled_reshape.shape[0],:,:,:] 
test_X, test_y = test[:,0, :-1], test[:,0, -1]

pyplot.plot(test_X)
pyplot.show()   
train = scaled_reshape[i,:,:,:]  #select first trial
train_X, train_y = train[:,0, :-1], train[:,0, -1]
pyplot.plot(train_X)
pyplot.show()  


for i in range(0,val_scaled_reshape.shape[0]):
    test = val_scaled_reshape[i % val_scaled_reshape.shape[0],:,:,:] 
    test_X, test_y = test[:,:, :-1], test[:,0, -1]
    yhat = model.predict(test_X, batch_size = local_batch_size)
    val_scaler = MinMaxScaler(feature_range=(0,1)).fit(raw_val_reshape[i% val_scaled_reshape.shape[0],:,0,2:3])
    inv_yhat_out = val_scaler.inverse_transform(yhat)
    inv_reshape = val_scaler.inverse_transform(val_scaled_reshape[i % val_scaled_reshape.shape[0],:,0,-1:])
    pyplot.plot(raw_val_reshape[i,:,0,0], label='Inner Temp Truth') #Inner Temp
    #pyplot.plot(raw_val_reshape[i,:,0,1]) #Time
    pyplot.plot(raw_val_reshape[i,:,0,2], label='Outer Temp') #Outer Temp
    #pyplot.plot(raw_val_reshape[i,:,0,3]) #Size
    pyplot.plot(inv_yhat_out,  label='Inner Temp NN')
    pyplot.legend()
    pyplot.show()   

    test_X = val_scaled_reshape[i,:, 0, :-1]
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]))
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
    #val_scalers[i] = MinMaxScaler(feature_range=(0, 1)) 
    
    # val_scaler = MinMaxScaler(feature_range=(0,1)).fit(raw_val_reshape[i,:,0,2:3])
    # inv_yhat_out = val_scaler.inverse_transform(yhat)
    # inv_reshape = val_scaler.inverse_transform(val_scaled_reshape[i,:,0,-1:])
    
    # pyplot.plot(inv_reshape) #select first trial
    # pyplot.plot(inv_yhat_out)
    # #pyplot.plot(val_scaled_reshape[i,:,0,0])
    # #pyplot.plot(val_scaled_reshape[i,:,0,1])
    # pyplot.show()

