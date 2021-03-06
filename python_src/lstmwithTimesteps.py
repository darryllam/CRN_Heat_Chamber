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

from sklearn.externals import joblib 
from sklearn.preprocessing import MinMaxScaler #Scaled inputs
from sklearn.metrics import mean_squared_error #Find errors
#Use these to build a LSTM model 
from keras.models import Sequential
from keras.models import load_model
from keras import optimizers
from keras.layers import Dense, Activation
from keras.layers import LSTM
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
#-e is number of epochs
#-tp is file path point to test data
#-vp is file path point to val data
#-o is a output file which model will be saved too
#-i is a input file which model is loaded from 
#-vcol is the col with truth/val data
#-stcol is for temp and time data
#-srcol is for data which changes on a per run basis 
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--epochs',type=int, help='Number of epochs', required=True)
    parser.add_argument('-tp', '--test_path', help='file path', type=dir_path,required=True)
    parser.add_argument('-vp', '--val_path', help='file path', type=dir_path,required=True)
    parser.add_argument('-i', '--in_file', help='Input file for Network weights',required=False)
    parser.add_argument('-o', '--out_file', help='Output file for Network weights',required=False)
    parser.add_argument('-w','--epochs',type=int, help='Size of window', required=True)
    parser.add_argument('-vcol','--part_col',type=int, help='Output Cols to verify with', required=True)
    parser.add_argument('-stcol','--scaled_trial_cols', nargs='+',type=int, help='cols that change during trial', required=True)
    parser.add_argument('-srcol','--scaled_run_cols', nargs='+',type=int, help='cols that change per trial', required=False)
    parser.add_argument('-tmcol','--air_temp_col', nargs='+',type=int, help='the air temp col', required=False)
    parser.add_argument('-min_temp','--min_temp', type=int, help='min_temp to scale temps to', required=False)
    parser.add_argument('-max_temp','--max_temp', type=int, help='max_temp to scale temps to', required=False)
    return parser.parse_args()


parsed_args = parse_arguments()
#init data
train_trials = 0 #Number of trials to train on 
number_of_trials = 27 #Number of trials to test on
data_len = 3600
only_predict_flag = 0 #Flag to determine if train or ONLY predict
local_batch_size = 180 #data_len/20, must be multiple of data_len
epochs_end = parsed_args.epochs #Number of epochs to train on
scalers = {}
val_scalers = {}
col_scalers = {}
timesteps = 180
#Filter parameters 
fs = 1
cutoff = .5
order = 15
plot_data = 1

parsed_args = parse_arguments()
in_file_name = parsed_args.in_file
out_file_name = parsed_args.out_file
if(parsed_args.scaled_run_cols == None):
    train_cols = parsed_args.scaled_trial_cols
    scaled_run_cols_arg = []
    len_scaled_run_cols_arg = None
else:
    scaled_run_cols_arg = parsed_args.scaled_run_cols
    len_scaled_run_cols_arg = -1*len(scaled_run_cols_arg)
    train_cols = parsed_args.scaled_trial_cols + scaled_run_cols_arg
print(train_cols)
print(parsed_args.verif_col)
if((in_file_name) != None):
    if os.path.isfile(in_file_name):
        only_predict_flag = 1
    else:
        raise FileNotFoundError(in_file_name)

val_data = get_data(train_cols,parsed_args.verif_col, parsed_args.val_path, 3600, True)
val_scaled = val_data
raw_val_reshape = reshape_with_timestep(val_scaled, 360,10) #360 * 10 is data length 3600

#Scale data that change during run this way
if(only_predict_flag == 0): 
    raw_data = get_data(train_cols,parsed_args.verif_col, parsed_args.test_path, 3600, True)
    scaled = raw_data
    raw_reshape = reshape_with_timestep(scaled, 360,10) #360 * 10 is data length 3600

    for j in range(scaled.shape[0]):
        scalers[j] = MinMaxScaler(feature_range=(0, 1))
        scaled[j,:, 2:3] = scalers[j].fit_transform(raw_data[j,:,2:3])  
        scaled[j,:, 0:1] = scalers[j].transform(raw_data[j,:,0:1])
        scalers[j+scaled.shape[0]] = MinMaxScaler(feature_range=(0, 1))
        scaled[j,:, 1:2] = scalers[j+scaled.shape[0]].fit_transform(raw_data[j,:,1:2])

for j in range(val_scaled.shape[0]):    
    scalers[j] = MinMaxScaler(feature_range=(0, 1))
    val_scaled[j,:, 2:3] = scalers[j].fit_transform(val_data[j,:,2:3])  
    val_scaled[j,:, 0:1] = scalers[j].transform(val_data[j,:,0:1])
    scalers[j+val_scaled.shape[0]] = MinMaxScaler(feature_range=(0, 1))
    val_scaled[j,:, 1:2] = scalers[j+val_scaled.shape[0]].fit_transform(val_data[j,:,1:2])
#Scale data that changes per run this way
if(len_scaled_run_cols_arg != None):
    for i in range(val_scaled.shape[2]-1, val_scaled.shape[2]-2, -1):
        if(only_predict_flag == 0): 
            col_scalers[i] = MinMaxScaler(feature_range=(0, 1))
            col_scalers[i].fit(np.vstack((raw_data[:,:,i], val_data[:,:,i])))
            scaled[:,:,i] = col_scalers[i].transform(raw_data[:,:,i])
            val_scaled[:,:,i] = col_scalers[i].transform(val_data[:,:,i])
            # scaled[:,:,i] = min_max_scaler(raw_data[:,:,i], 15, 70, 0, 1)
            # val_scaled[:,:,i] = min_max_scaler(val_data[:,:,i], 15, 70, 0, 1)
            joblib.dump(col_scalers[i], out_file_name[:-3] + str(i) + ".pkl") 
        else:
            col_scalers[i] = joblib.load(in_file_name[:-3] + str(i) + ".pkl") 
            val_scaled[:,:,i] = col_scalers[i].transform(val_data[:,:,i])

for t in range(0,val_scaled.shape[0]):
    val_scaled[t,:,:] = delay_series(val_scaled[t,:,1:],val_scaled[t,:,0],5)
val_scaled_reshape = reshape_with_timestep(val_scaled, 360,10) #360 * 10 is data length 3600

if(only_predict_flag == 0):
    for t in range(0,scaled.shape[0]):
        scaled[t,:,:] =  delay_series(scaled[t,:,1:],scaled[t,:,0],5)
    scaled_reshape = reshape_with_timestep(scaled, 360,10) #360 * 10 is data length 3600

if(plot_data == 1):
    train = val_scaled_reshape[0,:,:,:]  #select first trial
    train_X, train_y = train[:,:, :-1], train[:,0, -1]
    pyplot.title("Example Train_X and Train_y data")
    pyplot.plot(train_X[:,0,:], label='train_X features')
    pyplot.plot(train_y[:], label='train_y feature')
    pyplot.legend()
    pyplot.show()
local_batch_size = 36
#Build keras model
model = Sequential()
model.add(LSTM(25, batch_input_shape=(local_batch_size,val_scaled_reshape.shape[2], val_scaled_reshape.shape[3]-1),activation='softsign', stateful=True, return_sequences=False))
model.add(Dropout(0.005))
model.add(Dense(1))
model.add(Activation('linear'))
ad = optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='MSE', optimizer=ad)
if(only_predict_flag == 0): 
    train_trials = scaled_reshape.shape[0]
    history_log = {'loss' : [0]*epochs_end*train_trials, \
               'val': [0]*epochs_end*train_trials}
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
        model.reset_states()
    pyplot.title("Training Loss Curve")
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

val_data = get_data(train_cols,parsed_args.verif_col, parsed_args.val_path, 3600, True)
for i in range(0,val_scaled_reshape.shape[0]):
    test = val_scaled_reshape[i % val_scaled_reshape.shape[0],:,:,:] 
    test_X, test_y = test[:,:, :-1], test[:,0, -1]
    yhat = model.predict(test_X, batch_size = local_batch_size)
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
    if(plot_data == 1):
        pyplot.title("Prediction")
        pyplot.plot(val_scaled_reshape[i,:,0,:] , label='Truth Data') #Inner Temp
        pyplot.plot(yhat[:],  label='Inner Temp NN')
        pyplot.legend()
        pyplot.show()  
        val_reshape = reshape_with_timestep(val_data, 360,10) #360 * 10 is data length 3600
        val_scaler = MinMaxScaler(feature_range=(0,1)).fit(val_reshape[i,:,0,2:3])
        inv_yhat_out = val_scaler.inverse_transform(yhat)
        pyplot.plot(val_reshape[i,:,0,1],val_reshape[i,:,0,0] , label='Inner Temp Truth') #Inner Temp
        pyplot.plot(val_reshape[i,:,0,1],val_reshape[i,:,0,2], label='Outer Temp') #Outer Temp
        #pyplot.plot(val_reshape[i,:,0,1], inv_yhat_out[:],  label='Inner Temp NN')
        pyplot.xlabel('Time [s]')
        pyplot.ylabel('Temperature [C]')
        pyplot.legend()
        pyplot.show()    

