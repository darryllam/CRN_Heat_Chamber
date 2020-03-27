import math
import numpy as np
import os,sys
import argparse
import random
from scipy.signal import butter, lfilter,freqz
from scipy.interpolate import interp1d

from datafunction import min_max_scaler,addrandomnoise,delay_series,short_term_average,find_soak_time
from retrieveData import get_data

import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import load_model

timesteps = 0
scalers = {}
val_scalers = {}
local_batch_size = 180

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
    parser.add_argument('-vp', '--val_path', help='file path', type=dir_path,required=True)
    parser.add_argument('-i', '--in_file', help='Input file for Network weights',required=False)
    parser.add_argument('-stcol','--scaled_trial_cols', nargs='+',type=int, help='cols that change during trial', required=True)
    parser.add_argument('-srcol','--scaled_run_cols', nargs='+',type=int, help='cols that change per trial', required=False)
    parser.add_argument('-vcol','--verif_col',type=int, help='Output Cols to verify with', required=True)
    return parser.parse_args()


parsed_args = parse_arguments()


parsed_args = parse_arguments()
parsed_args = parse_arguments()
in_file_name = parsed_args.in_file
model = load_model(in_file_name)    


if(parsed_args.scaled_run_cols == None):
    train_cols = parsed_args.scaled_trial_cols
    scaled_run_cols_arg = []
    len_scaled_run_cols_arg = None
else:
    scaled_run_cols_arg = parsed_args.scaled_run_cols
    len_scaled_run_cols_arg = -1*len(scaled_run_cols_arg)
    train_cols = parsed_args.scaled_trial_cols + scaled_run_cols_arg

if((in_file_name) != None):
    if os.path.isfile(in_file_name):
        only_predict_flag = 1
    else:
        raise FileNotFoundError(in_file_name)

val_data = get_data(train_cols,parsed_args.verif_col, parsed_args.val_path, 3600, True)
val_scaled = val_data
outer_temp_data = np.zeros((val_data.shape[0], val_data.shape[1]))
real_outer_temp = np.zeros((val_data.shape[0], val_data.shape[1]))

for t in range(0, val_data.shape[0]):
    outer_temp_data[t,:] = val_data[t,:,2]
    for i in range(0, outer_temp_data.shape[1]):
        index, average = short_term_average(outer_temp_data[t,:i], 500, 4)
        if(index != -1):
            print(index)
            break       
    for i in range(index, outer_temp_data.shape[1]):
        outer_temp_data[t,i] = outer_temp_data[t,index]
    real_outer_temp[t,:] = val_data[t,:,2]
    val_data[t,:,2] = outer_temp_data[t,:]
#Scale data that change during run this way
for j in range(val_scaled.shape[0]):    
    scalers[j] = MinMaxScaler(feature_range=(0, 1))
    val_scaled[j,:, 2:3] = scalers[j].fit_transform(val_data[j,:,2:3])  
    val_scaled[j,:, 0:1] = scalers[j].transform(val_data[j,:,0:1])
    scalers[j+val_scaled.shape[0]] = MinMaxScaler(feature_range=(0, 1))
    val_scaled[j,:, 1:2] = scalers[j+val_scaled.shape[0]].fit_transform(val_data[j,:,1:2])
#Scale data that changes per run this way
if(parsed_args.scaled_run_cols != None):
    for i in range(val_scaled.shape[2]- len(scaled_run_cols_arg), val_scaled.shape[2]):
        col_scalers[i] = joblib.load(in_file_name[:-3] + str(i) + ".pkl") 
        val_scaled[:,:,i] = col_scalers[i].transform(val_data[:,:,i])

for t in range(0,val_scaled.shape[0]):
    val_scaled[t,:,:] = delay_series(val_scaled[t,:,1:],val_scaled[t,:,0],5)

if(timesteps == 1):
    val_scaled_reshape = reshape_with_timestep(val_scaled, 360,10) #360 * 10 is data length 3600

for i in range(0,val_scaled.shape[0]):

    test = val_scaled[i,:, :] 
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
    val_data = get_data(train_cols,parsed_args.verif_col, parsed_args.val_path, 3600,True)
    val_data[i,:,2] = outer_temp_data[i,:]
    val_scaler = MinMaxScaler(feature_range=(0,1)).fit(val_data[i,:,2:3])
    inv_yhat_out = val_scaler.inverse_transform(yhat)
    end_time = find_soak_time(outer_temp_data[i,-1], val_data[i,:,1], val_data[i,:,2], inv_yhat_out, .05)
    if(end_time == None):
        end_time = val_data[i,-1,1]
    print(end_time)
    pyplot.plot(val_scaled[i,:,2] , label='Inner Temp Truth') #Inner Temp
    pyplot.plot(val_scaled[i,:,1], label='Outer Temp') #Outer Temp
    pyplot.plot(val_scaled[i,:,0])
    pyplot.plot(yhat[:],  label='Inner Temp NN')
    pyplot.legend()
    pyplot.show()  
    pyplot.plot(val_data[i,:,1],real_outer_temp[i,:], label='Air Temp') #Outer Temp
    pyplot.plot(val_data[i,:,1],val_data[i,:,2] , label='Extrapolated Approximation of Air Temp') #Inner Temp
    pyplot.plot(val_data[i,:,1],val_data[i,:,0] , label='Inner Temp Truth') #Inner Temp
    pyplot.plot(val_data[i,:,1], inv_yhat_out[:],  label='Inner Temp Prediction')
    pyplot.axvline(x=end_time)
    pyplot.xlabel('Time [s]')
    pyplot.ylabel('Temperature [C]')
    pyplot.legend()
    pyplot.show()  
    pyplot.plot(val_data[i,:,1],real_outer_temp[i,:], label='Outer Temp') #Outer Temp
    pyplot.plot(val_data[i,:,1],val_data[i,:,0] , label='Part Internal Temprature') #Inner Temp
    # pyplot.axvline(x=end_time)
    pyplot.legend()
    pyplot.xlabel('Time [s]')
    pyplot.ylabel('Temperature [C]')
    pyplot.show()  