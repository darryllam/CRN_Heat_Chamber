import numpy as np
import argparse, os, sys
import math

from datafunction import min_max_scaler,addrandomnoise,delay_series,butter_lowpass_filter,find_soak_time
from retrieveData import get_data

import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import load_model
data_len = 3600
local_batch_size = 180 #data_len/20, must be multiple of data_len
windows = 50
temp_min = 20
temp_max = 70
predict_future = False
end_time = 3600
scalers = {}
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
val_data = np.zeros((3,end_time,3))
val_data[:,:3600,:] = get_data(train_cols,parsed_args.verif_col, parsed_args.val_path, 3600, True)
# added_time = val_data.shape[1] - 3600  
if(predict_future == True):
    for i in range(0,val_data.shape[0]):
        for j in range(3000,end_time):
            for k in range(0,3):
                if(k == 1):
                    val_data[i,j,k] = val_data[i,j-3000,k] + val_data[i,3000,k]
                else:
                    val_data[i,j,k] = val_data[i,3000,k]

val_scaled = val_data[:,:,:]
max_time = 0
for j in range(val_scaled.shape[0]):    
    val_scaled[j,:,1] = min_max_scaler(val_data[j,:,1], 0, val_data[j,3599,1], 0, 1)
    #scalers[j] = MinMaxScaler(feature_range=(0, 1))
    #scalers[j+val_scaled.shape[0]] = MinMaxScaler(feature_range=(0, 1))
    #val_scaled[j,:, 1:2] = scalers[j+val_scaled.shape[0]].fit_transform(val_data[j,:,1:2])

val_scaled[:,:,2] = min_max_scaler(val_data[:,:,2], temp_min, temp_max, 0, 1)
val_scaled[:,:,0] = min_max_scaler(val_data[:,:,0], temp_min, temp_max, 0, 1)

for t in range(0,val_scaled.shape[0]):
    val_scaled[t,:,:] = delay_series(val_scaled[t,:,1:],val_scaled[t,:,0],0)
    str1 = "Part Temperature Truth, Run "+str(t)
    str2 = "Air Temperature Truth, Run "+str(t)
    str3 = "Time, Run "+str(t)
    pyplot.plot(val_scaled[t,:,2] , label=str1) #Inner Temp
    pyplot.plot(val_scaled[t,:,1] , label=str2) #Inner Temp
    pyplot.plot(val_scaled[t,:,0],  label=str3)
pyplot.xlabel('Time')
pyplot.ylabel('Temperature C')
pyplot.legend()
pyplot.show()
val_scaled_copy = val_scaled
model = load_model("model_" + in_file_name)    
#val_data[:,:3600,:] = get_data(train_cols,parsed_args.verif_col, parsed_args.val_path, 3600, True)
for i in range(0,val_scaled.shape[0]):
    soak_time_arr = []
    for j in range(0, 36,3636):
        val_scaled_copy = np.zeros(val_scaled.shape)
        for k in range(0,end_time):
            if(k > j):
                val_scaled_copy[i,k,1] = val_scaled[i,j,1]
            else:
                val_scaled_copy[i,k,1] = val_scaled[i,k,1]

        val_scaled_copy[i,:,0] = val_scaled[i,:,0]
        val_scaled_copy[i,:,2] = val_scaled[i,:,0]
        
        # print(test_X[j,0,1])
        # print(j)

        test = val_scaled_copy[i,:, :] 
        
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
        rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.6f' % rmse)
        inv_yhat_out = min_max_scaler(yhat, 0, 1, temp_min, temp_max)
        val_data = get_data(train_cols,parsed_args.verif_col, parsed_args.val_path, 3600,True)
        soak_time = find_soak_time(val_data[i,3000,2], val_data[i,:,1], val_data[i,:,2], inv_yhat_out, .1)
        if(soak_time == None):
            soak_time = val_data[i,-1,1]
        soak_time_arr += [soak_time]
    x = np.linspace(0,val_data[i,-1,1],94)
    val_scaled_copy = min_max_scaler(val_scaled_copy, 0, 1, temp_min, temp_max)
        
    pyplot.plot(val_data[i,:,1],val_scaled_copy[i,:,1], label='Outer Temp Extrapolated') #Outer Temp
    pyplot.plot(val_data[i,:,1],val_data[i,:,0] , label='Part Internal Temprature') #Inner Temp
    pyplot.plot(val_data[i,:,1],inv_yhat_out, label = "Prediction")
    # for i in range(0,94):
    #     pyplot.axvline(x=soak_time_arr[i])
    pyplot.axvline(x=soak_time, color = 'b')
    true_soak_time = find_soak_time(val_data[i,3000,2], val_data[i,:,1], val_data[i,:,2], val_data[i,:,0], .1)
    pyplot.axvline(x=soak_time, color = 'r')
    
    pyplot.legend()
    pyplot.xlabel('Time [s]')
    pyplot.ylabel('Temperature [C]')
    pyplot.show()

    pyplot.plot(x, soak_time_arr, label = 'Stop Time Prediction')
    pyplot.plot(x, true_soak_time)
    pyplot.plot(x,x, label = 'y = x')
    pyplot.xlabel('Time Prediction was Made [s]')
    pyplot.ylabel('Soak Time Prediction [s]')
    pyplot.show()  

        # inv_yhat_out = min_max_scaler(yhat,0,1, 15 , 70 )
        # pyplot.plot(val_scaled[i,:,2] , label='Inner Temp Truth') #Inner Temp
        # pyplot.plot(val_scaled[i,:,1], label='Outer Temp') #Outer Temp
        # pyplot.plot(yhat[:],  label='Inner Temp NN')
        # pyplot.legend()
        # pyplot.show()  
        # pyplot.plot(val_data[i,:,1],val_data[i,:,0] , label='Inner Temp Truth') #Inner Temp
        # pyplot.plot(val_data[i,:,1],val_data[i,:,2], label='Outer Temp') #Outer Temp
        # pyplot.plot(val_data[i,:,1], inv_yhat_out[:],  label='Inner Temp NN')
        # pyplot.legend()
