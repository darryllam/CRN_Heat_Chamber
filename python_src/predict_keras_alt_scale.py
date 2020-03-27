import numpy as np
import argparse, os, sys
import math

from datafunction import min_max_scaler,addrandomnoise,delay_series,butter_lowpass_filter
from retrieveData import get_data

import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import load_model
timesteps = 0
scalers = {}
val_scalers = {}
local_batch_size = 120

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
val_data = np.zeros((3,3600,3))
val_data[:,:,:] = get_data(train_cols,parsed_args.verif_col, parsed_args.val_path, 3600, True)
# added_time = val_data.shape[1] - 3600  
# for i in range(0,val_data.shape[0]):
#     for j in range(3000,7200):
#         for k in range(0,3):
#             if(k == 1):
#                 val_data[i,j,k] = val_data[i,j-3000,k] + val_data[i,3000,k]
#             else:
#                 val_data[i,j,k] = val_data[i,3000,k]
val_scaled = val_data[:,:,:]
max_time = 0
for j in range(val_scaled.shape[0]):    
    scalers[j] = MinMaxScaler(feature_range=(0, 1))
    scalers[j+val_scaled.shape[0]] = MinMaxScaler(feature_range=(0, 1))
    val_scaled[j,:, 1:2] = scalers[j+val_scaled.shape[0]].fit_transform(val_data[j,:,1:2])

val_scaled[:,:,2] = min_max_scaler(val_data[:,:,2], 15, 70, 0, 1)
val_scaled[:,:,0] = min_max_scaler(val_data[:,:,0], 15, 70, 0, 1)

for t in range(0,val_scaled.shape[0]):
    val_scaled[t,:,:] = delay_series(val_scaled[t,:,1:],val_scaled[t,:,0],0)
    str1 = "Part Temperature Truth, Run "+str(t)
    str2 = "Air Temperature Truth, Run "+str(t)
    str3 = "Time, Run "+str(t)
    pyplot.plot(val_scaled[t,:,2] , label=str1) #Inner Temp
    pyplot.plot(val_scaled[t,:,1] , label=str2) #Inner Temp
    pyplot.plot(val_scaled[t,:,0],  label=str3)
    pyplot.legend()
pyplot.show()  #Scale data that change during run this way
val_scaled_copy = val_scaled
model = load_model("model_" + in_file_name)    

for i in range(0,val_scaled.shape[0]):
    # for j in range(0, 3600,359):
    #     val_scaled_copy = np.zeros(val_scaled.shape)
    #     for k in range(0,3600):
    #         if(k > j):
    #             val_scaled_copy[i,k,1] = val_scaled[i,j,1]
    #         else:
    #             val_scaled_copy[i,k,1] = val_scaled[i,k,1]

    #     val_scaled_copy[i,:,0] = val_scaled[i,:,0]
    #     val_scaled_copy[i,:,2] = val_scaled[i,:,0]
        
    #     # print(test_X[j,0,1])
    #     # print(j)

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
        rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.6f' % rmse)
        # #val_data = get_data(train_cols,parsed_args.verif_col, parsed_args.val_path, 3600,True)
        pyplot.plot(yhat, label='yhat Temp') #Outer Temp
        #pyplot.plot(val_scaled_copy[i,:,0], label='00 Temp') #Outer Temp
        pyplot.plot(val_scaled_copy[i,:,1], label='11 Temp') #Outer Temp
        pyplot.plot(val_scaled_copy[i,:,2], label='22 Temp') #Outer Temp
        pyplot.legend()
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