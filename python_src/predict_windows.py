import numpy as np #numpy
import argparse, os, sys 
import math #math 
import time #time for sleeps
import timeit
#Functions for importing and adjusting data
from datafunction import addrandomnoise,delay_series,butter_lowpass_filter, reshape_with_timestep,min_max_scaler,find_soak_time
from retrieveData import get_data
#Plotting functions 
import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn.externals import joblib 
from sklearn.preprocessing import MinMaxScaler #Scaled inputs
from sklearn.metrics import mean_squared_error, r2_score#Find errors
#Use these to build a LSTM model 
from keras.models import Sequential
from keras.models import load_model
from keras import optimizers
from keras.layers import Dense, Activation
from keras.layers import LSTM, SimpleRNN, GRU
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
    parser.add_argument('-vp', '--val_path', help='file path', type=dir_path,required=True)
    parser.add_argument('-i', '--in_file', help='Input file for Network weights',required=False)
    parser.add_argument('-w','--window_size',type=int, help='Size of window', required=True)
    parser.add_argument('-future', '--future', type=float, help='Predict Using fraction eg 1.5 would predict 1.5 length run',required=False)
    parser.add_argument('-live', '--live_predict', type=int, help='makes live prediction every step, size based on input',required=False)
    parser.add_argument('-vcol','--part_col',type=int, help='Output Cols to verify with', required=True)
    parser.add_argument('-tmcol','--air_temp_col', nargs='+',type=int, help='the air temp col', required=True)
    parser.add_argument('-stcol','--scaled_trial_cols', nargs='+',type=int, help='cols that change during trial', required=False)
    parser.add_argument('-srcol','--scaled_run_cols', nargs='+',type=int, help='cols that change per trial', required=False)
    parser.add_argument('-min_temp','--min_temp', type=int, help='min_temp to scale temps to', required=True)
    parser.add_argument('-max_temp','--max_temp', type=int, help='max_temp to scale temps to', required=True)
    return parser.parse_args()


parsed_args = parse_arguments()
#init data
data_len = 3600
only_predict_flag = 0 #Flag to determine if train or ONLY predict
local_batch_size = 180 #data_len/20, must be multiple of data_len
scalers = {}
val_scalers = {}
col_scalers = {}
plot_data = 1
windows = parsed_args.window_size
Neurons = 25
temp_min = parsed_args.min_temp
temp_max = parsed_args.max_temp

parsed_args = parse_arguments()
in_file_name = parsed_args.in_file
if(parsed_args.scaled_trial_cols == None):
    train_cols = parsed_args.air_temp_col
    len_scaled_trial_cols_arg=None
else:
    train_cols = parsed_args.air_temp_col + parsed_args.scaled_trial_cols
    scaled_trial_cols_arg = parsed_args.scaled_trial_cols
    len_scaled_trial_cols_arg = len(parsed_args.scaled_trial_cols)
if(parsed_args.scaled_run_cols == None):
    scaled_run_cols_arg = []
    len_scaled_run_cols_arg = None
else:
    scaled_run_cols_arg = parsed_args.scaled_run_cols
    len_scaled_run_cols_arg = -1*len(scaled_run_cols_arg)
    train_cols +=  scaled_run_cols_arg

print(train_cols)
print(parsed_args.part_col)

if((in_file_name) != None):
    if os.path.isfile(in_file_name):
        only_predict_flag = 1
    else:
        raise FileNotFoundError(in_file_name)
if(parsed_args.live_predict == None):
    live_predict_step = data_len
else: 
    live_predict_step = parsed_args.live_predict

if(parsed_args.future == None):
    predict_future = False
    end_time = data_len
else: 
    predict_future = True
    end_time = int(parsed_args.future * data_len)

if(predict_future == True):
    val_data = np.zeros((3,end_time,3))
    val_data[:,:3600,:] = get_data(train_cols,parsed_args.part_col, parsed_args.val_path, 3600, True)
    for i in range(0,val_data.shape[0]):
        for j in range(3000,end_time): #set to 3000 cause one run has weird uptick at end
            for k in range(0,3):
                if(k == 2):
                    val_data[i,j,k] = val_data[i,j-3000,k] + val_data[i,3000,k]
                else:
                    val_data[i,j,k] = val_data[i,3000,k]
else:
    val_data = get_data(train_cols,parsed_args.part_col, parsed_args.val_path, data_len, True)
val_data_copy = val_data
val_scaled = val_data
#raw_val_reshape = reshape_with_timestep(val_scaled, 360,10) #360 * 10 is data length 3600
if(len_scaled_trial_cols_arg != None):
    for i in range(2, val_scaled.shape[2] - len(scaled_run_cols_arg)):
        for j in range(val_scaled.shape[0]):    
            if(predict_future == True):
                val_scaled[j,:,2] = min_max_scaler(val_data[j,:,2], 0, val_data[j,-1,2], 0, val_data[j,-1,2]/val_data[j,3599,2])
            else:
                scalers[j+val_scaled.shape[0]] = MinMaxScaler(feature_range=(0, 1))
                val_scaled[j,:, i:i+1] = scalers[j+val_scaled.shape[0]].fit_transform(val_data[j,:,i:i+1])

val_scaled[:,:,1] = min_max_scaler(val_data[:,:,1], temp_min, temp_max, 0, 1)
val_scaled[:,:,0] = min_max_scaler(val_data[:,:,0], temp_min, temp_max, 0, 1)

val_scaled_reshape = np.zeros((val_scaled.shape[0], end_time, windows, val_scaled.shape[2]))
if(len_scaled_run_cols_arg != None):
    for i in range(val_scaled.shape[2]- len(scaled_run_cols_arg), val_scaled.shape[2]):
        if(only_predict_flag == 0): 
            col_scalers[i] = MinMaxScaler(feature_range=(0, 1))
            col_scalers[i].fit(np.vstack((raw_data[:,:,i], val_data[:,:,i])))
            scaled[:,:,i] = col_scalers[i].transform(raw_data[:,:,i])
            val_scaled[:,:,i] = col_scalers[i].transform(val_data[:,:,i])
            joblib.dump(col_scalers[i], out_file_name[:-3] + str(i) + ".pkl") 
        else:
            col_scalers[i] = joblib.load(in_file_name[:-3] + str(i) + ".pkl") 
            val_scaled[:,:,i] = col_scalers[i].transform(val_data[:,:,i])

for t in range(0,val_scaled.shape[0]):
    val_scaled[t,:,:] = delay_series(val_scaled[t,:,1:],val_scaled[t,:,0],0)
for t in range(0,val_scaled.shape[0]):
    for i in range(0,windows):
        data = val_scaled[t,:,:]
        for td in range(0,i-1):
            #for i in range(0, len_col):
            #extend all Cols
            data = np.append(data, data[-1,:][None], axis = 0 )
            data = np.delete(data, 0, axis = 0)
        val_scaled_reshape[t,:,-i,:] = data
    if(windows == 1):
        val_scaled_reshape[t,:,0,:] = data

model = load_model("model_" + in_file_name)
print(val_scaled_reshape.shape)
if(predict_future == True):
    val_data = np.zeros((3,end_time,3))
    val_data[:,:3600,:] = get_data(train_cols,parsed_args.part_col, parsed_args.val_path, 3600, True)
    for i in range(0,val_data.shape[0]):
        for j in range(3000,end_time): #set to 3000 cause one run has weird uptick at end
            for k in range(0,3):
                if(k == 2):
                    val_data[i,j,k] = val_data[i,j-3000,k] + val_data[i,3000,k]
                else:
                    val_data[i,j,k] = val_data[i,3000,k]
else:
    val_data = get_data(train_cols,parsed_args.part_col, parsed_args.val_path, data_len, True)

for i in range(0,val_data.shape[2]):
    soak_time_arr = []
    val_scaled_copy = np.zeros(val_scaled_reshape.shape)
    val_scaled_copy[:,:,:,1] = val_scaled_reshape[:,:,:,1]
    val_scaled_copy[:,:,:,2] = val_scaled_reshape[:,:,:,2]
    stop_time = -1
    iter_step = 0
    for j in range(0, end_time+1,live_predict_step):
        start = time.time()
        iter_step += 1
        for k in range(0,end_time):
            for l in range(0,windows):
                # print(k)
                if(k > j):
                    val_scaled_copy[i,k,l,0] = val_scaled_reshape[i,j,l,0]    
                else: 
                    val_scaled_copy[i,k,l,0] = val_scaled_reshape[i,k,l,0]
        test = val_scaled_copy[i % val_scaled_copy.shape[0],:,:,:] 
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
        r2error = r2_score(inv_y, inv_yhat)
        print('Test R2: %.9f' % (1-r2error))
        inv_yhat_out = min_max_scaler(yhat, 0, 1, temp_min, temp_max)
        inv_part_sacled = min_max_scaler(val_scaled_copy[i,:,0,0], 0, 1, temp_min, temp_max)
        soak_time = find_soak_time(val_data[i,3000,1], val_data[i,:,2], val_data[i,:,1], inv_yhat_out, .05)
        true_soak_time = find_soak_time(val_data[i,3000,1], val_data[i,:,2], val_data[i,:,1], val_data[i,:,0], .05)
        if(j >= val_data.shape[1]):
            current_run_time = val_data[i,-1,2]
        else:
            current_run_time = val_data[i,j,2]
        print("Prediction at time:  {:.7}".format(current_run_time))
        print("Estimated Soak Time: {:.7}".format(soak_time))
        print("Real Soak Time:      {:.7}".format(true_soak_time))
        if(soak_time == None):
            soak_time = val_data[i,-1,1]
        soak_time_arr += [soak_time]
        if(soak_time < current_run_time and stop_time == -1):
            stop_time = current_run_time
        elif(stop_time != -1):
            print("Run Has Ended Past Soak Time Predicted of: {}".format(soak_time))
        pyplot.plot(val_data[i,:,2], val_data[i,:, 1], label='Full Air Temperature') #Inner Temp
        pyplot.plot(val_data[i,:,2], val_data[i,:, 0], label='Part Temp') #Inner Temp
        pyplot.plot(val_data[i,:,2], inv_yhat_out,  label='Part Temperature Prediction')
        pyplot.plot(val_data[i,:,2], inv_part_sacled, label='Air Temperature Used to Predict') #Inner Temp
        pyplot.axvline(x=true_soak_time, label='True Soak Time', color = 'k')
        pyplot.axvline(x=soak_time, label="Soak Time Prediction", color = '#7f7f7f')
        pyplot.axvline(x=current_run_time, label="Time Prediction was Made",linestyle = "dashed")
        pyplot.hlines((val_data[i,3000,1] - val_data[i,3000,1]*.05),500,val_data[i,-1,2], color = 'g', label="Tolerance",linestyle = "dashed")
        pyplot.hlines((val_data[i,3000,1] + val_data[i,3000,1]*.05),500,val_data[i,-1,2], color = 'g',linestyle = "dashed")
        pyplot.title("Soak Time Predictions")
        pyplot.xlabel('Time [s]')
        pyplot.ylabel('Temperature [C]')
        pyplot.legend()
        fname = str(i)+"plot_prediction_{:03d}".format(iter_step)
        # pyplot.savefig(fname)
        # pyplot.close()
        pyplot.show()
        end = time.time()
        print("Time: {:.3}".format(end - start))


    for s in range(0,len(soak_time_arr)):
        pyplot.axvline(x=soak_time_arr[s], color = '#7f7f7f')
    pyplot.plot(val_data[i,:,2], val_data[i,:, 1], label='Air  Temperature') #Inner Temp
    pyplot.plot(val_data[i,:,2], val_data[i,:, 0], label='Part Temperature') #Inner Temp
    pyplot.plot(val_data[i,:,2], inv_yhat_out,  label='Part Temperature Prediction')
    #pyplot.plot(val_data[i,:,2], inv_part_sacled, label='Air Temperature Used to Predict') #Inner Temp
    pyplot.axvline(x=true_soak_time, label='True Soak Time', color = 'k')
    pyplot.axvline(x=soak_time, label="All Soak Time Predictions", color = '#7f7f7f')
    pyplot.axvline(x=stop_time, label="Final Soak Time Prediction", color = 'c')
    pyplot.hlines((val_data[i,3000,1] - val_data[i,3000,1]*.05),500,val_data[i,-1,2], color = 'g', label="Tolerance",linestyle = "dashed")
    pyplot.hlines((val_data[i,3000,1] + val_data[i,3000,1]*.05),500,val_data[i,-1,2], color = 'g',linestyle = "dashed")
    pyplot.title("Soak Time Predictions")
    pyplot.xlabel('Time [s]')
    pyplot.ylabel('Temperature [C]')
    pyplot.legend()
    fname = str(i)+"plot_prediction_" + str(iter_step)
    # pyplot.savefig(fname)
    # pyplot.close()S
    pyplot.show()
    x = np.linspace(0,current_run_time,int(end_time/live_predict_step)+1)
    pyplot.plot(x, soak_time_arr, label = 'Soak Time Prediction')
    pyplot.axhline(true_soak_time, label = 'True Soak Time', color = 'r')
    pyplot.plot(x,x, label = 'y = x')
    pyplot.plot("Soak Time Prediction vs When it was Made")
    pyplot.xlabel('Time Prediction was Made [s]')
    pyplot.ylabel('Soak Time Prediction [s]')
    pyplot.legend()
    pyplot.show()  

    # 