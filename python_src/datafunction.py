import math
import numpy as np
import os,sys
import random
from scipy.signal import butter, lfilter,freqz

# def convertfile(infile):
#     mat = loadmat(infile)
#     if(infile.find('outer') > 0):
#     #    print(infile.find('outer'))
#         mat = np.array(mat['outT'])
#     else:
#         mat = np.array(mat['centerT'])
#     return mat

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
        #for i in range(0, len_col):
        #extend all Cols
        data = np.append(data, data[-1,:][None], axis = 0 ) 
        data = np.delete(data, 0, axis = 0)
    data = np.column_stack((data, data_to_predict))
    return data
    
def reduce_data_size(data, new_size):
    #will take a 2d array and reduce the size of the cols
    #will take a 2d array and reduce the size of the cols
    #pass data as data np array 
    num_row = data.shape[0]
    num_col = data.shape[1]
    print(num_row)
    print(num_col)
    new_data = np.zeros((new_size,num_col), float)
    for i in range(0,num_col):
        row = 0 
        for j in range(0, num_row, int(num_row/new_size)):
            new_data[row,i] = data[j,i]
            row = row + 1
            
    return new_data
def reduce_data_size3d(data, new_size):
    #will take a 3d array and reduce the size of the cols
    #pass data as data np array 
    flag_2d = 0
    num_row = data.shape[1]
    num_col = data.shape[2]
    new_data = np.zeros((new_size,num_col), float)
    out_data = np.zeros((1,new_size,num_col),float)
    for k in range(0, data.shape[0]):
        for i in range(0,num_col):
            row = 0 
            for j in range(0, num_row, int(num_row/new_size)):
                new_data[row,i] = data[j,i]
                row = row + 1
        if flag_2d == 0:
            flag_2d = 1
            out_data[0,:,:] = new_data
        else:
            out_data = np.vstack((out_data, new_data[None]))
    return out_data
    
def randomize_data(data, num_trials, trial_size):
    num_row = data.shape[0]
    num_col = data.shape[1]
    print(num_row)
    print(num_col)
    new_data = np.zeros((num_row,num_col), float)
    data_order = list(range(0,num_trials))
    print(data_order)
    random.shuffle(data_order)
    print(data_order)
    for i in range(0,num_col):
        for j in range(0,num_trials):
            print(new_data[(trial_size*j):(trial_size*(j+1)),i])
            print(data_order[j])
            print(data[(trial_size*data_order[j]):(trial_size*(data_order[j]+1)),i])
            new_data[(trial_size*j):(trial_size*(j+1)),i] = data[(trial_size*data_order[j]):(trial_size*(data_order[j]+1)),i]        
    return new_data
