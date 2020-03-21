import argparse, os
import csv
from numpy import genfromtxt
import numpy as np
import math
import numpy as np
import os,sys
import random
from scipy.signal import butter, lfilter,freqz
from scipy.interpolate import interp1d

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
    #pass data as data np array
    num_row = data.shape[0]
    num_col = data.shape[1]
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

def reshape_with_timestep(data, samples, time_steps):
    num_col = data.shape[2]
    num_trials = data.shape[0]
    print(data.shape)
    new_data = np.zeros((num_trials,samples,time_steps,num_col), float)
    for t in range(0,num_trials):
        for i in range(0,num_col):
            for j in range(0, samples):
                for k in range(0, time_steps):
                    new_data[t,j,k,i] = data[t,k+(time_steps*j),i]
    return new_data

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


def interpolate_data(data,real_size,new_size):
    num_row = data.shape[1]
    num_col = data.shape[2]
    new_data = np.ones((data.shape[0],new_size,num_col))
    new_data = -1 * new_data
    index = new_size
    for t in range(0, data.shape[0]):
        for i in range(0,num_col):
            if(real_size[t] >= data[t,:,i].shape[0]):
                return data
            x = np.linspace(0,real_size[t],real_size[t])
            f = interp1d(x, data[t,:real_size[t],i])
            x_new = np.linspace(0,real_size[t],3600)
            new_data[t,:,i] = f(x_new)
    return new_data

def min_max_scaler(data, input_min, input_max, out_min, out_max):
    data_std = (data - input_min) / (input_max - input_min)
    data_scaled = data_std * (out_max - out_min) + out_min
    return data_scaled

def short_term_average(data, short_term_len, tolerance):
    length = data.shape[0]
    sum = 0
    for i in range(1,short_term_len):
        #Average
        if(i+1 > length):
            return -1, 0
        sum += abs(data[-i]-data[-1]) / 50
    #print(sum)
    if(sum < tolerance):
        return length, sum
    else:
        return -1, sum
        
def find_soak_time(target, time, air_T, part_T, tolerance):
    lower_margin = target - target*tolerance
    upper_margin = target + target*tolerance

    for i in range(len(time)):
        if (air_T[i] > lower_margin) and (air_T[i] <= upper_margin) and (part_T[i] >= lower_margin):
            return time[i]
