#retrieve data functions 
import argparse, os
import csv 
from numpy import genfromtxt
import numpy as np
from datafunction import interpolate_data
import feda
###
#Retrives CSV data from a folder specified by path
#Input cols_to_keep is list defining which columns should be loaded in
###
def get_data(feature_cols,truth_col,path,length, resize):
    # reshape input to be 3D [samples, timesteps, features]
    real_size = [] 
    if (path == None):
        print("ERROR: Use Command Line arg -p to set path to data")
        raise NotADirectoryError(path)
    vstack_flag = 0
    for filename in os.listdir(path):
        with open(path + filename) as csv_file:
            print(filename)
            my_data = genfromtxt(csv_file, delimiter=',')
            real_size += [my_data.shape[0]]
            mydata2 = np.zeros([1,length,len(feature_cols)+1])
            iter = 1
            for col in feature_cols:
                mydata2[0,:my_data.shape[0],iter] = my_data[:length,col]
                iter = iter + 1
            mydata2[0,:my_data.shape[0],0] = my_data[:length,truth_col]
            if vstack_flag == 0:
                raw_data = mydata2
                vstack_flag = 1
            else:
                raw_data = np.vstack((raw_data,mydata2))
    if(resize == True):
        raw_data = interpolate_data(raw_data,real_size, length)
        print(raw_data.shape)
    return raw_data

def get_data_source_transfer(feature_cols,truth_col,path,length, resize):
    # reshape input to be 3D [samples, timesteps, features]
    real_size = [] 
    if (path == None):
        print("ERROR: Use Command Line arg -p to set path to data")
        raise NotADirectoryError(path)
    vstack_flag = 0
    for filename in os.listdir(path):
        with open(path + filename) as csv_file:
            print(filename)
            my_data = genfromtxt(csv_file, delimiter=',')
            transformed_features, _truth = feda.source_transform(my_data, truth_col)
            real_size += [my_data.shape[0]]
            mydata2 = np.zeros([1,length,len(transformed_features[0])+1])
            iter = 1
            for col in range(len(transformed_features[0])):
                mydata2[0,:my_data.shape[0],iter] = transformed_features[:length,col]
                iter = iter + 1
            mydata2[0,:my_data.shape[0],0] = my_data[:length,truth_col]
            if vstack_flag == 0:
                raw_data = mydata2
                vstack_flag = 1
            else:
                raw_data = np.vstack((raw_data,mydata2))
    if(resize == True):
        raw_data = interpolate_data(raw_data,real_size, length)
        print(raw_data.shape)
    return raw_data

def get_data_target_transfer(feature_cols,truth_col,path,length, resize):
    # reshape input to be 3D [samples, timesteps, features]
    real_size = [] 
    if (path == None):
        print("ERROR: Use Command Line arg -p to set path to data")
        raise NotADirectoryError(path)
    vstack_flag = 0
    for filename in os.listdir(path):
        with open(path + filename) as csv_file:
            print(filename)
            my_data = genfromtxt(csv_file, delimiter=',')
            transformed_features, _truth = feda.target_transform(my_data, truth_col)
            real_size += [my_data.shape[0]]
            mydata2 = np.zeros([1,length,len(transformed_features[0])+1])
            iter = 1
            for col in range(len(transformed_features[0])):
                mydata2[0,:my_data.shape[0],iter] = transformed_features[:length,col]
                iter = iter + 1
            mydata2[0,:my_data.shape[0],0] = my_data[:length,truth_col]
            if vstack_flag == 0:
                raw_data = mydata2
                vstack_flag = 1
            else:
                raw_data = np.vstack((raw_data,mydata2))
    if(resize == True):
        raw_data = interpolate_data(raw_data,real_size, length)
        print(raw_data.shape)
    return raw_data