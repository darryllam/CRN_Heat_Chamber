#retrieve data functions 
import argparse, os
import csv 
from numpy import genfromtxt
import numpy as np
###
#Retrives CSV data from a folder specified by path
#Input cols_to_keep is list defining which columns should be loaded in
###
def get_data(cols_to_keep, path):
    # reshape input to be 3D [samples, timesteps, features]
    if (path == None):
        print("ERROR: Use Command Line arg -p to set path to data")
        raise NotADirectoryError(path)
    vstack_flag = 0
    for filename in os.listdir(path):
        with open(path + filename) as csv_file:
            print(filename)
            my_data = genfromtxt(csv_file, delimiter=',')
            mydata2 = np.zeros([1,my_data.shape[0],len(cols_to_keep)])
            iter = 0
            for col in cols_to_keep:
                mydata2[0,:,iter] = my_data[:,col]
                iter = iter + 1
            if vstack_flag == 0:
                raw_data = mydata2
                vstack_flag = 1
            else:
                raw_data = np.vstack((raw_data,mydata2))

    return raw_data