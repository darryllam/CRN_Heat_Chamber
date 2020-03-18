#retrieve data functions 
import argparse, os
import csv 
from numpy import genfromtxt
import numpy as np
###
#Retrives CSV data from a folder specified by path
#Input cols_to_keep is list defining which columns should be loaded in
###
def get_data(feature_cols,truth_col, path,length):
    # reshape input to be 3D [samples, timesteps, features]
    if (path == None):
        print("ERROR: Use Command Line arg -p to set path to data")
        raise NotADirectoryError(path)
    vstack_flag = 0
    for filename in os.listdir(path):
        with open(path + filename) as csv_file:
            #print(filename)
            my_data = genfromtxt(csv_file, delimiter=',')
            mydata2 = np.zeros([1,length,len(feature_cols)+1])
            iter = 1
            for col in feature_cols:
                mydata2[0,:,iter] = my_data[:length,col]
                iter = iter + 1
            mydata2[0,:,0] = my_data[:length,truth_col]
            if vstack_flag == 0:
                raw_data = mydata2
                vstack_flag = 1
            else:
                raw_data = np.vstack((raw_data,mydata2))

    return raw_data