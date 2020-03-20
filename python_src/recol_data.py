#retrieve data functions 
import argparse, os
import csv 
from numpy import genfromtxt
import numpy as np
from datafunction import interpolate_data
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
            #print(filename)
            my_data = genfromtxt(csv_file, delimiter=',')
            real_size += [my_data.shape[0]]
            mydata2 = np.zeros([length,6])
            print(mydata2.shape)
            mydata2[:,0] = my_data[:,5] #T
            mydata2[:,1] = my_data[:,1] #out
            mydata2[:,2] = my_data[:,0] #center
            mydata2[:,3] = my_data[:,2] #init 
            mydata2[:,4] = my_data[:,3] #rise
            mydata2[:,5] = my_data[:,4] #final
            np.savetxt(filename, mydata2, delimiter=",",fmt='%10.5f')
            


get_data([1,5],0, './data2/train/', 3600,True)
