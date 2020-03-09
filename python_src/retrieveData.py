#retrieve data functions 
import argparse, os
import csv 
from numpy import genfromtxt
import numpy as np

def dir_path(string):
    string2 = os.getcwd() + string
    if os.path.isdir(string2):
        return string2
    elif os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
 
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='file path', type=dir_path)
    return parser.parse_args()

def get_data(cols_to_keep):
    # reshape input to be 3D [samples, timesteps, features]
    vstack_flag = 0
    parsed_args = parse_arguments()
    path = parsed_args.path
    for filename in os.listdir(parsed_args.path):
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