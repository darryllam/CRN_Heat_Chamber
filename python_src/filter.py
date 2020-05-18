import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib import pyplot
import csv
from itertools import zip_longest

def kalman_filter(one_d_data):
    kf = KalmanFilter()

    kf.em(
    X       = one_d_data,
    n_iter  = 100,
    em_vars = [
        'initial_state_covariance',
        'transition_covariance',
        'observation_covariance'
    ])

    (smoothed_state_means, smoothed_state_covariances) = kf.smooth(one_d_data)
    return smoothed_state_means


def savitzky_filter(air_T, part_T):
    y1 = savgol_filter(air_T, 101, 3)
    y2 = savgol_filter(part_T, 101, 3)
    return y1, y2


data_path = '/home/daara/Documents/School/4th_Year/ENGR_499_Capstone/real_data_test/'
filename = 'max60run2.csv' # file name
col_list = ['time', 'air_temp', 'part_temp']

data = pd.read_csv(data_path+filename, usecols=col_list)

plt.title("Unfiltered Results")
plt.plot(data.time, data.air_temp, color = 'blue')
plt.plot(data.time,data.part_temp, color = 'green')
plt.legend(["air temp", "part temp"])
plt.show()

SG_air_temp, SG_part_temp = savitzky_filter(data.air_temp, data.part_temp)

# plt.title("Savitzky-Golay Filtered Results")
# plt.plot(data.time, SG_air_temp, color = 'blue')
# plt.plot(data.time, SG_part_temp, color = 'green')
# plt.legend(["air temp", "part temp"])
# plt.show()

KF_air_temp = kalman_filter(data.air_temp)
KF_part_temp = kalman_filter(data.part_temp)

# plt.title("Kalman Filtered Results")
# plt.plot(data.time, KF_air_temp, color = 'blue')
# plt.plot(data.time, KF_part_temp, color = 'green')
# plt.legend(["air temp", "part temp"])
# plt.show()

plt.title("Kalman Filtered vs Unfiltered Results")
plt.plot(data.time, data.air_temp, color = 'blue')
plt.plot(data.time, KF_air_temp, color = 'yellow')
plt.plot(data.time,data.part_temp, color = 'green')
plt.plot(data.time, KF_part_temp, color = 'red')
plt.legend(["Unfiltered air temp","Filtered air temp", "Unfiltered part temp", "Filtered part temp"])
plt.xlabel("Time [s]")
plt.ylabel("Temperature [°C]")
plt.show()

plt.title("Savitzky-Golay Filtered vs Unfiltered Results")
plt.plot(data.time, data.air_temp, color = 'blue')
plt.plot(data.time, SG_air_temp, color = 'yellow')
plt.plot(data.time,data.part_temp, color = 'green')
plt.plot(data.time, SG_part_temp, color = 'red')
plt.legend(["Unfiltered air temp","Filtered air temp", "Unfiltered part temp", "Filtered part temp"])
plt.xlabel("Time [s]")
plt.ylabel("Temperature [°C]")
plt.show()

# df = pd.DataFrame(data.time)

# df1 = pd.DataFrame(data.air_temp)
# df2 = pd.DataFrame(data.part_temp)
# df = df.merge(df1, left_index = True, right_index = True)
# df = df.merge(df2, left_index = True, right_index = True)
# df.to_csv("kalman_max40.csv", header=None, index=None)

kf = pd.DataFrame(data.time)
kf_air_df = pd.DataFrame(KF_air_temp)
kf_part_df = pd.DataFrame(KF_part_temp)
kf = kf.merge(kf_air_df, left_index = True, right_index = True)
kf = kf.merge(kf_part_df, left_index = True, right_index = True)
kf.to_csv("kalman_max60run2.csv", header=None, index=None)

sg = pd.DataFrame(data.time)
sg_air_df = pd.DataFrame(SG_air_temp)
sg_part_df = pd.DataFrame(SG_part_temp)
sg = sg.merge(sg_air_df, left_index = True, right_index = True)
sg = sg.merge(sg_part_df, left_index = True, right_index = True)
sg.to_csv("savitzky_max60run2.csv", header=None, index=None)
