from operator import sub
import os
import matplotlib.pyplot as plt
import numpy as np
import math

plot_type ='all'

cyclical_tl_50 = [6.391, 9.422, 2,285, 8,3615]
cyclical_tl_100 = [10.655, 9.876, 12.709, 3.205]
cyclical_tl_1011 = [575.452, 10.324, 2.925, 3.199]

tl_50 = [10.655, 9.876, 12.709, 3.205]
tl_100 = [71.906, 10.264, 8.733, 1.512]
tl_1011 = [957.161, 7.689, 3.649, 4.088]

cyclical_50 = [37.573, 12.907, 17.681, 11.453]
cyclical_100 = [185.046, 11.281, 2.266, 8.69]
cyclical_1011 = [95.795, 10.999, 2.431, 1.99]

standard_50 = [9.769, 12.188, 11.76, 6.472]
standard_100 = [83.893, 10.264, 8.733, 1.512]
standard_1011 = [167.373, 9.592, 4.485, 47.426]

if plot_type == 'all':
    all_bin_max = 1000
    all_bin_count = math.ceil(all_bin_max/5)
    bins = np.linspace(0, all_bin_max, all_bin_count)
    cyclical_tl_bin_max = 580
    tl_max = 955
    cyclical_max = 190
    standard_max = 170
else:
    if plot_type == 'cyclical_tl':   
        cyclical_tl_bin_count = math.ceil(cyclical_tl_bin_max/5)
        bins = np.linspace(0, cyclical_tl_bin_max, cyclical_tl_bin_count)
    if plot_type == 'cyclical':
        cyclical_bin_count = math.ceil(190/5)
        bins = np.linspace(0, cyclical_max, cyclical_bin_count)
    if plot_type == 'tl':
        tl_bin_count = math.ceil(tl_max/5)
        bins = np.linspace(0, tl_max, tl_bin_count)
    if plot_type == 'standard':
        standard_bin_count = math.ceil(standard_max/5)
        bins = np.linspace(standard_max, standard_bin_count)

overall_max = 0

histogram = plt.figure()

ax = histogram.add_subplot(1,1,1)

ax.set_title('Histogram of LSTM RMSE Values')

if plot_type == 'cyclical_tl' or plot_type == 'all':
    ax.hist(cyclical_tl_50, bins, alpha=0.5, label = 'Cyclical TL 8% Target Data')
    ax.hist(cyclical_tl_100, bins, alpha=0.5, label = 'Cyclical TL 4% Target Data')
    ax.hist(cyclical_tl_1011, bins, alpha=0.5, label = 'Cyclical TL 0.4% Target Data')
    if cyclical_tl_bin_max > overall_max:
        overall_max = cyclical_tl_bin_max
if plot_type == 'tl' or plot_type == 'all':
    ax.hist(tl_50, bins, alpha=0.5, label = 'TL 8% Target Data')
    ax.hist(tl_100, bins, alpha=0.5, label = 'TL 4% Target Data')
    ax.hist(tl_1011, bins, alpha=0.5, label = 'TL 0.4% Target Data')
    if tl_max > overall_max:
        overall_max = tl_max
if plot_type == 'cyclical' or plot_type == 'all':
    ax.hist(cyclical_50, bins, alpha=0.5, label = 'Cyclical LSTM 8% Target Data')
    ax.hist(cyclical_100, bins, alpha=0.5, label = 'Cyclical LSTM 4% Target Data')
    ax.hist(cyclical_1011, bins, alpha=0.5, label = 'Cyclical LSTM 0.4% Target Data')
    if cyclical_max > overall_max:
        overall_max = cyclical_max
if plot_type == 'standard' or plot_type == 'all':
    ax.hist(standard_50, bins, alpha=0.5, label = 'LSTM 8% Target Data')
    ax.hist(standard_100, bins, alpha=0.5, label = 'LSTM 4% Target Data')
    ax.hist(standard_1011, bins, alpha=0.5, label = 'LSTM 0.4% Target Data')
    if standard_max > overall_max:
        overall_max = standard_max


ax.minorticks_on()
ax.set_xticks(np.arange(0, overall_max, 50))
for label in ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)

plt.legend(loc='upper right')
if plot_type == 'all':
    plt.legend(fontsize='x-small', ncol=2,handleheight=2.4, labelspacing=0.05)
# plt.xlim([0,580])
plt.show()

