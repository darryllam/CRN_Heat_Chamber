from operator import sub
import os
import matplotlib.pyplot as plt
import numpy as np
import math

plot_type ='all'

trans_1011 = [4.098300868836252, 3.6272868234648157, 4.487169185349144, 2.4616173863496438]
rf_1011 = [6.912708715200209, 5.667522443674665, 3.2222678628468215, 6.785496382174964]

trans_100 = [3.081145428660213, 9.303035006346791, 4.89743726779099, 3.779045157723486]
rf_100 = [5.45532928518972, 9.53996667803461, 7.500056798746185, 7.254354336368928]

trans_50 = [3.770871823560886, 4.844408690869378, 5.0241718030955065, 2.4441788863515477]
rf_50 = [5.462521908935398, 10.059120706772125, 5.104070827199658, 6.902170658023943]


if plot_type == 'all':
    all_bin_max = 11
    bins = np.linspace(0, 11, 20)

histogram = plt.figure()

ax = histogram.add_subplot(1,1,1)

ax.set_title('Histogram of Random Forest RMSE Values')

ax.hist(trans_1011, bins, alpha=0.5, label = 'TL 0.4% Target Data')
ax.hist(rf_1011, bins, alpha=0.5, label = 'RF 0.4% Target Data')
ax.hist(trans_100, bins, alpha=0.5, label = 'TL 4% Target Data')
ax.hist(rf_100, bins, alpha=0.5, label='RF 4% Target Data')
ax.hist(trans_50, bins, alpha=0.5, label='TL 8% Target Data')
ax.hist(rf_50, bins, alpha=0.5, label='RF 8% Target Data')

ax.minorticks_on()
# ax.set_xticks(np.arange(0, 11, 20))
# for label in ax.xaxis.get_ticklabels()[::2]:
#     label.set_visible(False)

plt.legend(loc='upper right')
if plot_type == 'all':
    plt.legend(fontsize='x-small', ncol=1,handleheight=2.4, labelspacing=0.05)
# plt.xlim([0,580])
plt.show()

