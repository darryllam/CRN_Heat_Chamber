import os
import pickle
import matplotlib.pyplot as plt

fig_title = 'LSTM Cyclical 4% Target Samples'
dir_name = 'Jan2021_Results\\Reduced_Data_Cyclical\\New Folder'
loss_files = []
for file in os.listdir(dir_name):
    if file.endswith('.p'):
        loss_files.append(os.path.join(dir_name, file))
        print('loading ' + file)

train = []
test = []

max_val = 0

for i in range(0,len(loss_files)):
    with open(loss_files[i], 'rb') as file:
        pickle_model = pickle.load(file)
        train.append(pickle_model.get('loss'))
        test.append(pickle_model.get('val'))
        # if max(train[i]) > max_val:
        #     max_val = max(train[i])
        # if max(test[i]) > max_val:
        #     max_val = max(test[i])

fig, axs = plt.subplots(2,2)
axs[0, 0].plot(test[1], label='test')
axs[0, 0].plot(train[0], label='train', alpha=0.75)
axs[0, 0].set_title('Run 1 Loss')
axs[0, 0].legend(loc="upper right")
# axs[0, 0].set_ylim(0, max_val)
axs[0, 1].plot(test[1], label='test')
axs[0, 1].plot(train[1], label='train', alpha=0.75)
axs[0, 1].set_title('Run 2 Loss')
axs[0, 1].legend(loc="upper right")

# axs[0, 1].set_ylim(0, max_val)
axs[1, 0].plot(test[2], label='test')
axs[1, 0].plot(train[2], label='train', alpha=0.75)
axs[1, 0].set_title('Run 3 Loss')
axs[1, 0].legend(loc="upper right")
# axs[1, 0].set_ylim(0, max_val)
axs[1, 1].plot(test[3], label='test')
axs[1, 1].plot(train[3], label='train', alpha=0.75)
axs[1, 1].set_title('Run 4 Loss')
axs[1, 1].legend(loc="upper right")
# axs[1, 1].set_ylim(0, max_val)
fig.suptitle(fig_title)
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
    # for label in ax.xaxis.get_ticklabels()[::2]:
    #     label.set_visible(False)

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.show()
fig_name = fig_title + '.png'
plt.savefig(fig_name)
plt.clf()