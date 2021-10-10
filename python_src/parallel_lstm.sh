start python lstmMethod.py -tp ../Updated_Dataset_Jan_2021/Formatted_Data/No_TL_LSTM/Train1/ -vp ../Updated_Dataset_Jan_2021/Formatted_Data/No_TL_LSTM/Test1/  -o relu_cyclical_weights_train_lstm_simple_Jan2021_run1.h5 -w 1 -vcol 1 -stcol 0 -tmcol 2 -srcol 3 4 5 6 7 -min_temp 20 -max_temp 60 -e 1000 -state 0 -cyclical &
start python lstmMethod.py -tp ../Updated_Dataset_Jan_2021/Formatted_Data/No_TL_LSTM/Train2/ -vp ../Updated_Dataset_Jan_2021/Formatted_Data/No_TL_LSTM/Test2/  -o relu_cyclical_weights_train_lstm_simple_Jan2021_run2.h5 -w 1 -vcol 1 -stcol 0 -tmcol 2 -srcol 3 4 5 6 7 -min_temp 20 -max_temp 60 -e 1000 -state 0 -cyclical &
start python lstmMethod.py -tp ../Updated_Dataset_Jan_2021/Formatted_Data/No_TL_LSTM/Train3/ -vp ../Updated_Dataset_Jan_2021/Formatted_Data/No_TL_LSTM/Test3/  -o relu_cyclical_weights_train_lstm_simple_Jan2021_run3.h5 -w 1 -vcol 1 -stcol 0 -tmcol 2 -srcol 3 4 5 6 7 -min_temp 20 -max_temp 60 -e 1000 -state 0 -cyclical &
start python lstmMethod.py -tp ../Updated_Dataset_Jan_2021/Formatted_Data/No_TL_LSTM/Train4/ -vp ../Updated_Dataset_Jan_2021/Formatted_Data/No_TL_LSTM/Test4/  -o relu_cyclical_weights_train_lstm_simple_Jan2021_run4.h5 -w 1 -vcol 1 -stcol 0 -tmcol 2 -srcol 3 4 5 6 7 -min_temp 20 -max_temp 60 -e 1000 -state 0 -cyclical


echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running