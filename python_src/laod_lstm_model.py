from keras.models import load_model

data_len = 3600

raw_data = get_data([0,1,5])
scalers = {}
scaled = raw_data
for i in range(scaled.shape[0]):
    scalers[i] = MinMaxScaler(feature_range=(0, 1))
    scaled[i,:, :] = scalers[i].fit_transform(raw_data[i,:,:]) 
#scaled = scaler.fit_transform(raw_data)

for t in range(0,scaled.shape[0]):
    scaled[t,:,:] =  delay_series(scaled[t,:,1:],scaled[t,:,0],5)

model = load_model(file_name)

for i in range(0,number_of_trials):
    test = scaled[i,:, :] 
    test_X, test_y = test[:, :-1], test[:, -1]
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    yhat = model.predict(test_X, batch_size = local_batch_size)
    test_shape = test_X.shape[2]

    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    scaler = MinMaxScaler(feature_range=(0,1)).fit(inv_yhat)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
    model.save_weights(file_name)
    print('Test RMSE: %.3f' % rmse)
    #yhat = model.predict(test_X[data_len*i:data_len*(i+1)-1,:,:])
    pyplot.plot(scaled[i,:, -1]) #select first trial
    pyplot.plot(yhat)
    pyplot.plot(scaled[i,:,0])
    pyplot.plot(scaled[i,:,1])
    pyplot.show()