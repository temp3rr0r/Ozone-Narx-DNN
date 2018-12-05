from __future__ import print_function
import sys
import time
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import GaussianNoise, Dense, LSTM, Bidirectional, BatchNormalization
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import pandas as pd
import gc
from sklearn.model_selection import TimeSeriesSplit


def trainModel(x, *args):
    trainModel.counter += 1
    modelLabel = trainModel.label
    dataManipulation = trainModel.dataManipulation
    x_data, y_data = args
    full_model_parameters = x.copy()

    print("\n=============\n")
    print("{} iteration {} using:\n\t{}".format(modelLabel, trainModel.counter, x[6:15]))

    dropout1 = x[6]
    dropout2 = x[7]
    dropout3 = x[8]
    recurrent_dropout1 = x[9]
    recurrent_dropout2 = x[10]
    recurrent_dropout3 = x[11]

    # Gaussian noise
    noise_stddev1 = x[12]
    noise_stddev2 = x[13]
    noise_stddev3 = x[14]

    x = np.rint(x).astype(np.int32)
    optimizers = ['adadelta', 'adagrad', 'nadam', 'adamax',
                  'adam']  # , 'rmsprop', 'sgd'] # Avoid loss NaNs, by removing rmsprop & sgd
    batch_size = x[0]
    epoch_size = x[1]
    optimizer = optimizers[x[2]]
    units1 = x[3]
    units2 = x[4]
    units3 = x[5]

    # Batch normalization
    useBatchNormalization1 = x[15]
    useBatchNormalization2 = x[16]
    useBatchNormalization3 = x[17]
    use_gaussian_noise1 = x[18]
    use_gaussian_noise2 = x[19]
    use_gaussian_noise3 = x[20]

    print("\tbatch_size: {}, epoch_size: {} Optimizer: {}, LSTM Unit sizes: {} Batch Normalization/Gaussian Noise: {}"
          .format(x[0], x[1], optimizers[x[2]], x[3:6], x[15:21]))

    x_data, x_data_holdout = x_data[:-365], x_data[-365:]
    y_data, y_data_holdout = y_data[:-365], y_data[-365:]

    totalFolds = 2
    timeSeriesCrossValidation = TimeSeriesSplit(n_splits=totalFolds)
    # timeSeriesCrossValidation = KFold(n_splits=totalFolds)

    smape_scores = []
    mse_scores = []
    current_fold = 0
    for train, validation in timeSeriesCrossValidation.split(x_data, y_data):
        current_fold += 1
        print("Current Fold: {}/{}".format(current_fold, totalFolds))

        # create model
        model = Sequential()
        lstm_kwargs = {'units': units1, 'dropout': dropout1, 'recurrent_dropout': recurrent_dropout1,
                       'return_sequences': True,
                       'implementation': 2}
        model.add(Bidirectional(LSTM(**lstm_kwargs), input_shape=(
        x_data.shape[1], x_data.shape[2])))  # input_shape: rows: n, timestep: 1, features: m

        if use_gaussian_noise1 == 1:
            model.add(GaussianNoise(noise_stddev1))
        if useBatchNormalization1 == 1:
            model.add(BatchNormalization())
        lstm_kwargs['units'] = units2
        lstm_kwargs['dropout'] = dropout2
        lstm_kwargs['recurrent_dropout'] = recurrent_dropout2
        model.add(Bidirectional(LSTM(**lstm_kwargs)))
        if use_gaussian_noise2 == 1:
            model.add(GaussianNoise(noise_stddev2))
        if useBatchNormalization2 == 1:
            model.add(BatchNormalization())
        lstm_kwargs['units'] = units3
        lstm_kwargs['dropout'] = dropout3
        lstm_kwargs['recurrent_dropout'] = recurrent_dropout3
        lstm_kwargs['return_sequences'] = False
        model.add(Bidirectional(LSTM(**lstm_kwargs)))
        if use_gaussian_noise3 == 1:
            model.add(GaussianNoise(noise_stddev3))
        if useBatchNormalization3 == 1:
            model.add(BatchNormalization())
        model.add(Dense(y_data.shape[1]))
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        # TODO: do not store model on every step
        # early_stop = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto'),
        #               ReduceLROnPlateau(patience=3, verbose=1),
        #               ModelCheckpoint(filepath='foundModels/best_model_{}.h5'.format(modelLabel), monitor='val_loss',
        #                               save_best_only=True)]

        early_stop = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto'),
                      ReduceLROnPlateau(patience=3, verbose=1)]

        try:
            history = model.fit(x_data[train], y_data[train],
                                verbose=2,
                                batch_size=batch_size,
                                epochs=epoch_size,
                                validation_data=(x_data[validation], y_data[validation]),
                                callbacks=early_stop)
        except ValueError:
            print("Value Error exception: Model fit exception. Trying again...")
            history = model.fit(x_data[train], y_data[train],
                                verbose=2,
                                batch_size=batch_size,
                                epochs=epoch_size,
                                validation_data=(x_data[validation], y_data[validation]),
                                callbacks=early_stop)
        except:
            print("Exception: Returning max float value for this iteration.")

            del model  # Manually delete model
            from keras import backend as K
            K.clear_session()  # Manually clear_session with keras 2.1.6
            gc.collect()

            return sys.float_info.max

        prediction = model.predict(x_data[validation])
        y_validation = y_data[validation]

        if dataManipulation["scale"] == 'standardize':
            sensor_mean = pd.read_pickle("data/BETN073_ts_mean.pkl")
            sensor_std = pd.read_pickle("data/BETN073_ts_std.pkl")
            if trainModel.counter == 1:
                print("Un-standardizing...")
                print("sensor_mean:")
                print(sensor_mean)
                print("sensor_std:")
                print(sensor_std)
                print(np.array(sensor_mean)[0:y_data.shape[1]])
            sensor_mean = np.array(sensor_mean)
            sensor_std = np.array(sensor_std)
            prediction = (prediction * sensor_std[0:y_data.shape[1]]) + sensor_mean[0:y_data.shape[1]]
            y_validation = (y_validation * sensor_std[0:y_data.shape[1]]) + sensor_mean[0:y_data.shape[1]]
        elif dataManipulation["scale"] == 'normalize':
            sensor_min = pd.read_pickle("data/BETN073_ts_min.pkl")
            sensor_max = pd.read_pickle("data/BETN073_ts_max.pkl")
            if trainModel.counter == 1:
                print("Un-normalizing...")
                print("sensor_min:")
                print(sensor_min)
                print("sensor_max:")
                print(sensor_max)
                print(np.array(sensor_min)[0:y_data.shape[1]])
            sensor_min = np.array(sensor_min)
            sensor_max = np.array(sensor_max)
            prediction = prediction * (sensor_max[0:y_data.shape[1]] - sensor_min[0:y_data.shape[1]]) + sensor_min[0:y_data.shape[1]]
            y_validation = y_validation * (sensor_max[0:y_data.shape[1]] - sensor_min[0:y_data.shape[1]]) + sensor_min[0:y_data.shape[1]]

        # Calc mse/rmse
        mse = mean_squared_error(prediction, y_validation)
        print('Validation MSE: %.3f' % mse)
        mse_scores.append(mse)
        rmse = sqrt(mse)
        print('Validation RMSE: %.3f' % rmse)

        smape = 0.01 * (100 / len(y_validation) * np.sum(2 * np.abs(prediction - y_validation) /
                                                         (np.abs(y_validation) + np.abs(prediction))))
        print('Validation SMAPE: {}'.format(smape))
        smape_scores.append(smape)

        full_x = x_data.copy()
        full_y = y_data.copy()
        full_prediction = model.predict(x_data)
        full_expected_ts = y_data

        if dataManipulation["scale"] == 'standardize':
            full_prediction = (full_prediction * sensor_std[0:y_data.shape[1]]) + sensor_mean[0:y_data.shape[1]]
            full_expected_ts = (full_expected_ts * sensor_std[0:y_data.shape[1]]) + sensor_mean[0:y_data.shape[1]]
        elif dataManipulation["scale"] == 'normalize':
            full_prediction = full_prediction * (sensor_max[0:y_data.shape[1]] - sensor_min[0:y_data.shape[1]]) + sensor_min[0:y_data.shape[1]]
            full_expected_ts = full_expected_ts * (sensor_max[0:y_data.shape[1]] - sensor_min[0:y_data.shape[1]]) + sensor_min[0:y_data.shape[1]]

        full_rmse = sqrt(mean_squared_error(full_prediction, full_expected_ts))
        print('Full Data RMSE: %.3f' % full_rmse)

        full_smape = 0.01 * (100 / len(full_expected_ts) * np.sum(
            2 * np.abs(full_prediction - full_expected_ts) / (np.abs(full_expected_ts) + np.abs(full_prediction))))
        print('Full Data SMAPE: {}'.format(full_smape))


    # Plot model architecture
    plot_model(model, show_shapes=True, to_file='foundModels/{}ModelIter{}.png'.format(modelLabel, trainModel.counter))
    SVG(model_to_dot(model).create(prog='dot', format='svg'))

    mean_smape = np.mean(smape_scores)
    std_smape = np.std(smape_scores)
    print('Cross validation Full Data SMAPE: {} +/- {}'.format(round(mean_smape * 100, 2), round(std_smape * 100, 2)))

    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    print('Cross validation Full Data MSE: {} +/- {}'.format(round(mean_mse * 100, 2), round(std_mse * 100, 2)))
    min_mse = pd.read_pickle("foundModels/min_mse.pkl")['min_mse'][0]

    holdout_prediction = model.predict(x_data_holdout)

    if dataManipulation["scale"] == 'standardize':
        holdout_prediction = (holdout_prediction * sensor_std[0:y_data.shape[1]]) + sensor_mean[0:y_data.shape[1]]
        y_data_holdout = (y_data_holdout * sensor_std[0:y_data.shape[1]]) + sensor_mean[0:y_data.shape[1]]
    elif dataManipulation["scale"] == 'normalize':
        holdout_prediction = holdout_prediction * (sensor_max[0:y_data.shape[1]] - sensor_min[0:y_data.shape[1]]) \
                             + sensor_min[0:y_data.shape[1]]
        y_data_holdout = y_data_holdout * (sensor_max[0:y_data.shape[1]] - sensor_min[0:y_data.shape[1]]) \
                         + sensor_min[0:y_data.shape[1]]

    holdout_rmse = sqrt(mean_squared_error(holdout_prediction, y_data_holdout))
    print('Holdout Data RMSE: %.3f' % holdout_rmse)
    holdout_smape = 0.01 * (100/len(y_data_holdout) * np.sum(2 * np.abs(holdout_prediction - y_data_holdout) /
                                                             (np.abs(y_data_holdout) + np.abs(holdout_prediction))))

    print('Holdout Data SMAPE: {}'.format(holdout_smape))
    holdout_mape = np.mean(np.abs((y_data_holdout - holdout_prediction) / y_data_holdout))
    print('Holdout Data MAPE: {}'.format(holdout_mape))
    holdout_mse = mean_squared_error(holdout_prediction, y_data_holdout)
    print('Holdout Data MSE: {}'.format(holdout_mse))
    # Index Of Agreement: https://cirpwiki.info/wiki/Statistics#Index_of_Agreement
    holdout_ioa = 1 - (np.sum((y_data_holdout - holdout_prediction) ** 2)) / (np.sum(
        (np.abs(holdout_prediction - np.mean(y_data_holdout)) + np.abs(y_data_holdout - np.mean(y_data_holdout))) ** 2))
    print('Holdout Data IOA: {}'.format(holdout_ioa))
    with open('logs/{}Runs.csv'.format(modelLabel), 'a') as file:
        # Data to store:
        # datetime, iteration, cvMseMean, cvMseStd,
        # cvSmapeMean, cvSmapeStd, holdoutRmse, holdoutSmape, holdoutMape,
        # holdoutMse, holdoutIoa, full_pso_parameters
        file.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(str(int(time.time())), str(trainModel.counter),
            str(mean_mse), str(std_mse), str(mean_smape), str(std_smape), str(holdout_rmse), str(holdout_smape),
            str(holdout_mape), str(holdout_mse), str(holdout_ioa), full_model_parameters.tolist()))

    if mean_mse < min_mse:
        print("New min_mse: {}".format(mean_mse))
        original_df1 = pd.DataFrame({"min_mse": [mean_mse]})
        original_df1.to_pickle("foundModels/min_mse.pkl")
        original_df2 = pd.DataFrame({"full_{}_parameters".format(modelLabel): [full_model_parameters]})
        original_df2.to_pickle("foundModels/full_{}_parameters.pkl".format(modelLabel))
        model.summary()  # print layer shapes and model parameters

        # Plot history
        pyplot.figure(figsize=(8, 6))  # Resolution 800 x 600
        pyplot.title("{} (iter: {}): Training History Last Fold".format(modelLabel, trainModel.counter))
        pyplot.plot(history.history['loss'], label='loss')
        pyplot.plot(history.history['val_loss'], label='val_loss')
        pyplot.xlabel("Training Epoch")
        pyplot.ylabel("MSE")
        pyplot.grid(True)
        pyplot.legend()
        # pyplot.show()
        pyplot.savefig("foundModels/{}Iter{}History.png".format(modelLabel, trainModel.counter))

        # Plot test data
        for i in range(holdout_prediction.shape[1]):
            pyplot.figure(figsize=(16, 12))  # Resolution 800 x 600
            pyplot.title("{} (iter: {}): Test data - Series {} (RMSE: {}, MAPE: {}%, IOA: {}%)"
                    .format(modelLabel, trainModel.counter, i, np.round(holdout_rmse, 2),
                            np.round(holdout_mape * 100, 2), np.round(holdout_ioa * 100, 2)))
            pyplot.plot(holdout_prediction[:,i], label='prediction')
            pyplot.plot(y_data_holdout[:,i], label='expected')
            pyplot.xlabel("Time step")
            pyplot.ylabel("Sensor Value")
            pyplot.grid(True)
            pyplot.legend()
            # pyplot.show()
            pyplot.savefig("foundModels/{}Iter{}Series{}Test.png".format(modelLabel,trainModel.counter, i))

        # Store model
        model_json = model.to_json() # serialize model to JSON
        with open("foundModels/bestModelArchitecture.json".format(modelLabel), "w") as json_file:
            json_file.write(model_json)
            print("Saved model to disk")
        model.save_weights("foundModels/bestModelWeights.h5".format(modelLabel)) # serialize weights to HDF5
        print("Saved weights to disk")

    del model  # Manually delete model
    from keras import backend as K
    K.clear_session()  # Manually clear_session with keras 2.1.6
    gc.collect()

    return mean_mse
