from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use the 970 only
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the 1070Ti only
os.environ["PATH"] += os.pathsep + 'C:/Users/temp3rr0r/Anaconda3/Library/bin/graphviz'

import random
import sys, time
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from pyswarm.pso import pso
from matplotlib import pyplot
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import GaussianNoise, Dense, LSTM, SimpleRNN, Bidirectional, BatchNormalization
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import pandas as pd
import os.path
import gc
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def trainNar(x, *args):

    trainNar.counter += 1
    x_data, y_data = args
    full_nar_parameters = x.copy()

    print("\n=============\n")
    print("Nar iteration {} using:\n\t{}".format(trainNar.counter, x[6:15]))

    x = np.rint(x).astype(np.int32)
    # TODO: test avoid loss NaNs, by removing rmsprop & sgd
    optimizers = ['adadelta', 'adagrad', 'nadam', 'adamax', 'adam']  # , 'rmsprop', 'sgd']
    batch_size = x[0]
    epoch_size = x[1]
    print("\tbatch_size: {}, epoch_size: {} Optimizer: {}, LSTM Unit sizes: {} Batch Normalization/Gaussian Noise: {}"
          .format(x[0], x[1], optimizers[x[2]], x[3:6], x[15:21]))

    # x_data, x_data_holdout = x_data[:-365, :], x_data[-365:, :]
    # y_data, y_data_holdout = y_data[:-365, :], y_data[-365:, :]
    x_data, x_data_holdout = x_data[:-365], x_data[-365:]
    y_data, y_data_holdout = y_data[:-365], y_data[-365:]

    totalFolds = 10
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
        rnn_kwargs = {'units': 64, 'return_sequences': False,'implementation': 2, 'input_shape': (x_data_3d.shape[1], x_data_3d.shape[2]), 'activation': 'sigmoid'}
        model.add(SimpleRNN(**rnn_kwargs))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mean_squared_error', optimizer='sgd')

        # Plot model architecture
        plot_model(model, show_shapes=True, to_file='narModelIter{}.png'.format(trainNar.counter))
        SVG(model_to_dot(model).create(prog='dot', format='svg'))

        early_stop = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto'),
                      ReduceLROnPlateau(patience=3, verbose=1),
                      ModelCheckpoint(filepath='best_model_nar.h5', monitor='val_loss', save_best_only=True)]

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

        # Undo standardization
        # y_test = (y_data[test] * sensor_std) + sensor_mean
        # prediction = (prediction * sensor_std) + sensor_mean
        y_validation = y_data[validation]

        # Calc mse/rmse
        mse = mean_squared_error(prediction, y_validation)
        print('Validation MSE: %.3f' % mse)
        mse_scores.append(mse)
        rmse = sqrt(mse)
        print('Validation RMSE: %.3f' % rmse)

        prediction.resize((prediction.shape[0], 1))
        y_validation.resize((prediction.shape[0], 1))
        smape = (2.0 / prediction.shape[0]) * \
                sum(np.nan_to_num(np.fabs(prediction - y_validation)
                / (prediction + y_validation)))  # Symmetric Mean Absolute Percent Error (SMAPE)
        print('Validation SMAPE: {}'.format(smape))
        smape_scores.append(smape)

        full_x = x_data.copy()
        full_y = y_data.copy()
        full_prediction = model.predict(x_data)
        # full_expected_ts = (y_data * sensor_std) + sensor_mean
        # full_prediction = (full_prediction * sensor_std) + sensor_mean
        full_expected_ts = y_data

        full_rmse = sqrt(mean_squared_error(full_prediction, full_expected_ts))
        print('Full Data RMSE: %.3f' % full_rmse)

        full_prediction.resize((full_prediction.shape[0], 1))
        full_expected_ts.resize((full_prediction.shape[0], 1))
        full_smape = (2.0 / full_prediction.shape[0]) * \
                sum(np.nan_to_num(np.fabs(full_prediction - full_expected_ts))
                / (full_prediction + full_expected_ts))  # Symmetric Mean Absolute Percent Error (SMAPE)
        print('Full Data SMAPE: {}'.format(full_smape))

    # TODO: Undo standardization
    mean_smape = np.mean(smape_scores)
    std_smape = np.std(smape_scores)
    print('Cross validation Full Data SMAPE: {} +/- {}'.format(round(mean_smape * 100, 2), round(std_smape * 100, 2)))
    min_smape = pd.read_pickle("min_smape.pkl")['min_smape'][0]

    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    print('Cross validation Full Data MSE: {} +/- {}'.format(round(mean_mse * 100, 2), round(std_mse * 100, 2)))
    min_mse = pd.read_pickle("min_mse.pkl")['min_mse'][0]

    holdout_prediction = model.predict(x_data_holdout)

    # TODO: Undo standardization
    # y_data_holdout = (y_data_holdout * sensor_std) + sensor_mean
    # holdout_prediction = (holdout_prediction * sensor_std) + sensor_mean

    holdout_rmse = sqrt(mean_squared_error(holdout_prediction, y_data_holdout))
    print('Holdout Data RMSE: %.3f' % holdout_rmse)
    holdout_prediction.resize((holdout_prediction.shape[0], 1))
    y_data_holdout.resize((holdout_prediction.shape[0], 1))
    holdout_smape = (2.0 / holdout_prediction.shape[0]) * \
                    sum(np.nan_to_num(np.fabs(holdout_prediction - y_data_holdout))
                        / (holdout_prediction + y_data_holdout))  # Symmetric Mean Absolute Percent Error (SMAPE)
    print('Holdout Data SMAPE: {}'.format(holdout_smape))
    holdout_mape = np.mean(np.abs((y_data_holdout - holdout_prediction) / y_data_holdout))
    print('Holdout Data MAPE: {}'.format(holdout_mape))
    holdout_mse = mean_squared_error(holdout_prediction, y_data_holdout)
    print('Holdout Data MSE: {}'.format(holdout_mse))
    # Index Of Agreement: https://cirpwiki.info/wiki/Statistics#Index_of_Agreement
    #  IA = 1 - mean((xc(:)-xm(:)).^2)/max(mean((abs(xc(:)-mean(xm(:)))+abs(xm(:)-mean(xm(:)))).^2),eps)
    # x_m: measured values, x_c: final calculated values
    # holdout_ioa = 1 - ((holdout_mse) /
    #                    (np.fabs(holdout_prediction - np.mean(holdout_prediction))
    #                     + np.fabs(y_data_holdout - np.mean(y_data_holdout))) ** 2)
    holdout_ioa = 1 - (np.sum((y_data_holdout - holdout_prediction) ** 2)) / (np.sum(
        (np.abs(holdout_prediction - np.mean(y_data_holdout)) + np.abs(y_data_holdout - np.mean(y_data_holdout))) ** 2))
    print('Holdout Data IOA: {}'.format(holdout_ioa))
    with open('narRuns.csv', 'a') as file:
        # TODO: data to store:
        # datetime, iteration, cvMseMean, cvMseStd,
        # cvSmapeMean, cvSmapeStd, holdoutRmse, holdoutSmape, holdoutMape,
        # holdoutMse, holdoutIoa, full_mlp_parameters
        file.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(str(int(time.time())), str(trainNar.counter),
                                                                  str(mean_mse), str(std_mse), str(mean_smape), str(std_smape), str(holdout_rmse), str(holdout_smape[0]),
                                                                  str(holdout_mape), str(holdout_mse), str(holdout_ioa), 0
                                                                  #''.join(str(e) for e in full_mlp_parameters)
                                                                  ))

    # if mean_smape < min_smape:
    if mean_mse < min_mse:  # TODO: do not pickle
    # if False:

        print("New min_smape: {}".format(mean_smape))
        print("New min_mse: {}".format(mean_mse))
        original_df = pd.DataFrame({"min_smape": [mean_smape]})
        original_df.to_pickle("min_smape.pkl")
        original_df1 = pd.DataFrame({"min_mse": [mean_mse]})
        original_df1.to_pickle("min_mse.pkl")
        original_df2 = pd.DataFrame({"full_nar_parameters": [full_nar_parameters]})
        original_df2.to_pickle("full_nar_parameters.pkl")
        model.summary()  # print layer shapes and model parameters

        # Plot history
        # pyplot.figure(1, figsize=(8, 6))  # Resolution 800 x 600
        # pyplot.title("Rand (iter: {}): Training History '{}'(Min MSE,Standardized,Early-stoppage patience: 5)".format(trainLstm.counter, optimizer))
        # pyplot.plot(history.history['loss'], label='train')
        # # if useValidation == True:
        # #     pyplot.plot(history.history['val_loss'], label='test')
        # pyplot.xlabel("Training Epoch")
        # pyplot.ylabel("MSE")
        # pyplot.grid(True)
        # pyplot.legend()
        # pyplot.show()

        # Plot validation data
        # pyplot.figure(1, figsize=(8, 6))  # Resolution 800 x 600
        # pyplot.title("Rand (iter: {}): Validation(last fold) data (RMSE: {}, SMAPE: {}%)".format(trainLstm.counter, np.round(rmse, 2), np.round(smape * 100, 2)))
        # pyplot.plot(prediction, label='prediction')
        # pyplot.plot(y_validation, label='expected')
        # pyplot.xlabel("Time step")
        # pyplot.ylabel("Sensor Value")
        # pyplot.grid(True)
        # pyplot.legend()
        # pyplot.show()

        # Plot test data
        pyplot.figure(1, figsize=(12, 10))  # Resolution 800 x 600
        pyplot.title("Nar (iter: {}): Test data (RMSE: {}, MAPE: {}%, IOA: {}%)".format(trainNar.counter, np.round(holdout_rmse, 2), np.round(holdout_mape * 100, 2), np.round(holdout_ioa * 100, 2)))
        pyplot.plot(holdout_prediction, label='prediction')
        pyplot.plot(y_data_holdout, label='expected')
        pyplot.xlabel("Time step")
        pyplot.ylabel("Sensor Value")
        pyplot.grid(True)
        pyplot.legend()
        pyplot.show()

        # Plot full data
        # pyplot.figure(1, figsize=(16, 12))  # Resolution 800 x 600
        # pyplot.title("Rand (iter: {}): Full data (RMSE: {}, SMAPE: {}%)".format(trainLstm.counter, np.round(full_rmse, 2), np.round(full_smape * 100, 2)))
        # pyplot.plot(full_prediction, label='prediction')
        # pyplot.plot(full_expected_ts, label='expected')
        # pyplot.xlabel("Time step")
        # pyplot.ylabel("Sensor Value")
        # pyplot.grid(True)
        # pyplot.legend()
        # pyplot.show()

        # Store model
        # serialize model to JSON
        model_json = model.to_json()
        with open("bidirectionalLstmArIotGardenerModel.json", "w") as json_file:
            json_file.write(model_json)
            print("Saved weights to disk")

        # serialize weights to HDF5
        model.save_weights("bidirectionalLstmArIotGardenerModelWeights.h5")
        print("Saved model to disk")

    del model  # Manually delete model
    from keras import backend as K
    K.clear_session()  # Manually clear_session with keras 2.1.6
    gc.collect()

    return mean_mse

def trainNarRnn(x_data, y_data):

    # TODO: loss parameter?
    # Rand with: batch_size, epoch_size, optimizer, 3x {'units', 'dropout', 'recurrent_dropout',
    #  gaussian noise, batch normalization }
    lb = [7,  # batch_size
          1000, 0,  # epoch_size, optimizer
          16, 16, 16,  # units
          0.01, 0.01, 0.01,  # dropout
          0.01, 0.01, 0.01,  # recurrent_dropout
          0.01, 0.01, 0.01,  # gaussian noise std
          0, 0, 0,  # gaussian_noise
          0, 0, 0]  # batch normalization
    ub = [512,  # TODO: 1024, # batch_size
          1000, 3,  # epoch_size, optimizer
          256, 256, 256,  # TODO: 1024, 1024, 1024  # units
          0.25, 0.25, 0.25,  # dropout
          0.25, 0.25, 0.25,  # recurrent_dropout
          1, 1, 1,  # gaussian noise std
          1, 1, 1,  # gaussian_noise
          1, 1, 1]  # batch normalization

    args = (x_data, y_data)

    trainNar.counter = 0  # Function call counter
    for i in range(1):
        x = [random.randint(lb[0], ub[0]),  # batch_size
             random.randint(lb[1], ub[1]), random.randint(lb[2], ub[2]),  # epoch_size, optimizer
             random.randint(lb[3], ub[3]), random.randint(lb[4], ub[4]), random.randint(lb[5], ub[5]),  # units
             random.uniform(lb[6], ub[6]), random.uniform(lb[7], ub[7]), random.uniform(lb[8], ub[8]),  # dropout
             random.uniform(lb[9], ub[9]), random.uniform(lb[10], ub[10]), random.uniform(lb[11], ub[11]),  # recurrent_dropout
             random.uniform(lb[12], ub[12]), random.uniform(lb[13], ub[13]), random.uniform(lb[14], ub[14]),  # gaussian noise std
             random.randint(lb[15], ub[15]), random.randint(lb[16], ub[16]), random.randint(lb[17], ub[17]), # gaussian_noise
             random.randint(lb[18], ub[18]), random.randint(lb[19], ub[19]), random.randint(lb[20], ub[20])]  # batch normalization
        trainNar(x, *args)

    # xopt1, fopt1 = pso(trainLstm, lb, ub, args=args)
    # # TODO: test larger swarm, more iterations
    # # pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
    # #     swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, minstep=1e-8,
    # #     minfunc=1e-8, debug=False)
    #
    # print('The optimum is at:')
    # print('    {}'.format(xopt1))
    # print('Optimal function value:')
    # print('    myfunc: {}'.format(fopt1))

if __name__ == "__main__":

    # TODO: Online learning: Only if new model is better... with batches of 512? with "1 month data" sliding window, updated every day
    # TODO: TimeDistributed? TimeDistributed wrapper layer and the need for some LSTM layers to return sequences rather than single values.
    # TODO: masking layer? Skips timesteps

    # TODO: GridSearch CV http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        # TODO: and https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

    sensor_mean = []
    sensor_std = []

    batch_size = 512  # 512 # Ful data RMSE, 60: 1.889, 32: 1.991, 256: 1.854, 512: 1.749
    epoch_size = 25  # 40

    print('Loading data...')
    # r = np.genfromtxt("BETN073_ts_standardized.csv", delimiter=',')#
    r = np.genfromtxt("BETN073.csv", delimiter=',')  #
    r = np.delete(r, [0], axis=1)  # Remove dates

    # dataFrameColumns = ['sensor.temperature', 'sensor.brightness', 'sensor.pressure', 'sensor.humidity',
    #     'sensor.flower1_temperature', 'sensor.flower1_conductivity', 'sensor.flower1_light_intensity',
    #     'sensor.flower1_moisture', 'sensor.yr_symbol', 'sun.sun', 'sensor.dark_sky_apparent_temperature',
    #     'sensor.dark_sky_cloud_coverage', 'sensor.dark_sky_humidity', 'sensor.dark_sky_temperature',
    #     'sensor.dark_sky_visibility', 'sensor.dark_sky_precip_intensity', 'sensor.dark_sky_precip_probability']
    r = np.delete(r, [1, 2, 3, 4, 5, 6], axis=1)  # Remove miFlora data & in house data except temperature

    # TODO: Undo standardization
    # sensor_mean = pd.read_pickle("BETN073_ts_mean.pkl")
    # sensor_std = pd.read_pickle("BETN073_ts_std.pkl")

    # print("\nStart Array r:\n {}".format(r[::5]))
    print("\nStart Array r:\n {}".format(r[0,0]))

    maxlen = r.shape[1] - 1
    print('Variables: {}'.format(maxlen))
    print('TimeSteps: {}'.format(r.shape[0]))
    proportion = 0.15

    x_data = r[:, 1:maxlen + 1]
    y_data = r[:, 0]

    print('x_train shape:', x_data.shape)

    y_data = np.array(y_data)

    x_data_3d = x_data.reshape(x_data.shape[0], 1, x_data.shape[1])  # reshape input to be 3D [samples, timesteps, features]

    if not os.path.exists("min_smape.pkl"):
        min_smape = 1
        original_df = pd.DataFrame({"min_smape": [min_smape]})
        original_df.to_pickle("min_smape.pkl")
    else:
        min_smape = pd.read_pickle("min_smape.pkl")['min_smape'][0]
        print("Previous min_smape: {}".format(min_smape))

        if os.path.exists("full_nar_parameters.pkl"):
            full_nar_parameters = pd.read_pickle("full_nar_parameters.pkl")['full_nar_parameters'][0]
            print("Previous full_nar_parameters: {}".format(full_nar_parameters))

    if not os.path.exists("min_mse.pkl"):
        min_mse = sys.float_info.max
        print("Previous min_mse: {}".format(min_mse))
        original_df = pd.DataFrame({"min_mse": [min_mse]})
        original_df.to_pickle("min_mse.pkl")
    else:
        min_mse = pd.read_pickle("min_mse.pkl")['min_mse'][0]
        print("Previous min_mse: {}".format(min_mse))

    trainNarRnn(x_data_3d, y_data)