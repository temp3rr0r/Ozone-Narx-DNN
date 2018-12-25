from __future__ import print_function
import gc
import sys
import time
import numpy as np
import tensorflow as tf  # TODO: Do use the faster (and less features) CudnnLSTM, cudnnGRU
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from sklearn.model_selection import TimeSeriesSplit


def trainModel(x, *args):

    startTime = time.time()  # training time per model

    trainModel.counter += 1
    modelLabel = trainModel.label
    modelFolds = trainModel.folds
    dataManipulation = trainModel.dataManipulation
    rank = dataManipulation["rank"]
    master = dataManipulation["master"]
    directory = dataManipulation["directory"]
    filePrefix = dataManipulation["filePrefix"]
    island = dataManipulation["island"]
    verbosity = dataManipulation["verbose"]
    multi_gpu = dataManipulation["multi_gpu"]

    x_data, y_data = args

    x = [32.269684115953126, 478.4579158867764, 2.4914987273745344, 291.55476719406147, 32.0, 512.0, 0.0812481431483004,
         0.01, 0.1445004524623349, 0.22335740221774894, 0.03443050512961357, 0.05488258021289669, 1.0,
         0.620275664519184, 0.34191582396595566, 0.9436131979280933, 0.4991752935129543, 0.4678261851228459, 0.0,
         0.355287972380982, 0.0]  # TODO: Temp set the same model to benchmark a specific DNN

    full_model_parameters = np.array(x.copy())
    if dataManipulation["fp16"]:
        full_model_parameters.astype(np.float32, casting='unsafe')

    print("\n=============\n")
    print("--- Rank {}: {} iteration {} using: {}".format(rank, modelLabel, trainModel.counter, x[6:15]))

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

    print("--- Rank {}: batch_size: {}, epoch_size: {} Optimizer: {}, LSTM Unit sizes: {} "
          "Batch Normalization/Gaussian Noise: {}"
          .format(rank, x[0], x[1], optimizers[x[2]], x[3:6], x[15:21]))

    x_data, x_data_holdout = x_data[:-365], x_data[-365:]
    y_data, y_data_holdout = y_data[:-365], y_data[-365:]

    totalFolds = modelFolds
    timeSeriesCrossValidation = TimeSeriesSplit(n_splits=totalFolds)
    # timeSeriesCrossValidation = KFold(n_splits=totalFolds)

    smape_scores = []
    mse_scores = []
    current_fold = 0

    # TODO: why re-build model after every fold? test just making it once bfr entering the folds

    # create model  # TODO: Naive LSTM
    # model = Sequential()
    # lstm_kwargs = {'units': 64, 'return_sequences': False,
    #                'implementation': 2}
    # model.add(Bidirectional(LSTM(**lstm_kwargs), input_shape=(
    #     x_data.shape[1], x_data.shape[2])))  # input_shape: rows: n, timestep: 1, features: m
    # model.add(Dense(y_data.shape[1]))
    # model.compile(loss='mean_squared_error', optimizer=optimizer)

    # create model  # TODO: 3 layers
<<<<<<< HEAD
    # model = tf.keras.models.Sequential()
    # lstm_kwargs = {'units': units1, 'dropout': dropout1, 'recurrent_dropout': recurrent_dropout1,
    #                'return_sequences': True,
    #                'implementation': 2,
    #                # 'kernel_regularizer': tf.keras.regularizers.l2(0.01),
    #                # 'activity_regularizer': tf.keras.regularizers.l1_l2(0.01),
    #                # 'bias_regularizer': tf.keras.regularizers.l2(0.01)    # TODO: test with kernel, activity, bias regularizers
    #                }
    # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs), input_shape=(
    #     x_data.shape[1], x_data.shape[2])))  # input_shape: rows: n, timestep: 1, features: m
    # if use_gaussian_noise1 == 1:
    #     model.add(tf.keras.layers.GaussianNoise(noise_stddev1))
    # if useBatchNormalization1 == 1:
    #     model.add(tf.keras.layers.BatchNormalization())
    # lstm_kwargs['units'] = units2
    # lstm_kwargs['dropout'] = dropout2
    # lstm_kwargs['recurrent_dropout'] = recurrent_dropout2
    # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs)))
    # if use_gaussian_noise2 == 1:
    #     model.add(tf.keras.layers.GaussianNoise(noise_stddev2))
    # if useBatchNormalization2 == 1:
    #     model.add(tf.keras.layers.BatchNormalization())
    # lstm_kwargs['units'] = units3
    # lstm_kwargs['dropout'] = dropout3
    # lstm_kwargs['recurrent_dropout'] = recurrent_dropout3
    # lstm_kwargs['return_sequences'] = False
    # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs)))
    # if use_gaussian_noise3 == 1:
    #     model.add(tf.keras.layers.GaussianNoise(noise_stddev3))
    # if useBatchNormalization3 == 1:
    #     model.add(tf.keras.layers.BatchNormalization())
    # # model.add(Dense(units3))  # TODO: test with 2 extra dense layers
    # # model.add(Dense(y_data.shape[1]))
    # model.add(tf.keras.layers.Dense(y_data.shape[1]))
    # if multi_gpu:  # TODO: Temp set the same model to benchmark 1x 1070Ti vs 2x (970 + 1070ti)
    #     model = tf.keras.utils.multi_gpu_model(model, gpus=2)
    # model.compile(loss='mean_squared_error', optimizer=optimizer)

    # # TODO: Small model for GA course
    # # create model  # TODO: 3 moar layers (6)
    # model = tf.keras.models.Sequential()
    # lstm_kwargs = {'units': units1, 'dropout': dropout1, 'recurrent_dropout': recurrent_dropout1,
    #                'return_sequences': False,
    #                'implementation': 2,
    #                # 'kernel_regularizer': l2(0.01),
    #                # 'activity_regularizer': l2(0.01),
    #                # 'bias_regularizer': l2(0.01)    # TODO: test with kernel, activity, bias regularizers
    #                }
    # if use_gaussian_noise2 == 1 and use_gaussian_noise3 == 1:  # TODO: gene contraption: added kernel regularizer
    #     lstm_kwargs['kernel_regularizer'] = tf.keras.regularizers.l1_l2(noise_stddev2)
    # elif use_gaussian_noise2 == 1:  # TODO: gene contraption: added kernel regularizer
    #     lstm_kwargs['kernel_regularizer'] = tf.keras.regularizers.l1(noise_stddev2)
    # elif use_gaussian_noise3 == 1:  # TODO: gene contraption: added kernel regularizer
    #     lstm_kwargs['kernel_regularizer'] = tf.keras.regularizers.l2(noise_stddev3)
    #
    # if useBatchNormalization2 == 1 and useBatchNormalization3 == 1:  # TODO: gene contraption: added activity_regularizer
    #     lstm_kwargs['activity_regularizer'] = tf.keras.regularizers.l1_l2(dropout2)
    # elif useBatchNormalization2 == 1:  # TODO: gene contraption: added activity_regularizer
    #     lstm_kwargs['activity_regularizer'] = tf.keras.regularizers.l1(dropout2)
    # elif useBatchNormalization3 == 1:  # TODO: gene contraption: added activity_regularizer
    #     lstm_kwargs['activity_regularizer'] = tf.keras.regularizers.l2(dropout3)
    #
    # if useBatchNormalization2 == 1 and useBatchNormalization3 == 1:  # TODO: gene contraption: added activity_regularizer
    #     lstm_kwargs['bias_regularizer'] = tf.keras.regularizers.l1_l2(recurrent_dropout2)
    # elif useBatchNormalization2 == 1:  # TODO: gene contraption: added activity_regularizer
    #     lstm_kwargs['bias_regularizer'] = tf.keras.regularizers.l1(recurrent_dropout2)
    # elif useBatchNormalization3 == 1:  # TODO: gene contraption: added activity_regularizer
    #     lstm_kwargs['bias_regularizer'] = tf.keras.regularizers.l2(recurrent_dropout3)
    #
    # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs), input_shape=(
    #     x_data.shape[1], x_data.shape[2])))  # input_shape: rows: n, timestep: 1, features: m
    # if use_gaussian_noise1 == 1:
    #     model.add(tf.keras.layers.GaussianNoise(noise_stddev1))
    # if useBatchNormalization1 == 1:
    #     model.add(tf.keras.layers.BatchNormalization())
    #
    # model.add(tf.keras.layers.Dense(y_data.shape[1]))
    #
    # if multi_gpu:
    #     from keras.utils import multi_gpu_model # TODO: Temp set the same model to benchmark 1x 1070Ti vs 2x (970 + 1070ti)
    #     model = multi_gpu_model(model, gpus=2)
    # model.compile(loss='mean_squared_error', optimizer=optimizer)

    # TODO: 6 layer large model
    # create model
=======
>>>>>>> parent of 39d5d51... Fixed some jupyter typos
    model = tf.keras.models.Sequential()
    lstm_kwargs = {'units': units1, 'dropout': dropout1, 'recurrent_dropout': recurrent_dropout1,
                   'return_sequences': True,
                   'implementation': 2,
                   # 'kernel_regularizer': tf.keras.regularizers.l2(0.01),
                   # 'activity_regularizer': tf.keras.regularizers.l1_l2(0.01),
                   # 'bias_regularizer': tf.keras.regularizers.l2(0.01)    # TODO: test with kernel, activity, bias regularizers
                   }
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs), input_shape=(
        x_data.shape[1], x_data.shape[2])))  # input_shape: rows: n, timestep: 1, features: m
    if use_gaussian_noise1 == 1:
        model.add(tf.keras.layers.GaussianNoise(noise_stddev1))
    if useBatchNormalization1 == 1:
        model.add(tf.keras.layers.BatchNormalization())
<<<<<<< HEAD

=======
>>>>>>> parent of 39d5d51... Fixed some jupyter typos
    lstm_kwargs['units'] = units2
    lstm_kwargs['dropout'] = dropout2
    lstm_kwargs['recurrent_dropout'] = recurrent_dropout2
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs)))
    if use_gaussian_noise2 == 1:
        model.add(tf.keras.layers.GaussianNoise(noise_stddev2))
    if useBatchNormalization2 == 1:
        model.add(tf.keras.layers.BatchNormalization())
<<<<<<< HEAD

    lstm_kwargs['units'] = units3
    lstm_kwargs['dropout'] = dropout3
    lstm_kwargs['recurrent_dropout'] = recurrent_dropout3
=======
    lstm_kwargs['units'] = units3
    lstm_kwargs['dropout'] = dropout3
    lstm_kwargs['recurrent_dropout'] = recurrent_dropout3
    lstm_kwargs['return_sequences'] = False
>>>>>>> parent of 39d5d51... Fixed some jupyter typos
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs)))
    if use_gaussian_noise3 == 1:
        model.add(tf.keras.layers.GaussianNoise(noise_stddev3))
    if useBatchNormalization3 == 1:
        model.add(tf.keras.layers.BatchNormalization())
<<<<<<< HEAD

    lstm_kwargs['units'] = units1
    lstm_kwargs['dropout'] = dropout1
    lstm_kwargs['recurrent_dropout'] = recurrent_dropout1
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs)))
    if use_gaussian_noise1 == 1:
        model.add(tf.keras.layers.GaussianNoise(noise_stddev2))
    if useBatchNormalization1 == 1:
        model.add(tf.keras.layers.BatchNormalization())

    lstm_kwargs['units'] = units2
    lstm_kwargs['dropout'] = dropout2
    lstm_kwargs['recurrent_dropout'] = recurrent_dropout2
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs)))
    if use_gaussian_noise2 == 1:
        model.add(tf.keras.layers.GaussianNoise(noise_stddev3))
    if useBatchNormalization2 == 1:
        model.add(tf.keras.layers.BatchNormalization())

    lstm_kwargs['units'] = units3
    lstm_kwargs['dropout'] = dropout3
    lstm_kwargs['recurrent_dropout'] = recurrent_dropout3
    lstm_kwargs['return_sequences'] = False  # Last layer should return sequences
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs)))
    if use_gaussian_noise3 == 1:
        model.add(tf.keras.layers.GaussianNoise(noise_stddev3))
    if useBatchNormalization3 == 1:
        model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(units3))  # TODO: test with 2 extra dense layers
    model.add(tf.keras.layers.Dense(y_data.shape[1]))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
=======
    # model.add(Dense(units3))  # TODO: test with 2 extra dense layers
    # model.add(Dense(y_data.shape[1]))
    model.add(tf.keras.layers.Dense(y_data.shape[1]))
    if multi_gpu:  # TODO: Temp set the same model to benchmark 1x 1070Ti vs 2x (970 + 1070ti)
        model = tf.keras.utils.multi_gpu_model(model, gpus=2)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # # TODO: Small model for GA course
    # # create model  # TODO: 3 moar layers (6)
    # model = Sequential()
    # lstm_kwargs = {'units': units1, 'dropout': dropout1, 'recurrent_dropout': recurrent_dropout1,
    #                'return_sequences': False,
    #                'implementation': 2,
    #                # 'kernel_regularizer': l2(0.01),
    #                # 'activity_regularizer': l2(0.01),
    #                # 'bias_regularizer': l2(0.01)    # TODO: test with kernel, activity, bias regularizers
    #                }
    # if use_gaussian_noise2 == 1 and use_gaussian_noise3 == 1:  # TODO: gene contraption: added kernel regularizer
    #     lstm_kwargs['kernel_regularizer'] = l1_l2(noise_stddev2)
    # elif use_gaussian_noise2 == 1:  # TODO: gene contraption: added kernel regularizer
    #     lstm_kwargs['kernel_regularizer'] = l1(noise_stddev2)
    # elif use_gaussian_noise3 == 1:  # TODO: gene contraption: added kernel regularizer
    #     lstm_kwargs['kernel_regularizer'] = l2(noise_stddev3)
    #
    # if useBatchNormalization2 == 1 and useBatchNormalization3 == 1:  # TODO: gene contraption: added activity_regularizer
    #     lstm_kwargs['activity_regularizer'] = l1_l2(dropout2)
    # elif useBatchNormalization2 == 1:  # TODO: gene contraption: added activity_regularizer
    #     lstm_kwargs['activity_regularizer'] = l1(dropout2)
    # elif useBatchNormalization3 == 1:  # TODO: gene contraption: added activity_regularizer
    #     lstm_kwargs['activity_regularizer'] = l2(dropout3)
    #
    # if useBatchNormalization2 == 1 and useBatchNormalization3 == 1:  # TODO: gene contraption: added activity_regularizer
    #     lstm_kwargs['bias_regularizer'] = l1_l2(recurrent_dropout2)
    # elif useBatchNormalization2 == 1:  # TODO: gene contraption: added activity_regularizer
    #     lstm_kwargs['bias_regularizer'] = l1(recurrent_dropout2)
    # elif useBatchNormalization3 == 1:  # TODO: gene contraption: added activity_regularizer
    #     lstm_kwargs['bias_regularizer'] = l2(recurrent_dropout3)
    #
    # model.add(Bidirectional(LSTM(**lstm_kwargs), input_shape=(
    #     x_data.shape[1], x_data.shape[2])))  # input_shape: rows: n, timestep: 1, features: m
    # if use_gaussian_noise1 == 1:
    #     model.add(GaussianNoise(noise_stddev1))
    # if useBatchNormalization1 == 1:
    #     model.add(BatchNormalization())
    #
    # model.add(Dense(y_data.shape[1]))
    #
    # if multi_gpu:
    #     from keras.utils import multi_gpu_model # TODO: Temp set the same model to benchmark 1x 1070Ti vs 2x (970 + 1070ti)
    #     model = multi_gpu_model(model, gpus=2)
    # model.compile(loss='mean_squared_error', optimizer=optimizer)

    # TODO: 6 layer large model
    # create model
    # model = Sequential()
    # lstm_kwargs = {'units': units1, 'dropout': dropout1, 'recurrent_dropout': recurrent_dropout1,
    #                'return_sequences': True,
    #                'implementation': 2,
    #                # 'kernel_regularizer': l2(0.01),
    #                # 'activity_regularizer': l2(0.01),
    #                # 'bias_regularizer': l2(0.01)    # TODO: test with kernel, activity, bias regularizers
    #                }
    # model.add(Bidirectional(LSTM(**lstm_kwargs), input_shape=(
    #     x_data.shape[1], x_data.shape[2])))  # input_shape: rows: n, timestep: 1, features: m
    # if use_gaussian_noise1 == 1:
    #     model.add(GaussianNoise(noise_stddev1))
    # if useBatchNormalization1 == 1:
    #     model.add(BatchNormalization())
    #
    # lstm_kwargs['units'] = units2
    # lstm_kwargs['dropout'] = dropout2
    # lstm_kwargs['recurrent_dropout'] = recurrent_dropout2
    # model.add(Bidirectional(LSTM(**lstm_kwargs)))
    # if use_gaussian_noise2 == 1:
    #     model.add(GaussianNoise(noise_stddev2))
    # if useBatchNormalization2 == 1:
    #     model.add(BatchNormalization())
    #
    # lstm_kwargs['units'] = units3
    # lstm_kwargs['dropout'] = dropout3
    # lstm_kwargs['recurrent_dropout'] = recurrent_dropout3
    # model.add(Bidirectional(LSTM(**lstm_kwargs)))
    # if use_gaussian_noise3 == 1:
    #     model.add(GaussianNoise(noise_stddev3))
    # if useBatchNormalization3 == 1:
    #     model.add(BatchNormalization())
    #
    # lstm_kwargs['units'] = units1
    # lstm_kwargs['dropout'] = dropout1
    # lstm_kwargs['recurrent_dropout'] = recurrent_dropout1
    # model.add(Bidirectional(LSTM(**lstm_kwargs)))
    # if use_gaussian_noise1 == 1:
    #     model.add(GaussianNoise(noise_stddev2))
    # if useBatchNormalization1 == 1:
    #     model.add(BatchNormalization())
    #
    # lstm_kwargs['units'] = units2
    # lstm_kwargs['dropout'] = dropout2
    # lstm_kwargs['recurrent_dropout'] = recurrent_dropout2
    # model.add(Bidirectional(LSTM(**lstm_kwargs)))
    # if use_gaussian_noise2 == 1:
    #     model.add(GaussianNoise(noise_stddev3))
    # if useBatchNormalization2 == 1:
    #     model.add(BatchNormalization())
    #
    # lstm_kwargs['units'] = units3
    # lstm_kwargs['dropout'] = dropout3
    # lstm_kwargs['recurrent_dropout'] = recurrent_dropout3
    # lstm_kwargs['return_sequences'] = False  # Last layer should return sequences
    # model.add(Bidirectional(LSTM(**lstm_kwargs)))
    # if use_gaussian_noise3 == 1:
    #     model.add(GaussianNoise(noise_stddev3))
    # if useBatchNormalization3 == 1:
    #     model.add(BatchNormalization())
    #
    # model.add(Dense(units3))  # TODO: test with 2 extra dense layers
    # model.add(Dense(y_data.shape[1]))
    # model.compile(loss='mean_squared_error', optimizer=optimizer)
>>>>>>> parent of 39d5d51... Fixed some jupyter typos
    # TODO: do not store model on every step
    # early_stop = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto'),
    #               ReduceLROnPlateau(patience=3, verbose=1),
    #               ModelCheckpoint(filepath='foundModels/best_model_{}.h5'.format(modelLabel), monitor='val_loss',
    #                               save_best_only=True)]

    early_stop = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                   # patience=25, verbose=1, mode='min'),  # TODO: test with large patience
                                                   patience=10, verbose=1, mode='auto'),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=1E-7,
                                                       patience=5,
                                                       verbose=1), tf.keras.callbacks.TerminateOnNaN()]

    for train, validation in timeSeriesCrossValidation.split(x_data, y_data):
        current_fold += 1
        print("--- Rank {}: Current Fold: {}/{}".format(rank, current_fold, totalFolds))

        try:
            history = model.fit(x_data[train], y_data[train],
                                verbose=verbosity,
                                batch_size=batch_size,
                                epochs=epoch_size,
                                validation_data=(x_data[validation], y_data[validation]),
                                callbacks=early_stop)
        except ValueError:
            print("--- Rank {}: Value Error exception: Model fit exception. Trying again...".format(rank))
            history = model.fit(x_data[train], y_data[train],
                                verbose=verbosity,
                                batch_size=batch_size,
                                epochs=epoch_size,
                                validation_data=(x_data[validation], y_data[validation]),
                                callbacks=early_stop)
        except:
            print("--- Rank {}: Exception: Returning max float value for this iteration.".format(rank))

            # Memory handling
            del model  # Manually delete model
            tf.reset_default_graph()
            tf.keras.backend.clear_session()
            gc.collect()

            return sys.float_info.max

        prediction = model.predict(x_data[validation])
        y_validation = y_data[validation]

        if dataManipulation["scale"] == 'standardize':
            sensor_mean = pd.read_pickle(directory + filePrefix + "_ts_mean.pkl")
            sensor_std = pd.read_pickle(directory + filePrefix + "_ts_std.pkl")
            # if trainModel.counter == 1:
            #     print("Un-standardizing...")
            #     print("sensor_mean:", sensor_mean)
            #     print("sensor_std:", sensor_std)
            #     print(np.array(sensor_mean)[0:y_data.shape[1]])
            sensor_mean = np.array(sensor_mean)
            sensor_std = np.array(sensor_std)
            prediction = (prediction * sensor_std[0:y_data.shape[1]]) + sensor_mean[0:y_data.shape[1]]
            y_validation = (y_validation * sensor_std[0:y_data.shape[1]]) + sensor_mean[0:y_data.shape[1]]
        elif dataManipulation["scale"] == 'normalize':
            sensor_min = pd.read_pickle(directory + filePrefix + "_ts_min.pkl")
            sensor_max = pd.read_pickle(directory + filePrefix + "_ts_max.pkl")
            # if trainModel.counter == 1:
            #     print("Un-normalizing...")
            #     print("sensor_min:", sensor_min)
            #     print("sensor_max:", sensor_max)
            #     print(np.array(sensor_min)[0:y_data.shape[1]])
            sensor_min = np.array(sensor_min)
            sensor_max = np.array(sensor_max)
            prediction = prediction * (sensor_max[0:y_data.shape[1]] - sensor_min[0:y_data.shape[1]]) + sensor_min[0:y_data.shape[1]]
            y_validation = y_validation * (sensor_max[0:y_data.shape[1]] - sensor_min[0:y_data.shape[1]]) + sensor_min[0:y_data.shape[1]]

        # Calc mse/rmse
        mse = mean_squared_error(prediction, y_validation)
        print("--- Rank {}: Validation MSE: {}".format(rank, mse))
        mse_scores.append(mse)
        rmse = sqrt(mse)
        print("--- Rank {}: Validation RMSE: {}".format(rank, rmse))

        smape = 0.01 * (100 / len(y_validation) * np.sum(2 * np.abs(prediction - y_validation) /
                                                         (np.abs(y_validation) + np.abs(prediction))))
        print("--- Rank {}: Validation SMAPE: {}".format(rank, smape))
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
        print("--- Rank {}: Full Data RMSE: {}".format(rank, full_rmse))

        full_smape = 0.01 * (100 / len(full_expected_ts) * np.sum(
            2 * np.abs(full_prediction - full_expected_ts) / (np.abs(full_expected_ts) + np.abs(full_prediction))))
        print('--- Rank {}: Full Data SMAPE: {}'.format(rank, full_smape))

    # Plot model architecture
    tf.keras.utils.plot_model(model, show_shapes=True, to_file='foundModels/{}Iter{}Rank{}Model.png'.format(modelLabel, trainModel.counter, rank))

    mean_smape = np.mean(smape_scores)
    std_smape = np.std(smape_scores)
    print("--- Rank {}: Cross validation Full Data SMAPE: {} +/- {}".format(rank, round(mean_smape * 100, 2), round(std_smape * 100, 2)))

    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    print("--- Rank {}: Cross validation Full Data MSE: {} +/- {}".format(rank, round(mean_mse * 100, 2), round(std_mse * 100, 2)))
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
    print('--- Rank {}: Holdout Data RMSE: {}'.format(rank, holdout_rmse))
    holdout_smape = 0.01 * (100/len(y_data_holdout) * np.sum(2 * np.abs(holdout_prediction - y_data_holdout) /
                                                             (np.abs(y_data_holdout) + np.abs(holdout_prediction))))

    print('--- Rank {}: Holdout Data SMAPE: {}'.format(rank, holdout_smape))
    holdout_mape = np.mean(np.abs((y_data_holdout - holdout_prediction) / y_data_holdout))
    print('--- Rank {}: Holdout Data MAPE: {}'.format(rank, holdout_mape))
    holdout_mse = mean_squared_error(holdout_prediction, y_data_holdout)
    print('--- Rank {}: Holdout Data MSE: {}'.format(rank, holdout_mse))
    # Index Of Agreement: https://cirpwiki.info/wiki/Statistics#Index_of_Agreement
    holdout_ioa = 1 - (np.sum((y_data_holdout - holdout_prediction) ** 2)) / (np.sum(
        (np.abs(holdout_prediction - np.mean(y_data_holdout)) + np.abs(y_data_holdout - np.mean(y_data_holdout))) ** 2))
    print('--- Rank {}: Holdout Data IOA: {}'.format(rank, holdout_ioa))
    with open('logs/{}Runs.csv'.format(modelLabel), 'a') as file:
        # Data to store:
        # datetime, iteration, gpu, cvMseMean, cvMseStd
        # cvSmapeMean, cvSmapeStd, holdoutRmse, holdoutSmape, holdoutMape,
        # holdoutMse, holdoutIoa, full_pso_parameters
        file.write("{},{},{},{},{},{},{},{},{},{},{},{},\"{}\"\n"
                   .format(str(int(time.time())), str(trainModel.counter), str(rank - 1),
            str(mean_mse), str(std_mse), str(mean_smape), str(std_smape), str(holdout_rmse), str(holdout_smape),
            str(holdout_mape), str(holdout_mse), str(holdout_ioa), full_model_parameters.tolist()))

    if mean_mse < min_mse:
        print("--- Rank {}: New min_mse: {}".format(rank, mean_mse))
        original_df1 = pd.DataFrame({"min_mse": [mean_mse]})
        original_df1.to_pickle("foundModels/min_mse.pkl")

        original_df2 = pd.DataFrame({"full_{}_rank{}_parameters".format(modelLabel, rank): [full_model_parameters]})
        original_df2.to_pickle("foundModels/full_{}_rank{}_parameters.pkl".format(modelLabel, rank))
        model.summary()  # print layer shapes and model parameters

        # Plot history
        pyplot.figure(figsize=(8, 6))  # Resolution 800 x 600
        pyplot.title("Rank {}: {} (iter: {}): Training History Last Fold".format(rank, modelLabel, trainModel.counter))
        pyplot.plot(history.history['val_loss'], label='val_loss')
        pyplot.plot(history.history['loss'], label='loss')
        pyplot.xlabel("Training Epoch")
        pyplot.ylabel("MSE")
        pyplot.grid(True)
        pyplot.legend()
        # pyplot.show()
        pyplot.savefig("foundModels/{}Iter{}Rank{}History.png".format(modelLabel, trainModel.counter, rank))
        pyplot.close()

        # Plot test data
        for i in range(holdout_prediction.shape[1]):
            pyplot.figure(figsize=(16, 12))  # Resolution 800 x 600
            pyplot.title("{} (iter: {}): Test data - Series {} (RMSE: {}, MAPE: {}%, IOA: {}%)"
                    .format(modelLabel, trainModel.counter, i, np.round(holdout_rmse, 2),
                            np.round(holdout_mape * 100, 2), np.round(holdout_ioa * 100, 2)))
            pyplot.plot(y_data_holdout[:, i], label='expected')
            pyplot.plot(holdout_prediction[:, i], label='prediction')
            pyplot.xlabel("Time step")
            pyplot.ylabel("Sensor Value")
            pyplot.grid(True)
            pyplot.legend()
            # pyplot.show()
            pyplot.savefig("foundModels/{}Iter{}Rank{}Series{}Test.png".format(modelLabel, trainModel.counter, rank, i))
            pyplot.close()
        pyplot.close("all")

        # Store model
        model_json = model.to_json() # serialize model to JSON
        with open("foundModels/bestModelArchitecture.json".format(modelLabel), "w") as json_file:
            json_file.write(model_json)
            print("--- Rank {}: Saved model to disk".format(rank))
        model.save_weights("foundModels/bestModelWeights.h5".format(modelLabel))  # serialize weights to HDF5
        print("--- Rank {}: Saved weights to disk".format(rank))

    # Memory handling
    del model  # Manually delete model
    tf.reset_default_graph()
    tf.keras.backend.clear_session()
    gc.collect()

    endTime = time.time()

    # Worker to master
    dataWorkerToMaster = {"worked": endTime - startTime, "rank": rank, "mean_mse": mean_mse, "agent": x,
                          "island": island, "iteration": trainModel.counter}
    comm = dataManipulation["comm"]
    req = comm.isend(dataWorkerToMaster, dest=master, tag=1)  # Send data async to master
    req.wait()

    # Master to worker
    agentToEa = {"swapAgent": False, "agent": None}
    dataMasterToWorker = comm.recv(source=master, tag=2)  # Receive data sync (blocking) from master
    swapAgent = dataMasterToWorker["swapAgent"]
    if swapAgent:
        outAgent = dataMasterToWorker["agent"]
        agentToEa = {"swapAgent": True, "agent": outAgent}  # Send agent copy

    return mean_mse, agentToEa


def ackley(x):
    arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e


def trainModelTester(x, *args):

    startTime = time.time()  # training time per model

    trainModel.counter += 1
    modelLabel = trainModel.label
    modelFolds = trainModel.folds
    dataManipulation = trainModel.dataManipulation
    island = dataManipulation["island"]
    rank = dataManipulation["rank"]
    master = dataManipulation["master"]
    x_data, y_data = args
    full_model_parameters = x.copy()

    # TODO: func to optimize
    # timeToSleep = np.random.uniform(0, 0.01)
    # time.sleep(timeToSleep)
    # mean_mse = 333.33 + timeToSleep
    mean_mse = ackley([x[12], x[13]])

    trainModel.counter += 1
    endTime = time.time()
    # Worker to master
    dataWorkerToMaster = {"worked": endTime - startTime, "rank": rank, "mean_mse": mean_mse, "agent": x,
                          "island": island, "iteration": trainModel.counter}
    comm = dataManipulation["comm"]
    # req = comm.isend(dataWorkerToMaster, dest=master, tag=1)  # TODO: test sync
    # req.wait()
    comm.send(dataWorkerToMaster, dest=master, tag=1)

    # Master to worker
    agentToEa = {"swapAgent": False, "agent": None}
    # dataMasterToWorker = comm.recv(source=0, tag=2)  # TODO: blocking or non-blocking?
    req = comm.irecv(source=0, tag=2)
    dataMasterToWorker = req.wait()

    swapAgent = dataMasterToWorker["swapAgent"]
    if swapAgent:
        outAgent = dataMasterToWorker["agent"]
        agentToEa = {"swapAgent": True, "agent": outAgent}

    return mean_mse, agentToEa


def trainModelTester2(x, *args):

    startTime = time.time()  # training time per model

    trainModel.counter += 1
    modelLabel = trainModel.label
    modelFolds = trainModel.folds
    dataManipulation = trainModel.dataManipulation
    island = dataManipulation["island"]
    rank = dataManipulation["rank"]
    master = dataManipulation["master"]
    x_data, y_data = args
    full_model_parameters = x.copy()

    # TODO: func to optimize
    timeToSleep = np.random.uniform(2, 5)
    time.sleep(timeToSleep)
    mean_mse = ackley([x[12], x[13]])

    trainModel.counter += 1
    endTime = time.time()
    # Worker to master
    dataWorkerToMaster = {"worked": endTime - startTime, "rank": rank, "mean_mse": mean_mse, "agent": x,
                          "island": island, "iteration": trainModel.counter}
    # comm = dataManipulation["comm"]
    # req = comm.isend(dataWorkerToMaster, dest=master, tag=1)  # TODO: test sync
    # req.wait()
    # comm.send(dataWorkerToMaster, dest=master, tag=1)

    # Master to worker
    agentToEa = {"swapAgent": False, "agent": None}
    # dataMasterToWorker = comm.recv(source=0, tag=2)  # TODO: blocking or non-blocking?
    # req = comm.irecv(source=0, tag=2)
    # dataMasterToWorker = req.wait()

    # swapAgent = dataMasterToWorker["swapAgent"]
    # if swapAgent:
    #     outAgent = dataMasterToWorker["agent"]
    #     agentToEa = {"swapAgent": True, "agent": outAgent}

    return mean_mse, {"swapAgent": True, "agent": x}
