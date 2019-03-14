from __future__ import print_function
import sys
import time
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import tensorflow as tf
import pandas as pd
import gc
from sklearn.model_selection import TimeSeriesSplit, train_test_split


def delete_model(model):
    """
    Clear a tensorflow model from memory & garbage collector.
    :param model: Tensorflow model to remove.
    :return:
    """
    # Memory handling
    del model  # Manually delete model
    tf.reset_default_graph()
    tf.keras.backend.clear_session()
    gc.collect()


def reduce_time_series_validation_fold_size(train, validation, max_validation_length=365):
    """
    If time-series fold validation array larger than the threshold (i.e 365 days), strip the starting validation excess
    and add it to the end of the training set.
    :param train: Indices of the time-series training set.
    :param validation: Indices of the time-series validation set.
    :param max_validation_length: Max validation array size.
    :return: 
    """
    if len(validation) > max_validation_length:
        train = np.append(train, validation[0:len(validation) - max_validation_length])
        validation = validation[-max_validation_length:]
    return train, validation


def train_model(x, *args):
    """
    Train a deep learning model.
    :param x: Model phenotype.
    :param args: Data (inputs and expected).
    :return: Average validation Mean Squared Error.
    """
    startTime = time.time()  # training time per model

    train_model.counter += 1
    modelLabel = train_model.label
    modelFolds = train_model.folds
    data_manipulation = train_model.data_manipulation
    rank = data_manipulation["rank"]
    master = data_manipulation["master"]
    directory = data_manipulation["directory"]
    filePrefix = data_manipulation["filePrefix"]
    island = data_manipulation["island"]
    verbosity = data_manipulation["verbose"]
    multi_gpu = data_manipulation["multi_gpu"]
    store_plots = data_manipulation["storePlots"]

    x_data, y_data = args

    # if island == "bh" or island == "sg":  # TODO: un-normalize data
    #     print("bounds ", data_manipulation["bounds"])
    #     print("x ", x)
    #     for i in range(len(x)):
    #         x[i] = x[i] * (data_manipulation["bounds"][i][1] - data_manipulation["bounds"][i][0]) \
    #                + data_manipulation["bounds"][i][0]
    #     x = np.array(x)
    #     print("un-normalized x ", x)

    # x = [32.269684115953126, 478.4579158867764, 2.4914987273745344, 291.55476719406147, 32.0, 512.0, 0.0812481431483004,
    #      0.01, 0.1445004524623349, 0.22335740221774894, 0.03443050512961357, 0.05488258021289669, 1.0,
    #      0.620275664519184, 0.34191582396595566, 0.9436131979280933, 0.4991752935129543, 0.4678261851228459, 0.0,
    #      0.355287972380982, 0.0]  # TODO: Temp set the same model to benchmark a specific DNN

    full_model_parameters = np.array(x.copy())
    if data_manipulation["fp16"]:
        full_model_parameters.astype(np.float32, casting='unsafe')  # TODO: temp test speed of keras with fp16

    print("\n=============\n")
    print("--- Rank {}: {} iteration {} using: {}".format(rank, modelLabel, train_model.counter, x[6:15]))

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
                  'adam', 'amsgrad']  # , 'rmsprop', 'sgd'] # Avoid loss NaNs, by removing rmsprop & sgd
    batch_size = x[0]
    epoch_size = x[1]
    optimizer = optimizers[x[2]]
    units1 = x[3]
    units2 = x[4]
    units3 = x[5]

    # Batch normalization
    use_batch_normalization1 = x[15]
    use_batch_normalization2 = x[16]
    use_batch_normalization3 = x[17]
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
    train_mse_scores = []
    # dev_mse_scores = []
    current_fold = 0

    # TODO: (Baldwin) phenotypic plasticity, using random uniform.
    min_regularizer = 0.0
    max_regularizer = 0.01
    regularizer_chance = 0.1
    regularizer_chance_randoms = np.random.rand(9)
    core_layers_randoms = np.random.randint(4, size=5)  # TODO: Dense, LSTM, BiLSTM, GRU, BiGRU

    l1_l2_randoms = np.random.uniform(low=min_regularizer, high=max_regularizer, size=(9, 2))

    for train, validation in timeSeriesCrossValidation.split(x_data, y_data):  # TODO: test train/dev/validation
    # for train, validation_full in timeSeriesCrossValidation.split(x_data, y_data):  # TODO: Nested CV?

        train, validation = reduce_time_series_validation_fold_size(train, validation)

        # dev, validation = train_test_split(validation_full, test_size=0.1, shuffle=False)  # TODO: 50-50 for dev/val

        # # create model  # TODO: Naive LSTM
        # model = tf.keras.models.Sequential()
        # lstm_kwargs = {'units': 16, 'return_sequences': False,
        #                'implementation': 2}
        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs), input_shape=(
        #     x_data.shape[1], x_data.shape[2])))  # input_shape: rows: n, timestep: 1, features: m
        # model.add(tf.keras.layers.Dense(y_data.shape[1]))
        # model.compile(loss='mean_squared_error', optimizer=optimizer)

        # # create model  # TODO: 1 lstm 1 dense, highly diverse
        # model = tf.keras.models.Sequential()
        # lstm_kwargs = {'units': units1,
        #                'dropout': dropout1,
        #                'recurrent_dropout': recurrent_dropout1,
        #                'return_sequences': True,
        #                'implementation': 2,
        #                "input_shape" : (x_data.shape[1], x_data.shape[2])
        #                }
        # # lstm_kwargs['return_sequences'] = False
        # lstm_kwargs['kernel_regularizer'] = tf.keras.regularizers.l1_l2(recurrent_dropout2, dropout2)  # TODO: mini rand: 50% for (0 - 0.01)
        # if use_gaussian_noise3 > 0.5:
        #     lstm_kwargs['activity_regularizer'] = tf.keras.regularizers.l1_l2(recurrent_dropout3, dropout3)
        # if use_gaussian_noise2 > 0.5:
        #     lstm_kwargs['bias_regularizer'] = tf.keras.regularizers.l1_l2(noise_stddev2, noise_stddev3)
        # # if np.random.uniform(0, 1) > 0.5:
        # #     lstm_kwargs['stateful'] = True
        # #     batch_input_shape = (batch_size, timesteps, data_dim)
        # #     lstm_kwargs['batch_input_shape'] = (x_data.shape[1], x_data.shape[2])
        # # model.add(tf.keras.layers.CuDNNLSTM(**lstm_kwargs, )  # TODO: test speed
        # # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(**lstm_kwargs)))  # TODO: test speed
        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs)))  # TODO: test speed
        # lstm_kwargs['return_sequences'] = False  # Last layer should return sequences
        # lstm_kwargs['units'] = units3
        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs)))
        # # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs), input_shape=(
        # #     x_data.shape[1], x_data.shape[2])
        #     # ,merge_mode=random.choice(['sum', 'mul', 'concat', 'ave', None])
        #       # input_shape: rows: n, timestep: 1, features: m
        # if useBatchNormalization2 > 0.5:
        #     model.add(tf.keras.layers.AlphaDropout(np.random.uniform(0.001, 0.1)))
        # if useBatchNormalization3 > 0.5:
        #     model.add(tf.keras.layers.GaussianDropout(np.random.uniform(0.001, 0.1)))
        # if use_gaussian_noise1 > 0.5:
        #     model.add(tf.keras.layers.GaussianNoise(noise_stddev1))
        # if useBatchNormalization1 > 0.5:
        #     model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.Dense(units2, activation=random.choice(
        #     ["tanh", "softmax", "elu", "selu", "softplus", "relu", "softsign", "hard_sigmoid",
        #      "linear"])))  # TODO: test with 2 extra dense layers
        # model.add(tf.keras.layers.Dense(y_data.shape[1]))
        # if multi_gpu:
        #     model = tf.keras.utils.multi_gpu_model(model, gpus=2)
        #
        # if optimizer == 'amsgrad':  # TODO: test amsgrad, special version of adam
        #     model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(amsgrad=True))
        # else:
        #     model.compile(loss='mean_squared_error', optimizer=optimizer)

        # TODO: 6 layer large model

        # TODO: test different initializers
        # TODO: layer type in bounds

        # create model
        model = tf.keras.models.Sequential()
        lstm_kwargs = {'units': units1, 'dropout': dropout1, 'recurrent_dropout': recurrent_dropout1,
                       'return_sequences': True,
                       'implementation': 2,
                       # 'kernel_regularizer': l2(0.01),
                       # 'activity_regularizer': l2(0.01),
                       # 'bias_regularizer': l2(0.01)    # TODO: test with kernel, activity, bias regularizers
                       }
        # Local mutation
        if regularizer_chance_randoms[0] < regularizer_chance:
            lstm_kwargs['activity_regularizer'] = tf.keras.regularizers.l1_l2(
                l1_l2_randoms[0, 0], l1_l2_randoms[0, 1])
        if regularizer_chance_randoms[1] < regularizer_chance:
            lstm_kwargs['bias_regularizer'] = tf.keras.regularizers.l1_l2(
                l1_l2_randoms[1, 0], l1_l2_randoms[2, 1])
        if regularizer_chance_randoms[2] < regularizer_chance:
            lstm_kwargs['kernel_regularizer'] = tf.keras.regularizers.l1_l2(
                l1_l2_randoms[2, 0], l1_l2_randoms[0, 1])

        # 1st base layer
        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs), input_shape=(x_data.shape[1], x_data.shape[2])))  # input_shape: rows: n, timestep: 1, features: m
        if core_layers_randoms[0] == 0:
            model.add(tf.keras.layers.LSTM(**lstm_kwargs))
        elif core_layers_randoms[0] == 1:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs)))
        elif core_layers_randoms[0] == 2:
            model.add(tf.keras.layers.GRU(**lstm_kwargs))
        elif core_layers_randoms[0] == 3:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(**lstm_kwargs)))
        else:
            model.add(tf.keras.layers.Dense(units1,
                                            activity_regularizer=tf.keras.regularizers.l1_l2(l1_l2_randoms[3, 0],
                                                                                             l1_l2_randoms[3, 1]),
                                            bias_regularizer=tf.keras.regularizers.l1_l2(l1_l2_randoms[4, 0],
                                                                                         l1_l2_randoms[4, 1]),
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1_l2_randoms[5, 0],
                                                                                           l1_l2_randoms[5, 1])))

        # 2nd base layer
        if use_gaussian_noise1 < 0.5:
            model.add(tf.keras.layers.GaussianNoise(noise_stddev1))
        if use_batch_normalization1 < 0.5:
            model.add(tf.keras.layers.BatchNormalization())

        lstm_kwargs['units'] = units2
        lstm_kwargs['dropout'] = dropout2
        lstm_kwargs['recurrent_dropout'] = recurrent_dropout2
        # TODO: Local mutation
        if regularizer_chance_randoms[3] < regularizer_chance:
            lstm_kwargs['activity_regularizer'] = tf.keras.regularizers.l1_l2(
                l1_l2_randoms[3, 0], l1_l2_randoms[3, 1])
        if regularizer_chance_randoms[4] < regularizer_chance:
            lstm_kwargs['bias_regularizer'] = tf.keras.regularizers.l1_l2(
                l1_l2_randoms[4, 0], l1_l2_randoms[4, 1])
        if regularizer_chance_randoms[5] < regularizer_chance:
            lstm_kwargs['kernel_regularizer'] = tf.keras.regularizers.l1_l2(
                l1_l2_randoms[5, 0], l1_l2_randoms[5, 1])
        # 2nd base layer
        if core_layers_randoms[1] == 0:
            model.add(tf.keras.layers.LSTM(**lstm_kwargs))
        elif core_layers_randoms[1] == 1:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs)))
        elif core_layers_randoms[1] == 2:
            model.add(tf.keras.layers.GRU(**lstm_kwargs))
        elif core_layers_randoms[1] == 3:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(**lstm_kwargs)))
        else:
            model.add(tf.keras.layers.Dense(units2,
                        activity_regularizer=tf.keras.regularizers.l1_l2(l1_l2_randoms[3, 0], l1_l2_randoms[3, 1]),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1_l2_randoms[4, 0], l1_l2_randoms[4, 1]),
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_l2_randoms[5, 0], l1_l2_randoms[5, 1])))

        if use_gaussian_noise2 < 0.5:
            model.add(tf.keras.layers.GaussianNoise(noise_stddev2))
        if use_batch_normalization2 < 0.5:
            model.add(tf.keras.layers.BatchNormalization())

        # 3rd base layer
        lstm_kwargs['units'] = units3
        lstm_kwargs['dropout'] = dropout3
        lstm_kwargs['recurrent_dropout'] = recurrent_dropout3
        lstm_kwargs['return_sequences'] = False  # Last layer should return sequences
        # TODO: Local mutation
        if regularizer_chance_randoms[6] < regularizer_chance:
            lstm_kwargs['activity_regularizer'] = tf.keras.regularizers.l1_l2(
                l1_l2_randoms[6, 0], l1_l2_randoms[6, 1])
        if regularizer_chance_randoms[7] < regularizer_chance:
            lstm_kwargs['bias_regularizer'] = tf.keras.regularizers.l1_l2(
                l1_l2_randoms[7, 0], l1_l2_randoms[7, 1])
        if regularizer_chance_randoms[8] < regularizer_chance:
            lstm_kwargs['kernel_regularizer'] = tf.keras.regularizers.l1_l2(
                l1_l2_randoms[8, 0], l1_l2_randoms[8, 1])
        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs)))
        if core_layers_randoms[2] == 0:
            model.add(tf.keras.layers.LSTM(**lstm_kwargs))
        elif core_layers_randoms[2] == 1:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**lstm_kwargs)))
        elif core_layers_randoms[2] == 2:
            model.add(tf.keras.layers.GRU(**lstm_kwargs))
        elif core_layers_randoms[2] == 3:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(**lstm_kwargs)))
        else:
            model.add(tf.keras.layers.Dense(units3,
                                            activity_regularizer=tf.keras.regularizers.l1_l2(l1_l2_randoms[3, 0],
                                                                                             l1_l2_randoms[3, 1]),
                                            bias_regularizer=tf.keras.regularizers.l1_l2(l1_l2_randoms[4, 0],
                                                                                         l1_l2_randoms[4, 1]),
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1_l2_randoms[5, 0],
                                                                                           l1_l2_randoms[5, 1])))
        if use_gaussian_noise3 < 0.5:
            model.add(tf.keras.layers.GaussianNoise(noise_stddev3))
        if use_batch_normalization3 < 0.5:
            model.add(tf.keras.layers.BatchNormalization())

        # model.add(tf.keras.layers.Dense(y_data.shape[1], activation=random.choice(
        #     ["tanh", "softmax", "elu", "selu", "softplus", "relu", "softsign", "hard_sigmoid",
        #      "linear"])))  # TODO: test with 2 extra dense layers
        model.add(tf.keras.layers.Dense(y_data.shape[1]))
        if multi_gpu:
            model = tf.keras.utils.multi_gpu_model(model, gpus=2)

        if optimizer == 'amsgrad':  # Adam variant: amsgrad (boolean), "On the Convergence of Adam and Beyond".
            model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(amsgrad=True))
        else:
            model.compile(loss='mean_squared_error', optimizer=optimizer)

        current_fold += 1  # TODO: train, trainValidation, validation
        print("--- Rank {}: Current Fold: {}/{}".format(rank, current_fold, totalFolds))

        early_stop = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, mode='auto',
                                                 cooldown=1, verbose=1),
            tf.keras.callbacks.TerminateOnNaN()
        ]

        # try:  # TODO: Use dev set
        #     history = model.fit(x_data[train], y_data[train],
        #                         verbose=verbosity,
        #                         batch_size=batch_size,
        #                         epochs=epoch_size,
        #                         validation_data=(x_data[dev], y_data[dev]),
        #                         callbacks=early_stop)
        # except ValueError:
        #     print("--- Rank {}: Value Error exception: Model fit exception. Trying again...".format(rank))
        #     history = model.fit(x_data[train], y_data[train],
        #                         verbose=verbosity,
        #                         batch_size=batch_size,
        #                         epochs=epoch_size,
        #                         validation_data=(x_data[dev], y_data[dev]),
        #                         callbacks=early_stop)
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
            delete_model(model)

            return sys.float_info.max

        prediction = model.predict(x_data[validation])
        train_prediction = model.predict(x_data[train])
        # dev_prediction = model.predict(x_data[dev])
        y_validation = y_data[validation]
        y_train = y_data[train]
        # y_dev = y_data[dev]

        if data_manipulation["scale"] == 'standardize':
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
            train_prediction = (train_prediction * sensor_std[0:y_data.shape[1]]) + sensor_mean[0:y_data.shape[1]]
            # dev_prediction = (dev_prediction * sensor_std[0:y_data.shape[1]]) + sensor_mean[0:y_data.shape[1]]
            y_validation = (y_validation * sensor_std[0:y_data.shape[1]]) + sensor_mean[0:y_data.shape[1]]
            y_train = (y_train * sensor_std[0:y_data.shape[1]]) + sensor_mean[0:y_data.shape[1]]
            # y_dev = (y_dev * sensor_std[0:y_data.shape[1]]) + sensor_mean[0:y_data.shape[1]]
        elif data_manipulation["scale"] == 'normalize':
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
            train_prediction = train_prediction * (sensor_max[0:y_data.shape[1]] - sensor_min[0:y_data.shape[1]]) + sensor_min[0:y_data.shape[1]]
            # dev_prediction = dev_prediction * (sensor_max[0:y_data.shape[1]] - sensor_min[0:y_data.shape[1]]) + sensor_min[0:y_data.shape[1]]
            y_validation = y_validation * (sensor_max[0:y_data.shape[1]] - sensor_min[0:y_data.shape[1]]) + sensor_min[0:y_data.shape[1]]
            y_train = y_train * (sensor_max[0:y_data.shape[1]] - sensor_min[0:y_data.shape[1]]) + sensor_min[0:y_data.shape[1]]
            # y_dev = y_dev * (sensor_max[0:y_data.shape[1]] - sensor_min[0:y_data.shape[1]]) + sensor_min[0:y_data.shape[1]]

        # Calc mse/rmse
        mse = mean_squared_error(prediction, y_validation)
        print("--- Rank {}: Validation MSE: {}".format(rank, mse))
        mse_scores.append(mse)
        train_mse_scores.append(mean_squared_error(train_prediction, y_train))
        # dev_mse_scores.append(mean_squared_error(dev_prediction, y_dev))
        rmse = sqrt(mse)
        print("--- Rank {}: Validation RMSE: {}".format(rank, rmse))

        smape = 0.01 * (100 / len(y_validation) * np.sum(2 * np.abs(prediction - y_validation) /
                                                         (np.abs(y_validation) + np.abs(prediction))))
        print("--- Rank {}: Validation SMAPE: {}".format(rank, smape))
        smape_scores.append(smape)

        full_x = x_data.copy()
        full_y = y_data.copy()
        full_prediction = model.predict(x_data)  # TODO: evaluate vs predict?
        full_expected_ts = y_data

        if data_manipulation["scale"] == 'standardize':
            full_prediction = (full_prediction * sensor_std[0:y_data.shape[1]]) + sensor_mean[0:y_data.shape[1]]
            full_expected_ts = (full_expected_ts * sensor_std[0:y_data.shape[1]]) + sensor_mean[0:y_data.shape[1]]
        elif data_manipulation["scale"] == 'normalize':
            full_prediction = full_prediction * (sensor_max[0:y_data.shape[1]] - sensor_min[0:y_data.shape[1]]) + sensor_min[0:y_data.shape[1]]
            full_expected_ts = full_expected_ts * (sensor_max[0:y_data.shape[1]] - sensor_min[0:y_data.shape[1]]) + sensor_min[0:y_data.shape[1]]

        full_rmse = sqrt(mean_squared_error(full_prediction, full_expected_ts))
        print("--- Rank {}: Full Data RMSE: {}".format(rank, full_rmse))

        full_smape = 0.01 * (100 / len(full_expected_ts) * np.sum(
            2 * np.abs(full_prediction - full_expected_ts) / (np.abs(full_expected_ts) + np.abs(full_prediction))))
        print('--- Rank {}: Full Data SMAPE: {}'.format(rank, full_smape))

        if current_fold < totalFolds - 1:
            delete_model(model)

    # Plot model architecture
    if store_plots:
        tf.keras.utils.plot_model(model, show_shapes=True, to_file='foundModels/{}Iter{}Rank{}Model.png'.
                                  format(modelLabel, train_model.counter, rank))

    train_mean_mse = np.mean(train_mse_scores)
    train_std_mse = np.std(train_mse_scores)
    print("--- Rank {}: Cross validation, Train Data MSE: {} +/- {}".format(rank, round(train_mean_mse, 2),
                                                                           round(train_std_mse, 2)))

    # dev_mean_mse = np.mean(dev_mse_scores)
    # dev_std_mse = np.std(dev_mse_scores)
    # print("--- Rank {}: Cross validation, Dev Data MSE: {} +/- {}".format(rank, round(dev_mean_mse, 2),
    #                                                                        round(dev_std_mse, 2)))

    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    print("--- Rank {}: Cross validation, Validation Data MSE: {} +/- {}".format(rank, round(mean_mse, 2),
                                                                                round(std_mse, 2)))
    train_mean_rmse = np.mean(np.sqrt(train_mse_scores))
    train_std_rmse = np.std(np.sqrt(train_mse_scores))
    print("--- Rank {}: Cross validation, Train Data RMSE: {} +/- {}".format(rank, round(train_mean_rmse, 2),
                                                                                round(train_std_rmse, 2)))

    # dev_mean_rmse = np.mean(np.sqrt(dev_mse_scores))
    # dev_std_rmse = np.std(np.sqrt(dev_mse_scores))
    # print("--- Rank {}: Cross validation, Dev Data RMSE: {} +/- {}".format(rank, round(dev_mean_rmse, 2),
    #                                                                             round(dev_std_rmse, 2)))

    mean_rmse = np.mean(np.sqrt(mse_scores))
    std_rmse = np.std(np.sqrt(mse_scores))
    print("--- Rank {}: Cross validation, Validation Data RMSE: {} +/- {}".format(rank, round(mean_rmse, 2),
                                                                                round(std_rmse, 2)))
    mean_smape = np.mean(smape_scores)
    std_smape = np.std(smape_scores)
    print("--- Rank {}: Cross validation, Validation Data SMAPE: {} +/- {}".format(rank, round(mean_smape * 100, 2),
                                                                                   round(std_smape * 100, 2)))
    min_mse = pd.read_pickle("foundModels/min_mse.pkl")['min_mse'][0]

    holdout_prediction = model.predict(x_data_holdout)

    if data_manipulation["scale"] == 'standardize':
        holdout_prediction = (holdout_prediction * sensor_std[0:y_data.shape[1]]) + sensor_mean[0:y_data.shape[1]]
        y_data_holdout = (y_data_holdout * sensor_std[0:y_data.shape[1]]) + sensor_mean[0:y_data.shape[1]]
    elif data_manipulation["scale"] == 'normalize':
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
                   .format(str(int(time.time())), str(train_model.counter), str(rank),
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
        if store_plots:
            pyplot.figure(figsize=(16, 12))  # Resolution 800 x 600
            pyplot.title("Rank {}: {} (iter: {}): Training History Last Fold".format(rank, modelLabel, train_model.counter))
            pyplot.plot(history.history['val_loss'], label='val_loss')
            pyplot.plot(history.history['loss'], label='loss')
            pyplot.xlabel("Training Epoch")
            pyplot.ylabel("MSE")
            pyplot.grid(True)
            pyplot.legend()
            # pyplot.show()
            pyplot.savefig("foundModels/{}Iter{}Rank{}History.png".format(modelLabel, train_model.counter, rank))
            pyplot.close()

            # Plot test data
            for i in range(holdout_prediction.shape[1]):
                pyplot.figure(figsize=(16, 12))  # Resolution 800 x 600
                pyplot.title("{} (iter: {}): Test data - Series {} (RMSE: {}, MAPE: {}%, IOA: {}%)"
                        .format(modelLabel, train_model.counter, i, np.round(holdout_rmse, 2),
                                np.round(holdout_mape * 100, 2), np.round(holdout_ioa * 100, 2)))
                pyplot.plot(y_data_holdout[:, i], label='expected')
                pyplot.plot(holdout_prediction[:, i], label='prediction')
                pyplot.xlabel("Time step")
                pyplot.ylabel("Sensor Value")
                pyplot.grid(True)
                pyplot.legend()
                # pyplot.show()
                pyplot.savefig("foundModels/{}Iter{}Rank{}Series{}Test.png".format(modelLabel, train_model.counter, rank, i))
                pyplot.close()
            pyplot.close("all")

        # Store model
        model_json = model.to_json()  # serialize model to JSON
        with open("foundModels/bestModelArchitecture.json".format(modelLabel), "w") as json_file:
            json_file.write(model_json)
            print("--- Rank {}: Saved model to disk".format(rank))
        model.save_weights("foundModels/bestModelWeights.h5".format(modelLabel))  # serialize weights to HDF5
        print("--- Rank {}: Saved weights to disk".format(rank))

    delete_model(model)

    endTime = time.time()
    return mean_mse


def ackley(x):
    """
    Ackley function, 2 dimensional.
    :param x: List of parameters.
    :return: Function result, using the given x parameters.
    """
    arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e


def train_model_tester3(x, *args):
    """
    Fake model training, for testing communication and workers. Tries to find ackley function minimum.
    :param x: Model phenotype.
    :param args: Data (inputs and expected).
    :return: Mean Squared Error.
    """
    startTime = time.time()  # training time per model

    train_model.counter += 1
    modelLabel = train_model.label
    modelFolds = train_model.folds
    data_manipulation = train_model.data_manipulation
    island = data_manipulation["island"]
    rank = data_manipulation["rank"]
    master = data_manipulation["master"]
    x_data, y_data = args
    full_model_parameters = x.copy()

    # TODO: func to optimize
    # timeToSleep = np.random.uniform(0, 0.01)
    # time.sleep(timeToSleep)
    # mean_mse = 333.33 + timeToSleep
    mean_mse = ackley([x[12], x[13]])

    train_model.counter += 1
    endTime = time.time()

    return mean_mse
