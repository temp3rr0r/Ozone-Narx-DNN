from __future__ import print_function
import sys
import numpy as np
import pandas as pd
from ModelSearch import randomModelSearch, particleSwarmOptimizationModelSearch, differentialEvolutionModelSearch, \
    basinHoppingpModelSearch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use the 970 only
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the 1070Ti only
os.environ["PATH"] += os.pathsep + 'C:/Users/temp3rr0r/Anaconda3/Library/bin/graphviz'

# modelLabel = 'rand'
modelLabel = 'de'
# modelLabel = 'pso'
# modelLabel = 'bh'

dataManipulation = {
    "detrend": False,
    # "scale": None,
    "scale": 'standardize',
    # "scale": 'normalize',
}

dataDetrend = False

if __name__ == "__main__":

    # TODO: TimeDistributed? TimeDistributed wrapper layer and the need for some LSTM layers to return sequences rather than single values.
    # TODO: masking layer? Skips timesteps

    # TODO: GridSearch CV http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        # TODO: and https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

    print('Loading data...')

    if dataManipulation["scale"] == 'standardize':
        r = np.genfromtxt("data/BETN073_ts_standardized.csv", delimiter=',')
    elif dataManipulation["scale"] == 'normalize':
        r = np.genfromtxt("data/BETN073_ts_normalized.csv", delimiter=',')
    else:
        r = np.genfromtxt("data/BETN073_ts.csv", delimiter=',')  # TODO: test with standardized data
    r = np.delete(r, [0], axis=1)  # Remove dates

    # r = np.delete(r, [1, 2, 3, 4, 5, 6], axis=1)  # Remove all other variables

    # print("\nStart Array r:\n {}".format(r[::5]))
    print("\nStart Array r:\n {}".format(r[0,0]))

    maxLen = r.shape[1] - 1
    print('Variables: {}'.format(maxLen))
    print('TimeSteps: {}'.format(r.shape[0]))

    # TODO: y_data 4 stations NOT 1
    x_data = r[:, 4:maxLen + 1]
    y_data = r[:, 0:4]

    print('x_data shape:', x_data.shape)

    print("y_data shape:")
    print(y_data.shape)

    y_data = np.array(y_data)
    x_data_3d = x_data.reshape(x_data.shape[0], 1, x_data.shape[1])  # reshape input to 3D[samples, timesteps, features]

    if not os.path.exists("foundModels/min_mse.pkl"):
        min_mse = sys.float_info.max
        print("Previous min_mse: {}".format(min_mse))
        original_df = pd.DataFrame({"min_mse": [min_mse]})
        original_df.to_pickle("foundModels/min_mse.pkl")
    else:
        min_mse = pd.read_pickle("foundModels/min_mse.pkl")['min_mse'][0]
        print("Previous min_mse: {}".format(min_mse))

        if os.path.exists("foundModels/full_{}_parameters.pkl".format(modelLabel)):
            full_model_parameters = pd.read_pickle("foundModels/full_{}_parameters.pkl"
                                                   .format(modelLabel))['full_{}_parameters'.format(modelLabel)][0]
            print("Previous full_{}_parameters: {}".format(modelLabel, full_model_parameters))

    if modelLabel == 'rand':
        randomModelSearch(x_data_3d, y_data, dataManipulation)
    elif modelLabel == 'pso':
        particleSwarmOptimizationModelSearch(x_data_3d, y_data, dataManipulation)
    elif modelLabel == 'de':
        differentialEvolutionModelSearch(x_data_3d, y_data, dataManipulation)
    elif modelLabel == 'bh':
        basinHoppingpModelSearch(x_data_3d, y_data, dataManipulation)

