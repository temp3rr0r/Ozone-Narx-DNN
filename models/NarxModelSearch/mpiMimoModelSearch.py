from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # These lines should be called asap, after the os import
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU only by default

import sys
import pandas as pd
from ModelSearch import randomModelSearchMpi, \
    differentialEvolutionModelSearchMpi, basinHoppingpModelSearchMpi, particleSwarmOptimizationModelSearchMpi, \
    bounds, getRandomModel
import time
from mpi4py import MPI
import numpy as np

os.environ["PATH"] += os.pathsep + 'C:/Users/temp3rr0r/Anaconda3/Library/bin/graphviz'
# os.environ["PATH"] += os.pathsep + 'C:/ProgramData/Anaconda3/pkgs/graphviz-2.38.0-h6538335_1009/Library/bin/graphviz'  # Path for the EC2 instance

# modelLabel = 'rand'
# modelLabel = 'de'
modelLabel = 'pso'
# modelLabel = 'bh'

dataManipulation = {
    "detrend": False,
    # "scale": None,
    "scale": 'standardize',
    # "scale": 'normalize',
    "swapEvery": 5,  # Do swap island agent every iterations
    "sendBestAgentFromBuffer": True,  # Do send the best agent from buffer
    "master": 0,
    "folds": 2,
    "iterations": 200,
    "agents": 20,
    "storeCheckpoints": 0,
    "verbose": 2,
    "fp16": True,
    "multi_gpu": False,  # Disabled: Rather slow for hybrid architectures (GTX970 + GTX1070 Ti, even with fp16)
}
dataDetrend = False  # TODO: de-trend

if dataManipulation["fp16"]:
    from keras import backend as K  # TODO: temp test fp16
    K.set_epsilon(1e-4)
    K.set_floatx('float16')
    print("--- Working with keras float precision: {}".format(K.floatx()))

def loadData(directory, filePrefix, mimoOutputs, rank=1):
    # TODO: TimeDistributed? TimeDistributed wrapper layer and the need for some LSTM layers to return sequences
    # TODO: rather than single values.
    # TODO: masking layer? Skips timesteps

    # TODO: GridSearch CV http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    # TODO: and https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

    print('Loading data...')

    if dataManipulation["scale"] == 'standardize':
        r = np.genfromtxt(directory + filePrefix + "_ts_standardized.csv", delimiter=',')
    elif dataManipulation["scale"] == 'normalize':
        r = np.genfromtxt(directory + filePrefix + "_ts_normalized.csv", delimiter=',')
    else:
        r = np.genfromtxt(directory + filePrefix + "_ts.csv", delimiter=',')
    r = np.delete(r, [0], axis=1)  # Remove dates

    if dataManipulation["fp16"]:
        r.astype(np.float16, casting='unsafe')  # TODO: temp test speed of keras with fp16

    # TODO: test 1 station only printouts
    # r = np.delete(r, [1, 2, 3], axis=1)  # Remove all other ts

    # TODO: BETN073 only training. Removing stations 12, 66, 121 (and lags-1 of those)
    # r = np.delete(r, [0, 1, 3, 55, 56, 58], axis=1)  # Remove all other ts

    # TODO: greatly decrease r length for testing (365 days + 2 x X amount) and remove 40 vars
    # r = r[1:(365+60):]
    # r = np.delete(r, range(5, 50), axis=1)

    # TODO: greatly decrease r length for testing: 2000-2009 training, 2010 for testing
    # row2000_01_01 = 3653 - 1
    # row2010_12_31 = 7670
    # r = r[row2000_01_01:row2010_12_31, :]

    # TODO: Greatly decrease r length for testing: 1990-2009 training, 2010 for testing
    # row2010_12_31 = 7670
    # r = r[0:row2010_12_31, :]
    print("\nStart Array r:\n {}".format(r[0, 0]))

    print("r[0, 0]", r[0, 0])
    print("r[-1, 0]", r[-1, 0])

    print("\nStart Array r:\n {}".format(r[0, 0]))

    maxLen = r.shape[1] - 1
    print('Variables: {}'.format(maxLen))
    print('TimeSteps: {}'.format(r.shape[0]))
    x_data = r[:, mimoOutputs:maxLen + 1]
    y_data = r[:, 0:mimoOutputs]
    print('x_data shape:', x_data.shape)
    print("y_data shape:", y_data.shape)

    # TODO: more time-steps instead of 1?
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

        if os.path.exists("foundModels/full_{}_rank{}_parameters.pkl".format(modelLabel, rank)):
            full_model_parameters = pd.read_pickle("foundModels/full_{}_rank{}_parameters.pkl".format(modelLabel, rank))['full_{}_rank{}_parameters'.format(modelLabel, rank)][0]
            print("Previous full_{}_parameters: {}".format(modelLabel, full_model_parameters))

    return x_data_3d, y_data


def getTotalMessageCount(islands, size, dataManipulation):

    # TODO: should all have close to equal iterations. rand most importantly
    totalMessageCount = 0
    iterations = dataManipulation["iterations"]
    psoMessageCount = (iterations + 1) * dataManipulation["agents"]
    randMessageCount = iterations
    bhMessageCount = 0  # TODO: basinHopping count
    deMessageCount = (# (dataManipulation["iterations"] + 1)
        2 * dataManipulation["agents"] * len(bounds))

    for i in range(1, size):
        if islands[i] == "pso":
            totalMessageCount += psoMessageCount
        elif islands[i] == "de":
            totalMessageCount += deMessageCount
        elif islands[i] == "rand":
            totalMessageCount += randMessageCount
        elif islands[i] == "bh":
            totalMessageCount += bhMessageCount

    return int(totalMessageCount)


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

# islands = ['bh', 'pso', 'de', 'rand']
islands = ['rand', 'pso', 'de', 'rand', 'pso', 'de', 'pso'] * 4
# islands = ['', 'pso', 'pso', 'rand', 'de', 'de'] * 4
# islands = ['rand', 'pso', 'pso', 'de', 'rand', 'de'] * 4
islands = ['rand'] * 32

if rank == 0:  # Master Node

    swappedAgent = -1  # Rand init buffer agent
    startTime = time.time()
    totalSecondsWork = 0
    mean_mse_threshold = 3000.0

    for worker in range(1, size):  # Init workers
        initDataToWorkers = {"command": "init", "island": islands[worker % 3]}
        comm.send(initDataToWorkers, dest=worker, tag=0)
        print("--- Rank {}. Sending data: {} to {}...".format(rank, initDataToWorkers, worker))

    swapCounter = 0
    agentBuffer = getRandomModel()
    overallMinMse = 10e4  # TODO: formalize it
    evaluations = 0
    bestIsland = ""

    totalMessageCount = getTotalMessageCount(islands, size, dataManipulation)
    print("--- Expecting {} total messages...".format(totalMessageCount))

    for messageId in range(totalMessageCount):  # TODO 1000-1200 bh iters
        swapCounter += 1

        # Worker to master

        req = comm.irecv(tag=1)  # TODO: test sync
        dataWorkerToMaster = req.wait()
        # dataWorkerToMaster = comm.recv(tag=1)

        # print("--- Rank {}. Data Received: {} from {}!".format(rank, dataWorkerToMaster, worker))
        totalSecondsWork += dataWorkerToMaster["worked"]
        print("mean_mse: {} ({}: {})".format(dataWorkerToMaster["mean_mse"], dataWorkerToMaster["island"], dataWorkerToMaster["iteration"]))
        evaluations += 1
        if dataWorkerToMaster["mean_mse"] < overallMinMse:
            overallMinMse = dataWorkerToMaster["mean_mse"]
            bestIsland = dataWorkerToMaster["island"]
            if dataManipulation["sendBestAgentFromBuffer"]:
                agentBuffer = dataWorkerToMaster["agent"]  # TODO: Send the best agent received so far
            print("--- New overall min MSE: {} ({}: {}) (overall: {})".format(
                overallMinMse, dataWorkerToMaster["island"], dataWorkerToMaster["iteration"], evaluations))
        # if dataWorkerToMaster["mean_mse"] <= mean_mse_threshold:  # TODO: stop condition if mean_mse <= threshold
            # print("Abort: mean_mse = {} less than ".format(dataWorkerToMaster["mean_mse"]))
            # comm.Abort()  # TODO: block for func call sync

        # Master to worker

        dataMasterToWorker = {"swapAgent": False, "agent": None}
        if swapCounter > dataManipulation["swapEvery"]:
            print("========= Swapping...")
            swapCounter = 0
            dataMasterToWorker["swapAgent"] = True
            dataMasterToWorker["agent"] = agentBuffer
            agentBuffer = dataWorkerToMaster["agent"]
        comm.send(dataMasterToWorker, dest=dataWorkerToMaster["rank"], tag=2)  # TODO: test send async
        # req = comm.isend(dataMasterToWorker, dest=dataWorkerToMaster["rank"], tag=2)  # TODO: test send async
        # req.wait()

    endTime = time.time()
    print("--- Overall min MSE (total evals: {}): {} ({})".format(evaluations, overallMinMse, bestIsland))
    print("--- Total work: %d secs in %.2f secs, speedup: %.2f / %d" % (
        totalSecondsWork, round(endTime - startTime, 2),
        totalSecondsWork / round(endTime - startTime, 2), size - 1))
    # comm.Disconnect()

else:  # Worker Node

    print("waiting({})...".format(rank))

    initData = comm.recv(source=0, tag=0)  # Block wait the init command by the master
    if initData["command"] == "init":

        if rank == 1:  # Rank per gpu
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        elif rank == 2:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        elif rank == 3:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        elif rank == 4:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        elif rank == 5:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "4"
        elif rank == 6:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "5"
        elif rank == 7:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "6"
        elif rank == 8:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "7"

        print("working({})...".format(rank))
        island = initData["island"]  # Get the island type from the master
        print("--- Rank {}. Data Received: {}!".format(rank, initData))
        print("--- Island: {}".format(island))

        dataManipulation["directory"] = "data/6vars/"
        dataManipulation["filePrefix"] = "BETN073"
        dataManipulation["mimoOutputs"] = 1

        # dataManipulation["directory"] = "data/4stations51vars/"
        # dataManipulation["filePrefix"] = "BETN_12_66_73_121_51vars_O3_O3-1_19900101To2000101"
        # dataManipulation["mimoOutputs"] = 4

        # dataManipulation["directory"] = "data/24stations51vars/"
        # dataManipulation["filePrefix"] = "ALL_BETN_51vars_O3_O3-1_19900101To2000101"
        # dataManipulation["mimoOutputs"] = 24

        # dataManipulation["directory"] = "data/46stations51vars/"
        # dataManipulation["filePrefix"] = "ALL_BE_51vars_O3_O3-1_19900101To20121231"
        # dataManipulation["mimoOutputs"] = 46

        # dataManipulation["directory"] = "data/PM1073stations51vars/"
        # dataManipulation["filePrefix"] = "ALL_BE_51vars_PM10_PM10-1_19940101To20121231"
        # dataManipulation["mimoOutputs"] = 73

        x_data_3d, y_data = loadData(dataManipulation["directory"], dataManipulation["filePrefix"],
                                     dataManipulation["mimoOutputs"])

        dataManipulation["rank"] = rank
        dataManipulation["island"] = island
        dataManipulation["comm"] = comm

        # TODO: add/test (single or multi-agent) optimizers:
        # TODO: - Reinforcement Learning
        # TODO: - Bayesian Optimization (no derivatives needed)
        # TODO: - (traditional) Genetic Algorithms
        # TODO: - XGBoost
        # TODO: - Ant Colony Optimization (layer types only or bounded numerical if possible)
        # TODO: - Inductive Learning Programming (Known ts-DL layers/techniques (legends) =(progol)=>
        # TODO:     ML learned rules =(prolog)=> candidate layers
        # TODO: - Differentiable optimizers (convex solvers, other gradient solvers)
        # TODO: - RBF (if ez to implement) optimizers
        # TODO: - Memetic (?) algorithms
        # TODO: - Tabu search (?)

        if island == 'rand':
            randomModelSearchMpi(x_data_3d, y_data, dataManipulation)
        elif island == 'pso':
            particleSwarmOptimizationModelSearchMpi(x_data_3d, y_data, dataManipulation)
        elif island == 'de':
            differentialEvolutionModelSearchMpi(x_data_3d, y_data, dataManipulation)
        elif island == 'bh':
            basinHoppingpModelSearchMpi(x_data_3d, y_data, dataManipulation)

        print("--- Done({})!".format(island))
