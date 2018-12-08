from __future__ import print_function
import sys
import pandas as pd
from ModelSearch import randomModelSearchMpi, particleSwarmOptimizationModelSearch, \
    differentialEvolutionModelSearchMpi, basinHoppingpModelSearch, particleSwarmOptimizationModelSearchMpi
import os
import time
from mpi4py import MPI
from random import randint
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PATH"] += os.pathsep + 'C:/Users/temp3rr0r/Anaconda3/Library/bin/graphviz'

# modelLabel = 'rand'
# modelLabel = 'de'
modelLabel = 'pso'  # TODO: PSO: 3 iterations x 20 swarmsize = 80 total iterations
# modelLabel = 'bh'

dataManipulation = {
    "detrend": False,
    # "scale": None,
    "scale": 'standardize',
    # "scale": 'normalize',
    "master": 0,
    "folds": 2,
    "iterations": 3,
    "agents": 4
}

dataDetrend = False
master = 0
# iterations = 3
iterations = dataManipulation["iterations"]

def loadData():
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

    # TODO: test 1 station only printouts
    # r = np.delete(r, [1, 2, 3], axis=1)  # Remove all other ts

    r = r[1:800,:]  # TODO: greately decrease r for testing

    # print("\nStart Array r:\n {}".format(r[::5]))
    print("\nStart Array r:\n {}".format(r[0, 0]))

    maxLen = r.shape[1] - 1
    print('Variables: {}'.format(maxLen))
    print('TimeSteps: {}'.format(r.shape[0]))

    # y_data 4 stations NOT 1
    mimoOutputs = 4
    # mimoOutputs = 1  # TODO: test 1 station only printouts
    x_data = r[:, mimoOutputs:maxLen + 1]
    y_data = r[:, 0:mimoOutputs]

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

    return x_data_3d, y_data

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

data = {'params': [0.3, 1.5, 500, 30, 500]}
islands = ['rand', 'pso', 'de']

if rank == 0:  # Master Node
    swappedAgent = -1  # Rand init buffer agent
    startTime = time.time()
    totalSecondsWork = 0
    mean_mse_threshold = 300.0

    for worker in range(1, size):  # Init workers
        comm.send({"command": "init", "island": islands[worker % 3]}, dest=worker, tag=0)
        print("-- Rank {}. Sending data: {} to {}...".format(rank, data, worker))

    # iterations = dataManipulation["iterations"]
    swapCounter = 0
    swapEvery = 5
    agentBuffer = 0
    for messageId in range((size - 1) * iterations):

        swapCounter += 1
        req = comm.irecv(tag=1)
        data = req.wait()

        dataToFitnessFunction = {"swapAgent": False, "agent": None}
        if swapCounter > swapEvery:  # TODO: decide to swap that agent
            swapCounter = 0
            agentBuffer = data["agent"]
            dataToFitnessFunction["swapAgent"] = True
            dataToFitnessFunction["agent"] = agentBuffer

        print("-- Rank {}. Data Received: {} from {}!".format(rank, data, worker))
        comm.send({"agentToReceive": swappedAgent}, dest=data["rank"], tag=2)
        swappedAgent = data["agentToSend"]
        totalSecondsWork += data["worked"]
        print("mean_mse: {}".format(data["mean_mse"]))
        if data["mean_mse"] <= mean_mse_threshold:
            print("Abort: mean_mse = {} less than ".format(data["mean_mse"]))
            # comm.Abort()
        # TODO: block for func call sync
        # TODO: stop condition if mean_mse >= 0.95
    endTime = time.time()
    print("-- Total work: %d secs in %.2f secs, speedup: %.2f / %d" % (
        totalSecondsWork, round(endTime - startTime, 2),
        totalSecondsWork / round(endTime - startTime, 2), size - 1))

    # comm.Disconnect()

else:  # Worker Node

    if rank == 1:  # TODO: rank per gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use the 1070Ti only
    elif rank == 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the 970 only

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank - 1)  # Use the 1070Ti or 970  # TODO: auto set gpu per rank

    print("waiting({})...".format(rank))

    initData = comm.recv(source=0, tag=0)  # Block wait the init command by the master
    if initData["command"] == "init":

        print("working({})...".format(rank))
        island = initData["island"]  # Get the island type from the master
        print("-- Rank {}. Data Received: {}!".format(rank, data))
        print("-- Island: {}".format(island))

        x_data_3d, y_data = loadData()

        islandAgents = np.array([rank] * 20)  # Populate agents
        dataManipulation["rank"] = rank
        dataManipulation["island"] = island
        dataManipulation["comm"] = comm

        # TODO: implement MPI versions
        if island == 'rand':
            # randomModelSearchMpi(x_data_3d, y_data, dataManipulation)  # TODO: test mpi pso
            particleSwarmOptimizationModelSearchMpi(x_data_3d, y_data, dataManipulation)
            # differentialEvolutionModelSearchMpi(x_data_3d, y_data, dataManipulation)
        elif island == 'pso':
            # randomModelSearchMpi(x_data_3d, y_data, dataManipulation)  # TODO: test mpi pso
            particleSwarmOptimizationModelSearchMpi(x_data_3d, y_data, dataManipulation)
            # differentialEvolutionModelSearchMpi(x_data_3d, y_data, dataManipulation)
        elif island == 'de':
            # differentialEvolutionModelSearchMpi(x_data_3d, y_data, dataManipulation)
            # randomModelSearchMpi(x_data_3d, y_data, dataManipulation)
            particleSwarmOptimizationModelSearchMpi(x_data_3d, y_data, dataManipulation)
        elif island == 'bh':
            # basinHoppingpModelSearch(x_data_3d, y_data, dataManipulation)
            randomModelSearchMpi(x_data_3d, y_data, dataManipulation)
            # differentialEvolutionModelSearchMpi(x_data_3d, y_data, dataManipulation)


        #     data['worked'] = timeWorked
        #     data['mean_mse'] = np.array(random.uniform(0.7, 0.95))
        #     data['iteration'] = i
        #     data['agentToSend'] = islandAgents[agentReplaceIndex]  # TODO: agent genotype -> phenotype
        #
        #     req = comm.isend(data, dest=master, tag=1)
        #     req.wait()
        #
        #     agent = comm.recv(source=0, tag=2)
        #
        #     islandAgents[agentReplaceIndex] = agent["agentToReceive"]  # TODO: inject speciated agent

        print("-- Done({}). Agents: {}.".format(island, islandAgents))
