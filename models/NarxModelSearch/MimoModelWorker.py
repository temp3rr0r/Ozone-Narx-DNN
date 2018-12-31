from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # These lines should be called asap, after the os import
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU only by default

import sys
import pandas as pd
from ModelSearch import random_model_search_mpi, \
    differential_evolution_model_search_mpi, basin_hopping_model_search_mpi, particle_swarm_optimization_model_search_mpi, \
    bounds, get_random_model
import time
from mpi4py import MPI
import numpy as np
import BaseNarxModelMpi as baseMpi

os.environ["PATH"] += os.pathsep + 'C:/Users/temp3rr0r/Anaconda3/Library/bin/graphviz'
# os.environ["PATH"] += os.pathsep + 'C:/ProgramData/Anaconda3/pkgs/graphviz-2.38.0-h6538335_1009/Library/bin/graphviz'

# modelLabel = 'rand'
# modelLabel = 'de'
modelLabel = 'pso'
# modelLabel = 'bh'

data_manipulation = {
    "detrend": False,
    # "scale": None,
    "scale": 'standardize',
    # "scale": 'normalize',
    "swapEvery": 500,  # Do swap island agent every iterations
    "sendBestAgentFromBuffer": True,  # Do send the best agent from buffer
    "master": 0,
    "folds": 2,
    "iterations": 200,
    "agents": 10,
    "storeCheckpoints": False,
    "verbose": 0,
    "fp16": False,  # Disabled: Faster than fp32 ONLY on very small architectures (1 LSTM) for ~ -10%
    "multi_gpu": False,  # Disabled: Rather slow for hybrid architectures (GTX970 + GTX1070 Ti, even with fp16)
}
dataDetrend = False  # TODO: de-trend


def loadData(directory, filePrefix, mimoOutputs, rank=1):
    print('Loading data...')

    if data_manipulation["scale"] == 'standardize':
        r = np.genfromtxt(directory + filePrefix + "_ts_standardized.csv", delimiter=',')
    elif data_manipulation["scale"] == 'normalize':
        r = np.genfromtxt(directory + filePrefix + "_ts_normalized.csv", delimiter=',')
    else:
        r = np.genfromtxt(directory + filePrefix + "_ts.csv", delimiter=',')
    r = np.delete(r, [0], axis=1)  # Remove dates

    if data_manipulation["fp16"]:
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

    # TODO: greatly decrease r length for testing: 2000-2012 training, 2013 for testing
    row2000_01_01 = 3653 - 1
    r = r[row2000_01_01:-1, :]

    print("r[0, 0]", r[0, 0])
    print("r[-1, 0]", r[-1, 0])

    maxLen = r.shape[1] - 1
    print('Variables: {}'.format(maxLen))
    print('TimeSteps: {}'.format(r.shape[0]))
    x_data = r[:, mimoOutputs:maxLen + 1]
    y_data = r[:, 0:mimoOutputs]
    print("x_data shape: ".format(x_data.shape))
    print("y_data shape: ".format(y_data.shape))

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


def get_total_message_count(islands, size, data_manipulation):

    # TODO: should all have close to equal iterations. rand most importantly
    totalMessageCount = 0
    iterations = data_manipulation["iterations"]
    psoMessageCount = (iterations + 1) * data_manipulation["agents"]
    randMessageCount = iterations
    bhMessageCount = 0  # TODO: basinHopping count
    bhMessageCount = iterations # TODO: bh == rand
    deMessageCount = (# (data_manipulation["iterations"] + 1)
        2 * data_manipulation["agents"] * len(bounds))

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
# islands = ['rand', 'pso', 'de', 'rand', 'pso', 'de', 'pso'] * 4
# islands = ['de', 'de', 'de', 'rand', 'de', 'pso', 'de'] * 4
# islands = ['', 'pso', 'pso', 'rand', 'de', 'de'] * 4
# islands = ['rand', 'pso', 'pso', 'de', 'rand', 'de'] * 4
islands = ['rand'] * 32
# islands = ['bh'] * 32
# islands = ['pso'] * 32
# islands = ['de'] * 32
# islands = ['pso', 'de'] * 32

rank = 1  # TODO: temp rank


print("waiting({})...".format(rank))

# initData = comm.recv(source=0, tag=0)  # Block wait the init command by the master

if rank == 1:  # Rank per gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
elif rank == 2:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

if data_manipulation["fp16"]:
    import tensorflow as tf
    tf.keras.backend.set_epsilon(1e-4)
    tf.keras.backend.set_floatx('float16')
    print("--- Working with tensorflow.keras float precision: {}".format(tf.keras.backend.floatx()))

print("working({})...".format(rank))
initData = {"command": "init", "island": "worker"}  # TODO: change

island = initData["island"]  # Get the island type from the master
print("--- Rank {}. Data Received: {}!".format(rank, initData))
print("--- Island: {}".format(island))

# data_manipulation["directory"] = "data/6vars/"  # Lerp on missing values, comparable with other thesis
# data_manipulation["filePrefix"] = "BETN073"
# data_manipulation["mimoOutputs"] = 1

# data_manipulation["directory"] = "data/6vars_ALL/"  # "closest station" data replacement strategy
# data_manipulation["filePrefix"] = "BETN073_ALL"
# data_manipulation["mimoOutputs"] = 1

data_manipulation["directory"] = "data/BETN073_BG/"  # TODO: "closest BG station" data replacement strategy
data_manipulation["filePrefix"] = "BETN073_BG"
data_manipulation["mimoOutputs"] = 1

# data_manipulation["directory"] = "data/4stations51vars/"
# data_manipulation["filePrefix"] = "BETN_12_66_73_121_51vars_O3_O3-1_19900101To2000101"
# data_manipulation["mimoOutputs"] = 4

# data_manipulation["directory"] = "data/BETN012_66_73_121_BG/"
# data_manipulation["filePrefix"] = "BETN012_66_73_121_BG"
# data_manipulation["mimoOutputs"] = 4

# data_manipulation["directory"] = "data/BETN113_121_132_BG/"
# data_manipulation["filePrefix"] = "BETN113_121_132_BG"
# data_manipulation["mimoOutputs"] = 3

# data_manipulation["directory"] = "data/24stations51vars/"
# data_manipulation["filePrefix"] = "ALL_BETN_51vars_O3_O3-1_19900101To2000101"
# data_manipulation["mimoOutputs"] = 24

# data_manipulation["directory"] = "data/46stations51vars/"
# data_manipulation["filePrefix"] = "ALL_BE_51vars_O3_O3-1_19900101To20121231"
# data_manipulation["mimoOutputs"] = 46

# data_manipulation["directory"] = "data/PM10_BETN/"
# data_manipulation["filePrefix"] = "PM10_BETN"
# data_manipulation["mimoOutputs"] = 16

# data_manipulation["directory"] = "data/PM1073stations51vars/"
# data_manipulation["filePrefix"] = "ALL_BE_51vars_PM10_PM10-1_19940101To20121231"
# data_manipulation["mimoOutputs"] = 73

x_data_3d, y_data = loadData(data_manipulation["directory"], data_manipulation["filePrefix"],
                             data_manipulation["mimoOutputs"])

data_manipulation["rank"] = rank
data_manipulation["island"] = island
data_manipulation["comm"] = comm

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

iterations = data_manipulation["iterations"]
agents = data_manipulation["agents"]
args = (x_data_3d, y_data)
baseMpi.train_model.counter = 0  # Function call counter
baseMpi.train_model.folds = data_manipulation["folds"]
baseMpi.train_model.data_manipulation = data_manipulation

# rabbit_mq_worker(x_data_3d, y_data, data_manipulation)

import pika
import time
import json
import numpy as np
from BaseNarxModelMpi import ackley

timeout = 600 * 10  # TODO: timeouts 10 mins * islands
params = pika.ConnectionParameters(heartbeat_interval=timeout, blocked_connection_timeout=timeout)
# params = pika.ConnectionParameters("localhost")
connection = pika.BlockingConnection(params)  # Connect with msg broker server
channel = connection.channel()  # Listen to channels
channel.queue_declare(queue="task_queue", durable=False)  # Open common task queue
channel.basic_qos(prefetch_count=1)  # Allow only 1 task in queue
results_queues = []  # List of declared results queues

def callback(ch, method, properties, body):  # Tasks receiver callback
    try:
        body = json.loads(body)
        print(" [x] Received %r" % body)
        # time.sleep(str(body["delay"]).count("."))

        # Do work
        print(" [x] Island: ", body["island"])

        data_manipulation["island"] = body["island"]
        baseMpi.train_model.label = body["island"]
        baseMpi.train_model.data_manipulation = data_manipulation

        results_queue = body["results_queue"]
        if not any(results_queue in s for s in results_queues):  # Add queue of results in case it doesn't exist yes
            results_queues.append(results_queue)
            channel.queue_declare(queue=results_queue, durable=False)

        array1 = np.array(body["array"])
        x = array1
        mse = np.mean(array1 * 3)
        # x = np.array(body["array"])
        # mse = ackley([x[0], x[1]])
        # mse = baseMpi.train_model(x, *args)  # TODO: rabbit Mq worker
        mse = baseMpi.train_model_rabbit_mq(x, *args)  # TODO: rabbit Mq worker
        print("ok4")
        print(" [x] mse: ", mse)

        ch.basic_ack(delivery_tag=method.delivery_tag)  # Ack receipt of task & work done

        # Send back result to the sender, on its unique receive channel
        mse_message = {"array1": x.tolist(), "mse": mse.tolist(), "island": body["island"]}
        mse_message = json.dumps(mse_message)
        channel.basic_publish(exchange="", routing_key=results_queue, body=mse_message,
                              properties=pika.BasicProperties(delivery_mode=2))  # make msg persistent
        print(" [x] Sent back '%s'" % mse_message)
        print(" [x] Done")

    except ValueError as ev:  # Handle exceptions
        print(" [x] Exception, sending rejection: %s" % str(ev))
        ch.basic_reject(delivery_tag=method.delivery_tag)

channel.basic_consume(callback, queue="task_queue")

print(" [*] Waiting for messages. To exit press CTRL+C")
channel.start_consuming()  # Listen for incoming tasks


print("--- Done({})!".format(island))
