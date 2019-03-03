from __future__ import print_function
import os
import pika
import json
import numpy as np
import sys
import pandas as pd
from base import NeuroevolutionModelTraining as baseMpi
from base.bounds import bounds

# from tensorflow.python.client import device_lib
# if "Tegra" in str(device_lib.list_local_devices()):  # If Nvidia Jetson TX2 -> dynamic GPU memory (for OOM avoidance)
#     print("---- Detected Nvidia Jetson TX2: Setting dynamic memory growth (out of memory work-around).")
#     import tensorflow as tf
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     tf.keras.backend.set_session(tf.Session(config=config))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # These lines should be called asap, after the os import
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU only by default
os.environ["PATH"] += os.pathsep + 'C:/Users/temp3rr0r/Anaconda3/Library/bin/graphviz'
# os.environ["PATH"] += os.pathsep + 'C:/ProgramData/Anaconda3/pkgs/graphviz-2.38.0-h6538335_1009/Library/bin/graphviz'


def init_gpu(gpu_rank):
    if gpu_rank == 1:  # Rank per gpu
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif gpu_rank == 2:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    elif gpu_rank == 3:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    elif gpu_rank == 4:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    elif gpu_rank == 5:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    elif gpu_rank == 6:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    elif gpu_rank == 7:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    elif gpu_rank == 8:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def load_data(directory, file_prefix, mimo_outputs, gpu_rank=1):
    if data_manipulation["scale"] == 'standardize':
        r = np.genfromtxt(directory + file_prefix + "_ts_standardized.csv", delimiter=',')
    elif data_manipulation["scale"] == 'normalize':
        r = np.genfromtxt(directory + file_prefix + "_ts_normalized.csv", delimiter=',')
    else:
        r = np.genfromtxt(directory + file_prefix + "_ts.csv", delimiter=',')
    r = np.delete(r, [0], axis=1)  # Remove dates

    if data_manipulation["fp16"]:
        r.astype(np.float16, casting='unsafe')

    # TODO: test 1 station only printouts
    # r = np.delete(r, [1, 2, 3], axis=1)  # Remove all other ts

    # TODO: BETN073 only training. Removing stations 12, 66, 121 (and lags-1 of those)
    # r = np.delete(r, [0, 1, 3, 55, 56, 58], axis=1)  # Remove all other ts  # Lerp on missing values, comparable with other thesis

    # TODO: greatly decrease r length for testing (365 days + 2 x X amount) and remove 40 vars
    # r = r[1:(365+60):]
    # r = np.delete(r, range(5, 50), axis=1)

    # TODO: greatly decrease r length for testing: 2000-2009 training, 2010 for testing
    # row2000_01_01 = 3653 - 1  # Lerp on missing values, comparable with other thesis
    # row2010_12_31 = 7670
    # r = r[row2000_01_01:row2010_12_31, :]

    # TODO: Greatly decrease r length for testing: 1990-2009 training, 2010 for testing
    # row2010_12_31 = 7670
    # r = r[0:row2010_12_31, :]

    # TODO: greatly decrease r length for testing: 2000-2012 training, 2013 for testing
    # row2000_01_01 = 3653 - 1
    # r = r[row2000_01_01:-1, :]

    # TODO: BETN073 training from O3_BETN016, BETN066, BETN073, O3_BETN121. Remove all other stations and lags.
    # TODO: O3_BETN016 -> 7, 104(lag 0, lag 1) O3_BETN066 -> 22, 119 O3_BETN073 -> 24, 121 O3_BETN121 -> 29, 126. Weather vars: 46 - 96
    # stations_range = [24, 121]  # Only BETN073 and lag-1
    stations_range = [7, 22, 24, 29, 121, 104, 119, 126]  # 4 stations & lag-1:_BETN016, BETN066, BETN073, O3_BETN121
    weather_variables_range = np.array(range(46, 96 + 1))
    columns_range = np.append(stations_range, weather_variables_range)
    r = r[:, columns_range]

    # TODO: greatly decrease r length for testing: 2014-2017 training, 2018 for testing
    # row2014_01_01 = 8777 - 1
    # r = r[row2014_01_01:-1, :]

    # TODO: greatly decrease r length for testing: 2010-2017 training, 2018 for testing
    row2010_01_01 = 7307 - 1
    r = r[row2010_01_01:-1, :]

    print("r[0, 0]", r[0, 0])
    print("r[-1, 0]", r[-1, 0])

    max_len = r.shape[1] - 1
    print('Variables: {}'.format(max_len))
    print('TimeSteps: {}'.format(r.shape[0]))
    x_data = r[:, mimo_outputs:max_len + 1]
    y_data_in = r[:, 0:mimo_outputs]
    print("x_data shape: ".format(x_data.shape))
    print("y_data shape: ".format(y_data_in.shape))

    # TODO: more time-steps instead of 1?
    y_data_in = np.array(y_data_in)
    x_data_3d_in = x_data.reshape(x_data.shape[0], 1, x_data.shape[1])  # reshape to 3D[samples, timesteps, features]

    # TODO: normalize + standardize

    if not os.path.exists("foundModels/min_mse.pkl"):
        min_mse = sys.float_info.max
        print("Previous min_mse: {}".format(min_mse))
        original_df = pd.DataFrame({"min_mse": [min_mse]})
        original_df.to_pickle("foundModels/min_mse.pkl")
    else:
        min_mse = pd.read_pickle("foundModels/min_mse.pkl")['min_mse'][0]
        print("Previous min_mse: {}".format(min_mse))

        if os.path.exists("foundModels/full_{}_rank{}_parameters.pkl".format(modelLabel, gpu_rank)):
            full_model_parameters = pd.read_pickle("foundModels/full_{}_rank{}_parameters.pkl".format(modelLabel,
                                                                                                      gpu_rank))[
                'full_{}_rank{}_parameters'.format(modelLabel, gpu_rank)][0]
            print("Previous full_{}_parameters: {}".format(modelLabel, full_model_parameters))

    return x_data_3d_in, y_data_in


def model_training_callback(ch, method, properties, body):  # Tasks receiver callback
    try:
        body = json.loads(body.decode("utf-8"))
        # print(" [x] Received %r" % body)
        print(" [x] Island: ", body["island"])
        data_manipulation["island"] = body["island"]
        data_manipulation["rank"] = body["rank"]
        baseMpi.train_model.label = body["island"]
        baseMpi.train_model.data_manipulation = data_manipulation
        results_queue = body["results_queue"]
        if not any(results_queue in s for s in results_queues):  # Add queue of results in case it doesn't exist yes
            results_queues.append(results_queue)
            channel.queue_declare(queue=results_queue, durable=False)

        x = np.array(body["array"])
        mse = baseMpi.train_model(x, *args)  # Do train model
        # mse = baseMpi.train_model_tester3(x, *args)  # TODO: ackley for island communications tests
        print(" [x] mse: ", mse)

        ch.basic_ack(delivery_tag=method.delivery_tag)  # Ack receipt of task & work done

        # Send back result to the sender, on its unique receive channel
        mse_message = {"array": x.tolist(), "mse": mse, "island": body["island"]}
        mse_message = json.dumps(mse_message)
        channel.basic_publish(exchange="", routing_key=results_queue, body=mse_message,
                              properties=pika.BasicProperties(delivery_mode=2))  # make msg persistent
        # print(" [x] Sent back '%s'" % mse_message)
        print(" [x] Done")

    except ValueError as ev:  # Handle exceptions
        print(" [x] Exception, sending rejection: %s" % str(ev))
        ch.basic_reject(delivery_tag=method.delivery_tag)


gpu_device = 2  # Set GPU

print("--- Loading GPU {}...".format(gpu_device))
init_gpu(gpu_device)

print("--- Loading simulation settings...")
with open('settings/data_manipulation.json') as f:
    data_manipulation = json.load(f)
modelLabel = data_manipulation["modelLabel"]

if data_manipulation["fp16"]:
    import tensorflow as tf
    tf.keras.backend.set_epsilon(1e-4)
    tf.keras.backend.set_floatx('float16')
    print("--- Working with tensorflow.keras float precision: {}".format(tf.keras.backend.floatx()))

# Choose data

# data_manipulation["directory"] = "data/6vars/"  # Lerp on missing values, comparable with other thesis
# data_manipulation["filePrefix"] = "BETN073"
# data_manipulation["mimoOutputs"] = 1

# data_manipulation["directory"] = "data/6vars_ALL/"  # "closest station" data replacement strategy
# data_manipulation["filePrefix"] = "BETN073_ALL"
# data_manipulation["mimoOutputs"] = 1

# data_manipulation["directory"] = "data/BETN073_BG/"  # TODO: "closest BG station" data replacement strategy
# data_manipulation["filePrefix"] = "BETN073_BG"
# data_manipulation["mimoOutputs"] = 1

# data_manipulation["directory"] = "data/4stations51vars/"  # Lerp on missing values, comparable with other thesis
# data_manipulation["filePrefix"] = "BETN_12_66_73_121_51vars_O3_O3-1_19900101To2000101"
# data_manipulation["mimoOutputs"] = 1

# data_manipulation["directory"] = "data/4stations51vars/"  # Lerp on missing values, comparable with other thesis
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

# data_manipulation["directory"] = "data/PM10_BETN_1995To2019/"
# data_manipulation["filePrefix"] = "PM10_BETN"
# data_manipulation["mimoOutputs"] = 16

# data_manipulation["directory"] = "data/O3_BETN_1990To2019/"
# data_manipulation["filePrefix"] = "O3_BETN"
# data_manipulation["mimoOutputs"] = 46

# TODO: BETN073 training from O3_BETN016, BETN066, BETN073, O3_BETN121. Remove all other stations and lags
# TODO: O3_BETN016 -> 7, 104(lag 0, lag 1) O3_BETN066 -> 22, 119 O3_BETN073 -> 24, 121 O3_BETN121 -> 29, 126. Weather vars: 46 - 96
# TODO: Columns to keep O3_BETN073: 24, 121, 46-96
data_manipulation["directory"] = "data/O3_BETN_1990To2019/"
data_manipulation["filePrefix"] = "O3_BETN"
data_manipulation["mimoOutputs"] = 4

# data_manipulation["directory"] = "data/PM1073stations51vars/"
# data_manipulation["filePrefix"] = "ALL_BE_51vars_PM10_PM10-1_19940101To20121231"
# data_manipulation["mimoOutputs"] = 73

# data_manipulation["directory"] = "data/PM1083stations51vars/"
# data_manipulation["filePrefix"] = "ALL_BE_51vars_PM10_PM10-1_19940101To20190125"
# data_manipulation["mimoOutputs"] = 83

print("--- Loading data...")
x_data_3d, y_data = load_data(data_manipulation["directory"], data_manipulation["filePrefix"],
                              data_manipulation["mimoOutputs"])

data_manipulation["rank"] = gpu_device
data_manipulation["bounds"] = bounds  # TODO: add bounds from modelsearch

iterations = data_manipulation["iterations"]
agents = data_manipulation["agents"]
args = (x_data_3d, y_data)
baseMpi.train_model.counter = 0  # Function call counter
baseMpi.train_model.folds = data_manipulation["folds"]
baseMpi.train_model.data_manipulation = data_manipulation

timeout = 3600 * 10  # Timeouts 60 mins * islands
credentials = pika.PlainCredentials("madks", "ma121284")
params = pika.ConnectionParameters(host="temp3rr0r-pc", heartbeat_interval=timeout, blocked_connection_timeout=timeout,
                                   credentials=credentials)
connection = pika.BlockingConnection(params)  # Connect with msg broker server
channel = connection.channel()  # Listen to channels
channel.queue_declare(queue="task_queue", durable=False)  # Open common task queue
channel.basic_qos(prefetch_count=1)  # Allow only 1 task in queue
results_queues = []  # List of declared results queues

channel.basic_consume(model_training_callback, queue="task_queue")
print(" [*] Waiting for messages. To exit press CTRL+C")
channel.start_consuming()  # Listen for incoming training tasks

print("--- Done(GPU {})!".format(gpu_device))
