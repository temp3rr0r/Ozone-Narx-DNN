import pika
import json
import random
import uuid
import time
import numpy as np
from BaseNarxModelMpi import train_model


def train_model_requester_rabbit_mq(x, *args):

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
    mean_mse = 0

    # TODO: func to optimize
    timeToSleep = np.random.uniform(2, 5)
    time.sleep(timeToSleep)

    train_model.counter += 1
    endTime = time.time()
    # Worker to master
    dataWorkerToMaster = {"worked": endTime - startTime, "rank": rank, "mean_mse": mean_mse, "agent": x,
                          "island": island, "iteration": train_model.counter}
    # Master to worker
    agentToEa = {"swapAgent": False, "agent": None}

    # TODO: send to worker

    timeout = 600 * 10  # TODO: timeouts 10 mins * islands
    params = pika.ConnectionParameters(heartbeat_interval=timeout, blocked_connection_timeout=timeout)
    connection = pika.BlockingConnection(params)  # Connect with msg broker server
    channel = connection.channel()  # Listen to channels
    channel.queue_declare(queue="task_queue", durable=False)  # Open common task queue

    results_queue = "results_queue" + "_" + island + "_" + str(uuid.uuid4())[:5]
    channel.queue_declare(queue=results_queue, durable=False)  # Open unique results channel for island

    channel.basic_qos(prefetch_count=1)

    array1 = {"array": x.tolist(), "delay": ["."] * random.randint(0, 8),  # TODO: np array x
              "island": island, "results_queue": results_queue}
    message = json.dumps(array1)  # Serialize msg
    channel.basic_publish(exchange="", routing_key="task_queue", body=message,  # Use common task queue
                          properties=pika.BasicProperties(delivery_mode=2))  # make msg persistent

    def trained_model_callback(ch, method, properties, body):  # Results receiver callback
        try:
            body = json.loads(body)
            print(" [x] Received %r" % body)
            print(" [x] Received mse: ", body["mse"])

            # TODO: store mean_mse
            mean_mse = body["mse"]

            print(" [x] Done")
            ch.basic_ack(delivery_tag=method.delivery_tag)  # Acknowledge msg delivery
            channel.stop_consuming()  # Stop listening for results
        except ValueError as ev:  # Handle exceptions
            print(" [x] Exception, sending rejection %s" % str(ev))
            ch.basic_reject(delivery_tag=method.delivery_tag)

    channel.basic_consume(trained_model_callback, queue=results_queue)  # Listen for task results on unique results channel
    print(" [x] Sent '%s'" % message)
    channel.start_consuming()  # Start listening for results
    print(" [*] Waiting for messages. To exit press CTRL+C")
    channel.queue_delete(queue=results_queue)  # Delete the results queue
    connection.close()  # Stop all connections

    return mean_mse, {"swapAgent": True, "agent": x}  # TODO: test send only if swap agent is enabled

    # # Worker to master  # TODO: test send only if swap agent is enabled
    # dataWorkerToMaster = {"worked": endTime - startTime, "rank": rank, "mean_mse": mean_mse, "agent": x,
    #                       "island": island, "iteration": train_model.counter}
    # comm = data_manipulation["comm"]
    # req = comm.isend(dataWorkerToMaster, dest=master, tag=1)  # Send data async to master
    # req.wait()
    # # Master to worker
    # agentToEa = {"swapAgent": False, "agent": None}
    # dataMasterToWorker = comm.recv(source=master, tag=2)  # Receive data sync (blocking) from master
    # swapAgent = dataMasterToWorker["swapAgent"]
    # if swapAgent:
    #     outAgent = dataMasterToWorker["agent"]
    #     agentToEa = {"swapAgent": True, "agent": outAgent}  # Send agent copy
    # if island == "bh":
    #     return mean_mse
    # return mean_mse, agentToEa
