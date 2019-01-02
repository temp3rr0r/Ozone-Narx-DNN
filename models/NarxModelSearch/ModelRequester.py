import pika
import json
import random
import uuid
import time
import numpy as np
from BaseNarxModelMpi import train_model


def train_model_requester_rabbit_mq(x):

    start_time = time.time()  # training time per model

    train_model.counter += 1
    data_manipulation = train_model.data_manipulation
    island = data_manipulation["island"]
    rank = data_manipulation["rank"]
    master = data_manipulation["master"]

    # TODO: send to worker

    timeout = 600 * 10  # TODO: timeouts 10 mins * islands
    params = pika.ConnectionParameters(heartbeat_interval=timeout, blocked_connection_timeout=timeout)
    connection = pika.BlockingConnection(params)  # Connect with msg broker server
    channel = connection.channel()  # Listen to channels
    channel.queue_declare(queue="task_queue", durable=False)  # Open common task queue

    results_queue = "results_queue" + "_" + island + "_" + str(uuid.uuid4())[:5]
    channel.queue_declare(queue=results_queue, durable=False)  # Open unique results channel for island
    channel.basic_qos(prefetch_count=1)

    msg = {"array": x.tolist(), "island": island, "results_queue": results_queue}
    message = json.dumps(msg)  # Serialize msg
    channel.basic_publish(exchange="", routing_key="task_queue", body=message,  # Use common task queue
                          properties=pika.BasicProperties(delivery_mode=2))  # make msg persistent

    def trained_model_callback(ch, method, properties, body):  # Results receiver callback
        try:
            body = json.loads(body)
            print(" [x] Received mse: ", body["mse"])
            trained_model_callback.mse = body["mse"]  # TODO: store mean_mse
            print(" [x] Done")
            ch.basic_ack(delivery_tag=method.delivery_tag)  # Acknowledge msg delivery
            channel.stop_consuming()  # Stop listening for results
        except ValueError as ev:  # Handle exceptions
            print(" [x] Exception, sending rejection %s" % str(ev))
            ch.basic_reject(delivery_tag=method.delivery_tag)

    trained_model_callback.mse = -1.0
    channel.basic_consume(trained_model_callback, queue=results_queue)  # Listen on unique results channel
    channel.start_consuming()  # Start listening for results
    print(" [*] Waiting for messages. To exit press CTRL+C")
    channel.queue_delete(queue=results_queue)  # Delete the results queue
    connection.close()  # Stop all connections
    print(" [x] Closed connection.")

    end_time = time.time()
    mean_mse = trained_model_callback.mse

    # Worker to master
    data_worker_to_master = {
        "worked": end_time - start_time, "rank": rank, "mean_mse": mean_mse, "agent": x,
        "island": island, "iteration": train_model.counter}
    comm = data_manipulation["comm"]
    req = comm.isend(data_worker_to_master, dest=master, tag=1)  # Send data async to master
    req.wait()

    # Master to worker
    agent_to_ea = {"swapAgent": False, "agent": None}
    data_master_to_worker = comm.recv(source=master, tag=2)  # Receive data sync (blocking) from master
    swap_agent = data_master_to_worker["swapAgent"]
    if swap_agent:
        out_agent = data_master_to_worker["agent"]
        agent_to_ea = {"swapAgent": True, "agent": out_agent}  # Send agent copy
    if island == "bh":
        return mean_mse

    return mean_mse, agent_to_ea
