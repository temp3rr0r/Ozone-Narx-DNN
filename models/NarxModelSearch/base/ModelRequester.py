import pika
import json
import uuid
import time
from base.NeuroevolutionModelTraining import train_model


def train_model_requester_rabbit_mq(x):
    """
    Sends model training request to the Rabbit MQ message broker. Also block-waits for response.
    :param x: Full model parameters.
    :return: None.
    """
    start_time = time.time()  # training time per model

    train_model.counter += 1
    data_manipulation = train_model.data_manipulation
    island = data_manipulation["island"]
    rank = data_manipulation["rank"]
    master = data_manipulation["master"]

    # Send to worker

    timeout = 3600 * 10  # Timeouts 60 mins * islands
    # credentials = pika.PlainCredentials("madks", "asdf")
    # params = pika.ConnectionParameters(heartbeat_interval=timeout, blocked_connection_timeout=timeout, credentials=credentials)  # TODO: check heartbeat_interval -> heartbeat
    params = pika.ConnectionParameters(heartbeat=timeout, blocked_connection_timeout=timeout, credentials=credentials)
    connection = pika.BlockingConnection(params)  # Connect with msg broker server
    channel = connection.channel()  # Listen to channels
    channel.queue_declare(queue="task_queue", durable=False)  # Open common task queue

    results_queue = "results_queue" + "_" + island + "_" + str(uuid.uuid4())[:5]
    channel.queue_declare(queue=results_queue, durable=False)  # Open unique results channel for island
    channel.basic_qos(prefetch_count=1)

    msg = {"array": x.tolist(), "island": island, "results_queue": results_queue, "rank": data_manipulation["rank"]}
    message = json.dumps(msg)  # Serialize msg
    channel.basic_publish(exchange="", routing_key="task_queue", body=message,  # Use common task queue
                          properties=pika.BasicProperties(delivery_mode=2))  # make msg persistent

    def trained_model_callback(ch, method, properties, body):
        """
        Callback method that handles trained model message.
        :param ch: Channel. For declaring acceptance/rejection of acks.
        :param method: For declaring ack of the delivery tag.
        :param properties: None for now.
        :param body: Received serialized json message containing the full message from the worker/trainers.
        """
        try:
            body = json.loads(body)
            print(" [x] Received mse: ", body["mse"])
            trained_model_callback.mse = body["mse"]
            print(" [x] Done")
            ch.basic_ack(delivery_tag=method.delivery_tag)  # Acknowledge msg delivery
            channel.stop_consuming()  # Stop listening for results
        except ValueError as ev:  # Handle exceptions
            print(" [x] Exception, sending rejection %s" % str(ev))
            ch.basic_reject(delivery_tag=method.delivery_tag)

    trained_model_callback.mse = -1.0
    # channel.basic_consume(trained_model_callback, queue=results_queue)  # Listen on unique results channel # TODO: check TypeError: basic_consume() got multiple values for argument 'queue'
    channel.basic_consume(queue=results_queue, on_message_callback=trained_model_callback)  # Listen on unique results channel # TODO: check TypeError: basic_consume() got multiple values for argument 'queue'
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

    return mean_mse, data_worker_to_master
