import pika
import time
import json
import numpy as np
from BaseNarxModelMpi import ackley

params = pika.ConnectionParameters(heartbeat_interval=600, blocked_connection_timeout=600)
# params = pika.ConnectionParameters("localhost")
connection = pika.BlockingConnection(params)  # Connect with msg broker server
channel = connection.channel()  # Listen to channels
channel.queue_declare(queue="task_queue", durable=False)  # Open common task queue
channel.basic_qos(prefetch_count=1)  # Allow only 1 task in queue
results_queues = []  # List of declared results queues

print('pika.ConnectionParameters("socket_timeout")', pika.ConnectionParameters("socket_timeout"))
print('pika.ConnectionParameters("blocked_connection_timeout")', pika.ConnectionParameters("blocked_connection_timeout"))
print('pika.ConnectionParameters("heartbeat")', pika.ConnectionParameters("heartbeat"))

def callback(ch, method, properties, body):  # Tasks receiver callback
    try:
        body = json.loads(body)
        print(" [x] Received %r" % body)
        time.sleep(str(body["delay"]).count("."))

        # Do work
        array1 = np.array(body["array"])
        x = array1
        mse = np.mean(array1 * 3)
        # x = np.array(body["array"])
        # mse = ackley([x[0], x[1]])
        print(" [x] mse: ", mse)

        results_queue = body["results_queue"]
        if not any(results_queue in s for s in results_queues):  # Add queue of results in case it doesn't exist yes
            results_queues.append(results_queue)
            channel.queue_declare(queue=results_queue, durable=False)

        ch.basic_ack(delivery_tag=method.delivery_tag)  # Ack receipt of task & work done

        # Send back result to the sender, on its unique receive channel
        mse_message = {"array1": x.tolist(), "mse": mse.tolist(), "island": body["island"]}
        mse_message = json.dumps(mse_message)
        channel.basic_publish(exchange="", routing_key=results_queue, body=mse_message,
                              properties=pika.BasicProperties(delivery_mode=2))  # make msg persistent
        print(" [x] Sent back '%s'" % mse_message)
        print(" [x] Done")

    except ValueError as ev:  # Handle exceptions
        print(" [x] Exception, sending rejection %s" % str(ev))
        ch.basic_reject(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)  # Allow only 1 task in queue
channel.basic_consume(callback, queue="task_queue")

print(" [*] Waiting for messages. To exit press CTRL+C")
channel.start_consuming()  # Listen for incoming tasks
