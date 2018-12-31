import pika
import time
import json
import numpy as np

connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()
channel.queue_declare(queue="task_queue", durable=False)
# channel.queue_declare(queue="results_queue", durable=False)
results_queues = []

def callback(ch, method, properties, body):
    try:
        body = json.loads(body)
        print(" [x] Received %r" % body)
        time.sleep(str(body["delay"]).count("."))

        # Do work
        array1 = np.array(body["array"])
        mse = np.mean(array1 * 3)
        print(" [x] mse: ", mse)

        # TODO: add queue of results in case it doesn't exist
        # island = body["island"]
        results_queue = body["results_queue"]
        if not any(results_queue in s for s in results_queues):
            results_queues.append(results_queue)
            channel.queue_declare(queue=results_queue, durable=False)

        ch.basic_ack(delivery_tag=method.delivery_tag)

        # TODO: send back result
        mse_message = {"array1": array1.tolist(), "mse": mse.tolist(), "island": body["island"]}
        mse_message = json.dumps(mse_message)
        channel.basic_publish(exchange="", routing_key=results_queue, body=mse_message,
                              properties=pika.BasicProperties(delivery_mode=2))  # make msg persistent
        print(" [x] Sent back'%s'" % mse_message)
        print(" [x] Done")

    except ValueError as ev:  # Handle exceptions
        print(" [x] Exception, sending rejection %s" % str(ev))
        ch.basic_reject(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(callback,
                      queue="task_queue")

print(" [*] Waiting for messages. To exit press CTRL+C")
channel.start_consuming()
