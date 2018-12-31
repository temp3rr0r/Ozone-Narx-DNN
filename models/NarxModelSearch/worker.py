import pika
import time
import json
import numpy as np

connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()
channel.queue_declare(queue="task_queue", durable=True)


def callback(ch, method, properties, body):
    try:
        body = json.loads(body)
        print(" [x] Received %r" % body)
        time.sleep(body.count(b"."))

        # Do work
        array1 = np.array(body)
        mse = array1 * 3
        print(" [x] mse: ", np.mean(mse))

        print(" [x] Done")
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except ValueError as ev:  # Handle exceptions
        print(" [x] Exception, sending rejection %s" % str(ev))
        ch.basic_reject(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(callback,
                      queue="task_queue")

print(" [*] Waiting for messages. To exit press CTRL+C")
channel.start_consuming()
