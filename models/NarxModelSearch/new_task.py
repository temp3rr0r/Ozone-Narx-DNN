import pika
import json
import random
import uuid

connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()
channel.queue_declare(queue="task_queue", durable=False)

island = random.choice(["pso", "rand", "de"])
results_queue = "results_queue" + "_" + island + "_" + str(uuid.uuid4())[:5]
channel.queue_declare(queue=results_queue, durable=False)


def callback(ch, method, properties, body):
    try:
        body = json.loads(body)
        print(" [x] Received %r" % body)
        print(" [x] Received mse: ", body["mse"])
        print(" [x] Done")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        channel.stop_consuming()
    except ValueError as ev:  # Handle exceptions
        print(" [x] Exception, sending rejection %s" % str(ev))
        ch.basic_reject(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)

for i in range(0, 5):

    array1 = {"array": [random.uniform(0, 1), random.uniform(100, 300)], "delay": ["."] * random.randint(0, 8),
              "island": island, "results_queue": results_queue}
    message = json.dumps(array1)
    channel.basic_publish(exchange="", routing_key="task_queue", body=message,
                      properties=pika.BasicProperties(delivery_mode=2))  # make msg persistent
    channel.basic_consume(callback, queue=results_queue)
    print(" [x] Sent '%s'" % message)
    channel.start_consuming()
print(" [*] Waiting for messages. To exit press CTRL+C")

channel.queue_delete(queue=results_queue)
connection.close()

#


