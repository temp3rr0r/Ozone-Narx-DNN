import pika
import time

connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()
channel.queue_declare(queue="task_queue", durable=True)


def callback(ch, method, properties, body):
    try:
        print(" [x] Received %r" % body)
        # if "error" in str(body):
        #     raise ValueError("Value error thrown exception")
        time.sleep(body.count(b"."))
        print(" [x] Done")
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except ValueError as ev:
        print(" [x] Exception, sending rejection %s" % str(ev))
        ch.basic_reject(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(callback,
                      queue="task_queue")

print(" [*] Waiting for messages. To exit press CTRL+C")
channel.start_consuming()
