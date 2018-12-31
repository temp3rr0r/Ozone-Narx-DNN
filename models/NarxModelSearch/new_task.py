import pika
import sys

connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()
channel.queue_declare(queue="task_queue", durable=True)

message = " ".join(sys.argv[1:]) or "Hello World!"

message = "long........."
message = "short."
message = "error..."


channel.basic_publish(exchange="", routing_key="task_queue", body=message,
                      properties=pika.BasicProperties(delivery_mode=2))  # make msg persistent
print(" [x] Sent '%s'" % message)

connection.close()

