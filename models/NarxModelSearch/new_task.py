import pika
import sys
import json
# import numpy as np

connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()
channel.queue_declare(queue="task_queue", durable=True)

message = " ".join(sys.argv[1:]) or "Hello World!"

message = "long........."
message = "short."
message = "error..."

array1 = [234.2, 34, 0.005, 0.18, 1.0, 99]
message = json.dumps(array1)

channel.basic_publish(exchange="", routing_key="task_queue", body=message,
                      properties=pika.BasicProperties(delivery_mode=2))  # make msg persistent
print(" [x] Sent '%s'" % message)

connection.close()

