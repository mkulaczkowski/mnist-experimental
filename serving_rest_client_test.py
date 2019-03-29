from random import randint

import requests
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
image_index = randint(0, 9999)  # You may select anything up to 60,000

test_image_data = x_test[image_index].tolist()

vector = []
for item in test_image_data:
    vector.extend(item)

json = {
    "inputs": [vector]
}

response = requests.post('http://127.0.0.1:8501/v1/models/mnist:predict', json=json)

print(response.status_code)
print(response.text)