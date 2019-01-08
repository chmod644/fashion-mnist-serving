#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pprint import pprint

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import json
import requests

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num', 3, "number of prediction image", short_name='n')


def main(argv=None):

    ### Import the Fashion MNIST datasets
    train_images, train_labels, test_images, test_labels = load_data()
    train_images, test_images = preprocess_images(train_images, test_images)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    data = {"signature_name": "serving_default", "instances": test_images[0:FLAGS.num].tolist()}
    pprint(type(data['instances']))
    pprint(np.array(data['instances']).shape)
    data = json.dumps(data)
    print('Data: {} ... {}'.format(data[:50], data[len(data) - 52:]))

    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/fashion_model:predict', data=data, headers=headers)
    response = json.loads(json_response.text)
    predictions = response['predictions']
    pprint('Response: {}'.format(response))

    for i in range(FLAGS.num):
        predict_label = np.argmax(predictions[i])
        gt_label = test_labels[i]
        show(test_images[i], 'The model thought this was a {} (class {}, score {}), and it was actually a {} (class {})'.format(
            class_names[predict_label], predict_label, predictions[i][predict_label], class_names[gt_label], gt_label))


def load_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    return train_images, train_labels, test_images, test_labels


def preprocess_images(train_images, test_images):
    # scale the values to 0.0 to 1.0
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # reshape for feeding into the model
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    return train_images, test_images


def show(test_image, title):
    plt.figure()
    plt.imshow(test_image.reshape(28,28))
    plt.axis('off')
    plt.title('\n\n{}'.format(title), fontdict={'size': 16})
    plt.show()


if __name__ == '__main__':
    tf.app.run()