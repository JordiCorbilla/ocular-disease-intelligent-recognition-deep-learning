# Copyright 2019 Jordi Corbilla. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import logging
import logging.config
import time
import pydot
import graphviz

import tensorflow
import tensorflow as tf
from absl import app
import numpy as np
import matplotlib.pyplot as plt
import os

import odir
from odir_model_factory import Factory, ModelTypes
from odir_model_googlenet import GoogleNet
from odir_model_vggnet import VggNet

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras import Model, datasets, models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras import Input
from tensorflow.python.keras.utils import plot_model

#gpu_options.allocator_type = 'BFC'

def main(argv):
    # Print TF version
    print(tf.version.VERSION)

    # Load ODIR Data
    (x_train, y_train), (x_test, y_test) = odir.load_data()

    # format data [0..1]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    #factory = Factory((224,224,3))
    #model = factory.compile(ModelTypes.vgg_net)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = Conv2D(32, (3,3), activation='relu')  # , input_shape=(28,28,3)
            self.max = MaxPooling2D()
            self.conv2 = Conv2D(32, (3,3), activation='relu')  # , input_shape=(28,28,3)
            self.conv3 = Conv2D(64, (3,3), activation='relu')  # , input_shape=(28,28,3)
            self.flatten = Flatten()
            self.d1 = Dense(512, activation='relu')
            self.d2 = Dense(8, activation='sigmoid')
            self.dropout = Dropout(0.2)

        def call(self, x):
            x = self.conv1(x)
            x = self.max(x)
            x = self.dropout(x)
            x = self.conv2(x)
            x = self.max(x)
            x = self.conv3(x)
            x = self.max(x)
            x = self.dropout(x)
            x = self.flatten(x)
            x = self.d1(x)
            x = self.d2(x)
            return x

    # Create an instance of the model
    model = MyModel()

    loss_object = tf.keras.losses.BinaryCrossentropy()

    optimizer = tf.keras.optimizers.Adam()
    tf.keras.backend.set_floatx('float64')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryCrossentropy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryCrossentropy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)
        # logger.debug(predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)
        # print(test_accuracy.result().numpy())

    EPOCHS = 5
    # summary_writer = tf.summary.create_file_writer('./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    for epoch in range(EPOCHS):
        start = time.time()
        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Time: {}s'
        end = time.time()

        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100,
                              str(end - start)))
        #
        # for i in test_accuracy.metrics():
        #     print(i)

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


if __name__ == '__main__':
    # create logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('odir')
    app.run(main)