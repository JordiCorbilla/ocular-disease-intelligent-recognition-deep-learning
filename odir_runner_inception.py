# Copyright 2019-2020 Jordi Corbilla. All Rights Reserved.
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
import logging.config
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import app

from odir_advance_plotting import Plotter
from odir_kappa_score import FinalScore
from odir_model_factory import Factory, ModelTypes
from odir_predictions_writer import Prediction

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from sklearn import metrics
import odir


def generator(train_a, labels_a):
    while True:
        for i in range(len(train_a)):
            yield train_a[i].reshape(1, 128, 128, 3), labels_a[i].reshape(1, 8)


def main(argv):
    print(tf.version.VERSION)
    image_size = 128

    (x_train, y_train), (x_test, y_test) = odir.load_data(image_size)

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = (x_train - x_train.mean()) / x_train.std()
    x_test = (x_test - x_test.mean()) / x_test.std()

    defined_metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]

    factory = Factory((image_size, image_size, 3), defined_metrics)

    model = factory.compile(ModelTypes.inception_v1)
    class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']

    # plot data input
    plotter = Plotter(class_names)
    print("Training")

    class_weight = {0: 1.,
                    1: 1.583802025,
                    2: 8.996805112,
                    3: 10.24,
                    4: 10.05714286,
                    5: 14.66666667,
                    6: 10.7480916,
                    7: 2.505338078}

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)

    history = model.fit_generator(generator=generator(x_train, y_train), steps_per_epoch=len(x_train),
                                  epochs=30, verbose=1, callbacks=[callback], validation_data=(x_test, y_test),
                                  shuffle=True)

    print("plotting")
    plotter.plot_metrics(history, 'inception_1', 2)
    print("saving")
    model.save('model_inception_30.h5')

    # Hide meanwhile for now
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('image_run2' + 'inception_1' + '.png')
    plt.show()

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(test_acc)

    test_predictions_baseline = model.predict(x_test)
    plotter.plot_confusion_matrix_generic(y_test, test_predictions_baseline, 'inception_1', 0)

    # save the predictions
    prediction_writer = Prediction(test_predictions_baseline, 400)
    prediction_writer.save()
    prediction_writer.save_all(y_test)

    # show the final score
    score = FinalScore()
    score.output()


if __name__ == '__main__':
    # create logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('odir')
    app.run(main)
