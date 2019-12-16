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

from odir_model_factory import Factory, ModelTypes
from odir_predictions_writer import Prediction

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from sklearn import metrics
import odir


def main(argv):
    print(tf.version.VERSION)
    image_size = 224

    (x_train, y_train), (x_test, y_test) = odir.load_data(image_size)

    x_train, x_test = x_train / 1.0, x_test / 1.0

    x_train = x_train[..., ::-1]
    x_test = x_test[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    x_train[..., 0] -= mean[0]
    x_train[..., 1] -= mean[1]
    x_train[..., 2] -= mean[2]
    x_test[..., 0] -= mean[0]
    x_test[..., 1] -= mean[1]
    x_test[..., 2] -= mean[2]

    x_train = (x_train - x_train.mean()) / x_train.std()
    x_test = (x_test - x_test.mean()) / x_test.std()

    plt.figure(figsize=(9, 9))
    for i in range(100):
        plt.subplot(10, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i])
    plt.subplots_adjust(bottom=0.04, right=0.94, top=0.95, left=0.06, wspace=0.20, hspace=0.17)
    plt.show()

    factory = Factory((image_size, image_size, 3))
    model = factory.compile(ModelTypes.vgg16)

    print("Training")

    class_weight = {0: 1.,
                    1: 1.583802025,
                    2: 8.996805112,
                    3: 10.24,
                    4: 10.05714286,
                    5: 14.66666667,
                    6: 10.7480916,
                    7: 2.505338078}

    history = model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1, shuffle=True,
                        validation_data=(x_test, y_test), class_weight=class_weight)

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(test_acc)

    predictions = model.predict(x_test)

    def odir_metrics(gt_data, pr_data):
        th = 0.5
        gt = gt_data.flatten()
        pr = pr_data.flatten()
        kappa = metrics.cohen_kappa_score(gt, pr > th)
        f1 = metrics.f1_score(gt, pr > th, average='micro')
        auc = metrics.roc_auc_score(gt, pr)
        final_score = (kappa + f1 + auc) / 3.0
        return kappa, f1, auc, final_score

    def import_data(filepath):
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            pr_data = [[int(row[0])] + list(map(float, row[1:])) for row in reader]
        pr_data = np.array(pr_data)
        return pr_data

    prediction_writer = Prediction(predictions, 400)
    prediction_writer.save()
    prediction_writer.save_all(y_test)

    gt_data = import_data('odir_ground_truth.csv')
    pr_data = import_data('odir_predictions.csv')
    kappa, f1, auc, final_score = odir_metrics(gt_data[:, 1:], pr_data[:, 1:])
    print("kappa score:", kappa, " f-1 score:", f1, " AUC vlaue:", auc, " Final Score:", final_score)


if __name__ == '__main__':
    # create logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('odir')
    app.run(main)
