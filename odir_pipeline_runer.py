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
import logging.config
import tensorflow as tf
from absl import app
import numpy as np
import matplotlib.pyplot as plt
import os

from odir_model_factory import Factory, ModelTypes
from odir_predictions_writer import Prediction
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from sklearn import metrics

import odir


def main(argv):
    print(tf.version.VERSION)

    (x_train, y_train), (x_test, y_test) = odir.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = (x_train - x_train.mean()) / x_train.std()
    x_test = (x_test - x_test.mean()) / x_test.std()

    factory = Factory((128,128,3))
    model = factory.compile(ModelTypes.vgg_net)

    print("Training")
    history = model.fit(x_train, y_train, epochs=1,batch_size=32,
                        validation_data=(x_test, y_test))

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


    ##Additional test against the training dataset
    test_loss, test_acc = model.evaluate(x_train, y_train, verbose=2)
    print(test_acc)

    predictions = model.predict(x_train)
    prediction_writer = Prediction(predictions, 400)
    prediction_writer.save()
    prediction_writer.save_all(y_train)

    gt_data = import_data('odir_ground_truth.csv')
    pr_data = import_data('odir_predictions.csv')
    kappa, f1, auc, final_score = odir_metrics(gt_data[:, 1:], pr_data[:, 1:])
    print("kappa score:", kappa, " f-1 score:", f1, " AUC vlaue:", auc, " Final Score:", final_score)


if __name__ == '__main__':
    # create logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('odir')
    app.run(main)
