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

import logging.config
import os

import tensorflow as tf
from absl import app
from tensorflow.keras.applications import vgg16

import odir
from odir_advance_plotting import Plotter
from odir_kappa_score import FinalScore
from odir_predictions_writer import Prediction


def main(argv):
    print(tf.version.VERSION)
    image_size = 224
    test_run = 'zCSA'
    new_folder = r'C:\Users\thund\Source\Repos\TFM-ODIR\models\image_classification'

    # load the data
    (x_train, y_train), (x_test, y_test) = odir.load_data(image_size, 1)

    class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']

    # plot data input
    plotter = Plotter(class_names)
    plotter.plot_input_images(x_test, y_test)

    x_test_drawing = x_test

    # normalize input based on model
    x_test = vgg16.preprocess_input(x_test)

    # load one of the test runs
    model = tf.keras.models.load_model(os.path.join(new_folder , 'modelvgg200.h5'))
    model.summary()

    # display the content of the model
    baseline_results = model.evaluate(x_test, y_test, verbose=2)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
    print()

    # test a prediction
    test_predictions_baseline = model.predict(x_test)
    plotter.plot_confusion_matrix_generic(y_test, test_predictions_baseline, test_run, 0)

    # save the predictions
    prediction_writer = Prediction(test_predictions_baseline, 400)
    prediction_writer.save()
    prediction_writer.save_all(y_test)

    # show the final score
    score = FinalScore(new_folder)
    score.output()

    # plot output results
    plotter.plot_output(test_predictions_baseline, y_test, x_test_drawing)


if __name__ == '__main__':
    # create logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('odir')
    app.run(main)
