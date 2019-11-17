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

    factory = Factory((128,128,3))
    model = factory.compile(ModelTypes.vgg_net)


if __name__ == '__main__':
    # create logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('odir')
    app.run(main)