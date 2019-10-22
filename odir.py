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
import numpy as np


def load_data():
    """Loads the ODIR dataset.

    Arguments:
      none

    Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    """
    x_train = np.load('odir_training.npy')
    y_train = np.load('odir_training_labels.npy')

    x_test = np.load('odir_testing.npy')
    y_test = np.load('odir_testing_labels.npy')

    return (x_train, y_train), (x_test, y_test)
