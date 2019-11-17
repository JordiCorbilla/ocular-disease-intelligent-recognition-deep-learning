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
from abc import abstractmethod
import tensorflow as tf


class ModelBase:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def show_summary(self, model):
        model.summary()

    def plot_summary(self, model, file_name):
        tf.keras.utils.plot_model(model, to_file=file_name, show_shapes=True, show_layer_names=True)

    @abstractmethod
    def compile(self):
        pass
