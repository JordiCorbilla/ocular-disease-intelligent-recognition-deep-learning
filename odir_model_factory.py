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
import enum

from odir_model_advanced import Advanced
from odir_model_inception_v1 import InceptionV1
from odir_model_vgg16 import Vgg16


class ModelTypes(enum.Enum):
    vgg16 = 1
    inception_v1 = 2
    advanced_testing = 3


class Factory:

    def __init__(self, input_shape, metrics):
        self.Makers = {
            ModelTypes.vgg16: Vgg16(input_shape, metrics),
            ModelTypes.inception_v1: InceptionV1(input_shape, metrics),
            ModelTypes.advanced_testing: Advanced(input_shape, metrics)
        }

    def compile(self, model_type):
        return self.Makers[model_type].compile()
