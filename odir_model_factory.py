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
import enum

from odir_model_advanced import Advanced
from odir_model_googlenet import GoogleNet
from odir_model_vggnet import VggNet


class ModelTypes(enum.Enum):
    vgg_net = 1
    google_net = 2
    advanced = 3


class Factory:

    def __init__(self, input_shape):
        self.Makers = {
            ModelTypes.vgg_net: VggNet(input_shape),
            ModelTypes.google_net: GoogleNet(input_shape),
            ModelTypes.advanced: Advanced(input_shape)
        }

    def compile(self, model_type):
        return self.Makers[model_type].compile()
