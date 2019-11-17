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
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import SGD
from odir_model_base import ModelBase


class VggNet(ModelBase):

    def compile(self):
        x = models.Sequential()
        x.add(layers.ZeroPadding2D((1, 1), input_shape=self.input_shape))
        # Block 1
        x.add(layers.Conv2D(64, (3, 3), activation='relu'))
        x.add(layers.ZeroPadding2D((1, 1)))
        x.add(layers.Conv2D(64, (3, 3), activation='relu'))
        x.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        x.add(layers.ZeroPadding2D((1, 1)))

        # Block 2
        x.add(layers.Conv2D(128, (3, 3), activation='relu'))
        x.add(layers.ZeroPadding2D((1, 1)))
        x.add(layers.Conv2D(128, (3, 3), activation='relu'))
        x.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        x.add(layers.ZeroPadding2D((1, 1)))

        # Block 3
        x.add(layers.Conv2D(256, (3, 3), activation='relu'))
        x.add(layers.ZeroPadding2D((1, 1)))
        x.add(layers.Conv2D(256, (3, 3), activation='relu'))
        x.add(layers.ZeroPadding2D((1, 1)))
        x.add(layers.Conv2D(256, (3, 3), activation='relu'))
        x.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 4
        x.add(layers.ZeroPadding2D((1, 1)))
        x.add(layers.Conv2D(512, (3, 3), activation='relu'))
        x.add(layers.ZeroPadding2D((1, 1)))
        x.add(layers.Conv2D(512, (3, 3), activation='relu'))
        x.add(layers.ZeroPadding2D((1, 1)))
        x.add(layers.Conv2D(512, (3, 3), activation='relu'))
        x.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 5
        x.add(layers.ZeroPadding2D((1, 1)))
        x.add(layers.Conv2D(512, (3, 3), activation='relu'))
        x.add(layers.ZeroPadding2D((1, 1)))
        x.add(layers.Conv2D(512, (3, 3), activation='relu'))
        x.add(layers.ZeroPadding2D((1, 1)))
        x.add(layers.Conv2D(512, (3, 3), activation='relu'))
        x.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

        x.add(layers.Flatten())
        x.add(layers.Dense(4096, activation='relu'))
        x.add(layers.Dropout(0.5))
        x.add(layers.Dense(4096, activation='relu'))
        x.add(layers.Dropout(0.5))
        x.add(layers.Dense(8, activation='sigmoid'))

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        x.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
        self.show_summary(x)
        self.plot_summary(x, 'model_vggnet.png')
        return x
