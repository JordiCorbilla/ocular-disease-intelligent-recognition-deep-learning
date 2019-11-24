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
from odir_model_base import ModelBase


class Vgg16(ModelBase):

    def compile(self):
        x = models.Sequential()

        # Block 1
        x.add(layers.Conv2D(input_shape=self.input_shape, filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        x.add(layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        x.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 2
        x.add(layers.Conv2D(128, kernel_size=(3,3),padding="same", activation="relu"))
        x.add(layers.Conv2D(128, kernel_size=(3,3),padding="same", activation="relu"))
        x.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 3
        x.add(layers.Conv2D(256, kernel_size=(3,3),padding="same", activation="relu"))
        x.add(layers.Conv2D(256, kernel_size=(3,3),padding="same", activation="relu"))
        x.add(layers.Conv2D(256, kernel_size=(3,3),padding="same", activation="relu"))
        x.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 4
        x.add(layers.Conv2D(512, kernel_size=(3,3),padding="same", activation="relu"))
        x.add(layers.Conv2D(512, kernel_size=(3,3),padding="same", activation="relu"))
        x.add(layers.Conv2D(512, kernel_size=(3,3),padding="same", activation="relu"))
        x.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 5
        x.add(layers.Conv2D(512, kernel_size=(3,3),padding="same", activation="relu"))
        x.add(layers.Conv2D(512, kernel_size=(3,3),padding="same", activation="relu"))
        x.add(layers.Conv2D(512, kernel_size=(3,3),padding="same", activation="relu"))
        x.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

        x.add(layers.Flatten())
        x.add(layers.Dense(4096, activation='relu'))
        x.add(layers.Dense(4096, activation='relu'))
        x.add(layers.Dense(1000, activation='softmax'))

        # Transfer learning, load previous weights
        x.load_weights(r'C:\temp\vgg16_weights_tf_dim_ordering_tf_kernels.h5')

        # Remove last layer
        x.pop()

        # Add new dense layer
        x.add(layers.Dense(8, activation='sigmoid'))
        x.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.show_summary(x)
        self.plot_summary(x, 'model_vggnet.png')
        return x
