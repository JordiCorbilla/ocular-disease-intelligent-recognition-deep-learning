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
import tensorflow
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import SGD
from odir_model_base import ModelBase


class Vgg16(ModelBase):

    def compile(self):
        x = models.Sequential()
        trainable = False
        # Block 1
        layer = layers.Conv2D(input_shape=self.input_shape, filters=64,kernel_size=(3,3),padding="same", activation="relu")
        layer.trainable = trainable
        x.add(layer)
        layer = layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")
        layer.trainable = trainable
        x.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        x.add(layer)

        # Block 2
        layer = layers.Conv2D(128, kernel_size=(3,3),padding="same", activation="relu")
        layer.trainable = trainable
        x.add(layer)
        layer = layers.Conv2D(128, kernel_size=(3,3),padding="same", activation="relu")
        layer.trainable = trainable
        x.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        x.add(layer)

        # Block 3
        layer = layers.Conv2D(256, kernel_size=(3,3),padding="same", activation="relu")
        layer.trainable = trainable
        x.add(layer)
        layer = layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu")
        layer.trainable = trainable
        x.add(layer)
        layer = layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu")
        layer.trainable = trainable
        x.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        x.add(layer)

        # Block 4
        layer = layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu")
        layer.trainable = trainable
        x.add(layer)
        layer = layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu")
        layer.trainable = trainable
        x.add(layer)
        layer = layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu")
        layer.trainable = trainable
        x.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        x.add(layer)

        # Block 5
        layer = layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu")
        layer.trainable = trainable
        x.add(layer)
        layer = layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu")
        layer.trainable = trainable
        x.add(layer)
        layer = layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu")
        layer.trainable = trainable
        x.add(layer)
        layer = layers.MaxPooling2D((2, 2), strides=(2, 2))
        layer.trainable = trainable
        x.add(layer)

        layer = layers.Flatten()
        layer.trainable = trainable
        x.add(layer)
        layer = layers.Dense(4096, activation='relu')
        layer.trainable = trainable
        x.add(layer)
        layer = layers.Dense(4096, activation='relu')
        layer.trainable = trainable
        x.add(layer)
        layer = layers.Dense(1000, activation='softmax')
        layer.trainable = trainable
        x.add(layer)

        # Transfer learning, load previous weights
        x.load_weights(r'C:\temp\vgg16_weights_tf_dim_ordering_tf_kernels.h5')

        # Remove last layer
        x.pop()

        # Add new dense layer
        x.add(layers.Dense(8, activation='sigmoid'))
        #optimizer = tensorflow.keras.optimizers.SGD(learning_rate=1e-3)
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        x.compile(optimizer=sgd, loss='binary_crossentropy', metrics=self.metrics)

        self.show_summary(x)
        self.plot_summary(x, 'model_vggnet.png')
        return x
