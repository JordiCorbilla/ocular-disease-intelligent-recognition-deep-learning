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
import tensorflow
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.python.keras import Input
from odir_model_base import ModelBase


class GoogleNet(ModelBase):

    def compile(self):
        input_img = Input(shape=self.input_shape)
        layer_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input_img)
        layer_1 = Conv2D(10, (3, 3), padding='same', activation='relu')(layer_1)

        layer_2 = Conv2D(10, (1, 1), padding='same', activation='relu')(input_img)
        layer_2 = Conv2D(10, (5, 5), padding='same', activation='relu')(layer_2)

        layer_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
        layer_3 = Conv2D(10, (1, 1), padding='same', activation='relu')(layer_3)

        mid_1 = tensorflow.keras.layers.concatenate([layer_1, layer_2, layer_3], axis=3)
        flat_1 = Flatten()(mid_1)

        dense_1 = Dense(1200, activation='relu')(flat_1)
        dense_2 = Dense(600, activation='relu')(dense_1)
        dense_3 = Dense(150, activation='relu')(dense_2)
        output = Dense(8, activation='sigmoid')(dense_3)
        model = Model([input_img], output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.show_summary(model)
        self.plot_summary(model, 'model_googlenet.png')

        return model
