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

# Remove all future warnings thrown by numpy
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import tensorflow as tf
import os
from matplotlib.image import imread
from PIL import Image


class GenerateTFRecord:
    def __init__(self, images_path):
        self.images_path = images_path

    def patients_to_tfrecord(self, patients, tfrecord_file_name):
        # Iterate through the dictionary of patients
        with tf.io.TFRecordWriter(tfrecord_file_name) as writer:
            for key, value in patients.items():
                eye_image = os.path.join(self.images_path, key)
                train_data = self._patient_image(eye_image, value)
                writer.write(train_data.SerializeToString())

    @staticmethod
    def _patient_image(eye_image, patient):
        # Read the actual image
        image_data = imread(eye_image)
        # Convert image to string data
        image_str = image_data.tostring()
        # Store shape of image for reconstruction purposes
        img_shape = image_data.shape
        # Get filename
        filename = os.path.basename(eye_image)
        example = tf.train.Example(features=tf.train.Features(feature={
            'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
            'rows': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_shape[0]])),
            'cols': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_shape[1]])),
            'channels': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_shape[2]])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_str])),
            'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[patient.id])),
            'age': tf.train.Feature(int64_list=tf.train.Int64List(value=[patient.age])),
            'sex': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patient.sex.encode('utf-8')])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=patient.DiseaseVectorGenerated))
        }))
        return example
