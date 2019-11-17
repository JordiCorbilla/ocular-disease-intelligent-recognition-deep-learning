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
# import warnings
#
# warnings.filterwarnings("ignore", category=FutureWarning)
import tensorflow as tf
import os
import logging
from matplotlib.image import imread
from PIL import Image


class GenerateTFRecord:
    def __init__(self, images_path):
        self.images_path = images_path
        self.logger = logging.getLogger('odir')

    def patients_to_tfrecord(self, patients, tfrecord_file_name):
        # Iterate through the dictionary of patients
        with tf.io.TFRecordWriter(tfrecord_file_name) as writer:
            for key, value in patients.items():
                self.logger.debug("Processing Image into TF Record:" + key + " - Start")
                eye_image = os.path.join(self.images_path, key)
                train_data = self._patient_image(eye_image, value)
                writer.write(train_data.SerializeToString())
                self.logger.debug("Processing Image into TF Record:" + key + " - End")

    @staticmethod
    def _patient_image(eye_image, patient):
        image_string = open(eye_image, 'rb').read()
        image_shape = tf.image.decode_jpeg(image_string).shape
        # Get filename
        filename = os.path.basename(eye_image)
        # feature = {
        #     'filename': _bytes_feature(filename.encode('utf-8')),
        #     'rows': _int64_feature(image_shape[0]),
        #     'cols': _int64_feature(image_shape[1]),
        #     'channels': _int64_feature(image_shape[2]),
        #     'label': _int64_feature(label),
        #     'image': _bytes_feature(image_string),
        # }

        feature = {
            'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
            'rows': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[0]])),
            'cols': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[1]])),
            'channels': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[2]])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
            'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[patient.id])),
            'age': tf.train.Feature(int64_list=tf.train.Int64List(value=[patient.age])),
            'sex': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patient.sex.encode('utf-8')])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=patient.DiseaseVectorGenerated))
        }

        # Get image shape
        img_shape = imread(eye_image).shape

        # Read the actual image in bytes
        #with tf.io.gfile.GFile(eye_image, 'rb') as fid:
        #    image_data = fid.read()

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example
