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
import matplotlib.pyplot as plt
import os


class TFRecordReader:
    def __init__(self, tfrecord_file):
        self.tfrecord_file = os.path.abspath(tfrecord_file)

    @staticmethod
    def _extract_features(tfrecord):
        # Extract features using the keys set during creation
        record_features = {
            'filename': tf.io.FixedLenFeature([], tf.string),
            'rows': tf.io.FixedLenFeature([], tf.int64),
            'cols': tf.io.FixedLenFeature([], tf.int64),
            'channels': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string),
            'id': tf.io.FixedLenFeature([], tf.int64),
            'age': tf.io.FixedLenFeature([], tf.int64),
            'sex': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
        }

        # Extract the data record
        example = tf.io.parse_single_example(tfrecord, record_features)
        filename = example['filename']
        image_record = tf.image.decode_image(example['image'])
        image_shape = tf.stack([example['rows'], example['cols'], example['channels']])
        id_record = example['id']
        age_record = example['age']
        sex_record = example['sex']
        label_record = example['label']
        return [filename, image_record, image_shape, id_record, age_record, sex_record, label_record]

    def show_images(self):
        # Pipeline of dataset and iterator
        dataset = tf.data.TFRecordDataset([self.tfrecord_file])
        dataset = dataset.map(self._extract_features)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        next_element = iterator.get_next()

        with tf.compat.v1.Session() as sess:
            try:
                while True:
                    data_record = sess.run(next_element)
                    plt.imshow(data_record[1])
                    label = data_record[6]
                    title = data_record[0]
                    if label[0] == 1:
                        title = title + " Normal"
                    if label[1] == 1:
                        title = title + " Diabetes"
                    if label[2] == 1:
                        title = title + " Glaucoma"
                    if label[3] == 1:
                        title = title + " Cataract"
                    if label[4] == 1:
                        title = title + " AMD"
                    if label[5] == 1:
                        title = title + " Hypertension"
                    if label[6] == 1:
                        title = title + " Myopia"
                    if label[7] == 1:
                        title = title + " Others"
                    plt.title(title)
                    plt.show()
            except:
                pass


if __name__ == '__main__':
    t = TFRecordReader('images.tfrecord')
    t.show_images()
