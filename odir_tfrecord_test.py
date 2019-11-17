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
import math
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import tensorflow as tf
import matplotlib.pyplot as plt
import os

BATCH_SIZE = 32
AUTO = tf.data.experimental.AUTOTUNE

class TFRecordReader:
    def __init__(self, tfrecord_file):
        self.tfrecord_file = os.path.abspath(tfrecord_file)

    # def read_tfrecord(example):
    #     features = {
    #         "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
    #         "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar
    #     }
    #     # decode the TFRecord
    #     example = tf.io.parse_single_example(example, features)
    #
    #     image = tf.image.decode_jpeg(example['image'], channels=3)
    #     image = tf.cast(image, tf.float32) / 255.0
    #     image = tf.reshape(image, [TARGET_SIZE, TARGET_SIZE, 3])
    #
    #     class_label = tf.cast(example['class'], tf.int32)
    #
    #     return image, class_label

    def get_dataset_from_tfrecord(self):
        option_no_order = tf.data.Options()
        option_no_order.experimental_deterministic = False

        dataset = tf.data.TFRecordDataset([self.tfrecord_file])

        # dataset = tf.data.Dataset.list_files(filenames)

        dataset = dataset.with_options(option_no_order)
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)

        dataset = dataset.map(self._extract_features, num_parallel_calls=AUTO)
        #dataset = dataset.map(self._extract_features) #, num_parallel_calls=AUTO

        dataset = dataset.cache()  # This dataset fits in RAM
        dataset = dataset.repeat()
        #dataset = dataset.shuffle(2048)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        dataset = dataset.prefetch(AUTO)

        return dataset

    def make_iterator(dataset):
        iterator = dataset.make_one_shot_iterator()
        next_val = iterator.get_next()

        with tf.compat.v1.Session() as sess:
            while True:
                *inputs, labels = sess.run(next_val)
                yield inputs, labels

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
        #image_record = tf.image.decode_image(example['image'])

        #image = tf.image.decode_jpeg(example['image'], channels=3)
        #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        #image = tf.expand_dims(image, 0)
        #image = tf.image.resize_bilinear(image, [28, 28], align_corners=False)

        image = tf.image.decode_jpeg(example['image'], channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.reshape(image, [28, 28, 3])

        #image = tf.cast(image, tf.float32) / 255.0
        #image = tf.reshape(image, [150, 150, 3])

        image_shape = tf.stack([example['rows'], example['cols'], example['channels']])
        id_record = example['id']
        age_record = example['age']
        sex_record = example['sex']
        label_record = example['label']
        class_label = tf.cast(example['label'], tf.int32)
        # return filename, image_record, image_shape, id_record, age_record, sex_record, label_record

        return image, class_label

    def show(self):
        dataset = tf.data.TFRecordDataset([self.tfrecord_file])
        dataset = dataset.map(self._extract_features)

        BATCH_SIZE = 32

        ds = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=7000))
        # ds = dataset.shuffle(buffer_size=image_count)
        ds = ds.repeat()
        # ds = ds.batch(BATCH_SIZE)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        for filename, image_record, image_shape, id_record, age_record, sex_record, label_record in ds.take(1):
            plt.title(filename)
            plt.imshow(image_record)

    def extract_images_and_labels(self):
        dataset = tf.data.TFRecordDataset([self.tfrecord_file])
        dataset = dataset.map(self._extract_features)
        images, labels = tf.train.shuffle_batch([dataset[1], dataset[6]], batch_size=10, capacity=30, num_threads=1,
                                                min_after_dequeue=10)
        return images, labels

    def get_iterator(self):
        dataset = tf.data.TFRecordDataset([self.tfrecord_file])
        iterator = dataset.make_one_shot_iterator()
        next_val = iterator.get_next()

        with tf.compat.v1.Session() as sess:
            while True:
                *inputs, labels = sess.run(next_val)
                yield inputs, labels

    def show_images(self):
        # Pipeline of dataset and iterator
        dataset = tf.data.TFRecordDataset([self.tfrecord_file])
        dataset = dataset.map(self._extract_features)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        next_element = iterator.get_next()
        class_names = ['Normal', 'Diabetes', 'Glaucoma', 'AMD', 'Hypertension', 'Myopia', 'Others']

        with tf.compat.v1.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                while True:
                    data_record = sess.run(next_element)
                    plt.imshow(data_record[1])
                    plt.colorbar()
                    plt.grid(False)
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
                print("Error!", sys.exc_info()[0], "occurred.")
                pass

    def tfdata_generator(self, is_training, batch_size=128):
        '''Construct a data generator using `tf.Dataset`. '''

    #     def map_fn(image, label):
    #         '''Preprocess raw data to trainable input. '''
    #
    #     x = tf.reshape(tf.cast(image, tf.float32), (28, 28, 1))
    #     y = tf.one_hot(tf.cast(label, tf.uint8), _NUM_CLASSES)
    #     return x, y
    #
    # dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        dataset = tf.data.TFRecordDataset([self.tfrecord_file])

        #if is_training:
        #    dataset = dataset.shuffle(1000)  # depends on sample size
        dataset = dataset.map(self._extract_features)
        #dataset = dataset.batch(batch_size)
        #dataset = dataset.repeat()
        #dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    def simple_pipeline(self, dataset):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(kernel_size=3, filters=16, padding='same', activation='relu', input_shape=[28, 28, 3]),
            tf.keras.layers.Conv2D(kernel_size=3, filters=32, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(kernel_size=3, filters=64, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(kernel_size=3, filters=128, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(kernel_size=3, filters=256, padding='same', activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, 'sigmoid')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        training_set = self.tfdata_generator(is_training=True)

        model.fit(dataset, steps_per_epoch=BATCH_SIZE, epochs=5, verbose =1)

if __name__ == '__main__':
    t = TFRecordReader('images.tfrecord')
    data = t.get_dataset_from_tfrecord()
    t.simple_pipeline(data)
    # t.show()
    # t.show_images()
    # images, labels = t.extract_images_and_labels()
