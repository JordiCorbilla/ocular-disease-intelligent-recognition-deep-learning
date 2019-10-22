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
from absl import app
import logging
import logging.config
import time
import csv
import cv2
import os
import numpy as np
import glob


class NumpyDataGenerator:
    def __init__(self, training_path, testing_path, csv_path):
        self.training_path = training_path
        self.testing_path = testing_path
        self.csv_path = csv_path
        self.logger = logging.getLogger('odir')

    def npy_training_files(self, file_name_training, file_name_training_labels):
        training = []
        training_labels = []

        self.logger.debug("Opening CSV file")
        with open(self.csv_path) as csvDataFile:
            csv_reader = csv.reader(csvDataFile)
            for row in csv_reader:
                id = row[0]
                label = row[1]
                # just discard the first row
                if id != "ID":
                    self.logger.debug("Processing image: " + id)
                    # load first the image from the folder
                    eye_image = os.path.join(self.training_path, id)
                    image = cv2.imread(eye_image)
                    training.append(image)
                    training_labels.append(label)

        training = np.array(training, dtype='float32')
        training_labels = np.array(training_labels, dtype='float64')
        # convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
        # for example (120 * 40 * 40 * 3)-> (120 * 4800)
        training = np.reshape(training, [training.shape[0], training.shape[1] * training.shape[2] * training.shape[3]])

        # save numpy array as .npy formats
        np.save(file_name_training, training)
        np.save(file_name_training_labels, training_labels)

    def npy_testing_files(self, file_name_testing, file_name_testing_labels):
        testing = []
        testing_labels = []

        files = glob.glob(self.testing_path + "/*.jpg")
        for myFile in files:
            self.logger.debug("Processing image: " + myFile)
            image = cv2.imread(myFile)
            testing.append(image)
            testing_labels.append(0)

        testing = np.array(testing, dtype='float32')
        testing_labels = np.array(testing_labels, dtype='float64')
        testing = np.reshape(testing, [testing.shape[0], testing.shape[1] * testing.shape[2] * testing.shape[3]])

        # save numpy array as .npy formats
        np.save(file_name_testing, testing)
        np.save(file_name_testing_labels, testing_labels)


def main(argv):
    start = time.time()
    images_path = r'C:\temp\ODIR-5K_Training_Dataset_treated'
    training_path = r'C:\temp\ODIR-5K_Testing_Images_treated'
    csv_file = 'odir_classes.csv'
    logger.debug('Generating npy files')
    generator = NumpyDataGenerator(images_path, training_path, csv_file)
    generator.npy_training_files('odir_training', 'odir_training_labels')
    generator.npy_testing_files('odir_testing', 'odir_testing_labels')
    end = time.time()
    logger.debug('All Done in ' + str(end - start) + ' seconds')


if __name__ == '__main__':
    # create logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('odir')
    app.run(main)
