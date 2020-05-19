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
    def __init__(self, training_path, testing_path, csv_path, csv_testing_path, augmented_path, csv_augmented_file):
        self.training_path = training_path
        self.testing_path = testing_path
        self.csv_path = csv_path
        self.csv_testing_path = csv_testing_path
        self.logger = logging.getLogger('odir')
        self.total_records_training = 0
        self.total_records_testing = 0
        self.csv_augmented_path = csv_augmented_file
        self.augmented_path = augmented_path

    def npy_training_files(self, file_name_training, file_name_training_labels):
        training = []
        training_labels = []

        self.logger.debug("Opening CSV file")
        with open(self.csv_path) as csvDataFile:
            csv_reader = csv.reader(csvDataFile)
            self.total_records_training = 0
            for row in csv_reader:
                column_id = row[0]
                normal = row[1]
                diabetes = row[2]
                glaucoma = row[3]
                cataract = row[4]
                amd = row[5]
                hypertension = row[6]
                myopia = row[7]
                others = row[8]
                # just discard the first row
                if column_id != "ID":
                    self.logger.debug("Processing image: " + column_id)
                    # load first the image from the folder
                    eye_image = os.path.join(self.training_path, column_id)
                    image = cv2.imread(eye_image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    training.append(image)
                    training_labels.append([normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                    self.total_records_training = self.total_records_training + 1

        training = np.array(training, dtype='uint8')
        training_labels = np.array(training_labels, dtype='uint8')
        # convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
        # for example (6069 * 28 * 28 * 3)-> (6069 x 2352) (14,274,288)
        training = np.reshape(training, [training.shape[0], training.shape[1], training.shape[2], training.shape[3]])

        # save numpy array as .npy formats
        np.save(file_name_training, training)
        self.logger.debug("Saving NPY File: " + file_name_training)
        np.save(file_name_training_labels, training_labels)
        self.logger.debug("Saving NPY File: " + file_name_training_labels)
        self.logger.debug("Closing CSV file")

    def npy_testing_files(self, file_name_testing, file_name_testing_labels):
        testing = []
        testing_labels = []

        self.logger.debug("Opening CSV file")
        with open(self.csv_testing_path) as csvDataFile:
            csv_reader = csv.reader(csvDataFile)
            self.total_records_testing = 0
            for row in csv_reader:
                column_id = row[0]
                normal = row[1]
                diabetes = row[2]
                glaucoma = row[3]
                cataract = row[4]
                amd = row[5]
                hypertension = row[6]
                myopia = row[7]
                others = row[8]
                # just discard the first row
                if column_id != "ID":
                    self.logger.debug("Processing image: " + column_id + "_left.jpg")
                    # load first the image from the folder
                    eye_image = os.path.join(self.testing_path, column_id + "_left.jpg")
                    image = cv2.imread(eye_image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    testing.append(image)
                    testing_labels.append([normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                    self.total_records_testing = self.total_records_testing + 1

                    self.logger.debug("Processing image: " + column_id + "_right.jpg")
                    eye_image = os.path.join(self.testing_path, column_id + "_right.jpg")
                    image = cv2.imread(eye_image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    testing.append(image)
                    testing_labels.append([normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                    self.total_records_testing = self.total_records_testing + 1

        testing = np.array(testing, dtype='uint8')
        training_labels = np.array(testing_labels, dtype='uint8')
        # convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
        # for example (6069 * 28 * 28 * 3)-> (6069 x 2352) (14,274,288)
        testing = np.reshape(testing, [testing.shape[0], testing.shape[1], testing.shape[2], testing.shape[3]])

        # save numpy array as .npy formats
        np.save(file_name_testing, testing)
        self.logger.debug("Saving NPY File: " + file_name_testing)
        np.save(file_name_testing_labels, training_labels)
        self.logger.debug("Saving NPY File: " + file_name_testing_labels)
        self.logger.debug("Closing CSV file")

    def npy_training_files_split(self, split_number, file_name_training, file_name_training_labels, file_name_testing,
                                 file_name_testing_labels):
        training = []
        training_labels = []
        testing = []
        testing_labels = []

        self.logger.debug("Opening CSV file")
        count = 0
        with open(self.csv_path) as csvDataFile:
            csv_reader = csv.reader(csvDataFile)
            self.total_records_training = 0
            self.total_records_testing = 0
            for row in csv_reader:
                column_id = row[0]
                label = row[1]
                # just discard the first row
                if column_id != "ID":
                    self.logger.debug("Processing image: " + column_id)
                    # load first the image from the folder
                    eye_image = os.path.join(self.training_path, column_id)
                    image = cv2.imread(eye_image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if count < split_number:
                        testing.append(image)
                        testing_labels.append(label)
                        self.total_records_testing = self.total_records_testing + 1
                    else:
                        training.append(image)
                        training_labels.append(label)
                        self.total_records_training = self.total_records_training + 1
                    count = count + 1

        testing = np.array(testing, dtype='uint8')
        testing_labels = np.array(testing_labels, dtype='uint8')
        testing = np.reshape(testing, [testing.shape[0], testing.shape[1], testing.shape[2], testing.shape[3]])

        # save numpy array as .npy formats
        np.save(file_name_testing, testing)
        np.save(file_name_testing_labels, testing_labels)

        training = np.array(training, dtype='uint8')
        training_labels = np.array(training_labels, dtype='uint8')
        # convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
        # for example (6069 * 28 * 28 * 3)-> (6069 x 2352) (14,274,288)
        training = np.reshape(training, [training.shape[0], training.shape[1], training.shape[2], training.shape[3]])

        # save numpy array as .npy formats
        np.save(file_name_training, training)
        self.logger.debug("Saving NPY File: " + file_name_training)
        np.save(file_name_training_labels, training_labels)
        self.logger.debug("Saving NPY File: " + file_name_training_labels)
        self.logger.debug("Closing CSV file")

    def is_sickness(self, row, sickness):
        switcher = {
            "normal": row[1] == '1' and row[2] == '0' and row[3] == '0' and row[4] == '0' and row[5] == '0' and row[
                6] == '0' and row[7] == '0' and row[8] == '0',
            "diabetes": row[1] == '0' and row[2] == '1' and row[3] == '0' and row[4] == '0' and row[5] == '0' and row[
                6] == '0' and row[7] == '0' and row[8] == '0',
            "glaucoma": row[1] == '0' and row[2] == '0' and row[3] == '1' and row[4] == '0' and row[5] == '0' and row[
                6] == '0' and row[7] == '0' and row[8] == '0',
            "cataract": row[1] == '0' and row[2] == '0' and row[3] == '0' and row[4] == '1' and row[5] == '0' and row[
                6] == '0' and row[7] == '0' and row[8] == '0',
            "amd": row[1] == '0' and row[2] == '0' and row[3] == '0' and row[4] == '0' and row[5] == '1' and row[
                6] == '0' and row[7] == '0' and row[8] == '0',
            "hypertension": row[1] == '0' and row[2] == '0' and row[3] == '0' and row[4] == '0' and row[5] == '0' and
                            row[6] == '1' and row[7] == '0' and row[8] == '0',
            "myopia": row[1] == '0' and row[2] == '0' and row[3] == '0' and row[4] == '0' and row[5] == '0' and row[
                6] == '0' and row[7] == '1' and row[8] == '0',
            "others": row[1] == '0' and row[2] == '0' and row[3] == '0' and row[4] == '0' and row[5] == '0' and row[
                6] == '0' and row[7] == '0' and row[8] == '1'
        }
        return switcher.get(sickness, False)

    def npy_training_files_split_all(self, split_number, file_name_training, file_name_training_labels,
                                     file_name_testing,
                                     file_name_testing_labels, include_augmented):
        split_factor = 10820
        training = []
        training_labels = []
        training_2 = []
        training_labels_2 = []
        testing = []
        testing_labels = []
        images_used = []
        count_images = 0

        class_names = ['normal', 'diabetes', 'glaucoma', 'cataract', 'amd',
                       'hypertension', 'myopia', 'others']

        self.logger.debug("Opening CSV file")

        class_count = {'normal': 0, 'diabetes': 0, 'glaucoma': 0, 'cataract': 0, 'amd': 0, 'hypertension': 0,
                       'myopia': 0, 'others': 0}
        split_pocket = split_number / 8
        with open(self.csv_path) as csvDataFile:
            csv_reader = csv.reader(csvDataFile)
            self.total_records_training = 0
            self.total_records_testing = 0
            for row in csv_reader:
                column_id = row[0]
                normal = row[1]
                diabetes = row[2]
                glaucoma = row[3]
                cataract = row[4]
                amd = row[5]
                hypertension = row[6]
                myopia = row[7]
                others = row[8]
                # just discard the first row
                if column_id != "ID":
                    self.logger.debug("Processing image: " + column_id)
                    # load first the image from the folder
                    eye_image = os.path.join(self.training_path, column_id)
                    image = cv2.imread(eye_image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    found = False
                    for sickness in class_names:
                        if self.is_sickness(row, sickness) and class_count[sickness] < split_pocket:
                            testing.append(image)
                            images_used.append(row[0] + ',' + sickness + ',' + str(class_count[sickness]))
                            testing_labels.append([normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                            self.total_records_testing = self.total_records_testing + 1
                            class_count[sickness] = class_count[sickness] + 1
                            found = True
                            logger.debug('found ' + sickness + ' ' + str(class_count[sickness]))

                    if not found:
                        training.append(image)
                        training_labels.append([normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                        self.total_records_training = self.total_records_training + 1
                        count_images = count_images + 1

        if include_augmented:
            with open(self.csv_augmented_path) as csvDataFile:
                csv_reader = csv.reader(csvDataFile)
                for row in csv_reader:
                    column_id = row[0]
                    normal = row[1]
                    diabetes = row[2]
                    glaucoma = row[3]
                    cataract = row[4]
                    amd = row[5]
                    hypertension = row[6]
                    myopia = row[7]
                    others = row[8]
                    # just discard the first row
                    if column_id != "ID":
                        self.logger.debug("Processing image: " + column_id)
                        # load first the image from the folder
                        eye_image = os.path.join(self.augmented_path, column_id)
                        image = cv2.imread(eye_image)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        if count_images >= split_factor:
                            training_2.append(image)
                            training_labels_2.append([normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                        else:
                            training.append(image)
                            training_labels.append([normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                        self.total_records_training = self.total_records_training + 1
                        count_images = count_images + 1

        testing = np.array(testing, dtype='uint8')
        testing_labels = np.array(testing_labels, dtype='uint8')
        testing = np.reshape(testing, [testing.shape[0], testing.shape[1], testing.shape[2], testing.shape[3]])

        # save numpy array as .npy formats
        np.save(file_name_testing, testing)
        np.save(file_name_testing_labels, testing_labels)

        training = np.array(training, dtype='uint8')
        training_labels = np.array(training_labels, dtype='uint8')
        # convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
        # for example (6069 * 28 * 28 * 3)-> (6069 x 2352) (14,274,288)
        training = np.reshape(training, [training.shape[0], training.shape[1], training.shape[2], training.shape[3]])

        # convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
        # for example (6069 * 28 * 28 * 3)-> (6069 x 2352) (14,274,288)
        if include_augmented:
            training_2 = np.array(training_2, dtype='uint8')
            training_labels_2 = np.array(training_labels_2, dtype='uint8')
            training_2 = np.reshape(training_2, [training_2.shape[0], training_2.shape[1], training_2.shape[2], training_2.shape[3]])

        self.logger.debug(testing.shape)
        self.logger.debug(testing_labels.shape)
        self.logger.debug(training.shape)
        self.logger.debug(training_labels.shape)
        if include_augmented:
            self.logger.debug(training_2.shape)
            self.logger.debug(training_labels_2.shape)

        # save numpy array as .npy formats
        np.save(file_name_training + '_1', training)
        np.save(file_name_training_labels + '_1', training_labels)
        if include_augmented:
            np.save(file_name_training + '_2', training_2)
            np.save(file_name_training_labels + '_2', training_labels_2)
        self.logger.debug("Closing CSV file")
        for sickness in class_names:
            self.logger.debug('found ' + sickness + ' ' + str(class_count[sickness]))
        csv_writer = csv.writer(open("files_used.csv", 'w', newline=''))
        for item in images_used:
            self.logger.debug(item)
            entries = item.split(",")
            csv_writer.writerow(entries)


def main(argv):
    start = time.time()
    image_width = 224
    training_path = r'C:\temp\ODIR-5K_Training_Dataset_treated' + '_' + str(image_width)
    testing_path = r'C:\temp\ODIR-5K_Testing_Images_treated' + '_' + str(image_width)
    augmented_path = r'C:\temp\ODIR-5K_Training_Dataset_augmented' + '_' + str(image_width)
    csv_file = r'ground_truth\odir.csv'
    csv_augmented_file = r'ground_truth\odir_augmented.csv'
    training_file = r'ground_truth\testing_default_value.csv'
    logger.debug('Generating npy files')
    generator = NumpyDataGenerator(training_path, testing_path, csv_file, training_file, augmented_path,
                                   csv_augmented_file)

    # Generate testing file
    generator.npy_testing_files('odir_testing_challenge' + '_' + str(image_width), 'odir_testing_labels_challenge' + '_' + str(image_width))

    # Generate training file
    # generator.npy_training_files('odir_training', 'odir_training_labels')
    # generator.npy_training_files_split(1000, 'odir_training',
    # 'odir_training_labels', 'odir_testing', 'odir_testing_labels')

    # generator.npy_training_files_split_all(400, 'odir_training' + '_' + str(image_width),
    #                                        'odir_training_labels' + '_' + str(image_width),
    #                                        'odir_testing' + '_' + str(image_width),
    #                                        'odir_testing_labels' + '_' + str(image_width),
    #                                        True)
    end = time.time()
    logger.debug('Training Records ' + str(generator.total_records_training))
    logger.debug('Testing Records ' + str(generator.total_records_testing))
    logger.debug('All Done in ' + str(end - start) + ' seconds')


if __name__ == '__main__':
    # create logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('odir')
    app.run(main)
