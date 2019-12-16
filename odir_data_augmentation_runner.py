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
import csv
import logging.config
import os
from absl import app

from odir_data_augmentation_strategies import DataAugmentationStrategy
from odir_load_ground_truth_files import GroundTruthFiles


def write_header():
    with open(r'ground_truth\odir_augmented.csv', 'w', newline='') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(['ID', 'Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension',
                              'Myopia', 'Others'])
        return file_writer


def process_files(images, cache, files):
    total = 0
    for strategy in range(len(images)):
        images_to_process = images[strategy][0]
        samples_per_image = images[strategy][1]
        for image_index in range(images_to_process):
            image_vector = files[image_index]
            file_name = image_vector[0]

            # Only check during the first strategy
            if strategy == 0:
                if file_name not in cache:
                    cache[file_name] = 1
                else:
                    cache[file_name] = cache[file_name] * 20

            # print('Processing: ' + file_name)
            augment = DataAugmentationStrategy(image_size, file_name)
            count = augment.generate_images(samples_per_image, image_vector, cache[file_name])
            total = total + count
    return total


def main(argv):
    # load the ground truth file
    files = GroundTruthFiles()
    files.populate_vectors(csv_path)

    print('files record count order by size ASC')
    print('hypertension ' + str(len(files.hypertension)))
    print('myopia ' + str(len(files.myopia)))
    print('cataract ' + str(len(files.cataract)))
    print('amd ' + str(len(files.amd)))
    print('glaucoma ' + str(len(files.glaucoma)))
    print('others ' + str(len(files.others)))
    print('diabetes ' + str(len(files.diabetes)))

    images_hypertension = [[len(files.hypertension), 13], [128, 14]]
    images_myopia = [[len(files.myopia), 9], [196, 14]]
    images_cataract = [[len(files.cataract), 9], [66, 14]]
    images_amd = [[len(files.amd), 9], [16, 14]]
    images_glaucoma = [[len(files.glaucoma), 7], [312, 14]]
    images_others = [[len(files.others), 1], [568, 14]]
    images_diabetes = [[1038, 1]]

    # Delete previous file
    exists = os.path.isfile(r'ground_truth\odir_augmented.csv')
    if exists:
        os.remove(r'ground_truth\odir_augmented.csv')

    write_header()

    images_processed = {}

    total_hypertension = process_files(images_hypertension, images_processed, files.hypertension)
    total_myopia = process_files(images_myopia, images_processed, files.myopia)
    total_cataract = process_files(images_cataract, images_processed, files.cataract)
    total_amd = process_files(images_amd, images_processed, files.amd)
    total_glaucoma = process_files(images_glaucoma, images_processed, files.glaucoma)
    total_others = process_files(images_others, images_processed, files.others)
    total_diabetes = process_files(images_diabetes, images_processed, files.diabetes)

    print("total generated hypertension: " + str(total_hypertension))
    print("total generated myopia: " + str(total_myopia))
    print("total generated cataract: " + str(total_cataract))
    print("total generated amd: " + str(total_amd))
    print("total generated glaucoma: " + str(total_glaucoma))
    print("total generated others: " + str(total_others))
    print("total generated diabetes: " + str(total_diabetes))

if __name__ == '__main__':
    # create logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('odir')
    image_size = 224
    csv_path = 'ground_truth\odir.csv'
    app.run(main)
