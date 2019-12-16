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
import os
import cv2
from odir_image_treatment import ImageTreatment


class DataAugmentationStrategy:
    def __init__(self, image_size, file_name):
        self.base_image = file_name
        self.treatment = ImageTreatment(image_size)
        self.file_path = r'C:\temp\ODIR-5K_Training_Dataset_treated_' + str(image_size)
        self.saving_path = r'C:\temp\ODIR-5K_Training_Dataset_augmented_' + str(image_size)
        self.file_id = file_name.replace('.jpg', '')

    def save_image(self, original_vector, image, sample):
        central = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        file = self.file_id + '_'+str(sample)+'.jpg'
        file_name = os.path.join(self.saving_path, file)
        exists = os.path.isfile(file_name)
        if exists:
            print("duplicate file found: " + file_name)

        status = cv2.imwrite(file_name, central)

        with open(r'ground_truth\odir_augmented.csv', 'a', newline='') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow([file, original_vector[1], original_vector[2], original_vector[3], original_vector[4],
                                   original_vector[5], original_vector[6], original_vector[7], original_vector[8]])

        #print(file_name + " written to file-system : ", status)

    def generate_images(self, number_samples, original_vector, weights):
        eye_image = os.path.join(self.file_path, self.base_image)
        image = cv2.imread(eye_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image
        saved = 0

        # For any repeating elements, just give the other output
        # We are only expecting up to 3 repetitions
        if weights == 20:
            original_image = self.treatment.rot90(original_image, 2)
        if weights == 400:
            original_image = self.treatment.rot90(original_image, 3)
        if weights > 401:
            print(str(self.file_id) + ' samples:' + str(number_samples))
            raise ValueError('this cannot happen')

        # for the sample type 14, just generate 1 image and leave the method
        if number_samples == 14:
            central = self.treatment.rot90(original_image, 1)
            self.save_image(original_vector, central, weights+14)
            saved = saved +1
            return saved

        if number_samples > 0:
            central = self.treatment.crop_to_bounding_box(original_image, 0, 0, 112, 112)
            self.save_image(original_vector, central, weights+0)
            saved = saved + 1

        if number_samples > 1:
            central = self.treatment.crop_to_bounding_box(original_image, 112, 0, 112, 112)
            self.save_image(original_vector, central, weights+1)
            saved = saved + 1

        if number_samples > 2:
            central = self.treatment.crop_to_bounding_box(original_image, 0, 112, 112, 112)
            self.save_image(original_vector, central, weights+2)
            saved = saved + 1

        if number_samples > 3:
            central = self.treatment.crop_to_bounding_box(original_image, 112, 112, 112, 112)
            self.save_image(original_vector, central, weights+3)
            saved = saved + 1

        if number_samples > 4:
            vector = [0.50]
            central = self.treatment.scaling(original_image, vector)
            self.save_image(original_vector, central[0], weights+4)
            saved = saved + 1

        if number_samples > 5:
            vector = [0.70]
            central = self.treatment.scaling(original_image, vector)
            self.save_image(original_vector, central[0], weights+5)
            saved = saved + 1

        if number_samples > 6:
            vector = [0.80]
            central = self.treatment.scaling(original_image, vector)
            self.save_image(original_vector, central[0], weights+6)
            saved = saved + 1

        if number_samples > 7:
            vector = [0.90]
            central = self.treatment.scaling(original_image, vector)
            self.save_image(original_vector, central[0], weights+7)
            saved = saved + 1

        if number_samples > 8:
            central = self.treatment.rescale_intensity(original_image)
            self.save_image(original_vector, central, weights+8)
            saved = saved + 1

        if number_samples > 9:
            central = self.treatment.contrast(original_image, 2)
            self.save_image(original_vector, central, weights+9)
            saved = saved + 1

        if number_samples > 10:
            central = self.treatment.saturation(original_image, 0.5)
            self.save_image(original_vector, central, weights+10)
            saved = saved + 1

        if number_samples > 11:
            central = self.treatment.gamma(original_image, 0.5)
            self.save_image(original_vector, central, weights+11)
            saved = saved + 1

        if number_samples > 12:
            central = self.treatment.hue(original_image, 0.2)
            self.save_image(original_vector, central, weights+12)
            saved = saved + 1

        return saved

