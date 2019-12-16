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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging.config
import os

import numpy as np
from absl import app
import cv2
from odir_image_treatment import ImageTreatment
import matplotlib.pyplot as plt

def main(argv):
    treatment = ImageTreatment(image_size)
    file = '2_right.jpg'
    file_path = r'C:\temp\ODIR-5K_Training_Dataset_treated_' + str(image_size)
    saving_path = r'C:\temp\ODIR-5K_Training_Dataset_augmented_' + str(image_size)
    file_id = file.replace('.jpg', '')

    #Get the image in the correct format
    eye_image = os.path.join(file_path, file)
    image = cv2.imread(eye_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image

    ## Generate brightness images
    bright = treatment.brightness(image, 0.1)
    plt.subplots(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title('Base Image')
    plt.subplot(2, 2, 2)
    plt.imshow(bright)
    plt.title('Delta = 0.1')
    plt.show()
    plt.close()
    bright = cv2.cvtColor(bright, cv2.COLOR_BGR2RGB)
    status = cv2.imwrite(os.path.join(saving_path, file_id + '_a.jpg'), bright)
    print("Image written to file-system : ", status)

    ## Generate brightness images
    contrast = treatment.contrast(image, 2)
    plt.subplots(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title('Base Image')
    plt.subplot(2, 2, 2)
    plt.imshow(contrast)
    plt.title('Contrast Factor = 2')
    plt.show()
    plt.close()
    contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2RGB)
    status = cv2.imwrite(os.path.join(saving_path, file_id + '_b.jpg'), contrast)
    print("Image written to file-system : ", status)

    ## Generate brightness images
    saturation = treatment.saturation(image, 0.5)
    plt.subplots(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title('Base Image')
    plt.subplot(2, 2, 2)
    plt.imshow(saturation)
    plt.title('Saturation Factor = 2')
    plt.show()
    plt.close()
    saturation = cv2.cvtColor(saturation, cv2.COLOR_BGR2RGB)
    status = cv2.imwrite(os.path.join(saving_path, file_id + '_c.jpg'), saturation)
    print("Image written to file-system : ", status)

    ## Generate scaling images
    vector = [0.90, 0.80, 0.70, 0.50]
    newImages = treatment.scaling(image, vector)

    plt.subplots(figsize = (10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title('Base Image')
    plt.subplot(2, 2, 2)
    plt.imshow(newImages[0])
    plt.title('Scale = 0.90')
    plt.subplot(2, 2, 3)
    plt.imshow(newImages[1])
    plt.title('Scale = 0.80')
    plt.subplot(2, 2, 4)
    plt.imshow(newImages[2])
    plt.title('Scale = 0.70')
    plt.show()
    plt.close()
    for i in range(len(vector)):
        saving_image = cv2.cvtColor(newImages[i], cv2.COLOR_BGR2RGB)
        status = cv2.imwrite(os.path.join(saving_path, file_id + '_d'+str(i)+'.jpg'), saving_image)
        print("Image written to file-system : ", status)

    intensity = treatment.rescale_intensity(original_image)
    plt.subplots(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title('Base Image')
    plt.subplot(2, 2, 2)
    plt.imshow(intensity)
    plt.title('Rescale Intensity = 2-98%')
    plt.show()
    plt.close()
    intensity = cv2.cvtColor(intensity, cv2.COLOR_BGR2RGB)
    status = cv2.imwrite(os.path.join(saving_path, file_id + '_e.jpg'), intensity)
    print("Image written to file-system : ", status)

    gamma = treatment.gamma(original_image, 0.5)
    plt.subplots(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title('Base Image')
    plt.subplot(2, 2, 2)
    plt.imshow(gamma)
    plt.title('Gamma = 0.2')
    plt.show()
    plt.close()
    gamma = cv2.cvtColor(gamma, cv2.COLOR_BGR2RGB)
    status = cv2.imwrite(os.path.join(saving_path, file_id + '_f.jpg'), gamma)
    print("Image written to file-system : ", status)

    hue = treatment.hue(original_image, 0.2)
    plt.subplots(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title('Base Image')
    plt.subplot(2, 2, 2)
    plt.imshow(hue)
    plt.title('Gamma = 0.2')
    plt.show()
    plt.close()
    hue = cv2.cvtColor(hue, cv2.COLOR_BGR2RGB)
    status = cv2.imwrite(os.path.join(saving_path, file_id + '_g.jpg'), hue)
    print("Image written to file-system : ", status)

    central = treatment.crop_to_bounding_box(original_image, 0, 0, 112,112)
    plt.subplots(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title('Base Image')
    plt.subplot(2, 2, 2)
    plt.imshow(central)
    plt.title('Gamma = 0.2')
    plt.show()
    plt.close()
    central = cv2.cvtColor(central, cv2.COLOR_BGR2RGB)
    status = cv2.imwrite(os.path.join(saving_path, file_id + '_h.jpg'), central)
    print("Image written to file-system : ", status)

    central = treatment.crop_to_bounding_box(original_image, 112, 0, 112,112)
    plt.subplots(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title('Base Image')
    plt.subplot(2, 2, 2)
    plt.imshow(central)
    plt.title('Gamma = 0.2')
    plt.show()
    plt.close()
    central = cv2.cvtColor(central, cv2.COLOR_BGR2RGB)
    status = cv2.imwrite(os.path.join(saving_path, file_id + '_i.jpg'), central)
    print("Image written to file-system : ", status)

    central = treatment.crop_to_bounding_box(original_image, 0, 112, 112, 112)
    plt.subplots(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title('Base Image')
    plt.subplot(2, 2, 2)
    plt.imshow(central)
    plt.title('Gamma = 0.2')
    plt.show()
    plt.close()
    central = cv2.cvtColor(central, cv2.COLOR_BGR2RGB)
    status = cv2.imwrite(os.path.join(saving_path, file_id + '_j.jpg'), central)
    print("Image written to file-system : ", status)

    central = treatment.crop_to_bounding_box(original_image, 112, 112, 112,112)
    plt.subplots(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title('Base Image')
    plt.subplot(2, 2, 2)
    plt.imshow(central)
    plt.title('Gamma = 0.2')
    plt.show()
    plt.close()
    central = cv2.cvtColor(central, cv2.COLOR_BGR2RGB)
    status = cv2.imwrite(os.path.join(saving_path, file_id + '_k.jpg'), central)
    print("Image written to file-system : ", status)

    # central = treatment.central_crop(original_image, 0.5)
    # plt.subplots(figsize=(10, 10))
    # plt.subplot(2, 2, 1)
    # plt.imshow(original_image)
    # plt.title('Base Image')
    # plt.subplot(2, 2, 2)
    # plt.imshow(central)
    # plt.title('Gamma = 0.2')
    # plt.show()
    # plt.close()
    # central = cv2.cvtColor(central, cv2.COLOR_BGR2RGB)
    # status = cv2.imwrite(os.path.join(saving_path, file_id + '_h.jpg'), central)
    # print("Image written to file-system : ", status)

    # hist = treatment.equalize_histogram(original_image)
    # plt.subplots(figsize=(10, 10))
    # plt.subplot(2, 2, 1)
    # plt.imshow(original_image)
    # plt.title('Base Image')
    # plt.subplot(2, 2, 2)
    # plt.imshow(hist)
    # plt.title('Equialize Histogram')
    # plt.show()
    # plt.close()
    # #hist = cv2.cvtColor(hist, cv2.COLOR_BGR2RGB)
    # status = cv2.imwrite(os.path.join(saving_path, file_id + '_e.jpg'), hist)
    # print("Image written to file-system : ", status)
    #
    # equalize = treatment.equalize_adapthist(original_image)
    # plt.subplots(figsize=(10, 10))
    # plt.subplot(2, 2, 1)
    # plt.imshow(original_image)
    # plt.title('Base Image')
    # plt.subplot(2, 2, 2)
    # plt.imshow(equalize)
    # plt.title('equalize adapt hist - 0.03')
    # plt.show()
    # plt.close()
    # equalize = cv2.cvtColor(equalize, cv2.COLOR_BGR2RGB)
    # status = cv2.imwrite(os.path.join(saving_path, file_id + '_f.jpg'), equalize)
    # print("Image written to file-system : ", status)


if __name__ == '__main__':
    # create logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('odir')
    image_size = 224
    app.run(main)
