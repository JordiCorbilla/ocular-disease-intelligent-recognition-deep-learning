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
import numpy as np
import tensorflow as tf
from skimage import exposure


class ImageTreatment:
    def __init__(self, image_size):
        self.image_size = image_size

    def scaling(self, image, scale_vector):
        # Resize to 4-D vector
        image = np.reshape(image, (1, self.image_size, self.image_size, 3))
        boxes = np.zeros((len(scale_vector), 4), dtype=np.float32)
        for index, scale in enumerate(scale_vector):
            x1 = y1 = 0.5 - 0.5 * scale
            x2 = y2 = 0.5 + 0.5 * scale
            boxes[index] = np.array([y1, x1, y2, x2], dtype=np.float32)
        box_ind = np.zeros((len(scale_vector)), dtype=np.int32)
        crop_size = np.array([self.image_size, self.image_size], dtype=np.int32)

        output = tf.image.crop_and_resize(image, boxes, box_ind, crop_size)
        output = np.array(output, dtype=np.uint8)
        return output

    def brightness(self, image, delta):
        output = tf.image.adjust_brightness(image, delta)
        output = np.array(output, dtype=np.uint8)
        return output

    def contrast(self, image, contrast_factor):
        output = tf.image.adjust_contrast(image, contrast_factor)
        output = np.array(output, dtype=np.uint8)
        return output

    def saturation(self, image, saturation_factor):
        output = tf.image.adjust_saturation(image, saturation_factor)
        output = np.array(output, dtype=np.uint8)
        return output

    def hue(self, image, delta):
        output = tf.image.adjust_hue(image, delta)
        output = np.array(output, dtype=np.uint8)
        return output

    def central_crop(self, image, central_fraction):
        output = tf.image.central_crop(image, central_fraction)
        output = np.array(output, dtype=np.uint8)
        return output

    def crop_to_bounding_box(self, image, offset_height, offset_width, target_height, target_width):
        output = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
        output = tf.image.resize(output, (self.image_size, self.image_size))
        output = np.array(output, dtype=np.uint8)
        return output

    def gamma(self, image, gamma):
        output = tf.image.adjust_gamma(image, gamma)
        output = np.array(output, dtype=np.uint8)
        return output

    def rot90(self, image, k):
        output = tf.image.rot90(image, k)
        output = np.array(output, dtype=np.uint8)
        return output

    def rescale_intensity(self, image):
        p2, p98 = np.percentile(image, (2, 98))
        img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
        return img_rescale

    def equalize_histogram(self, image):
        img_eq = exposure.equalize_hist(image)
        return img_eq

    def equalize_adapthist(self, image):
        img_adapted = exposure.equalize_adapthist(image, clip_limit=0.03)
        return img_adapted
