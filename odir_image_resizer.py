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
import logging
import PIL
import os
from PIL import Image

# This class allows you to resize and mirror an image of the dataset according to specific rules


class ImageResizer:
    def __init__(self, image_width, quality, source_folder, destination_folder, file_name, keep_aspect_ratio):
        self.logger = logging.getLogger('odir')
        self.image_width = image_width
        self.quality = quality
        self.source_folder = source_folder
        self.destination_folder= destination_folder
        self.file_name = file_name
        self.keep_aspect_ration = keep_aspect_ratio

    def run(self):
        """ Runs the image library using the constructor arguments.
        Args:
          No arguments are required.
        Returns:
          Saves the treated image into a separate folder.
        """
        # We load the original file, we resize it to a smaller width and correspondent height and
        # also mirror the image when we find a right eye image so they are all left eyes

        file = os.path.join(self.source_folder, self.file_name)
        img = Image.open(file)
        if self.keep_aspect_ration:
            # it will have the exact same width-to-height ratio as the original photo
            width_percentage = (self.image_width / float(img.size[0]))
            height_size = int((float(img.size[1]) * float(width_percentage)))
            img = img.resize((self.image_width, height_size), PIL.Image.ANTIALIAS)
        else:
            # This will force the image to be square
            img = img.resize((self.image_width, self.image_width), PIL.Image.ANTIALIAS)
        if "right" in self.file_name:
            self.logger.debug("Right eye image found. Flipping it")
            img.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(self.destination_folder, self.file_name), optimize=True, quality=self.quality)
        else:
            img.save(os.path.join(self.destination_folder, self.file_name), optimize=True, quality=self.quality)
        self.logger.debug("Image saved")
