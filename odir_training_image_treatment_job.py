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
import logging.config
from os import listdir
from os.path import isfile, join
from odir_image_resizer import ImageResizer

# This default job to 224px images, will shrink the dataset from 1,439,776,768 bytes
# to 116,813,824 bytes 91.8% size reduction


def process_all_images():
    files = [f for f in listdir(source_folder) if isfile(join(source_folder, f))]
    for file in files:
        logger.debug('Processing image: ' + file)
        ImageResizer(image_width, quality, source_folder, destination_folder, file, keep_aspect_ratio).run()


if __name__ == '__main__':
    # Set the base width of the image to 200 pixels
    image_width = 224
    keep_aspect_ratio = False
    # set the quality of the resultant jpeg to 100%
    quality = 100
    source_folder = r'C:\temp\ODIR-5K_Training_Dataset_cropped'
    destination_folder = r'C:\temp\ODIR-5K_Training_Dataset_treated' + '_' + str(image_width)
    # create logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('odir')
    process_all_images()
