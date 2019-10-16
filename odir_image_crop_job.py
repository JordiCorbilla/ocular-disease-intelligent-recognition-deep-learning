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
import logging
import logging.config
from os import listdir
from os.path import isfile, join

from odir_image_crop import ImageCrop


# Note that this will alter the current training image set folder

def process_all_images():
    files = [f for f in listdir(source_folder) if isfile(join(source_folder, f))]
    for file in files:
        logger.debug('Processing image: ' + file)
        ImageCrop(source_folder, destination_folder, file).remove_black_pixels()


if __name__ == '__main__':
    source_folder = r'C:\temp\ODIR-5K_Training_Dataset'
    destination_folder = r'C:\temp\ODIR-5K_Training_Dataset_cropped'
    # create logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('odir')
    process_all_images()
