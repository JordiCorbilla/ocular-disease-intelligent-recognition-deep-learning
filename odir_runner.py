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

from odir_patients_to_tfrecord import GenerateTFRecord
from odir_training_data_parser import DataParser


def main(argv):
    # Path to the annotation data
    file = r'dataset\ODIR-5K_Training_Annotations(Updated)_V2.xlsx'
    # Produce all the patient information and store it in an dictionary
    parser = DataParser(file, 'Sheet1')
    patients = parser.generate_patients()
    # Additional quality check, can be commented out
    parser.check_data_quality()
    # Transform patients into TFRecord
    images_path = r'C:\temp\ODIR-5K_Training_Dataset'
    generator = GenerateTFRecord(images_path)
    generator.patients_to_tfrecord(patients, 'images.tfrecord')


if __name__ == '__main__':
    app.run(main)
