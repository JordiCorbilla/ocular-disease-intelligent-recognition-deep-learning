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


class Prediction:
    def __init__(self, prediction, num_images_test):
        self.prediction = prediction
        self.num_images_test = num_images_test

    def save(self):
        """Generate a CSV that contains the output of all the classes.
        Args:
          No arguments are required.
        Returns:
          File with the output
        """
        # The process here is to generate a CSV file with the content of the data annotations file
        # and also the total of labels per eye. This will help us later to process the images
        with open('predictions.csv', 'w', newline='') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['ID', 'Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others'])
            count = 0
            for sub in self.prediction:
                normal = sub[0]
                diabetes = sub[1]
                glaucoma = sub[2]
                cataract = sub[3]
                amd = sub[4]
                hypertension = sub[5]
                myopia = sub[6]
                others = sub[7]
                file_writer.writerow([count, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                count = count + 1

    def save_all(self, y_test):
        """Generate a CSV that contains the output of all the classes.
        Args:
          No arguments are required.
        Returns:
          File with the output
        """
        # The process here is to generate a CSV file with the content of the data annotations file
        # and also the total of labels per eye. This will help us later to process the images
        with open('odir_predictions.csv', 'w', newline='') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['ID', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])
            count = 0
            for i in range(self.num_images_test):
                normal = self.prediction[i][0]
                diabetes = self.prediction[i][1]
                glaucoma = self.prediction[i][2]
                cataract = self.prediction[i][3]
                amd = self.prediction[i][4]
                hypertension = self.prediction[i][5]
                myopia = self.prediction[i][6]
                others = self.prediction[i][7]
                file_writer.writerow([count, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                count = count + 1

        with open('odir_ground_truth.csv', 'w', newline='') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['ID', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])
            count = 0
            for i in range(self.num_images_test):
                normal2 = y_test[i][0]
                diabetes2 = y_test[i][1]
                glaucoma2 = y_test[i][2]
                cataract2 = y_test[i][3]
                amd2 = y_test[i][4]
                hypertension2 = y_test[i][5]
                myopia2 = y_test[i][6]
                others2 = y_test[i][7]

                file_writer.writerow([count, normal2, diabetes2, glaucoma2, cataract2, amd2, hypertension2, myopia2, others2])
                count = count + 1