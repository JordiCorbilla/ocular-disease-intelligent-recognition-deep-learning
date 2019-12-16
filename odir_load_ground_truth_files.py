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


class GroundTruthFiles:
    def __init__(self):
        self.amd = []
        self.cataract = []
        self.diabetes = []
        self.glaucoma = []
        self.hypertension = []
        self.myopia = []
        self.others = []

    def populate_vectors(self, ground_truth_file):
        with open(ground_truth_file) as csvDataFile:
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
                    print("Processing image: " + column_id + "_left.jpg")
                    if diabetes == '1':
                        self.diabetes.append([column_id, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                    if glaucoma == '1':
                        self.glaucoma.append([column_id, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                    if cataract == '1':
                        self.cataract.append([column_id, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                    if amd == '1':
                        self.amd.append([column_id, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                    if hypertension == '1':
                        self.hypertension.append([column_id, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                    if myopia == '1':
                        self.myopia.append([column_id, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                    if others == '1':
                        self.others.append([column_id, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])

