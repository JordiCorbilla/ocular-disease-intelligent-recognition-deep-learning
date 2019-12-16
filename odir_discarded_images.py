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
import pandas as pd
import xlrd as x
import csv

spreadsheet = r"C:\Users\thund\Source\Repos\TFM-ODIR\models\image_classification\DiscardedImages.xlsx"
sheet = pd.read_excel(spreadsheet, sheet_name="Sheet1")
Patients = {}

with open('discarded.csv', 'w', newline='') as csv_file:
    file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow(
        ['ID', 'Fundus', 'Diagnostic', 'Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension',
         'Myopia', 'Others'])
    for i in sheet.index:
        # load the data from the excel sheet
        patient_id = sheet['ID'][i]
        left_fundus = sheet['Left-Fundus'][i]
        right_fundus = sheet['Right-Fundus'][i]
        left_keywords = sheet['Left-Diagnostic Keywords'][i]
        right_keywords = sheet['Right-Diagnostic Keywords'][i]
        normal = sheet['N'][i]
        diabetes = sheet['D'][i]
        glaucoma = sheet['G'][i]
        cataract = sheet['C'][i]
        amd = sheet['A'][i]
        hypertension = sheet['H'][i]
        myopia = sheet['M'][i]
        others = sheet['O'][i]
        left_keywords = left_keywords.replace(",", "|")
        right_keywords = right_keywords.replace(",", "|")
        left_keywords = left_keywords.replace("，", "|")
        right_keywords = right_keywords.replace("，", "|")
        print(patient_id)
        file_writer.writerow([patient_id, left_fundus, left_keywords, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia,
                              others])
        file_writer.writerow([patient_id, right_fundus, right_keywords, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia,
                              others])

