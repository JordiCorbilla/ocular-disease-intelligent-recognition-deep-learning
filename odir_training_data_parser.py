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
import pandas as pd
import xlrd as x
import csv

from odir_eye_patient import EyePatient
from odir_rule_engine import RuleEngine


# This class reads excels files using Pandas library and applies additional logic
# to find the details for each patient eye


class DataParser:
    def __init__(self, spreadsheet, sheet):
        self.logger = logging.getLogger('odir')
        self.spreadsheet = spreadsheet
        self.sheet = pd.read_excel(self.spreadsheet, sheet_name=sheet)
        self.Patients = {}

    def generate_patients(self):
        """ Generate the Dictionary of patients defined by their eye id: "0_left.jpg".
        Args:
          No arguments are required.
        Returns:
          Dictionary of patient eye information.
        """
        # The process is very simple, we load the data from the excel and we store it
        # into a patient object. we use the rule engine to get the details for each eye and we
        # store them both (generated and non-generated vector):

        engine = RuleEngine()
        for i in self.sheet.index:
            # load the data from the excel sheet
            patient_id = self.sheet['ID'][i]
            age = self.sheet['Patient Age'][i]
            sex = self.sheet['Patient Sex'][i]
            left_fundus = self.sheet['Left-Fundus'][i]
            right_fundus = self.sheet['Right-Fundus'][i]
            left_keywords = self.sheet['Left-Diagnostic Keywords'][i]
            right_keywords = self.sheet['Right-Diagnostic Keywords'][i]
            normal = self.sheet['N'][i]
            diabetes = self.sheet['D'][i]
            glaucoma = self.sheet['G'][i]
            cataract = self.sheet['C'][i]
            amd = self.sheet['A'][i]
            hypertension = self.sheet['H'][i]
            myopia = self.sheet['M'][i]
            others = self.sheet['O'][i]

            # create left eye information
            left_eye = EyePatient(patient_id, age, sex, True, False, left_fundus, left_keywords)
            # create basic vector information with the data from the excel
            left_eye.set_disease(normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others)
            # create generated vector based on the the keyword information
            vector, empty = engine.process_keywords(left_keywords)
            # if the generated vector is not empty and this image does not belong to the black listed list
            # then continue and populate the vector and add it to the dictionary of patients
            if not empty and not engine.is_blacklisted(left_fundus):
                left_eye.set_disease_generated_vector(vector)
                self.Patients[left_eye.ImagePath] = left_eye

            # create right eye information
            right_eye = EyePatient(patient_id, age, sex, False, True, right_fundus, right_keywords)
            # create basic vector information with the data from the excel
            right_eye.set_disease(normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others)
            # create generated vector based on the the keyword information
            vector, empty = engine.process_keywords(right_keywords)
            # if the generated vector is not empty and this image does not belong to the black listed list
            # then continue and populate the vector and add it to the dictionary of patients
            if not empty and not engine.is_blacklisted(right_fundus):
                right_eye.set_disease_generated_vector(vector)
                self.Patients[right_eye.ImagePath] = right_eye

        return self.Patients

    def check_data_quality(self):
        """Determine the data quality of the generated vector and the provided vector.
        Args:
          No arguments are required.
        Returns:
          Displays the list of images that were not processed
        """
        # The idea is to check if the algorithm from the Engine works and we are not doing anything silly.
        # We need to ensure that the ground truth is kept and that no data is modified.
        # As the strategy was to use the diagnostic keywords and parse each record to populate the vector
        # instead of using the original vector that exists in the file

        discarded_images = 0
        for i in self.sheet.index:
            patient_id = self.sheet['ID'][i]
            left_fundus = self.sheet['Left-Fundus'][i]
            right_fundus = self.sheet['Right-Fundus'][i]

            # Check if the left eye exists in the dictionary
            left_eye = None
            if left_fundus in self.Patients:
                left_eye = self.Patients[left_fundus]

            # Check if the Right eye exists in the dictionary
            right_eye = None
            if right_fundus in self.Patients:
                right_eye = self.Patients[right_fundus]

            # if both eyes exists, then
            if left_eye is not None and right_eye is not None:
                stored_vector = left_eye.DiseaseVector
                left_vector = left_eye.DiseaseVectorGenerated
                right_vector = right_eye.DiseaseVectorGenerated

                normal = 0
                # Only consider if both are set to 1
                if left_vector[0] and right_vector[0]:
                    normal = 1

                diabetes = left_vector[1] or right_vector[1]
                glaucoma = left_vector[2] or right_vector[2]
                cataract = left_vector[3] or right_vector[3]
                amd = left_vector[4] or right_vector[4]
                hypertension = left_vector[5] or right_vector[5]
                myopia = left_vector[6] or right_vector[6]
                others = left_vector[7] or right_vector[7]

                combined_vector = [normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others]
                difference, position = self.vector_differences(stored_vector, combined_vector)
                if difference:
                    self.logger.debug("  Difference Id:" + str(patient_id) + ", Index: " + str(position))
                    self.logger.debug("                  [N,D,G,C,A,H,M,O]")
                    self.print_vector("  from source", stored_vector)
                    self.print_vector("  generated  ", combined_vector)

            elif left_eye is None and right_eye is not None:
                self.logger.debug(
                    "Left fundus not found: [" + left_fundus + "] as it has been discarded, working with *right* "
                                                               "fundus only")
                discarded_images = discarded_images + 1
                stored_vector = right_eye.DiseaseVector
                right_vector = right_eye.DiseaseVectorGenerated

                normal = right_vector[0]
                diabetes = right_vector[1]
                glaucoma = right_vector[2]
                cataract = right_vector[3]
                amd = right_vector[4]
                hypertension = right_vector[5]
                myopia = right_vector[6]
                others = right_vector[7]

                combined_vector = [normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others]
                difference, position = self.vector_differences(stored_vector, combined_vector)
                if difference:
                    self.logger.debug("  Difference Id:" + str(patient_id) + ", Index: " + str(position))
                    self.logger.debug("                  [N,D,G,C,A,H,M,O]")
                    self.print_vector("  from source", stored_vector)
                    self.print_vector("  generated  ", combined_vector)

            elif left_eye is not None and right_eye is None:
                self.logger.debug("Right fundus not found: [" + right_fundus + "] as it has been discarded, working "
                                                                               "with *left* fundus only")
                discarded_images = discarded_images + 1
                stored_vector = left_eye.DiseaseVector
                left_vector = left_eye.DiseaseVectorGenerated

                normal = left_vector[0]
                diabetes = left_vector[1]
                glaucoma = left_vector[2]
                cataract = left_vector[3]
                amd = left_vector[4]
                hypertension = left_vector[5]
                myopia = left_vector[6]
                others = left_vector[7]

                combined_vector = [normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others]
                difference, position = self.vector_differences(stored_vector, combined_vector)
                if difference:
                    self.logger.debug("  Difference Id:" + str(patient_id) + ", Index: " + str(position))
                    self.logger.debug("                  [N,D,G,C,A,H,M,O]")
                    self.print_vector("  from source", stored_vector)
                    self.print_vector("  generated  ", combined_vector)
            else:
                discarded_images = discarded_images + 2
                self.logger.debug(
                    "Left and Right fundus not found: [" + left_fundus + "],[" + right_fundus + "] as they have "
                                                                                                "been discarded")

        self.logger.debug("Total discarded images: " + str(discarded_images))
        self.logger.debug("Total training images: " + str(len(self.Patients)))

    def print_vector(self, title, vector):
        self.logger.debug(
            title + ":    [" + str(vector[0]) + "," + str(vector[1]) + "," + str(vector[2]) + "," + str(vector[3]) +
            "," + str(vector[4]) + "," + str(vector[5]) + "," + str(vector[6]) + "," + str(vector[7]) + "]")

    @staticmethod
    def vector_differences(left_vector, right_vector):
        match = True
        position = 0
        for i in range(len(left_vector)):
            match = match and left_vector[i] == right_vector[i]
            # Just output the first difference found
            if not match and position == 0:
                position = i

        return not match, position

    def generate_ground_truth_csv(self):
        """Generate a CSV that contains the output of all the classes.
        Args:
          No arguments are required.
        Returns:
          File with the output
        """
        # The process here is to generate a CSV file with the content of the data annotations file
        # and also the total of labels per eye. This will help us later to process the images
        with open('odir.csv', 'w', newline='') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['ID', 'Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension',
                                  'Myopia', 'Others', 'Total'])
            for i in self.sheet.index:
                left_fundus = self.sheet['Left-Fundus'][i]
                right_fundus = self.sheet['Right-Fundus'][i]

                if left_fundus in self.Patients:
                    left_eye = self.Patients[left_fundus]
                    left_vector = left_eye.DiseaseVectorGenerated
                    normal = left_vector[0]
                    diabetes = left_vector[1]
                    glaucoma = left_vector[2]
                    cataract = left_vector[3]
                    amd = left_vector[4]
                    hypertension = left_vector[5]
                    myopia = left_vector[6]
                    others = left_vector[7]
                    file_writer.writerow([left_fundus, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia,
                                          others,
                                          normal + diabetes + glaucoma + cataract + amd + hypertension + myopia + others])

                # Check if the Right eye exists in the dictionary
                if right_fundus in self.Patients:
                    right_eye = self.Patients[right_fundus]
                    right_vector = right_eye.DiseaseVectorGenerated
                    normal = right_vector[0]
                    diabetes = right_vector[1]
                    glaucoma = right_vector[2]
                    cataract = right_vector[3]
                    amd = right_vector[4]
                    hypertension = right_vector[5]
                    myopia = right_vector[6]
                    others = right_vector[7]
                    file_writer.writerow([right_fundus, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia,
                                          others,
                                          normal + diabetes + glaucoma + cataract + amd + hypertension + myopia + others])

    def generate_ground_truth_class_csv(self):
        """Generate a CSV that contains the output of all the classes once.
        Args:
          No arguments are required.
        Returns:
          File with the output
        """
        # The ground truth generation is different here and we will discard any images that are multi-labelled
        with open('odir_classes.csv', 'w', newline='') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['ID', 'Class'])
            for i in self.sheet.index:
                left_fundus = self.sheet['Left-Fundus'][i]
                right_fundus = self.sheet['Right-Fundus'][i]

                if left_fundus in self.Patients:
                    left_eye = self.Patients[left_fundus]
                    left_vector = left_eye.DiseaseVectorGenerated
                    normal = left_vector[0]
                    diabetes = left_vector[1]
                    glaucoma = left_vector[2]
                    cataract = left_vector[3]
                    amd = left_vector[4]
                    hypertension = left_vector[5]
                    myopia = left_vector[6]
                    others = left_vector[7]
                    # Only save the image that is labelled once. the rest will be discarded
                    if normal + diabetes + glaucoma + cataract + amd + hypertension + myopia + others == 1:
                        if normal == 1:
                            file_writer.writerow([left_fundus, 1])
                        if diabetes == 1:
                            file_writer.writerow([left_fundus, 2])
                        if glaucoma == 1:
                            file_writer.writerow([left_fundus, 3])
                        if cataract == 1:
                            file_writer.writerow([left_fundus, 4])
                        if amd == 1:
                            file_writer.writerow([left_fundus, 5])
                        if hypertension == 1:
                            file_writer.writerow([left_fundus, 6])
                        if myopia == 1:
                            file_writer.writerow([left_fundus, 7])
                        if others == 1:
                            file_writer.writerow([left_fundus, 8])

                # Check if the Right eye exists in the dictionary
                if right_fundus in self.Patients:
                    right_eye = self.Patients[right_fundus]
                    right_vector = right_eye.DiseaseVectorGenerated
                    normal = right_vector[0]
                    diabetes = right_vector[1]
                    glaucoma = right_vector[2]
                    cataract = right_vector[3]
                    amd = right_vector[4]
                    hypertension = right_vector[5]
                    myopia = right_vector[6]
                    others = right_vector[7]
                    # Only save the image that is labelled once. the rest will be discarded
                    if normal + diabetes + glaucoma + cataract + amd + hypertension + myopia + others == 1:
                        if normal == 1:
                            file_writer.writerow([right_fundus, 1])
                        if diabetes == 1:
                            file_writer.writerow([right_fundus, 2])
                        if glaucoma == 1:
                            file_writer.writerow([right_fundus, 3])
                        if cataract == 1:
                            file_writer.writerow([right_fundus, 4])
                        if amd == 1:
                            file_writer.writerow([right_fundus, 5])
                        if hypertension == 1:
                            file_writer.writerow([right_fundus, 6])
                        if myopia == 1:
                            file_writer.writerow([right_fundus, 7])
                        if others == 1:
                            file_writer.writerow([right_fundus, 8])