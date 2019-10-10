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


class EyePatient:
    """
        Basic Patient Eye class that will hold the information for each eye.
    """

    # Constructor that includes all the fields that are required in the Dataset. DiseaseVector represents the details
    # from the dataset DiseaseVectorGenerated represents the details from auto-generated vector using the engine to
    # parse the keywords for each eye
    def __init__(self, patient_id, age, sex, left, right, path, keywords):
        self.id = patient_id
        self.age = age
        self.sex = sex
        self.isLeftEye = left
        self.isRightEye = right
        self.ImagePath = path
        self.DiagnosticKeywords = keywords
        self.DiseaseVector = []
        self.DiseaseVectorGenerated = []

    # Vector:
    def set_disease(self, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others):
        self.DiseaseVector = [normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others]

    # Vector:
    def set_disease_vector(self, vector):
        self.DiseaseVector = vector

    # Vector generator:
    def set_disease_generated(self, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others):
        self.DiseaseVectorGenerated = [normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others]

    # Vector generator:
    def set_disease_generated_vector(self, vector):
        self.DiseaseVectorGenerated = vector
