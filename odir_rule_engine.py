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


class RuleEngine:
    @staticmethod
    def is_not_decisive(keyword):
        """Determine if keyword should not be processed.
        Args:
          keyword: string, keywords.
        Returns:
          boolean indicating if the keyword is blacklisted.
        """
        # The keywords "lens dust", "optic disk photographically invisible", "low image quality"
        # and "image offset" do not play a decisive role in determining patient's labels
        blacklist = {"lens dust", "optic disk photographically invisible",
                     "low image quality", "image offset"}
        return keyword in blacklist

    @staticmethod
    def is_blacklisted_keyword(keyword):
        """Determine if keyword should not be processed.
        Args:
          keyword: string, keywords.
        Returns:
          boolean indicating if the keyword is blacklisted.
        """
        # the set of keywords to avoid during the algorithm processing are the following ones:
        blacklist = {"anterior segment image", "no fundus image"}
        return keyword in blacklist

    @staticmethod
    def is_blacklisted(image_name):
        """Determine if file should not be processed.
        Args:
          image_name: string, path of the image file.
        Returns:
          boolean indicating if the image is blacklisted.
        """
        # The background of the following images is quite different from the rest ones. They are fundus images
        # uploaded from the hospital: We are sure that these images are preprocessed. You can decide by yourself
        # whether or not to train these images in the model
        # https://drive.google.com/file/d/1KteV8Z5fJ0kD9i64bQ4OAgW8bH-889gT/view?usp=sharing
        blacklist = {'2174_right.jpg', '2175_left.jpg', '2176_left.jpg', '2177_left.jpg', '2177_right.jpg',
                     '2178_right.jpg', '2179_left.jpg', '2179_right.jpg', '2180_left.jpg', '2180_right.jpg',
                     '2181_left.jpg', '2181_right.jpg', '2182_left.jpg', '2182_right.jpg', '2957_left.jpg',
                     '2957_right.jpg'}
        return image_name in blacklist

    def process_keywords(self, keywords):
        """Process a set of keywords and return a new vector as ground truth.
        Args:
          keywords: string, keywords.
        Returns:
          disease vector.
        """
        # Split the keywords into a list:
        # moderate non proliferative retinopathy，hypertensive retinopathy
        # [0] = moderate non proliferative retinopathy
        # [1] = hypertensive retinopathy
        # then run these keywords through the algorithm to find an specific match
        # and generate the vector that corresponds to the illnesses found
        # return [0,1,0,0,0,0,0,0]
        listkeywords = [x.strip() for x in keywords.split('，')]
        normal = 0
        diabetes = 0
        glaucoma = 0
        cataract = 0
        amd = 0
        hypertension = 0
        myopia = 0
        others = 0
        empty_vector = 0
        not_decisive = 0

        for keyword in listkeywords:
            if "normal fundus" in keyword:
                normal = 1
            elif "diabetic retinopathy" in keyword or "proliferative retinopathy" in keyword:
                diabetes = 1
            elif "glaucoma" in keyword:
                glaucoma = 1
            elif "cataract" in keyword:
                cataract = 1
            elif "macular degeneration" in keyword:
                amd = 1
            elif "hypertensive retinopathy" in keyword:
                hypertension = 1
            elif "myopi" in keyword:
                myopia = 1
            else:
                if not self.is_blacklisted_keyword(keyword):
                    others = 1
                if self.is_not_decisive(keyword):
                    not_decisive = 1

        # Special case to discard any image as part of the not decisive process
        if not_decisive == 1:
            normal = 0
            diabetes = 0
            glaucoma = 0
            cataract = 0
            amd = 0
            hypertension = 0
            myopia = 0
            others = 0

        if normal == 0 and diabetes == 0 and glaucoma == 0 and cataract == 0 and amd == 0 and hypertension == 0 \
                and myopia == 0 and others == 0:
            empty_vector = 1

        return [normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others], empty_vector
