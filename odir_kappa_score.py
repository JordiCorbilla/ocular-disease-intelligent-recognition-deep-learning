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
import csv
import os

import numpy as np
from sklearn import metrics


class FinalScore:
    def __init__(self, new_folder):
        self.new_folder = new_folder


    def odir_metrics(self, gt_data, pr_data):
        th = 0.5
        gt = gt_data.flatten()
        pr = pr_data.flatten()
        kappa = metrics.cohen_kappa_score(gt, pr > th)
        f1 = metrics.f1_score(gt, pr > th, average='micro')
        auc = metrics.roc_auc_score(gt, pr)
        final_score = (kappa + f1 + auc) / 3.0
        return kappa, f1, auc, final_score

    def import_data(self, filepath):
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            pr_data = [[int(row[0])] + list(map(float, row[1:])) for row in reader]
        pr_data = np.array(pr_data)
        return pr_data

    def output(self):
        gt_data = self.import_data(os.path.join(self.new_folder, 'odir_ground_truth.csv'))
        pr_data = self.import_data(os.path.join(self.new_folder, 'odir_predictions.csv'))
        kappa, f1, auc, final_score = self.odir_metrics(gt_data[:, 1:], pr_data[:, 1:])
        print("Kappa score:", kappa)
        print("F-1 score:", f1)
        print("AUC value:", auc)
        print("Final Score:", final_score)
