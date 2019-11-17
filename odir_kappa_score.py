import csv
import numpy as np
from sklearn import metrics

def import_data(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        pr_data = [[int(row[0])] + list(map(float, row[1:])) for row in reader]
    pr_data = np.array(pr_data)
    return pr_data

def odir_metrics(gt_data, pr_data):
    th = 0.5
    gt = gt_data.flatten()
    pr = pr_data.flatten()
    kappa = metrics.cohen_kappa_score(gt, pr > th)
    f1 = metrics.f1_score(gt, pr > th, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa + f1 + auc) / 3.0
    return kappa, f1, auc, final_score

gt_data = import_data('odir_ground_truth.csv')
pr_data = import_data('odir_predictions.csv')
kappa, f1, auc, final_score = odir_metrics(gt_data[:, 1:], pr_data[:, 1:])
print("kappa score:", kappa, " f-1 score:", f1, " AUC vlaue:", auc, " Final Score:", final_score)