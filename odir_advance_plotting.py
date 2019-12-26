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
from __future__ import absolute_import, division, print_function, unicode_literals

import sys

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib as mpl


class Plotter:
    def __init__(self, class_names):
        self.class_names = class_names

    def plot_metrics(self, history, test_run, index):
        metrics2 = ['loss', 'auc', 'precision', 'recall']
        for n, metric in enumerate(metrics2):
            name = metric.replace("_", " ").capitalize()
            plt.subplot(2, 2, n + 1)
            plt.plot(history.epoch, history.history[metric], color='green', label='Train')
            plt.plot(history.epoch, history.history['val_' + metric], color='green', linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0, 1])
            else:
                plt.ylim([0, 1])

            plt.legend()

        plt.savefig(test_run)
        plt.show()
        plt.close()

    def plot_input_images(self, x_train, y_train):
        plt.figure(figsize=(9, 9))
        for i in range(100):
            plt.subplot(10, 10, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(x_train[i])
            classes = ""
            for j in range(8):
                if y_train[i][j] >= 0.5:
                    classes = classes + self.class_names[j] + "\n"
            plt.xlabel(classes, fontsize=7, color='black', labelpad=1)

        plt.subplots_adjust(bottom=0.04, right=0.95, top=0.94, left=0.06, wspace=0.56, hspace=0.17)
        plt.show()

    def plot_image(self, i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img)
        label_check = [0,0,0,0,0,0,0,0]
        ground = ""
        count_true = 0
        predicted_true = 0

        for index in range(8):
            if true_label[index] >= 0.5:
                count_true = count_true + 1
                ground = ground + self.class_names[index] + "\n"
                label_check[index] = 1
            if predictions_array[index] >= 0.5:
                predicted_true = predicted_true + 1
                label_check[index] = label_check[index] + 1

        all_match = True
        for index in range(8):
            if label_check[index]==1:
                all_match = False

        if count_true == predicted_true and all_match:
            color = 'green'
        else:
            color = 'red'

        first, second, third, i, j, k = self.calculate_3_largest(predictions_array, 8)
        prediction = "{} {:2.0f}% \n".format(self.class_names[i], 100 * first)
        if second >= 0.5:
            prediction = prediction + "{} {:2.0f}% \n".format(self.class_names[j], 100 * second)
        if third >= 0.5:
            prediction = prediction + "{} {:2.0f}% \n".format(self.class_names[k], 100 * third)
        plt.xlabel("Predicted: {} Ground Truth: {}".format(prediction, ground), color=color)

    def calculate_3_largest(self, arr, arr_size):
        if arr_size < 3:
            print(" Invalid Input ")
            return

        third = first = second = -sys.maxsize
        index_1 = 0
        index_2 = 0
        index_3 = 0

        for i in range(0, arr_size):
            if arr[i] > first:
                third = second
                second = first
                first = arr[i]
            elif arr[i] > second:
                third = second
                second = arr[i]
            elif arr[i] > third:
                third = arr[i]

        for i in range(0, arr_size):
            if arr[i] == first:
                index_1 = i
        for i in range(0, arr_size):
            if arr[i] == second and i != index_1:
                index_2 = i
        for i in range(0, arr_size):
            if arr[i] == third and i != index_1 and i!= index_2:
                index_3 = i
        return first, second, third, index_1, index_2, index_3

    def plot_value_array(self, i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        bar_plot = plt.bar(range(8), predictions_array, color="#777777")
        plt.xticks(range(8), ('N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'))
        plt.ylim([0, 1])

        for j in range(8):
            if true_label[j] >= 0.5:
                bar_plot[j].set_color('green')

        for j in range(8):
            if predictions_array[j] >= 0.5 and true_label[j] < 0.5:
                bar_plot[j].set_color('red')

        def bar_label(rects):
            for rect in rects:
                height = rect.get_height()
                value = height * 100
                if value > 1:
                    plt.annotate('{:2.0f}%'.format(value),
                                 xy=(rect.get_x() + rect.get_width() / 2, height),
                                 xytext=(0, 3),  # 3 points vertical offset
                                 textcoords="offset points",
                                 ha='center', va='bottom')

        bar_label(bar_plot)

    def ensure_test_prediction_exists(self, predictions):
        exists = False
        for j in range(8):
            if predictions[j] >= 0.5:
                exists = True
        return exists

    def plot_output(self, test_predictions_baseline, y_test, x_test_drawing):
        mpl.rcParams["font.size"] = 7
        num_rows = 5
        num_cols = 3
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        j = 0
        i = 0
        while j < num_images:
            if self.ensure_test_prediction_exists(test_predictions_baseline[i]):
                plt.subplot(num_rows, 2 * num_cols, 2 * j + 1)
                self.plot_image(i, test_predictions_baseline, y_test, x_test_drawing)
                plt.subplot(num_rows, 2 * num_cols, 2 * j + 2)
                self.plot_value_array(i, test_predictions_baseline, y_test)
                j = j + 1
            i = i + 1
            if i > 400:
                break

        plt.subplots_adjust(bottom=0.08, right=0.95, top=0.94, left=0.05, wspace=0.11, hspace=0.56)
        plt.show()

    def plot_output_single(self, i, test_predictions_baseline, y_test, x_test_drawing):
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        self.plot_image(i, test_predictions_baseline, y_test, x_test_drawing)
        plt.subplot(1, 2, 2)
        self.plot_value_array(i, test_predictions_baseline, y_test)
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
        # Only use the labels that appear in the data
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               # xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        ax.set_ylim(8.0, -1.0)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

    def print_normalized_confusion_matrix(self, y_test, test_predictions_baseline):
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        self.plot_confusion_matrix(y_test, test_predictions_baseline, classes=self.class_names,
                                   title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        self.plot_confusion_matrix(y_test, test_predictions_baseline, classes=self.class_names, normalize=True,
                                   title='Normalized confusion matrix')

        plt.show()

    def plot_confusion_matrix_generic(self, labels2, predictions, test_run, p=0.5):
        cm = confusion_matrix(labels2.argmax(axis=1), predictions.argmax(axis=1))
        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(cm, annot=True, fmt="d")
        ax.set_ylim(8.0, -1.0)
        plt.title('Confusion matrix')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.savefig(test_run)
        plt.show()
        plt.close()
