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

from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import logging
import logging.config
import time

import sklearn
import tensorflow as tf
from absl import app
import numpy as np
import matplotlib.pyplot as plt
import os

# from matplotlib import colors
from sklearn.utils.multiclass import unique_labels

from odir_advance_plotting import Plotter
from odir_model_factory import Factory, ModelTypes
from odir_predictions_writer import Prediction

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from sklearn.utils import class_weight, compute_class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.applications.vgg16 import preprocess_input

# from imblearn.over_sampling import SMOTE

import odir


def normalize_vgg16(training, testing):
    training, testing = training / 1.0, testing / 1.0

    # training[:, :, 0] -= 103.939
    # training[:, :, 1] -= 116.779
    # training[:, :, 2] -= 123.68
    # training = training.transpose((1, 0, 2))
    # training = np.expand_dims(training, axis=0)
    #
    # testing[:, :, 0] -= 103.939
    # testing[:, :, 1] -= 116.779
    # testing[:, :, 2] -= 123.68
    # testing = testing.transpose((1, 0, 2))
    # testing = np.expand_dims(testing, axis=0)

    training = training[..., ::-1]
    testing = testing[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    training[..., 0] -= mean[0]
    training[..., 1] -= mean[1]
    training[..., 2] -= mean[2]
    testing[..., 0] -= mean[0]
    testing[..., 1] -= mean[1]
    testing[..., 2] -= mean[2]

    #training = (training - training.mean()) / training.std()
    #testing = (testing - testing.mean()) / testing.std()
    return training, testing

def generator(train_a, labels_a, train_b, labels_b):
    while True:
        for i in range(len(train_a)):
            yield train_a[i].reshape(1, 224, 224, 3), labels_a[i].reshape(1, 8)
        for i in range(len(train_b)):
            yield train_b[i].reshape(1, 224, 224, 3), labels_b[i].reshape(1, 8)

# def generator_validator():
#     (train_b, labels_b), (tests, labels) = odir.load_data(224, 1)
#     for i in range(len(tests)):
#         yield tests[i], labels[i]

def main(argv):
    print(tf.version.VERSION)
    image_size = 224
    model_type = "vgg16"
    epochs = 100
    test_run = 'fdc'
    plotter = Plotter()

    (train_a, labels_a), (x_test, y_test) = odir.load_data(224, 1)
    (train_b, labels_b), (x_test, y_test) = odir.load_data(224, 2)
    train_a, x_test = normalize_vgg16(train_a, x_test)
    train_b, x_test = normalize_vgg16(train_b, x_test)
    #(x_train, y_train), (x_test, y_test) = odir.load_data(image_size, 1)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD',
                   'Hypertension', 'Myopia', 'Others']

    # x_train, x_test = normalize_vgg16(x_train, x_test)

    defined_metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]

    factory = Factory((image_size, image_size, 3), defined_metrics)
    model = factory.compile(ModelTypes.vgg16)

    print("Training")

    # 1st batch
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

    # model.fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None,
    #                     validation_steps=None, validation_freq=1, class_weight=None, max_queue_size=10,
    #                     workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

    # history = model.fit_generator(generator=generator(train_a, labels_a, train_b, labels_b), steps_per_epoch=len(train_a),
    #                               epochs=epochs, verbose=1, callbacks=[callback], validation_data=(x_test, y_test), shuffle=True )

    history = model.fit(train_a, labels_a, epochs=epochs, batch_size=32, verbose=1, shuffle=True, validation_data=(x_test, y_test),
                         callbacks=[callback])

    # # 2nd batch
    #(x_train, y_train), (x_test, y_test) = odir.load_data(image_size, 2)
    # x_train, x_test = normalize_vgg16(x_train, x_test)
    # history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=1, shuffle=True, validation_data=(x_test, y_test),
    #                     callbacks=[callback])


    # prepare the image for the VGG model
    #image = preprocess_input(image)

    print("plotting")
    plotter.plot_metrics(history, test_run, 2)
    print("saving")
    model.save('model' + model_type + str(epochs) + '.h5')

    # Hide meanwhile for now
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #
    # #plt.ylim([0.5, 1]) --no
    plt.legend(loc='lower right')
    plt.savefig('image_run2' + test_run + '.png')
    plt.show()  # block=False

    baseline_results = model.evaluate(x_test, y_test, batch_size=32, verbose=2)  # test_loss, test_acc
    # print(test_acc)

    test_predictions_baseline = model.predict(x_test, batch_size=32)
    #train_predictions_baseline = model.predict(train_a, batch_size=32)

    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
    print()

    def plot_confusion_matrix(y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
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
        cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))  # >= 0.5
        # Only use the labels that appear in the data
        # classes = classes[unique_labels(y_true, y_pred)]
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

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test, test_predictions_baseline, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(y_test, test_predictions_baseline, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

    def plot_cm(labels2, predictions, p=0.5):
        cm = confusion_matrix(labels2.argmax(axis=1), predictions.argmax(axis=1))  # >= 0.5
        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(cm, annot=True, fmt="d")
        # plt.title('Confusion matrix @{:.2f}'.format(p))
        ax.set_ylim(8.0, -1.0)
        plt.title('Confusion matrix')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.savefig('image_run3' + test_run + '.png')
        plt.show()  # block=False
        plt.close()
        # print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
        # print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
        # print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
        # print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
        # print('Total Fraudulent Transactions: ', np.sum(cm[1]))

    plot_cm(y_test, test_predictions_baseline)

    def plot_roc(name2, labels2, predictions, **kwargs):
        fp, tp, _ = sklearn.metrics.roc_curve(labels2, predictions)

        plt.plot(100 * fp, 100 * tp, label=name2, linewidth=2, **kwargs)
        plt.xlabel('False positives [%]')
        plt.ylabel('True positives [%]')
        plt.xlim([-0.5, 20])
        plt.ylim([80, 100.5])
        plt.grid(True)
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.legend(loc='lower right')
        plt.savefig(name2 + 'image_run4' + test_run + '.png')
        plt.show()  # block=False
        plt.close()

    # plot_roc("Train Baseline", x_test, train_predictions_baseline, color='green')
    # plot_roc("Test Baseline", y_test, test_predictions_baseline, color='green', linestyle='--')

    # return

    # print(predictions[0])
    # print(np.argmax(predictions[0]))
    # print(y_test[0])

    def odir_metrics(gt_data, pr_data):
        th = 0.5
        gt = gt_data.flatten()
        pr = pr_data.flatten()
        kappa = metrics.cohen_kappa_score(gt, pr > th)
        f1 = metrics.f1_score(gt, pr > th, average='micro')
        auc = metrics.roc_auc_score(gt, pr)
        final_score = (kappa + f1 + auc) / 3.0
        return kappa, f1, auc, final_score

    def import_data(filepath):
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            pr_data = [[int(row[0])] + list(map(float, row[1:])) for row in reader]
        pr_data = np.array(pr_data)
        return pr_data

    prediction_writer = Prediction(test_predictions_baseline, 400)
    prediction_writer.save()
    prediction_writer.save_all(y_test)

    gt_data = import_data('odir_ground_truth.csv')
    pr_data = import_data('odir_predictions.csv')
    kappa, f1, auc, final_score = odir_metrics(gt_data[:, 1:], pr_data[:, 1:])
    print("Kappa score:", kappa)
    print("F-1 score:", f1)
    print("AUC value:", auc)
    print("Final Score:", final_score)

    ##Additional test against the training dataset
    # test_loss, test_acc = model.evaluate(x_train, y_train, verbose=2)
    # print(test_acc)
    #
    # predictions = model.predict(x_train)
    # prediction_writer = Prediction(predictions, 400)
    # prediction_writer.save()
    # prediction_writer.save_all(y_train)
    #
    # gt_data = import_data('odir_ground_truth.csv')
    # pr_data = import_data('odir_predictions.csv')
    # kappa, f1, auc, final_score = odir_metrics(gt_data[:, 1:], pr_data[:, 1:])
    # print("kappa score:", kappa, " f-1 score:", f1, " AUC vlaue:", auc, " Final Score:", final_score)

    def plot_image(i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array, true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        # if predicted_label == true_label:
        #    color = 'blue'
        # else:
        #    color = 'red'

        # plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
        #                                      100 * np.max(predictions_array),
        #                                      class_names[true_label]),
        #            color=color)

    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array, true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    # TODO for later
    # num_rows = 5
    # num_cols = 3
    # num_images = num_rows * num_cols
    # plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    # for i in range(num_images):
    #     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    #     plot_image(i, predictions[i], y_test[i], x_test)
    #     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    #     plot_value_array(i, predictions[i], y_test[i])
    # plt.tight_layout()
    # plt.show()
    # TODO for later
    # x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    # x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
    return


if __name__ == '__main__':
    # create logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('odir')
    app.run(main)
