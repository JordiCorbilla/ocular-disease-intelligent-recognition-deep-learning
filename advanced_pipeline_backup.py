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

#from matplotlib import colors
from sklearn.utils.multiclass import unique_labels

from odir_model_factory import Factory, ModelTypes
from odir_predictions_writer import Prediction

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from sklearn.utils import class_weight, compute_class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns

#from imblearn.over_sampling import SMOTE

import odir

def main(argv):
    print(tf.version.VERSION)
    image_size = 224
    model_type = "vgg16"
    epochs = 200
    test_run = 'zC'

    #train, test = tf.keras.datasets.fashion_mnist.load_data()

    (x_train, y_train), (x_test, y_test) = odir.load_data(image_size, 1)
    #weights = tf.gather(1. / class_weights, y_train)

    #classweights2 = compute_class_weight('balanced',
   #                                                   np.unique(y_train),
    #                                                  y_train)

   # print(classweights2)

   # return
    #class_names = ['Undefined', 'Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD',
    #               'Hypertension', 'Myopia', 'Others']

    class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD',
                   'Hypertension', 'Myopia', 'Others']

    #x_train, x_test = x_train / 255.0, x_test / 255.0

    #x_train, y_train = data_augmentation (x_train, y_train, 2000)

    # sm = SMOTE()
    # x_train, y_train = sm.fit_sample(x_train, y_train)

    #fff = tf.convert_to_tensor(VGG_MEAN, dtype=tf.uint8)
    #red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=x_train)
    #x_train = preprocess_input(x_train, 'channels_last', 'caffe')

    x_train, x_test = x_train / 1.0, x_test / 1.0
    # x_train /= 127.5
    # x_train -= 1.
    #
    # x_test /= 127.5
    # x_test -= 1.

    x_train = x_train[..., ::-1]
    x_test = x_test[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    x_train[..., 0] -= mean[0]
    x_train[..., 1] -= mean[1]
    x_train[..., 2] -= mean[2]
    #x_train = (x_train - x_train.mean())
    #x_test = (x_test - x_test.mean())

    #x_test = x_test[..., ::-1]
    #mean = [103.939, 116.779, 123.68]
    x_test[..., 0] -= mean[0]
    x_test[..., 1] -= mean[1]
    x_test[..., 2] -= mean[2]

    #x_test = preprocess_input(x_test, 'channels_last', 'caffe')
    # vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1, 1, 3))
    # x_train[1] = x_train[1] - vgg_mean[0][0][0]
    # x_train[2] = x_train[2] - vgg_mean[0][0][1]
    # x_train[3] = x_train[3] - vgg_mean[0][0][2]
    # x_train = x_train[:, ::-1]  # reverse axis rgb->bgr

    # x_test[1] = x_test[1] - vgg_mean[0][0][0]
    # x_test[2] = x_test[2] - vgg_mean[0][0][1]
    # x_test[3] = x_test[3] - vgg_mean[0][0][2]
    # x_test = x_test[:, ::-1]  # reverse axis rgb->bgr
    # assert red.get_shape().as_list()[1:] == [128, 128, 1]
    # assert green.get_shape().as_list()[1:] == [128, 128, 1]
    # assert blue.get_shape().as_list()[1:] == [128, 128, 1]
    # x_train = tf.concat(axis=3, values=[
    #     blue - VGG_MEAN[0],
    #     green - VGG_MEAN[1],
    #     red - VGG_MEAN[2],
    # ])

    # red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=x_test)
    # assert red.get_shape().as_list()[1:] == [128, 128, 1]
    # assert green.get_shape().as_list()[1:] == [128, 128, 1]
    # assert blue.get_shape().as_list()[1:] == [128, 128, 1]
    # x_test = tf.concat(axis=3, values=[
    #     blue - VGG_MEAN[0],
    #     green - VGG_MEAN[1],
    #     red - VGG_MEAN[2],
    # ])

    x_train = (x_train - x_train.mean()) / x_train.std()
    x_test = (x_test - x_test.mean()) / x_test.std()



    # datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    # datagen.fit(x_train)

    # plt.figure(figsize=(9, 9))
    # for i in range(100):
    #     plt.subplot(10, 10, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(x_train[i]) #, cmap=plt.cm.binary
    # #     plt.xlabel(class_names[y_train[i][0]], fontsize=7, color='black', labelpad=1)
    # #
    # plt.subplots_adjust(bottom=0.04, right=0.94, top=0.95, left=0.06, wspace=0.20, hspace=0.17)
    # plt.show()

    # tf.keras.metrics.TruePositives(name='tp'),
    # tf.keras.metrics.FalsePositives(name='fp'),
    # tf.keras.metrics.TrueNegatives(name='tn'),
    # tf.keras.metrics.FalseNegatives(name='fn'),

    defined_metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]

    factory = Factory((image_size,image_size,3), defined_metrics)
    model = factory.compile(ModelTypes.vgg16)

    def plot_metrics(history):
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
                plt.ylim([0.8, 1])
            else:
                plt.ylim([0, 1])

            plt.legend()

        plt.savefig('image_run1'+test_run+'.png')
        plt.show() #block=False
        plt.close()

    print("Training")

    # evey instance of class 3 as 10 instances of class 0
    # class_weight = { 0:1.,
    #                 1:1.583802025,
    #                 2:8.996805112,
    #                 3:10.24,
    #                 4:10.05714286,
    #                 5:14.66666667,
    #                 6:10.7480916,
    #                 7:2.505338078 }
    #
    class_weight = { 0:1.,
                    1:1.583802025,
                    2:8.996805112,
                    3:10.24,
                    4:10.05714286,
                    5:1.,
                    6:1.,
                    7:2.505338078 }

    #fmnist_train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    #fmnist_train_ds = fmnist_train_ds.shuffle(5000).batch(32)
    #train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
    #history = model.fit(train_ds, epochs=2)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    #process twice the data and see what happens

    history = model.fit(x_train, y_train, epochs=epochs,batch_size=32,verbose=1,shuffle=True,
                        validation_data=(x_test, y_test)) #, class_weight=class_weight

    print("plotting")
    plot_metrics(history)
    print("saving")
    model.save('model'+model_type + str(epochs)+'.h5')

    # Additional print for metrics





    #return

    #Hide meanwhile for now
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #
    # #plt.ylim([0.5, 1]) --no
    plt.legend(loc='lower right')
    plt.savefig('image_run2'+test_run+'.png')
    plt.show() #block=False

    baseline_results = model.evaluate(x_test, y_test, batch_size=32, verbose=2) #test_loss, test_acc
    #print(test_acc)

    test_predictions_baseline  = model.predict(x_test, batch_size=32)
    train_predictions_baseline  = model.predict(x_train, batch_size=32)

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
        cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1)) # >= 0.5
        # Only use the labels that appear in the data
        #classes = classes[unique_labels(y_true, y_pred)]
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
               #xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

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
        cm = confusion_matrix(labels2.argmax(axis=1), predictions.argmax(axis=1)) # >= 0.5
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d")
        #plt.title('Confusion matrix @{:.2f}'.format(p))
        plt.title('Confusion matrix')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.savefig('image_run3'+test_run+'.png')
        plt.show() #block=False
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
        plt.show() #block=False
        plt.close()

    #plot_roc("Train Baseline", x_test, train_predictions_baseline, color='green')
    #plot_roc("Test Baseline", y_test, test_predictions_baseline, color='green', linestyle='--')

    #return

    #print(predictions[0])
    #print(np.argmax(predictions[0]))
    #print(y_test[0])

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
        #if predicted_label == true_label:
        #    color = 'blue'
        #else:
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


    # Add a channels dimension
    # x_train = x_train[..., tf.newaxis]
    # x_test = x_test[..., tf.newaxis]
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = Conv2D(16, 3, activation='relu')  # , input_shape=(28,28,3)
            self.max = MaxPooling2D()
            self.conv2 = Conv2D(32, 3, activation='relu')  # , input_shape=(28,28,3)
            self.conv3 = Conv2D(64, 3, activation='relu')  # , input_shape=(28,28,3)
            self.flatten = Flatten()
            self.d1 = Dense(512, activation='relu')
            self.d2 = Dense(10, activation='softmax')
            self.dropout = Dropout(0.2)

        def call(self, x):
            x = self.conv1(x)
            x = self.max(x)
            x = self.dropout(x)
            x = self.conv2(x)
            x = self.max(x)
            x = self.conv3(x)
            x = self.max(x)
            x = self.dropout(x)
            x = self.flatten(x)
            x = self.d1(x)
            x = self.d2(x)
            return x

    # Create an instance of the model
    model = MyModel()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam()
    tf.keras.backend.set_floatx('float64')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)
        # logger.debug(predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)
        # print(test_accuracy.result().numpy())

    EPOCHS = 5
    # summary_writer = tf.summary.create_file_writer('./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    for epoch in range(EPOCHS):
        start = time.time()
        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Time: {}s'
        end = time.time()

        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100,
                              str(end - start)))
        #
        # for i in test_accuracy.metrics():
        #     print(i)

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


if __name__ == '__main__':
    # create logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('odir')
    app.run(main)
