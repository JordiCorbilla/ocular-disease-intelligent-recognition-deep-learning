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
import os
import tensorflow as tf
from tensorflow.keras.applications import xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.python.keras.optimizers import SGD

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import secrets
import odir
from odir_advance_plotting import Plotter
from odir_kappa_score import FinalScore
from odir_predictions_writer import Prediction
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from sklearn.utils import class_weight
import numpy as np

batch_size = 32
num_classes = 8
epochs = 100
patience = 8

# class_weight = {0: 1.,
#                 1: 1.583802025,
#                 2: 8.996805112,
#                 3: 10.24,
#                 4: 10.05714286,
#                 5: 1.,
#                 6: 1.,
#                 7: 2.505338078}

token = secrets.token_hex(16)
folder = r'C:\Users\thund\Source\Repos\TFM-ODIR\models\image_classification\test_run'

new_folder = os.path.join(folder, token)

if not os.path.exists(new_folder):
    os.makedirs(new_folder)

base_model = xception.Xception

base_model = base_model(weights='imagenet', include_top=False)

# Comment this out if you want to train all layers
#for layer in base_model.layers:
#    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

tf.keras.utils.plot_model(model, to_file=os.path.join(new_folder, 'model_xception.png'), show_shapes=True, show_layer_names=True)

defined_metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]

# Adam Optimizer Example
# model.compile(loss='binary_crossentropy',
#               optimizer=Adam(lr=0.001),
#               metrics=defined_metrics)

# RMSProp Optimizer Example
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=defined_metrics)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
print('Configuration Start -------------------------')
print(sgd.get_config())
print('Configuration End -------------------------')
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=defined_metrics)

(x_train, y_train), (x_test, y_test) = odir.load_data(224)

x_test_drawing = x_test

x_train = xception.preprocess_input(x_train)
x_test = xception.preprocess_input(x_test)

class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']

# plot data input
plotter = Plotter(class_names)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1)

#class_weight = class_weight.compute_class_weight('balanced', np.unique(x_train), x_train)

history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True, #class_weight= class_weight,
                    validation_data=(x_test, y_test), callbacks=[callback])

print("saving weights")
model.save(os.path.join(new_folder, 'model_weights.h5'))

print("plotting metrics")
plotter.plot_metrics(history, os.path.join(new_folder, 'plot1.png'), 2)

print("plotting accuracy")
plotter.plot_accuracy(history, os.path.join(new_folder, 'plot2.png'))

print("display the content of the model")
baseline_results = model.evaluate(x_test, y_test, verbose=2)
for name, value in zip(model.metrics_names, baseline_results):
    print(name, ': ', value)
print()

# test a prediction
test_predictions_baseline = model.predict(x_test)
print("plotting confusion matrix")
plotter.plot_confusion_matrix_generic(y_test, test_predictions_baseline, os.path.join(new_folder, 'plot3.png'), 0)

# save the predictions
prediction_writer = Prediction(test_predictions_baseline, 400, new_folder)
prediction_writer.save()
prediction_writer.save_all(y_test)

# show the final score
score = FinalScore(new_folder)
score.output()

# plot output results
plotter.plot_output(test_predictions_baseline, y_test, x_test_drawing, os.path.join(new_folder, 'plot4.png'))
