## Configuration

- 'Optimizer': 'SGD'
- 'learning_rate': 0.01
- 'decay': 1e-06
- 'momentum': 0.9
- 'nesterov': True
- 'Dropout': False

## Execution Output
```
C:\Users\thund\Source\Repos\TFM-ODIR\models\image_classification>python odir_xception_training_basic.py
2020-06-12 08:34:04.204639: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_100.dll
2020-06-12 08:34:38.284160: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
2020-06-12 08:34:39.433341: E tensorflow/stream_executor/cuda/cuda_driver.cc:318] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2020-06-12 08:34:39.524593: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: LAPTOP-NB3QKBHM
2020-06-12 08:34:39.537359: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: LAPTOP-NB3QKBHM
2020-06-12 08:34:39.547964: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, None, None,  0
__________________________________________________________________________________________________
block1_conv1 (Conv2D)           (None, None, None, 3 864         input_1[0][0]
__________________________________________________________________________________________________
block1_conv1_bn (BatchNormaliza (None, None, None, 3 128         block1_conv1[0][0]
__________________________________________________________________________________________________
block1_conv1_act (Activation)   (None, None, None, 3 0           block1_conv1_bn[0][0]
__________________________________________________________________________________________________
block1_conv2 (Conv2D)           (None, None, None, 6 18432       block1_conv1_act[0][0]
__________________________________________________________________________________________________
block1_conv2_bn (BatchNormaliza (None, None, None, 6 256         block1_conv2[0][0]
__________________________________________________________________________________________________
block1_conv2_act (Activation)   (None, None, None, 6 0           block1_conv2_bn[0][0]
__________________________________________________________________________________________________
block2_sepconv1 (SeparableConv2 (None, None, None, 1 8768        block1_conv2_act[0][0]
__________________________________________________________________________________________________
block2_sepconv1_bn (BatchNormal (None, None, None, 1 512         block2_sepconv1[0][0]
__________________________________________________________________________________________________
block2_sepconv2_act (Activation (None, None, None, 1 0           block2_sepconv1_bn[0][0]
__________________________________________________________________________________________________
block2_sepconv2 (SeparableConv2 (None, None, None, 1 17536       block2_sepconv2_act[0][0]
__________________________________________________________________________________________________
block2_sepconv2_bn (BatchNormal (None, None, None, 1 512         block2_sepconv2[0][0]
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, None, None, 1 8192        block1_conv2_act[0][0]
__________________________________________________________________________________________________
block2_pool (MaxPooling2D)      (None, None, None, 1 0           block2_sepconv2_bn[0][0]
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, None, None, 1 512         conv2d[0][0]
__________________________________________________________________________________________________
add (Add)                       (None, None, None, 1 0           block2_pool[0][0]
                                                                 batch_normalization[0][0]
__________________________________________________________________________________________________
block3_sepconv1_act (Activation (None, None, None, 1 0           add[0][0]
__________________________________________________________________________________________________
block3_sepconv1 (SeparableConv2 (None, None, None, 2 33920       block3_sepconv1_act[0][0]
__________________________________________________________________________________________________
block3_sepconv1_bn (BatchNormal (None, None, None, 2 1024        block3_sepconv1[0][0]
__________________________________________________________________________________________________
block3_sepconv2_act (Activation (None, None, None, 2 0           block3_sepconv1_bn[0][0]
__________________________________________________________________________________________________
block3_sepconv2 (SeparableConv2 (None, None, None, 2 67840       block3_sepconv2_act[0][0]
__________________________________________________________________________________________________
block3_sepconv2_bn (BatchNormal (None, None, None, 2 1024        block3_sepconv2[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, None, None, 2 32768       add[0][0]
__________________________________________________________________________________________________
block3_pool (MaxPooling2D)      (None, None, None, 2 0           block3_sepconv2_bn[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, None, None, 2 1024        conv2d_1[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, None, None, 2 0           block3_pool[0][0]
                                                                 batch_normalization_1[0][0]
__________________________________________________________________________________________________
block4_sepconv1_act (Activation (None, None, None, 2 0           add_1[0][0]
__________________________________________________________________________________________________
block4_sepconv1 (SeparableConv2 (None, None, None, 7 188672      block4_sepconv1_act[0][0]
__________________________________________________________________________________________________
block4_sepconv1_bn (BatchNormal (None, None, None, 7 2912        block4_sepconv1[0][0]
__________________________________________________________________________________________________
block4_sepconv2_act (Activation (None, None, None, 7 0           block4_sepconv1_bn[0][0]
__________________________________________________________________________________________________
block4_sepconv2 (SeparableConv2 (None, None, None, 7 536536      block4_sepconv2_act[0][0]
__________________________________________________________________________________________________
block4_sepconv2_bn (BatchNormal (None, None, None, 7 2912        block4_sepconv2[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, None, None, 7 186368      add_1[0][0]
__________________________________________________________________________________________________
block4_pool (MaxPooling2D)      (None, None, None, 7 0           block4_sepconv2_bn[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, None, None, 7 2912        conv2d_2[0][0]
__________________________________________________________________________________________________
add_2 (Add)                     (None, None, None, 7 0           block4_pool[0][0]
                                                                 batch_normalization_2[0][0]
__________________________________________________________________________________________________
block5_sepconv1_act (Activation (None, None, None, 7 0           add_2[0][0]
__________________________________________________________________________________________________
block5_sepconv1 (SeparableConv2 (None, None, None, 7 536536      block5_sepconv1_act[0][0]
__________________________________________________________________________________________________
block5_sepconv1_bn (BatchNormal (None, None, None, 7 2912        block5_sepconv1[0][0]
__________________________________________________________________________________________________
block5_sepconv2_act (Activation (None, None, None, 7 0           block5_sepconv1_bn[0][0]
__________________________________________________________________________________________________
block5_sepconv2 (SeparableConv2 (None, None, None, 7 536536      block5_sepconv2_act[0][0]
__________________________________________________________________________________________________
block5_sepconv2_bn (BatchNormal (None, None, None, 7 2912        block5_sepconv2[0][0]
__________________________________________________________________________________________________
block5_sepconv3_act (Activation (None, None, None, 7 0           block5_sepconv2_bn[0][0]
__________________________________________________________________________________________________
block5_sepconv3 (SeparableConv2 (None, None, None, 7 536536      block5_sepconv3_act[0][0]
__________________________________________________________________________________________________
block5_sepconv3_bn (BatchNormal (None, None, None, 7 2912        block5_sepconv3[0][0]
__________________________________________________________________________________________________
add_3 (Add)                     (None, None, None, 7 0           block5_sepconv3_bn[0][0]
                                                                 add_2[0][0]
__________________________________________________________________________________________________
block6_sepconv1_act (Activation (None, None, None, 7 0           add_3[0][0]
__________________________________________________________________________________________________
block6_sepconv1 (SeparableConv2 (None, None, None, 7 536536      block6_sepconv1_act[0][0]
__________________________________________________________________________________________________
block6_sepconv1_bn (BatchNormal (None, None, None, 7 2912        block6_sepconv1[0][0]
__________________________________________________________________________________________________
block6_sepconv2_act (Activation (None, None, None, 7 0           block6_sepconv1_bn[0][0]
__________________________________________________________________________________________________
block6_sepconv2 (SeparableConv2 (None, None, None, 7 536536      block6_sepconv2_act[0][0]
__________________________________________________________________________________________________
block6_sepconv2_bn (BatchNormal (None, None, None, 7 2912        block6_sepconv2[0][0]
__________________________________________________________________________________________________
block6_sepconv3_act (Activation (None, None, None, 7 0           block6_sepconv2_bn[0][0]
__________________________________________________________________________________________________
block6_sepconv3 (SeparableConv2 (None, None, None, 7 536536      block6_sepconv3_act[0][0]
__________________________________________________________________________________________________
block6_sepconv3_bn (BatchNormal (None, None, None, 7 2912        block6_sepconv3[0][0]
__________________________________________________________________________________________________
add_4 (Add)                     (None, None, None, 7 0           block6_sepconv3_bn[0][0]
                                                                 add_3[0][0]
__________________________________________________________________________________________________
block7_sepconv1_act (Activation (None, None, None, 7 0           add_4[0][0]
__________________________________________________________________________________________________
block7_sepconv1 (SeparableConv2 (None, None, None, 7 536536      block7_sepconv1_act[0][0]
__________________________________________________________________________________________________
block7_sepconv1_bn (BatchNormal (None, None, None, 7 2912        block7_sepconv1[0][0]
__________________________________________________________________________________________________
block7_sepconv2_act (Activation (None, None, None, 7 0           block7_sepconv1_bn[0][0]
__________________________________________________________________________________________________
block7_sepconv2 (SeparableConv2 (None, None, None, 7 536536      block7_sepconv2_act[0][0]
__________________________________________________________________________________________________
block7_sepconv2_bn (BatchNormal (None, None, None, 7 2912        block7_sepconv2[0][0]
__________________________________________________________________________________________________
block7_sepconv3_act (Activation (None, None, None, 7 0           block7_sepconv2_bn[0][0]
__________________________________________________________________________________________________
block7_sepconv3 (SeparableConv2 (None, None, None, 7 536536      block7_sepconv3_act[0][0]
__________________________________________________________________________________________________
block7_sepconv3_bn (BatchNormal (None, None, None, 7 2912        block7_sepconv3[0][0]
__________________________________________________________________________________________________
add_5 (Add)                     (None, None, None, 7 0           block7_sepconv3_bn[0][0]
                                                                 add_4[0][0]
__________________________________________________________________________________________________
block8_sepconv1_act (Activation (None, None, None, 7 0           add_5[0][0]
__________________________________________________________________________________________________
block8_sepconv1 (SeparableConv2 (None, None, None, 7 536536      block8_sepconv1_act[0][0]
__________________________________________________________________________________________________
block8_sepconv1_bn (BatchNormal (None, None, None, 7 2912        block8_sepconv1[0][0]
__________________________________________________________________________________________________
block8_sepconv2_act (Activation (None, None, None, 7 0           block8_sepconv1_bn[0][0]
__________________________________________________________________________________________________
block8_sepconv2 (SeparableConv2 (None, None, None, 7 536536      block8_sepconv2_act[0][0]
__________________________________________________________________________________________________
block8_sepconv2_bn (BatchNormal (None, None, None, 7 2912        block8_sepconv2[0][0]
__________________________________________________________________________________________________
block8_sepconv3_act (Activation (None, None, None, 7 0           block8_sepconv2_bn[0][0]
__________________________________________________________________________________________________
block8_sepconv3 (SeparableConv2 (None, None, None, 7 536536      block8_sepconv3_act[0][0]
__________________________________________________________________________________________________
block8_sepconv3_bn (BatchNormal (None, None, None, 7 2912        block8_sepconv3[0][0]
__________________________________________________________________________________________________
add_6 (Add)                     (None, None, None, 7 0           block8_sepconv3_bn[0][0]
                                                                 add_5[0][0]
__________________________________________________________________________________________________
block9_sepconv1_act (Activation (None, None, None, 7 0           add_6[0][0]
__________________________________________________________________________________________________
block9_sepconv1 (SeparableConv2 (None, None, None, 7 536536      block9_sepconv1_act[0][0]
__________________________________________________________________________________________________
block9_sepconv1_bn (BatchNormal (None, None, None, 7 2912        block9_sepconv1[0][0]
__________________________________________________________________________________________________
block9_sepconv2_act (Activation (None, None, None, 7 0           block9_sepconv1_bn[0][0]
__________________________________________________________________________________________________
block9_sepconv2 (SeparableConv2 (None, None, None, 7 536536      block9_sepconv2_act[0][0]
__________________________________________________________________________________________________
block9_sepconv2_bn (BatchNormal (None, None, None, 7 2912        block9_sepconv2[0][0]
__________________________________________________________________________________________________
block9_sepconv3_act (Activation (None, None, None, 7 0           block9_sepconv2_bn[0][0]
__________________________________________________________________________________________________
block9_sepconv3 (SeparableConv2 (None, None, None, 7 536536      block9_sepconv3_act[0][0]
__________________________________________________________________________________________________
block9_sepconv3_bn (BatchNormal (None, None, None, 7 2912        block9_sepconv3[0][0]
__________________________________________________________________________________________________
add_7 (Add)                     (None, None, None, 7 0           block9_sepconv3_bn[0][0]
                                                                 add_6[0][0]
__________________________________________________________________________________________________
block10_sepconv1_act (Activatio (None, None, None, 7 0           add_7[0][0]
__________________________________________________________________________________________________
block10_sepconv1 (SeparableConv (None, None, None, 7 536536      block10_sepconv1_act[0][0]
__________________________________________________________________________________________________
block10_sepconv1_bn (BatchNorma (None, None, None, 7 2912        block10_sepconv1[0][0]
__________________________________________________________________________________________________
block10_sepconv2_act (Activatio (None, None, None, 7 0           block10_sepconv1_bn[0][0]
__________________________________________________________________________________________________
block10_sepconv2 (SeparableConv (None, None, None, 7 536536      block10_sepconv2_act[0][0]
__________________________________________________________________________________________________
block10_sepconv2_bn (BatchNorma (None, None, None, 7 2912        block10_sepconv2[0][0]
__________________________________________________________________________________________________
block10_sepconv3_act (Activatio (None, None, None, 7 0           block10_sepconv2_bn[0][0]
__________________________________________________________________________________________________
block10_sepconv3 (SeparableConv (None, None, None, 7 536536      block10_sepconv3_act[0][0]
__________________________________________________________________________________________________
block10_sepconv3_bn (BatchNorma (None, None, None, 7 2912        block10_sepconv3[0][0]
__________________________________________________________________________________________________
add_8 (Add)                     (None, None, None, 7 0           block10_sepconv3_bn[0][0]
                                                                 add_7[0][0]
__________________________________________________________________________________________________
block11_sepconv1_act (Activatio (None, None, None, 7 0           add_8[0][0]
__________________________________________________________________________________________________
block11_sepconv1 (SeparableConv (None, None, None, 7 536536      block11_sepconv1_act[0][0]
__________________________________________________________________________________________________
block11_sepconv1_bn (BatchNorma (None, None, None, 7 2912        block11_sepconv1[0][0]
__________________________________________________________________________________________________
block11_sepconv2_act (Activatio (None, None, None, 7 0           block11_sepconv1_bn[0][0]
__________________________________________________________________________________________________
block11_sepconv2 (SeparableConv (None, None, None, 7 536536      block11_sepconv2_act[0][0]
__________________________________________________________________________________________________
block11_sepconv2_bn (BatchNorma (None, None, None, 7 2912        block11_sepconv2[0][0]
__________________________________________________________________________________________________
block11_sepconv3_act (Activatio (None, None, None, 7 0           block11_sepconv2_bn[0][0]
__________________________________________________________________________________________________
block11_sepconv3 (SeparableConv (None, None, None, 7 536536      block11_sepconv3_act[0][0]
__________________________________________________________________________________________________
block11_sepconv3_bn (BatchNorma (None, None, None, 7 2912        block11_sepconv3[0][0]
__________________________________________________________________________________________________
add_9 (Add)                     (None, None, None, 7 0           block11_sepconv3_bn[0][0]
                                                                 add_8[0][0]
__________________________________________________________________________________________________
block12_sepconv1_act (Activatio (None, None, None, 7 0           add_9[0][0]
__________________________________________________________________________________________________
block12_sepconv1 (SeparableConv (None, None, None, 7 536536      block12_sepconv1_act[0][0]
__________________________________________________________________________________________________
block12_sepconv1_bn (BatchNorma (None, None, None, 7 2912        block12_sepconv1[0][0]
__________________________________________________________________________________________________
block12_sepconv2_act (Activatio (None, None, None, 7 0           block12_sepconv1_bn[0][0]
__________________________________________________________________________________________________
block12_sepconv2 (SeparableConv (None, None, None, 7 536536      block12_sepconv2_act[0][0]
__________________________________________________________________________________________________
block12_sepconv2_bn (BatchNorma (None, None, None, 7 2912        block12_sepconv2[0][0]
__________________________________________________________________________________________________
block12_sepconv3_act (Activatio (None, None, None, 7 0           block12_sepconv2_bn[0][0]
__________________________________________________________________________________________________
block12_sepconv3 (SeparableConv (None, None, None, 7 536536      block12_sepconv3_act[0][0]
__________________________________________________________________________________________________
block12_sepconv3_bn (BatchNorma (None, None, None, 7 2912        block12_sepconv3[0][0]
__________________________________________________________________________________________________
add_10 (Add)                    (None, None, None, 7 0           block12_sepconv3_bn[0][0]
                                                                 add_9[0][0]
__________________________________________________________________________________________________
block13_sepconv1_act (Activatio (None, None, None, 7 0           add_10[0][0]
__________________________________________________________________________________________________
block13_sepconv1 (SeparableConv (None, None, None, 7 536536      block13_sepconv1_act[0][0]
__________________________________________________________________________________________________
block13_sepconv1_bn (BatchNorma (None, None, None, 7 2912        block13_sepconv1[0][0]
__________________________________________________________________________________________________
block13_sepconv2_act (Activatio (None, None, None, 7 0           block13_sepconv1_bn[0][0]
__________________________________________________________________________________________________
block13_sepconv2 (SeparableConv (None, None, None, 1 752024      block13_sepconv2_act[0][0]
__________________________________________________________________________________________________
block13_sepconv2_bn (BatchNorma (None, None, None, 1 4096        block13_sepconv2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, None, None, 1 745472      add_10[0][0]
__________________________________________________________________________________________________
block13_pool (MaxPooling2D)     (None, None, None, 1 0           block13_sepconv2_bn[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, None, None, 1 4096        conv2d_3[0][0]
__________________________________________________________________________________________________
add_11 (Add)                    (None, None, None, 1 0           block13_pool[0][0]
                                                                 batch_normalization_3[0][0]
__________________________________________________________________________________________________
block14_sepconv1 (SeparableConv (None, None, None, 1 1582080     add_11[0][0]
__________________________________________________________________________________________________
block14_sepconv1_bn (BatchNorma (None, None, None, 1 6144        block14_sepconv1[0][0]
__________________________________________________________________________________________________
block14_sepconv1_act (Activatio (None, None, None, 1 0           block14_sepconv1_bn[0][0]
__________________________________________________________________________________________________
block14_sepconv2 (SeparableConv (None, None, None, 2 3159552     block14_sepconv1_act[0][0]
__________________________________________________________________________________________________
block14_sepconv2_bn (BatchNorma (None, None, None, 2 8192        block14_sepconv2[0][0]
__________________________________________________________________________________________________
block14_sepconv2_act (Activatio (None, None, None, 2 0           block14_sepconv2_bn[0][0]
__________________________________________________________________________________________________
global_average_pooling2d (Globa (None, 2048)         0           block14_sepconv2_act[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 1024)         2098176     global_average_pooling2d[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 8)            8200        dense[0][0]
==================================================================================================
Total params: 22,967,856
Trainable params: 22,913,328
Non-trainable params: 54,528
__________________________________________________________________________________________________
Configuration Start -------------------------
{'name': 'SGD', 'learning_rate': 0.01, 'decay': 1e-06, 'momentum': 0.9, 'nesterov': True}
Configuration End -------------------------
Train on 13920 samples, validate on 400 samples
Epoch 1/100
13920/13920 [==============================] - 18979s 1s/sample - loss: 0.2952 - accuracy: 0.8732 - precision: 0.6646 - recall: 0.2848 - auc: 0.8644 - val_loss: 0.3392 - val_accuracy: 0.8806 - val_precision: 0.5536 - val_recall: 0.2325 - val_auc: 0.7576
Epoch 2/100
13920/13920 [==============================] - 9005s 647ms/sample - loss: 0.2138 - accuracy: 0.9081 - precision: 0.7537 - recall: 0.5601 - auc: 0.9365 - val_loss: 0.2999 - val_accuracy: 0.8813 - val_precision: 0.5340 - val_recall: 0.3925 - val_auc: 0.8287
Epoch 3/100
13920/13920 [==============================] - 15331s 1s/sample - loss: 0.1758 - accuracy: 0.9256 - precision: 0.8033 - recall: 0.6564 - auc: 0.9586 - val_loss: 0.3153 - val_accuracy: 0.8766 - val_precision: 0.5071 - val_recall: 0.4450 - val_auc: 0.8406
Epoch 4/100
13920/13920 [==============================] - 8375s 602ms/sample - loss: 0.1402 - accuracy: 0.9417 - precision: 0.8494 - recall: 0.7352 - auc: 0.9746 - val_loss: 0.3376 - val_accuracy: 0.8797 - val_precision: 0.5198 - val_recall: 0.4925 - val_auc: 0.8420
Epoch 5/100
13920/13920 [==============================] - 8496s 610ms/sample - loss: 0.1026 - accuracy: 0.9603 - precision: 0.9019 - recall: 0.8200 - auc: 0.9866 - val_loss: 0.3077 - val_accuracy: 0.8919 - val_precision: 0.5746 - val_recall: 0.5200 - val_auc: 0.8604
Epoch 6/100
13920/13920 [==============================] - 9191s 660ms/sample - loss: 0.0688 - accuracy: 0.9739 - precision: 0.9378 - recall: 0.8818 - auc: 0.9944 - val_loss: 0.3686 - val_accuracy: 0.8819 - val_precision: 0.5293 - val_recall: 0.4975 - val_auc: 0.8567
Epoch 7/100
13920/13920 [==============================] - 9433s 678ms/sample - loss: 0.0446 - accuracy: 0.9842 - precision: 0.9654 - recall: 0.9260 - auc: 0.9977 - val_loss: 0.4668 - val_accuracy: 0.8750 - val_precision: 0.5000 - val_recall: 0.5075 - val_auc: 0.8319
Epoch 8/100
13920/13920 [==============================] - 11866s 852ms/sample - loss: 0.0310 - accuracy: 0.9891 - precision: 0.9771 - recall: 0.9485 - auc: 0.9989 - val_loss: 0.5094 - val_accuracy: 0.8778 - val_precision: 0.5112 - val_recall: 0.5150 - val_auc: 0.8296
Epoch 9/100
13920/13920 [==============================] - 9725s 699ms/sample - loss: 0.0245 - accuracy: 0.9917 - precision: 0.9813 - recall: 0.9619 - auc: 0.9993 - val_loss: 0.5168 - val_accuracy: 0.8784 - val_precision: 0.5144 - val_recall: 0.4925 - val_auc: 0.8238
Epoch 10/100
13920/13920 [==============================] - 8530s 613ms/sample - loss: 0.0201 - accuracy: 0.9934 - precision: 0.9838 - recall: 0.9713 - auc: 0.9996 - val_loss: 0.4636 - val_accuracy: 0.8875 - val_precision: 0.5532 - val_recall: 0.5200 - val_auc: 0.8396
Epoch 00010: early stopping
saving weights
plotting metrics
plotting accuracy
display the content of the model
400/1 - 41s - loss: 0.4298 - accuracy: 0.8875 - precision: 0.5532 - recall: 0.5200 - auc: 0.8396
loss :  0.4635940623283386
accuracy :  0.8875
precision :  0.5531915
recall :  0.52
auc :  0.8395691

plotting confusion matrix
Kappa score: 0.47214076246334313
F-1 score: 0.8875
AUC value: 0.8611830357142858
Final Score: 0.740274599392543
```

## Plots
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/training_results/11%20-%20xception_basic/plot1.png)
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/training_results/11%20-%20xception_basic/plot2.png)
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/training_results/11%20-%20xception_basic/plot3.png)
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/training_results/11%20-%20xception_basic/plot4.png)

## Ground Truth
https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/training_results/11%20-%20xception_basic/odir_ground_truth.csv

## Output Predictions
https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/training_results/11%20-%20xception_basic/odir_predictions.csv
