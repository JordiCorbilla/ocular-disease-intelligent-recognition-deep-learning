## Execution Output
```
C:\Users\thund\Source\Repos\TFM-ODIR\models\image_classification>python odir_vgg_training_basic.py
2020-05-18 08:14:36.648073: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_100.dll
2020-05-18 08:15:12.406893: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
2020-05-18 08:15:13.692591: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
name: GeForce GTX 960M major: 5 minor: 0 memoryClockRate(GHz): 1.176
pciBusID: 0000:01:00.0
2020-05-18 08:15:13.705684: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2020-05-18 08:15:13.719335: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-05-18 08:15:13.730287: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2020-05-18 08:15:13.770959: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
name: GeForce GTX 960M major: 5 minor: 0 memoryClockRate(GHz): 1.176
pciBusID: 0000:01:00.0
2020-05-18 08:15:13.791001: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2020-05-18 08:15:13.826754: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-05-18 08:15:19.394016: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-18 08:15:19.402512: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0
2020-05-18 08:15:19.407801: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N
2020-05-18 08:15:19.416423: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3040 MB memory) -> physical GPU (device: 0, name: GeForce GTX 960M, pci bus id: 0000:01:00.0, compute capability: 5.0)
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 224, 224, 64)      1792
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 224, 224, 64)      36928
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 112, 112, 64)      0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 112, 112, 128)     73856
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 112, 112, 128)     147584
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 56, 56, 128)       0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 56, 56, 256)       295168
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 56, 56, 256)       590080
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 56, 56, 256)       590080
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 28, 28, 256)       0
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 28, 28, 512)       1180160
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 28, 28, 512)       2359808
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 28, 28, 512)       2359808
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 14, 14, 512)       0
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 14, 14, 512)       2359808
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 14, 14, 512)       2359808
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 14, 14, 512)       2359808
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 7, 7, 512)         0
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0
_________________________________________________________________
dense (Dense)                (None, 4096)              102764544
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              16781312
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 32776
=================================================================
Total params: 134,293,320
Trainable params: 32,776
Non-trainable params: 134,260,544
_________________________________________________________________
Train on 13920 samples, validate on 400 samples
Epoch 1/50
2020-05-18 08:20:02.781542: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_100.dll
2020-05-18 08:20:04.440177: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudnn64_7.dll
2020-05-18 08:20:13.491339: W tensorflow/stream_executor/cuda/redzone_allocator.cc:312] Internal: Invoking ptxas not supported on Windows
Relying on driver to perform ptx compilation. This message will be only logged once.
2020-05-18 08:20:15.727714: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 3.46GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-05-18 08:20:15.744891: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 3.04GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-05-18 08:20:15.975317: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.74GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-05-18 08:20:17.113921: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.74GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-05-18 08:20:19.285476: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.55GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-05-18 08:20:20.540206: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.34GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-05-18 08:20:21.755857: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.28GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
13888/13920 [============================>.] - ETA: 0s - loss: 0.3099 - accuracy: 0.8613 - precision: 0.5611 - recall: 0.2769 - auc: 0.85052020-05-18 08:25:27.127909: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.74GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-05-18 08:25:27.153712: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.54GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-05-18 08:25:28.144879: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.30GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
13920/13920 [==============================] - 334s 24ms/sample - loss: 0.3099 - accuracy: 0.8614 - precision: 0.5616 - recall: 0.2770 - auc: 0.8505 - val_loss: 0.3754 - val_accuracy: 0.8609 - val_precision: 0.3669 - val_recall: 0.1550 - val_auc: 0.7083
Epoch 2/50
13920/13920 [==============================] - 254s 18ms/sample - loss: 0.2717 - accuracy: 0.8773 - precision: 0.6432 - recall: 0.3781 - auc: 0.8902 - val_loss: 0.3433 - val_accuracy: 0.8641 - val_precision: 0.4112 - val_recall: 0.2025 - val_auc: 0.7538
Epoch 3/50
13920/13920 [==============================] - 311s 22ms/sample - loss: 0.2626 - accuracy: 0.8827 - precision: 0.6640 - recall: 0.4159 - auc: 0.8985 - val_loss: 0.3430 - val_accuracy: 0.8644 - val_precision: 0.4234 - val_recall: 0.2350 - val_auc: 0.7614
Epoch 4/50
13920/13920 [==============================] - 274s 20ms/sample - loss: 0.2575 - accuracy: 0.8849 - precision: 0.6709 - recall: 0.4327 - auc: 0.9029 - val_loss: 0.3340 - val_accuracy: 0.8719 - val_precision: 0.4742 - val_recall: 0.2300 - val_auc: 0.7693
Epoch 5/50
13920/13920 [==============================] - 272s 20ms/sample - loss: 0.2533 - accuracy: 0.8869 - precision: 0.6785 - recall: 0.4445 - auc: 0.9065 - val_loss: 0.3420 - val_accuracy: 0.8659 - val_precision: 0.4422 - val_recall: 0.2775 - val_auc: 0.7711
Epoch 6/50
13920/13920 [==============================] - 309s 22ms/sample - loss: 0.2501 - accuracy: 0.8890 - precision: 0.6864 - recall: 0.4563 - auc: 0.9093 - val_loss: 0.3226 - val_accuracy: 0.8728 - val_precision: 0.4833 - val_recall: 0.2525 - val_auc: 0.7876
Epoch 7/50
13920/13920 [==============================] - 353s 25ms/sample - loss: 0.2474 - accuracy: 0.8899 - precision: 0.6891 - recall: 0.4629 - auc: 0.9114 - val_loss: 0.3344 - val_accuracy: 0.8719 - val_precision: 0.4793 - val_recall: 0.2900 - val_auc: 0.7789
Epoch 8/50
13920/13920 [==============================] - 259s 19ms/sample - loss: 0.2451 - accuracy: 0.8918 - precision: 0.6968 - recall: 0.4731 - auc: 0.9133 - val_loss: 0.3319 - val_accuracy: 0.8719 - val_precision: 0.4784 - val_recall: 0.2775 - val_auc: 0.7818
Epoch 9/50
13920/13920 [==============================] - 272s 20ms/sample - loss: 0.2431 - accuracy: 0.8926 - precision: 0.7026 - recall: 0.4720 - auc: 0.9150 - val_loss: 0.3399 - val_accuracy: 0.8666 - val_precision: 0.4466 - val_recall: 0.2825 - val_auc: 0.7777
Epoch 10/50
13920/13920 [==============================] - 256s 18ms/sample - loss: 0.2411 - accuracy: 0.8943 - precision: 0.7078 - recall: 0.4837 - auc: 0.9165 - val_loss: 0.3329 - val_accuracy: 0.8731 - val_precision: 0.4870 - val_recall: 0.2800 - val_auc: 0.7828
Epoch 11/50
13920/13920 [==============================] - 270s 19ms/sample - loss: 0.2396 - accuracy: 0.8940 - precision: 0.7070 - recall: 0.4817 - auc: 0.9177 - val_loss: 0.3321 - val_accuracy: 0.8716 - val_precision: 0.4776 - val_recall: 0.2925 - val_auc: 0.7842
Epoch 12/50
13920/13920 [==============================] - 267s 19ms/sample - loss: 0.2378 - accuracy: 0.8955 - precision: 0.7112 - recall: 0.4920 - auc: 0.9192 - val_loss: 0.3290 - val_accuracy: 0.8794 - val_precision: 0.5343 - val_recall: 0.2725 - val_auc: 0.7836
Epoch 13/50
13920/13920 [==============================] - 328s 24ms/sample - loss: 0.2367 - accuracy: 0.8956 - precision: 0.7099 - recall: 0.4950 - auc: 0.9200 - val_loss: 0.3309 - val_accuracy: 0.8734 - val_precision: 0.4885 - val_recall: 0.2650 - val_auc: 0.7837
Epoch 14/50
13920/13920 [==============================] - 251s 18ms/sample - loss: 0.2352 - accuracy: 0.8966 - precision: 0.7147 - recall: 0.4984 - auc: 0.9213 - val_loss: 0.3241 - val_accuracy: 0.8775 - val_precision: 0.5189 - val_recall: 0.2750 - val_auc: 0.7894
Epoch 00014: early stopping
saving
plotting
400/1 - 6s - loss: 0.3398 - accuracy: 0.8775 - precision: 0.5189 - recall: 0.2750 - auc: 0.7894
loss :  0.32405226230621337
accuracy :  0.8775
precision :  0.5188679
recall :  0.275
auc :  0.7893544

Kappa score: 0.2987477638640429
F-1 score: 0.8775
AUC value: 0.7914375
Final Score: 0.6558950879546809
```

## Plots
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/output/vgg_basic/plot1.png)
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/output/vgg_basic/plot2.png)
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/output/vgg_basic/plot3.png)

## Ground Truth
https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/blob/master/output/vgg_basic/odir_ground_truth.csv

## Output Predictions
https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/blob/master/output/vgg_basic/odir_predictions.csv