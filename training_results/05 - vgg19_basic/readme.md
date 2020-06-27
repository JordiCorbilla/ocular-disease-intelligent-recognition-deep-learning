# VGG 19 Basic

## Configuration

- 'Optimizer': 'SGD'
- 'learning_rate': 0.001
- 'decay': 1e-06
- 'momentum': 0.9
- 'nesterov': False
- 'Dropout': False

## Execution Output

```cmd
C:\Users\thund\Source\Repos\TFM-ODIR\models\image_classification>python odir_vgg19_training_basic.py
2020-06-06 12:44:25.257215: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_100.dll
2020-06-06 12:44:52.797051: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
2020-06-06 12:44:53.824237: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
name: GeForce GTX 960M major: 5 minor: 0 memoryClockRate(GHz): 1.176
pciBusID: 0000:01:00.0
2020-06-06 12:44:53.837853: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2020-06-06 12:44:53.850429: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-06-06 12:44:53.857566: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2020-06-06 12:44:53.882580: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
name: GeForce GTX 960M major: 5 minor: 0 memoryClockRate(GHz): 1.176
pciBusID: 0000:01:00.0
2020-06-06 12:44:53.895024: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2020-06-06 12:44:53.910233: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-06-06 12:44:58.973294: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-06-06 12:44:58.979089: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0
2020-06-06 12:44:58.983915: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N
2020-06-06 12:44:58.989862: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3040 MB memory) -> physical GPU (device: 0, name: GeForce GTX 960M, pci bus id: 0000:01:00.0, compute capability: 5.0)
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
conv2d_7 (Conv2D)            (None, 56, 56, 256)       590080
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 28, 28, 256)       0
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 28, 28, 512)       1180160
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 28, 28, 512)       2359808
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 28, 28, 512)       2359808
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 28, 28, 512)       2359808
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 14, 14, 512)       0
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 14, 14, 512)       2359808
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 14, 14, 512)       2359808
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 14, 14, 512)       2359808
_________________________________________________________________
conv2d_15 (Conv2D)           (None, 14, 14, 512)       2359808
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
Total params: 139,603,016
Trainable params: 32,776
Non-trainable params: 139,570,240
_________________________________________________________________
Train on 13920 samples, validate on 400 samples
Epoch 1/50
2020-06-06 12:47:36.886429: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_100.dll
2020-06-06 12:47:38.274585: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudnn64_7.dll
2020-06-06 12:47:44.984139: W tensorflow/stream_executor/cuda/redzone_allocator.cc:312] Internal: Invoking ptxas not supported on Windows
Relying on driver to perform ptx compilation. This message will be only logged once.
2020-06-06 12:47:45.908022: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 3.46GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-06-06 12:47:45.927262: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 3.04GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-06-06 12:47:46.155349: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.74GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-06-06 12:47:47.174034: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.74GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-06-06 12:47:47.196860: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.55GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-06-06 12:47:48.420218: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.34GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-06-06 12:47:49.677236: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.28GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
13888/13920 [============================>.] - ETA: 0s - loss: 0.3197 - accuracy: 0.8592 - precision: 0.5429 - recall: 0.2927 - auc: 0.84272020-06-06 12:52:20.123284: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.74GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-06-06 12:52:20.203306: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.54GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-06-06 12:52:21.875865: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.30GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
13920/13920 [==============================] - 296s 21ms/sample - loss: 0.3196 - accuracy: 0.8593 - precision: 0.5430 - recall: 0.2928 - auc: 0.8428 - val_loss: 0.4051 - val_accuracy: 0.8272 - val_precision: 0.2571 - val_recall: 0.2025 - val_auc: 0.7196
Epoch 2/50
13920/13920 [==============================] - 283s 20ms/sample - loss: 0.2736 - accuracy: 0.8784 - precision: 0.6490 - recall: 0.3831 - auc: 0.8884 - val_loss: 0.3602 - val_accuracy: 0.8616 - val_precision: 0.4183 - val_recall: 0.2750 - val_auc: 0.7461
Epoch 3/50
13920/13920 [==============================] - 284s 20ms/sample - loss: 0.2636 - accuracy: 0.8828 - precision: 0.6655 - recall: 0.4142 - auc: 0.8977 - val_loss: 0.3627 - val_accuracy: 0.8706 - val_precision: 0.4646 - val_recall: 0.2300 - val_auc: 0.7412
Epoch 4/50
13920/13920 [==============================] - 284s 20ms/sample - loss: 0.2578 - accuracy: 0.8860 - precision: 0.6784 - recall: 0.4330 - auc: 0.9030 - val_loss: 0.3608 - val_accuracy: 0.8612 - val_precision: 0.4167 - val_recall: 0.2750 - val_auc: 0.7523
Epoch 5/50
13920/13920 [==============================] - 283s 20ms/sample - loss: 0.2541 - accuracy: 0.8877 - precision: 0.6828 - recall: 0.4463 - auc: 0.9061 - val_loss: 0.3534 - val_accuracy: 0.8609 - val_precision: 0.3963 - val_recall: 0.2150 - val_auc: 0.7567
Epoch 6/50
13920/13920 [==============================] - 284s 20ms/sample - loss: 0.2497 - accuracy: 0.8905 - precision: 0.6944 - recall: 0.4614 - auc: 0.9099 - val_loss: 0.3526 - val_accuracy: 0.8675 - val_precision: 0.4478 - val_recall: 0.2575 - val_auc: 0.7605
Epoch 7/50
13920/13920 [==============================] - 286s 21ms/sample - loss: 0.2468 - accuracy: 0.8920 - precision: 0.7008 - recall: 0.4681 - auc: 0.9122 - val_loss: 0.3419 - val_accuracy: 0.8725 - val_precision: 0.4780 - val_recall: 0.2175 - val_auc: 0.7684
Epoch 8/50
13920/13920 [==============================] - 285s 20ms/sample - loss: 0.2447 - accuracy: 0.8928 - precision: 0.7020 - recall: 0.4757 - auc: 0.9140 - val_loss: 0.3359 - val_accuracy: 0.8728 - val_precision: 0.4831 - val_recall: 0.2500 - val_auc: 0.7760
Epoch 9/50
13920/13920 [==============================] - 285s 20ms/sample - loss: 0.2427 - accuracy: 0.8934 - precision: 0.7039 - recall: 0.4797 - auc: 0.9156 - val_loss: 0.3378 - val_accuracy: 0.8691 - val_precision: 0.4570 - val_recall: 0.2525 - val_auc: 0.7762
Epoch 10/50
13920/13920 [==============================] - 286s 21ms/sample - loss: 0.2409 - accuracy: 0.8939 - precision: 0.7061 - recall: 0.4821 - auc: 0.9170 - val_loss: 0.3238 - val_accuracy: 0.8816 - val_precision: 0.5621 - val_recall: 0.2375 - val_auc: 0.7859
Epoch 11/50
13920/13920 [==============================] - 284s 20ms/sample - loss: 0.2396 - accuracy: 0.8948 - precision: 0.7072 - recall: 0.4903 - auc: 0.9180 - val_loss: 0.3189 - val_accuracy: 0.8803 - val_precision: 0.5470 - val_recall: 0.2475 - val_auc: 0.7909
Epoch 12/50
13920/13920 [==============================] - 290s 21ms/sample - loss: 0.2377 - accuracy: 0.8962 - precision: 0.7157 - recall: 0.4923 - auc: 0.9194 - val_loss: 0.3302 - val_accuracy: 0.8744 - val_precision: 0.4956 - val_recall: 0.2800 - val_auc: 0.7835
Epoch 13/50
13920/13920 [==============================] - 285s 21ms/sample - loss: 0.2362 - accuracy: 0.8969 - precision: 0.7155 - recall: 0.4999 - auc: 0.9206 - val_loss: 0.3531 - val_accuracy: 0.8681 - val_precision: 0.4522 - val_recall: 0.2600 - val_auc: 0.7637
Epoch 14/50
13920/13920 [==============================] - 287s 21ms/sample - loss: 0.2353 - accuracy: 0.8968 - precision: 0.7161 - recall: 0.4984 - auc: 0.9214 - val_loss: 0.3423 - val_accuracy: 0.8634 - val_precision: 0.4297 - val_recall: 0.2825 - val_auc: 0.7784
Epoch 15/50
13920/13920 [==============================] - 284s 20ms/sample - loss: 0.2333 - accuracy: 0.8982 - precision: 0.7215 - recall: 0.5048 - auc: 0.9228 - val_loss: 0.3434 - val_accuracy: 0.8656 - val_precision: 0.4370 - val_recall: 0.2600 - val_auc: 0.7784
Epoch 16/50
13920/13920 [==============================] - 288s 21ms/sample - loss: 0.2320 - accuracy: 0.8988 - precision: 0.7241 - recall: 0.5074 - auc: 0.9240 - val_loss: 0.3285 - val_accuracy: 0.8747 - val_precision: 0.4975 - val_recall: 0.2525 - val_auc: 0.7839
Epoch 00016: early stopping
saving
plotting
400/1 - 7s - loss: 0.3379 - accuracy: 0.8747 - precision: 0.4975 - recall: 0.2525 - auc: 0.7839
loss :  0.32848785161972044
accuracy :  0.8746875
precision :  0.49753696
recall :  0.2525
auc :  0.78385717

Kappa score: 0.2738795835219556
F-1 score: 0.8746875
AUC value: 0.7862857142857143
Final Score: 0.6449509326025565
```

## Plots

![plot1](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/training_results/05%20-%20vgg19_basic/plot1.png)
![plot2](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/training_results/05%20-%20vgg19_basic/plot2.png)
![plot3](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/training_results/05%20-%20vgg19_basic/plot3.png)
![plot4](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/training_results/05%20-%20vgg19_basic/plot4.png)

## Ground Truth

<https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/blob/master/training_results/05%20-%20vgg19_basic/odir_ground_truth.csv>

## Output Predictions

<https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/blob/master/training_results/05%20-%20vgg19_basic/odir_predictions.csv>


