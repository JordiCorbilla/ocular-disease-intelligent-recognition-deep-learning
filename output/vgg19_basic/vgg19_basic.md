## Execution Output
```
C:\Users\thund\Source\Repos\TFM-ODIR\models\image_classification>python odir_vgg19_training_basic.py
2020-06-06 08:21:17.582468: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_100.dll
2020-06-06 08:22:04.910744: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
2020-06-06 08:22:06.022405: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
name: GeForce GTX 960M major: 5 minor: 0 memoryClockRate(GHz): 1.176
pciBusID: 0000:01:00.0
2020-06-06 08:22:06.034396: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2020-06-06 08:22:06.046453: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-06-06 08:22:06.053775: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2020-06-06 08:22:06.093223: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
name: GeForce GTX 960M major: 5 minor: 0 memoryClockRate(GHz): 1.176
pciBusID: 0000:01:00.0
2020-06-06 08:22:06.105596: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2020-06-06 08:22:06.114867: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-06-06 08:22:10.818127: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-06-06 08:22:10.825222: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0
2020-06-06 08:22:10.830629: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N
2020-06-06 08:22:10.836479: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3040 MB memory) -> physical GPU (device: 0, name: GeForce GTX 960M, pci bus id: 0000:01:00.0, compute capability: 5.0)
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
dropout (Dropout)            (None, 4096)              0
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              16781312
_________________________________________________________________
dropout_1 (Dropout)          (None, 4096)              0
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 32776
=================================================================
Total params: 139,603,016
Trainable params: 32,776
Non-trainable params: 139,570,240
_________________________________________________________________
Train on 13920 samples, validate on 400 samples
Epoch 1/50
2020-06-06 08:26:23.827265: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_100.dll
2020-06-06 08:26:34.505201: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudnn64_7.dll
2020-06-06 08:26:37.331982: W tensorflow/stream_executor/cuda/redzone_allocator.cc:312] Internal: Invoking ptxas not supported on Windows
Relying on driver to perform ptx compilation. This message will be only logged once.
2020-06-06 08:26:38.265932: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 3.46GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-06-06 08:26:38.284293: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 3.04GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-06-06 08:26:38.514547: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.74GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-06-06 08:26:39.532449: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.74GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-06-06 08:26:39.548791: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.55GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-06-06 08:26:40.858094: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.34GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-06-06 08:26:42.109087: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.28GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-06-06 08:31:51.253153: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.54GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-06-06 08:31:52.828420: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.30GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
Epoch 1/50
13920/13920 [==============================] - 343s 25ms/sample - loss: 0.4579 - accuracy: 0.8400 - precision: 0.4511 - recall: 0.3868 - auc: 0.8022 - val_loss: 0.5919 - val_accuracy: 0.8144 - val_precision: 0.2775 - val_recall: 0.3025 - val_auc: 0.7022
Epoch 2/50
13920/13920 [==============================] - 285s 20ms/sample - loss: 0.4263 - accuracy: 0.8490 - precision: 0.4868 - recall: 0.4229 - auc: 0.8254 - val_loss: 0.4332 - val_accuracy: 0.8616 - val_precision: 0.3909 - val_recall: 0.1925 - val_auc: 0.7369
Epoch 3/50
13920/13920 [==============================] - 286s 21ms/sample - loss: 0.4211 - accuracy: 0.8511 - precision: 0.4947 - recall: 0.4281 - auc: 0.8291 - val_loss: 0.3957 - val_accuracy: 0.8656 - val_precision: 0.4257 - val_recall: 0.2150 - val_auc: 0.7476
Epoch 4/50
13920/13920 [==============================] - 286s 21ms/sample - loss: 0.4174 - accuracy: 0.8522 - precision: 0.4993 - recall: 0.4341 - auc: 0.8318 - val_loss: 0.4293 - val_accuracy: 0.8447 - val_precision: 0.3420 - val_recall: 0.2625 - val_auc: 0.7248
Epoch 5/50
13920/13920 [==============================] - 287s 21ms/sample - loss: 0.4193 - accuracy: 0.8529 - precision: 0.5018 - recall: 0.4382 - auc: 0.8325 - val_loss: 0.4078 - val_accuracy: 0.8566 - val_precision: 0.3958 - val_recall: 0.2800 - val_auc: 0.7659
Epoch 6/50
13920/13920 [==============================] - 291s 21ms/sample - loss: 0.4229 - accuracy: 0.8522 - precision: 0.4991 - recall: 0.4350 - auc: 0.8303 - val_loss: 0.4264 - val_accuracy: 0.8619 - val_precision: 0.4079 - val_recall: 0.2325 - val_auc: 0.7310
Epoch 7/50
13920/13920 [==============================] - 288s 21ms/sample - loss: 0.4170 - accuracy: 0.8539 - precision: 0.5057 - recall: 0.4442 - auc: 0.8344 - val_loss: 0.3909 - val_accuracy: 0.8669 - val_precision: 0.4449 - val_recall: 0.2625 - val_auc: 0.7592
Epoch 8/50
13920/13920 [==============================] - 290s 21ms/sample - loss: 0.4149 - accuracy: 0.8528 - precision: 0.5016 - recall: 0.4384 - auc: 0.8343 - val_loss: 0.4676 - val_accuracy: 0.8428 - val_precision: 0.3333 - val_recall: 0.2575 - val_auc: 0.7356
Epoch 9/50
13920/13920 [==============================] - 292s 21ms/sample - loss: 0.4159 - accuracy: 0.8539 - precision: 0.5058 - recall: 0.4473 - auc: 0.8351 - val_loss: 0.4138 - val_accuracy: 0.8609 - val_precision: 0.4170 - val_recall: 0.2825 - val_auc: 0.7647
Epoch 10/50
13920/13920 [==============================] - 292s 21ms/sample - loss: 0.4085 - accuracy: 0.8563 - precision: 0.5152 - recall: 0.4528 - auc: 0.8389 - val_loss: 0.3850 - val_accuracy: 0.8687 - val_precision: 0.4537 - val_recall: 0.2450 - val_auc: 0.7634
Epoch 11/50
13920/13920 [==============================] - 295s 21ms/sample - loss: 0.4157 - accuracy: 0.8527 - precision: 0.5010 - recall: 0.4432 - auc: 0.8355 - val_loss: 0.4740 - val_accuracy: 0.8559 - val_precision: 0.3822 - val_recall: 0.2475 - val_auc: 0.7107
Epoch 12/50
13920/13920 [==============================] - 288s 21ms/sample - loss: 0.4154 - accuracy: 0.8548 - precision: 0.5093 - recall: 0.4474 - auc: 0.8367 - val_loss: 0.4035 - val_accuracy: 0.8659 - val_precision: 0.4477 - val_recall: 0.3100 - val_auc: 0.7537
Epoch 13/50
13920/13920 [==============================] - 289s 21ms/sample - loss: 0.4094 - accuracy: 0.8553 - precision: 0.5113 - recall: 0.4502 - auc: 0.8407 - val_loss: 0.4392 - val_accuracy: 0.8422 - val_precision: 0.3312 - val_recall: 0.2575 - val_auc: 0.7362
Epoch 14/50
13920/13920 [==============================] - 289s 21ms/sample - loss: 0.4149 - accuracy: 0.8539 - precision: 0.5058 - recall: 0.4473 - auc: 0.8364 - val_loss: 0.4154 - val_accuracy: 0.8637 - val_precision: 0.4237 - val_recall: 0.2500 - val_auc: 0.7397
Epoch 15/50
13920/13920 [==============================] - 286s 21ms/sample - loss: 0.4120 - accuracy: 0.8554 - precision: 0.5116 - recall: 0.4479 - auc: 0.8386 - val_loss: 0.4742 - val_accuracy: 0.8366 - val_precision: 0.3394 - val_recall: 0.3250 - val_auc: 0.7186
Epoch 00015: early stopping
saving
plotting
400/1 - 8s - loss: 0.5083 - accuracy: 0.8366 - precision: 0.3394 - recall: 0.3250 - auc: 0.7186
loss :  0.4741806888580322
accuracy :  0.8365625
precision :  0.3394256
recall :  0.325
auc :  0.71860987

Kappa score: 0.23899599854492548
F-1 score: 0.8365625
AUC value: 0.7363544642857143
Final Score: 0.6039709876102132
```

## Plots
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/output/vgg19_basic/plot1.png)
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/output/vgg19_basic/plot2.png)
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/output/vgg19_basic/plot3.png)

## Ground Truth
https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/blob/master/output/vgg19_basic/odir_ground_truth.csv

## Output Predictions
https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/blob/master/output/vgg19_basic/odir_predictions.csv
