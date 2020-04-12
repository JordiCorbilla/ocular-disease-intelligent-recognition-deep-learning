# Ocular Disease Intelligent Recognition Through Deep Learning Architectures

This repository contains Jordi Corbilla's Msc dissertation: **Ocular Disease Intelligent recognition through deep learning architectures**, published by Universitat Oberta de Catalunya in 2020 [http://openaccess.uoc.edu/webapps/o2/handle/10609/113126]. The dissertation PDFs and the dissertation sources are licensed under the Creative Commons Attribution license, as described in the LICENSE file.

## Abstract
<div style="text-align: justify">Retinal pathologies are the most common cause of childhood blindness worldwide. Rapid and automatic detection of diseases is critical and urgent in reducing the ophthalmologist's workload. Ophthalmologists diagnose diseases based on pattern recognition through direct or indirect visualization of the eye and its surrounding structures. Dependence on the fundus of the eye and its analysis make the field of ophthalmology perfectly suited to benefit from deep learning algorithms. Each disease has different stages of severity that can be deduced by verifying the existence of specific lesions and each lesion is characterized by certain morphological features where several lesions of different pathologies have similar characteristics. We note that patients may be simultaneously affected by various pathologies, and consequently, the detection of eye diseases has a multi-label classification with a complex resolution principle.</div>
<p></p>
<div style="text-align: justify">Two deep learning solutions are being studied for the automatic detection of multiple eye diseases. The solutions chosen are due to their higher performance and final score in the ILSVRC challenge: GoogLeNet and VGGNet. First, we study the different characteristics of lesions and define the fundamental steps of data processing. We then identify the software and hardware needed to execute deep learning solutions. Finally, we investigate the principles of experimentation involved in evaluating the various methods, the public database used for the training and validation phases, and report the final detection accuracy with other important metrics.</div>

## Keywords:
Image classification, Deep learning, Retinography, Convolutional neural networks, Eye diseases, Medical imaging analysis.

## Pathologies:
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/images/pathologies.png)

According to 2010 World Health Organization data: There are more than 39 million blind people where 80% of them could have been prevented! This lack of prevention is especially true in developing countries where cataract is still the highest with 51% globally.

The current standard for the classification of diseases based on fundus photography, includes a manual estimation of injury locations  and an analysis of their severity, which requires a lot of time by the ophthalmologist, also incurring high costs in the healthcare system. Therefore, it would be important to have automated methods for performing the analysis.

Rapid and automatic detection of diseases is critical and urgent in reducing the ophthalmologist's workload.

## Deep learning architecture:
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/images/deeplearningdesign.png)

What is our methodology?

The dataset is first studied and its decomposition is placed correctly into different pathologies.

The images are then thoroughly analyzed and algorithms are generated for processing, which each network is capable of accepting. Here we get the ground truth and the final set of images to use.

A data augmentation module is also added to offset the imbalance found in the data set.

We generates data vectors that each model can consume and deep learning networks are trained through different experiments to provide image classification into 8 groups. The size of images and conversion blocks may vary in different applications. Convolutional layers try to extract relevant image features while reducing their dimensionality. The Sigmoid layer is responsible for the decision-making aspect of the network.
Finally, the different results from the experiments are generated and compared. This is the flowchart we created and describes in detail the different parts that make up this work.

## Training Details:
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/images/trainingdetails.png)

After introducing the models and performing hundreds of experiments with them, we can talk about the best experiment found for each model and their configuration.

For the **Inception** model, we used data augmentation, loaded the model with ImageNet weights, thus enabling transferred learning. Each model has two main components, one for the feature extraction and the other for the sorting. In this experiment, we enabled both components because the results obtained with this configuration were very satisfactory. As discussed earlier, the last layer has been modified to add a dense layer with a Sigmoid activation that allows us to calculate the loss for each of the 8 classes in the output. We also use a Stochastick Gradient Descent with a learning rate of 0.01. The loss function is of the binary cross entropy type due to the multi-tag configuration and we also add a patience feature that will stop the training if the validation loss does not decrease for 8 stages. The number of parameters we can train is of 23 million.

As far as the **VGG** model is concerned, we have noticed that transfer learning does better than training the model as a whole. Therefore, we loaded the weights of ImageNet and modified the last layer to consider the multi-label problem and only enabled the part of the classifier leaving us with only 32 thousand parameters to train. The only difference here with the Inception model is that the learning rate is 0.001 and the rest is configured in the same way.

## Model Comparison:
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/images/modelcomparison.png)

The final score of the models shows us a clear winner with 60% accuracy and a 55% of Recall with the Inception model. Giving you a final score of 76% (taking into account the mean value of the sum of the values of the Kappa coefficient of Cohen, F1-Score and AUC).
As for the VGG model, it is very close to the result of the Inception but has an accuracy of 57% with a recall of 36%, indicating that 36% of the returned values are correct with respect to the total number of images that are right.
For better visual representation, we can use the confusion matrix below.

## Confusion matrix:
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/images/ConfusionMatrix.png)

## Classification Output:
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/images/classificationoutput.png)

## Implementation Details:

### Works on:
[tensorflow-2.0](https://github.com/tensorflow/tensorflow) - use branch `master`

All the training images must be in JPEG format.

### Usage

#### Image Treatment Process

Place all the files in the following folders (Training and Validation images):

```cmd
c:\temp\ODIR-5K_Training_Dataset
c:\temp\ODIR-5K_Testing_Images
```

Create the following folders:

```cmd
c:\temp\ODIR-5K_Testing_Images_cropped
c:\temp\ODIR-5K_Testing_Images_treated
c:\temp\ODIR-5K_Training_Dataset_cropped
c:\temp\ODIR-5K_Training_Dataset_treated
```

run the following command to treat the training and validation images:

```cmd
python odir_image_crop_job.py
python odir_image_training_treatment_job.py
python odir_image_testing_treatment_job.py
```

#### Image to tf.Data conversion and .npy storage

run the following command to generate the dataset for training and validation:

```cmd
python odir_image_crop_job.py
```

```cmd
python.exe odir_runner.py
```

Put all the training images under dataset folder.

#### License
Apache License, Version 2.0
