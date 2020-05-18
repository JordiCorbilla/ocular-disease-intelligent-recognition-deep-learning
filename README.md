# Ocular Disease Intelligent Recognition Through Deep Learning Architectures

<p align="justify">This repository contains Jordi Corbilla's Msc dissertation: <br>
  - <b>Ocular Disease Intelligent recognition through deep learning architectures</b>, published by Universitat Oberta de Catalunya in 2020 [http://openaccess.uoc.edu/webapps/o2/handle/10609/113126]. 
<br>  The dissertation PDFs and the dissertation sources are licensed under the Creative Commons Attribution license, as described in the LICENSE file.</p>

## Abstract
<p align="justify">Retinal pathologies are the most common cause of childhood blindness worldwide. Rapid and automatic detection of diseases is critical and urgent in reducing the ophthalmologist's workload. Ophthalmologists diagnose diseases based on pattern recognition through direct or indirect visualization of the eye and its surrounding structures. Dependence on the fundus of the eye and its analysis make the field of ophthalmology perfectly suited to benefit from deep learning algorithms. Each disease has different stages of severity that can be deduced by verifying the existence of specific lesions and each lesion is characterized by certain morphological features where several lesions of different pathologies have similar characteristics. We note that patients may be simultaneously affected by various pathologies, and consequently, the detection of eye diseases has a multi-label classification with a complex resolution principle.</p>
<p></p>
<p align="justify">Two deep learning solutions are being studied for the automatic detection of multiple eye diseases. The solutions chosen are due to their higher performance and final score in the ILSVRC challenge: GoogLeNet and VGGNet. First, we study the different characteristics of lesions and define the fundamental steps of data processing. We then identify the software and hardware needed to execute deep learning solutions. Finally, we investigate the principles of experimentation involved in evaluating the various methods, the public database used for the training and validation phases, and report the final detection accuracy with other important metrics.</p>

## Keywords:
Image classification, Deep learning, Retinography, Convolutional neural networks, Eye diseases, Medical imaging analysis.

## Pathologies:
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/images/pathologies.png)

<p align="justify">According to 2010 World Health Organization data: There are more than 39 million blind people where 80% of them could have been prevented! This lack of prevention is especially true in developing countries where cataract is still the highest with 51% globally.</p>

<p align="justify">The current standard for the classification of diseases based on fundus photography, includes a manual estimation of injury locations  and an analysis of their severity, which requires a lot of time by the ophthalmologist, also incurring high costs in the healthcare system. Therefore, it would be important to have automated methods for performing the analysis.</p>

<p align="justify">Rapid and automatic detection of diseases is critical and urgent in reducing the ophthalmologist's workload.</p>

## Deep learning architecture:
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/images/deeplearningdesign.png)

**What is our methodology?**

The dataset is first studied and its decomposition is placed correctly into different pathologies.

<p align="justify">The images are then thoroughly analyzed and algorithms are generated for processing, which each network is capable of accepting. Here we get the ground truth and the final set of images to use.</p>

A data augmentation module is also added to offset the imbalance found in the data set.

<p align="justify">We generates data vectors that each model can consume and deep learning networks are trained through different experiments to provide image classification into 8 groups. The size of images and conversion blocks may vary in different applications. Convolutional layers try to extract relevant image features while reducing their dimensionality. The Sigmoid layer is responsible for the decision-making aspect of the network.</p>

<p align="justify">Finally, the different results from the experiments are generated and compared. This is the flowchart we created and describes in detail the different parts that make up this work.</p>

## Training Details:
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/images/trainingdetails.png)

<p align="justify">After introducing the models and performing hundreds of experiments with them, we can talk about the best experiment found for each model and their configuration.</p>

<p align="justify">For the <b>Inception</b> model, we used data augmentation, loaded the model with ImageNet weights, thus enabling transferred learning. Each model has two main components, one for the feature extraction and the other for the sorting. In this experiment, we enabled both components because the results obtained with this configuration were very satisfactory. As discussed earlier, the last layer has been modified to add a dense layer with a Sigmoid activation that allows us to calculate the loss for each of the 8 classes in the output. We also use a Stochastick Gradient Descent with a learning rate of 0.01. The loss function is of the binary cross entropy type due to the multi-tag configuration and we also add a patience feature that will stop the training if the validation loss does not decrease for 8 stages. The number of parameters we can train is of 23 million.</p>

<p align="justify">As far as the <b>VGG</b> model is concerned, we have noticed that transfer learning does better than training the model as a whole. Therefore, we loaded the weights of ImageNet and modified the last layer to consider the multi-label problem and only enabled the part of the classifier leaving us with only 32 thousand parameters to train. The only difference here with the Inception model is that the learning rate is 0.001 and the rest is configured in the same way.</p>

## Model Comparison:
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/images/modelcomparison.png)

<p align="justify">The final score of the models shows us a clear winner with 60% accuracy and a 55% of Recall with the Inception model. Giving you a final score of 76% (taking into account the mean value of the sum of the values of the Kappa coefficient of Cohen, F1-Score and AUC).</p>
<p align="justify">As for the VGG model, it is very close to the result of the Inception but has an accuracy of 57% with a recall of 36%, indicating that 36% of the returned values are correct with respect to the total number of images that are right.
For better visual representation, we can use the confusion matrix below.</p>

## Confusion matrix:
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/images/ConfusionMatrix.png)

<p align="justify">As we can see in these confusion matrices. Inception does a better job of sorting items on the diagonal of the array, indicating the correct classification. If we had a perfect matrix, we would have to see number 50 in each cell on the diagonal. Therefore we have classifications with 80% of successes and others like for example the hypertension named with a 5 where we have only been able to correctly classify 22%. We have more than 50% of correct classifications in each class except hypertension and other pathologies with 22% and 32% respectively. However, despite the increase in data (through data augmentation), there are still features that have not been learned by the model.</p>

<p align="justify">As for the VGG, we can see how the data is a bit more scattered but we also have different classifications on the diagonal. As for the minority hypertension class, we can also see that there was an issue here as it was unable to classify too many images in this category.</p>

## Classification Output:
![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/images/classificationoutput.png)

<p align="justify">Finally we can see the output that each model generates and where we can visually check the classification result towards its ground truth. With this work, all the code related to the training and validation of the data, as well as the inference check to validate the output of the models, are delivered in this repo.</p>

<p align="justify">We can see, then, that the two models have the same classification for the same image, but if we analyze in detail the response of each output we can see that it is quite different.</p>

## Conclusions:
- This project studies two deep learning models for the multiple classification of diseases.
- There is added complexity due to the multi-label and the initial data imbalance.
- We have seen that after the fine-tuning of the experiments we are able to obtain 60% accuracy on the validation set.
- The scenario is set for future applications, where the model could support the ophthalmologist during the capture of the fundus, and thus to classify pathologies faster.

## Implementation Details:

### Dataset:
The Dataset is part of the ODIR 2019 Grand Challenge. In order to use the data you need to register and download it from there: https://odir2019.grand-challenge.org/introduction/

### Works on Python 3.6:

[tensorflow-2.0](https://github.com/tensorflow/tensorflow) - use branch `master`

The full list of packages used can be seen below:
```
- tensorboard-2.0.0
- tensorflow-2.0.0
- tensorflow-estimator-2.0.1
- tensorflow-gpu-2.0
- matplotlib-3.1.1
- keras-applications-1.0.8
- keras-preprocessing-1.0.5
- opencv-python-4.1.1.26
- django-2.2.6
- image-1.5.27
- pillow-6.2.0
- sqlparse-0.3.0
- IPython-7.8.0
- keras-2.3.1
- scikit-learn-0.21.3
- pydot-1.4.1
- graphviz-0.13.2
- pylint-2.4.4
- imbalanced-learn-0.5.0
- seaborn-0.9.0
- scikit-image-0.16.2
- pandas-0.25.1
- numpy 1.17.2
- scipy-1.3.1
```
All the training images must be in JPEG format.

### Usage

#### 1) Image Treatment Process

Place all the files in the following folders (Training and Validation images):

```cmd
c:\temp\ODIR-5K_Training_Dataset
c:\temp\ODIR-5K_Testing_Images
```
The training images Dataset should contain 7000 images and the testing Dataset 1000 images. Below is a screenshot of the images in the training dataset folder:

![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/images/dataset.png)

Then, create the following folders:

```cmd
c:\temp\ODIR-5K_Testing_Images_cropped
c:\temp\ODIR-5K_Testing_Images_treated_128
c:\temp\ODIR-5K_Testing_Images_treated_224
c:\temp\ODIR-5K_Training_Dataset_cropped
c:\temp\ODIR-5K_Training_Dataset_treated_128
c:\temp\ODIR-5K_Training_Dataset_treated_224
c:\temp\ODIR-5K_Training_Dataset_augmented_128
c:\temp\ODIR-5K_Training_Dataset_augmented_224
```

run the following command to treat the training and validation images:

```cmd
//These two remove the black pixels
python odir_image_crop_job.py
python odir_image_testing_crop_job.py

//These two resize the images to 224 pixels
python odir_training_image_treatment_job.py
python odir_testing_image_treatment_job.py
```

The odir_image_crop_job.py job will treat all the Training Dataset images and remove the black area of the images so the images end up like in the image below (same job for the odir_image_testing_crop_job.py which will act upon the training images):

![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/images/cropped.png)

The second job will perform the resize and squaring functionality to 224 pixels x 224 pixels. The parameters image_width and keep_aspect_ratio variables can be edited in the python file to test different values/scenarios. This should give you images like the ones below:

![](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/images/squareimages.png)

#### 2) Data Augmentation (if you don't want to use this step you can skip it)

run the following command to generate the additional images:

```cmd
python.exe odir_data_augmentation_runner.py
```

This will generate the **odir_augmented.csv** file.

#### 3) Image to tf.Data conversion and .npy storage

Now that we have all the images. We need to translate them into a td.Data component so we can load them into our model. Run the following command to generate the dataset for training and validation:

```cmd
python.exe odir_patients_to_numpy.py
```

Note that any changes in the images will need a re-run of this script to rebuild the .npy files.

If you take a look at the arguments of the script you will see the following:

```python
image_width = 224
training_path = r'C:\temp\ODIR-5K_Training_Dataset_treated' + '_' + str(image_width)
testing_path = r'C:\temp\ODIR-5K_Testing_Images_treated' + '_' + str(image_width)
augmented_path = r'C:\temp\ODIR-5K_Training_Dataset_augmented' + '_' + str(image_width)
csv_file = r'ground_truth\odir.csv'
csv_augmented_file = r'ground_truth\odir_augmented.csv'
training_file = r'ground_truth\testing_default_value.csv'
```

**odir.csv** file contains the generated ground truth per eye. To generate the ground truth, you can take a look at odir_runner.py which contains the different procedures to generate the ground truth based on the ODIR-5K_Training_Annotations(Updated)_V2.xlsx file which is part of the provided file by ODIR.

**odir_augmented.csv** contains the generated ground truth per eye and sample of the data augmentation process generator. This makes things easier when trying to feed this into the model and compare the results.

**testing_default_value.csv** contains the vectors of the testing images.

#### 4) Run Inception

```cmd
python.exe odir_inception_training.py
```

#### 5) Run VGG

```cmd
-- Basic Run of the model
python.exe odir_vgg_training_basic.py
-- Sample output can be seen here: 
-- Enhanced Run of the model using Data Augmentation
python.exe odir_vgg_training.py
```

#### 6) Inception Inference

```cmd
python.exe odir_inception_testing_inference.py
```

#### 7) VGG Inference

```cmd
python.exe odir_vgg_testing_inference.py
```

#### License
Creative Commons Attribution-NonCommercial 4.0 International Public License
